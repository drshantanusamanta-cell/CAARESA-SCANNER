"""
CAARESA engine — data fetching + metric computation for Indian mutual funds.

Data sources (all public, no login, no upload):
  * AMFI NAVAll.txt  -> full scheme master with SEBI category, plan & option
  * api.mfapi.in      -> full daily NAV history per scheme code
  * Kuvera public API (via mf.captnemo.in ISIN redirect) -> expense ratio, AUM

CAARESA = Category, Age, AUM, Return, Expense ratio, Sortino, Alpha
(+ Std-Dev and Max-Drawdown as tie-breakers).
"""
from __future__ import annotations

import io
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

AMFI_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"
MFAPI_URL = "https://api.mfapi.in/mf/{code}"
KUVERA_URL = "https://mf.captnemo.in/kuvera/{isin}"

HEADERS = {"User-Agent": "Mozilla/5.0 (CAARESA-Screener)"}
TRADING_DAYS = 252

# ---------------------------------------------------------------- categories
# (group, canonical name, regex on the AMFI sub-category text)
CATEGORY_RULES = [
    ("Equity", "Large Cap", r"^large ?cap"),
    ("Equity", "Large & Mid Cap", r"large ?& ?mid ?cap"),
    ("Equity", "Mid Cap", r"^mid ?cap"),
    ("Equity", "Small Cap", r"^small ?cap"),
    ("Equity", "Multi Cap", r"^multi ?cap"),
    ("Equity", "Flexi Cap", r"^flexi ?cap"),
    ("Equity", "Focused", r"^focused"),
    ("Equity", "Value", r"^value"),
    ("Equity", "Contra", r"^contra"),
    ("Equity", "Dividend Yield", r"^dividend yield"),
    ("Equity", "ELSS (Tax Saver)", r"^elss"),
    ("Equity", "Sectoral / Thematic", r"sectoral|thematic"),
    # SEBI's 2025-26 debt renaming ("... Term Fund") is merged with the older "... Duration" names
    ("Debt", "Overnight", r"^overnight"),
    ("Debt", "Liquid", r"^liquid"),
    ("Debt", "Low Duration / Ultra Short to Short Term", r"^low duration|^ultra short to short"),
    ("Debt", "Ultra Short Duration / Term", r"^ultra short"),
    ("Debt", "Money Market", r"^money market"),
    ("Debt", "Short Duration / Term", r"^short (duration|term)"),
    ("Debt", "Medium to Long Duration / Term", r"^medium to long"),
    ("Debt", "Medium Duration / Term", r"^medium (duration|term)"),
    ("Debt", "Long Duration / Term", r"^long (duration|term)"),
    ("Debt", "Dynamic Bond / Term", r"^dynamic (bond|term)"),
    ("Debt", "Corporate Bond", r"^corporate bond"),
    ("Debt", "Credit Risk", r"^credit risk"),
    ("Debt", "Banking and PSU", r"^banking and psu"),
    ("Debt", "Gilt (10Y Constant Maturity)", r"10.?y(ea)?r|constant"),
    ("Debt", "Gilt", r"^gilt"),
    ("Debt", "Floater / Floating Rate", r"^float"),
    ("Hybrid", "Conservative Hybrid", r"^conservative hybrid"),
    ("Hybrid", "Balanced Hybrid", r"^balanced hybrid"),
    ("Hybrid", "Aggressive Hybrid", r"^aggressive hybrid"),
    ("Hybrid", "Balanced Advantage / Dynamic Asset Allocation",
     r"balanced advantage|dynamic asset allocation"),
    ("Hybrid", "Multi Asset Allocation", r"^multi asset"),
    ("Hybrid", "Arbitrage", r"^arbitrage"),
    ("Hybrid", "Equity Savings", r"^equity savings"),
    ("Solution Oriented", "Retirement", r"retirement"),
    ("Solution Oriented", "Children's", r"child"),
]

GROUP_PREFIX = {
    "equity scheme": "Equity",
    "debt scheme": "Debt",
    "income/debt oriented scheme": "Debt",
    "hybrid scheme": "Hybrid",
    "solution oriented scheme": "Solution Oriented",
}

PASSIVE_NAME = re.compile(r"\b(?:index|etf|fof|fund of funds?|nifty|sensex|bse \d+)\b", re.I)

# Benchmark proxies: long-history Direct-Growth index funds (TRI proxy, net of tiny TER)
BENCH_PROXIES = {
    "Nifty 50": {"120716": 1.0},          # UTI Nifty 50 Index Fund - Direct
    "Nifty 500": {"147625": 1.0},         # Motilal Oswal Nifty 500 Index - Direct
    "Nifty Midcap 150": {"147622": 1.0},  # Motilal Oswal Nifty Midcap 150 Index - Direct
    "Nifty Smallcap 250": {"147623": 1.0},# Motilal Oswal Nifty Smallcap 250 Index - Direct
    "Nifty LargeMidcap 250 (50:50 blend)": {"120716": 0.5, "147622": 0.5},
    "Category peer average": None,
}

AUTO_BENCH = {
    "Large Cap": "Nifty 50",
    "Large & Mid Cap": "Nifty LargeMidcap 250 (50:50 blend)",
    "Mid Cap": "Nifty Midcap 150",
    "Small Cap": "Nifty Smallcap 250",
    "Multi Cap": "Nifty 500",
    "Flexi Cap": "Nifty 500",
    "Focused": "Nifty 500",
    "Value": "Nifty 500",
    "Contra": "Nifty 500",
    "Dividend Yield": "Nifty 500",
    "ELSS (Tax Saver)": "Nifty 500",
}


def auto_benchmark(category: str) -> str:
    return AUTO_BENCH.get(category, "Category peer average")


def _classify(header: str):
    """'Equity Schemes - Flexi Cap Fund' -> ('Equity', 'Flexi Cap')"""
    if " - " not in header:
        return None
    grp_raw, sub = header.split(" - ", 1)
    grp_key = re.sub(r"schemes?", "scheme", grp_raw.lower()).replace("*", "").strip()
    group = GROUP_PREFIX.get(grp_key)
    if group is None:
        return None
    sub_l = sub.lower().strip()
    for g, name, rx in CATEGORY_RULES:
        if g == group and re.search(rx, sub_l):
            return group, name
    return None


# ---------------------------------------------------------------- HTTP
def _get(url: str, tries: int = 3, timeout: int = 30):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:  # noqa: PERF203
            last = str(e)
        time.sleep(0.8 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


# ---------------------------------------------------------------- AMFI master
def load_scheme_master() -> pd.DataFrame:
    """All open-ended, actively managed, Direct-Growth schemes with SEBI category."""
    text = _get(AMFI_URL, timeout=60).text
    rows, cur = [], None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^Open Ended Schemes\((.*)\)$", line)
        if m:
            cur = _classify(m.group(1))
            continue
        if line.startswith(("Close Ended", "Interval Fund")):
            cur = None
            continue
        if cur is None or ";" not in line:
            continue
        p = line.split(";")
        if len(p) < 8 or not p[0].isdigit():
            continue
        code, isin, _, name, plan, option, nav, date = p[:8]
        rows.append(dict(code=code, isin=isin.strip(), name=name.strip(),
                         plan=plan.strip(), option=option.strip(),
                         group=cur[0], category=cur[1], last_nav_date=date))
    df = pd.DataFrame(rows)
    df = df[df["plan"].str.contains("direct", case=False)
            & df["option"].str.contains("growth", case=False)
            & ~df["option"].str.contains("idcw|dividend|bonus", case=False)]
    df = df[~df["name"].str.contains(PASSIVE_NAME)]           # actively managed only
    df = df[~df["name"].str.contains(r"segregated", case=False)]
    df = df.drop_duplicates("isin").drop_duplicates("code").reset_index(drop=True)
    return df


# ---------------------------------------------------------------- per-fund data
def fetch_nav(code: str) -> pd.Series:
    js = _get(MFAPI_URL.format(code=code)).json()
    data = js.get("data") or []
    if not data:
        return pd.Series(dtype=float)
    s = pd.Series(
        pd.to_numeric([d["nav"] for d in data], errors="coerce"),
        index=pd.to_datetime([d["date"] for d in data], format="%d-%m-%Y"),
    )
    s = s[s > 0].sort_index()
    return s[~s.index.duplicated(keep="last")]


def fetch_kuvera(isin: str) -> dict:
    try:
        d = _get(KUVERA_URL.format(isin=isin), tries=2).json()
        d = d[0] if isinstance(d, list) else d
        aum = d.get("aum")
        return dict(
            expense_ratio=float(d["expense_ratio"]) if d.get("expense_ratio") is not None else np.nan,
            er_date=d.get("expense_ratio_date"),
            aum_cr=float(aum) / 10.0 if aum not in (None, "") else np.nan,  # Kuvera AUM is in ₹10-lakh units
            fund_manager=d.get("fund_manager"),
            start_date=d.get("start_date"),
            kuvera_category=d.get("fund_category"),
        )
    except Exception:  # noqa: BLE001
        return dict(expense_ratio=np.nan, er_date=None, aum_cr=np.nan,
                    fund_manager=None, start_date=None, kuvera_category=None)


def fetch_many(funds: pd.DataFrame, workers: int = 12, progress=None):
    """Parallel fetch of NAV history + Kuvera facts for every fund in `funds`."""
    navs, facts = {}, {}
    jobs = [(r.code, r.isin) for r in funds.itertuples()]

    def work(job):
        code, isin = job
        try:
            nav = fetch_nav(code)
        except Exception:  # noqa: BLE001
            nav = pd.Series(dtype=float)
        return code, nav, fetch_kuvera(isin)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for code, nav, fk in ex.map(work, jobs):
            navs[code], facts[code] = nav, fk
            done += 1
            if progress:
                progress(done / max(len(jobs), 1))
    return navs, facts


def build_benchmark(name: str, navs: dict[str, pd.Series], eligible_codes: list[str]) -> pd.Series:
    """Return a daily *return* series for the chosen benchmark."""
    spec = BENCH_PROXIES.get(name)
    if spec is None:  # equal-weight category peer average
        rets = pd.concat({c: navs[c].pct_change() for c in eligible_codes if len(navs.get(c, [])) > 1},
                         axis=1)
        return rets.mean(axis=1, skipna=True).dropna()
    parts = []
    for code, w in spec.items():
        parts.append(fetch_nav(code).pct_change() * w)
    return pd.concat(parts, axis=1).dropna().sum(axis=1)


# ---------------------------------------------------------------- metrics
def _nav_on_or_before(s: pd.Series, d: pd.Timestamp):
    sub = s.loc[:d]
    return (sub.index[-1], sub.iloc[-1]) if len(sub) else (None, np.nan)


def cagr(s: pd.Series, years: int, asof: pd.Timestamp) -> float:
    start = asof - pd.DateOffset(years=years)
    if s.index[0] > start + pd.Timedelta(days=7):
        return np.nan
    d0, v0 = _nav_on_or_before(s, start)
    if d0 is None:
        d0, v0 = s.index[0], s.iloc[0]
    v1 = s.loc[:asof].iloc[-1]
    yrs = (asof - d0).days / 365.25
    return ((v1 / v0) ** (1 / yrs) - 1) * 100


def _xirr(cfs: list[float], dates: list[pd.Timestamp]) -> float:
    t = np.array([(d - dates[0]).days / 365.0 for d in dates])
    c = np.array(cfs)

    def npv(r):
        return np.sum(c / (1 + r) ** t)

    lo, hi = -0.95, 5.0
    flo, fhi = npv(lo), npv(hi)
    if np.sign(flo) == np.sign(fhi):
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = npv(mid)
        if abs(fm) < 1e-10:
            break
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi = mid
    return mid * 100


def sip_xirr(s: pd.Series, years: int, asof: pd.Timestamp) -> float:
    n = years * 12
    first = asof - pd.DateOffset(months=n)
    if s.index[0] > first + pd.Timedelta(days=7):
        return np.nan
    units, cfs, dts = 0.0, [], []
    for k in range(n, 0, -1):
        d = asof - pd.DateOffset(months=k)
        after = s.loc[d:]
        if not len(after):
            continue
        units += 1.0 / after.iloc[0]
        cfs.append(-1.0)
        dts.append(after.index[0])
    end_val = units * s.loc[:asof].iloc[-1]
    cfs.append(end_val)
    dts.append(asof)
    return _xirr(cfs, dts)


def max_drawdown(s: pd.Series) -> float:
    if len(s) < 2:
        return np.nan
    return float((s / s.cummax() - 1).min() * 100)


@dataclass
class Params:
    rf: float = 6.5               # risk-free (FD) rate, % p.a.
    risk_years: int = 3           # window for Std-Dev / Sortino / Alpha
    min_age: float = 5.0
    min_aum: float = 500.0        # ₹ crore
    max_er: float = 1.0           # %
    min_sortino: float = 1.0
    min_alpha: float = 1.0        # % p.a.
    return_pctile: float = 75.0   # top-quartile threshold
    min_return_hits: int = 4      # how many of 4 return metrics must be top-quartile
    allow_missing_facts: bool = False


def compute_metrics(funds: pd.DataFrame, navs: dict, facts: dict,
                    bench_ret: pd.Series, p: Params) -> pd.DataFrame:
    rf_d = (1 + p.rf / 100) ** (1 / TRADING_DAYS) - 1
    out = []
    for r in funds.itertuples():
        s = navs.get(r.code, pd.Series(dtype=float))
        f = facts.get(r.code, {})
        row = dict(code=r.code, isin=r.isin, name=r.name, category=r.category,
                   expense_ratio=f.get("expense_ratio", np.nan),
                   aum_cr=f.get("aum_cr", np.nan), fund_manager=f.get("fund_manager"))
        if len(s) < 30:
            row["age_yrs"] = 0.0
            out.append(row)
            continue
        asof = s.index[-1]
        # stale schemes (merged/wound-up) are dropped
        row["nav_date"] = asof.date()
        row["stale"] = (pd.Timestamp.today() - asof).days > 15
        start_dates = [s.index[0]]
        if f.get("start_date"):
            try:
                start_dates.append(pd.Timestamp(f["start_date"]))
            except Exception:  # noqa: BLE001
                pass
        row["age_yrs"] = (asof - min(start_dates)).days / 365.25

        row["ret_3y"] = cagr(s, 3, asof)
        row["ret_5y"] = cagr(s, 5, asof)
        row["sip_3y"] = sip_xirr(s, 3, asof)
        row["sip_5y"] = sip_xirr(s, 5, asof)
        row["mdd_5y"] = max_drawdown(s.loc[asof - pd.DateOffset(years=5):])

        # ---- risk window
        w0 = asof - pd.DateOffset(years=p.risk_years)
        fr = s.pct_change().loc[w0:].dropna()
        df = pd.concat({"f": fr, "b": bench_ret.loc[w0:]}, axis=1).dropna()
        if len(df) > 120:
            ex_f, ex_b = df["f"] - rf_d, df["b"] - rf_d
            row["std_dev"] = df["f"].std() * np.sqrt(TRADING_DAYS) * 100
            downside = np.minimum(ex_f, 0)
            dd = np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS)
            ann_ret = (1 + df["f"]).prod() ** (TRADING_DAYS / len(df)) - 1
            row["sortino"] = (ann_ret - p.rf / 100) / dd if dd > 0 else np.nan
            beta = np.cov(ex_f, ex_b)[0, 1] / np.var(ex_b, ddof=1)
            row["beta"] = beta
            row["alpha"] = (ex_f.mean() - beta * ex_b.mean()) * TRADING_DAYS * 100
            # exit ("divorce") check: alpha in each of the last two 1-yr windows
            alphas = []
            for k in (1, 2):
                a0, a1 = asof - pd.DateOffset(years=k), asof - pd.DateOffset(years=k - 1)
                sub = df.loc[a0:a1]
                if len(sub) > 120:
                    xf, xb = sub["f"] - rf_d, sub["b"] - rf_d
                    bt = np.cov(xf, xb)[0, 1] / np.var(xb, ddof=1)
                    alphas.append((xf.mean() - bt * xb.mean()) * TRADING_DAYS * 100)
            row["alpha_y1"] = alphas[0] if len(alphas) > 0 else np.nan
            row["alpha_y2"] = alphas[1] if len(alphas) > 1 else np.nan
            bnav = (1 + bench_ret.loc[asof - pd.DateOffset(years=5):]).cumprod()
            row["bench_mdd_5y"] = max_drawdown(bnav)
        out.append(row)
    return pd.DataFrame(out)


RET_COLS = ["ret_3y", "ret_5y", "sip_3y", "sip_5y"]


def apply_caaresa(m: pd.DataFrame, p: Params) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    m = m.copy()
    if "stale" in m:
        m = m[m["stale"] != True]  # noqa: E712
    funnel = [("Active Direct-Growth funds in category", len(m))]

    m["G_age"] = m["age_yrs"] >= p.min_age
    miss_ok = p.allow_missing_facts
    m["G_aum"] = (m["aum_cr"] > p.min_aum) | (miss_ok & m["aum_cr"].isna())
    m["G_er"] = (m["expense_ratio"] < p.max_er) | (miss_ok & m["expense_ratio"].isna())

    # Return consistency is judged against age-eligible peers only
    peers = m["G_age"]
    for c in RET_COLS:
        m[c + "_pct"] = np.nan
        if c in m:
            m.loc[peers, c + "_pct"] = m.loc[peers, c].rank(pct=True) * 100
    pct_cols = [c + "_pct" for c in RET_COLS]
    m["ret_hits"] = (m[pct_cols] >= p.return_pctile).sum(axis=1)
    m["ret_consistency"] = m[pct_cols].mean(axis=1)
    m["G_ret"] = m["ret_hits"] >= p.min_return_hits

    for col in ("sortino", "alpha", "std_dev", "beta", "alpha_y1", "alpha_y2",
                "mdd_5y", "bench_mdd_5y"):
        if col not in m:
            m[col] = np.nan
    m["G_sortino"] = m["sortino"] > p.min_sortino
    m["G_alpha"] = m["alpha"] > p.min_alpha

    gates = ["G_age", "G_aum", "G_er", "G_ret", "G_sortino", "G_alpha"]
    labels = ["A · Age ≥ %.0f yrs" % p.min_age, "A · AUM > ₹%.0f cr" % p.min_aum,
              "E · Expense ratio < %.2f%%" % p.max_er, "R · Return consistency",
              "S · Sortino > %.1f" % p.min_sortino, "A · Alpha > %.1f" % p.min_alpha]
    mask = pd.Series(True, index=m.index)
    for g, lab in zip(gates, labels):
        mask &= m[g].fillna(False)
        funnel.append((lab, int(mask.sum())))
    m["gates_passed"] = m[gates].fillna(False).sum(axis=1)
    m["PASS"] = mask

    # ---- composite rank (higher better). Std-Dev & drawdown act as tie-breakers.
    def pr(col, asc=True):
        return m[col].rank(pct=True, ascending=asc) * 100

    m["score"] = (0.30 * m["ret_consistency"].fillna(0)
                  + 0.25 * pr("alpha").fillna(0)
                  + 0.25 * pr("sortino").fillna(0)
                  + 0.10 * pr("expense_ratio", asc=False).fillna(0)
                  + 0.05 * pr("std_dev", asc=False).fillna(0)
                  + 0.05 * pr("mdd_5y").fillna(0))          # mdd negative -> higher (less negative) better
    m["exit_flag"] = (m["alpha_y1"] < 0) & (m["alpha_y2"] < 0)
    m = m.sort_values(["PASS", "gates_passed", "score"], ascending=False).reset_index(drop=True)
    return m, funnel
