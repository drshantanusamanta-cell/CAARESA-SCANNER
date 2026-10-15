"""
CAARESA Mutual Fund Scanner — Streamlit app
Category · Age · AUM · Return · Expense ratio · Sortino · Alpha
(+ Std-Dev & Max-Drawdown tie-breakers, + exit/"divorce" check)

Run:  streamlit run app.py
All data is pulled live (AMFI, mfapi.in, Kuvera) — no uploads needed.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import caaresa_engine as eng
import entry_timing as et  # add-on: entry-timing tab (not used in CAARESA scoring)

st.set_page_config(page_title="CAARESA Fund Scanner", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.pick {border:1px solid rgba(128,128,128,.25); border-radius:14px; padding:14px 16px;
       height:100%; background:rgba(128,128,128,.04);}
.pick h4 {margin:0 0 4px 0; font-size:1.02rem; line-height:1.3;}
.rank {font-size:.78rem; letter-spacing:.06em; text-transform:uppercase; opacity:.65;}
.badge {display:inline-block; padding:2px 9px; border-radius:999px; font-size:.74rem; font-weight:600;}
.pass {background:#1f9d5522; color:#1f9d55; border:1px solid #1f9d5566;}
.near {background:#d9822b22; color:#d9822b; border:1px solid #d9822b66;}
.kv {display:grid; grid-template-columns:1fr 1fr; gap:4px 12px; margin-top:10px; font-size:.86rem;}
.kv span:nth-child(odd){opacity:.65}
.kv span:nth-child(even){text-align:right; font-variant-numeric:tabular-nums; font-weight:600}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ cached data
@st.cache_data(ttl=12 * 3600, show_spinner="Loading AMFI scheme master…")
def get_master():
    return eng.load_scheme_master()


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_category_data(category: str):
    master = get_master()
    funds = master[master["category"] == category].reset_index(drop=True)
    navs, facts = eng.fetch_many(funds, workers=12)
    return funds, navs, facts


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_benchmark(bench_name: str, category: str):
    funds, navs, _ = get_category_data(category)
    return eng.build_benchmark(bench_name, navs, list(funds["code"]))


def fmt(v, suf="", nd=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{nd}f}{suf}"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("📊 CAARESA Scanner")
try:
    master = get_master()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the AMFI scheme list: {e}")
    st.stop()

groups = ["Equity", "Hybrid", "Debt", "Solution Oriented"]
group = st.sidebar.selectbox("C · Category group", groups)
cats = (master[master["group"] == group].groupby("category").size()
        .sort_values(ascending=False))
category = st.sidebar.selectbox(
    "Fund category", cats.index.tolist(),
    format_func=lambda c: f"{c}  ({cats[c]} funds)")
top_n = st.sidebar.slider("How many top funds to show", 3, 5, 5)

auto_b = eng.auto_benchmark(category)
bench_opts = list(eng.BENCH_PROXIES.keys())
bench_choice = st.sidebar.selectbox(
    "Benchmark for Alpha", ["Auto: " + auto_b] + bench_opts,
    help="Equity categories use a long-history index fund as a TRI proxy. "
         "Debt/Hybrid/Thematic use the equal-weight average of all funds in the category.")
bench_name = auto_b if bench_choice.startswith("Auto") else bench_choice

with st.sidebar.expander("Thresholds (Dr. Ram defaults)", expanded=False):
    p = eng.Params(
        min_age=st.number_input("A · Minimum age (years)", 0.0, 20.0, 5.0, 0.5),
        min_aum=st.number_input("A · Minimum AUM (₹ crore)", 0.0, 50000.0, 500.0, 100.0),
        max_er=st.number_input("E · Max expense ratio (%)", 0.0, 3.0, 1.0, 0.05),
        min_sortino=st.number_input("S · Min Sortino", -5.0, 5.0, 1.0, 0.1),
        min_alpha=st.number_input("A · Min Alpha (% p.a.)", -10.0, 10.0, 1.0, 0.25),
        return_pctile=st.slider("R · Percentile = 'top quarter'", 50, 90, 75, 5),
        min_return_hits=st.slider("R · Return metrics that must be top-quarter (of 4)", 1, 4, 4,
                                  help="3Y CAGR, 5Y CAGR, 3Y SIP-XIRR, 5Y SIP-XIRR"),
        rf=st.number_input("Risk-free / FD rate (% p.a.)", 0.0, 12.0, 6.5, 0.25),
        risk_years=st.radio("Window for Sortino / Alpha / Std-Dev", [3, 5], horizontal=True),
        allow_missing_facts=st.checkbox("Let funds with missing AUM/TER data pass", False),
    )

run = st.sidebar.button("🔍 Scan category", type="primary", width="stretch")
if st.sidebar.button("↻ Refresh live data", width="stretch"):
    st.cache_data.clear()
    st.session_state.pop("scanned", None)
    st.rerun()

st.sidebar.caption("Data: AMFI · mfapi.in · Kuvera. Cached for 12 h.")

# ------------------------------------------------------------------ header
st.title("CAARESA Mutual Fund Scanner")
st.caption("**C**ategory · **A**ge · **A**UM · **R**eturn · **E**xpense ratio · **S**ortino · "
           "**A**lpha — Direct-Plan, actively managed funds only. Index funds, ETFs and FoFs are excluded.")

if run:
    st.session_state["scanned"] = category
if st.session_state.get("scanned") != category:
    st.info(f"Pick a category in the sidebar and press **Scan category**. "
            f"**{category}** has {cats[category]} active Direct-Growth funds to scan.")
    st.stop()

# ------------------------------------------------------------------ scan
with st.spinner(f"Fetching NAV history, TER and AUM for {cats[category]} {category} funds…"):
    funds, navs, facts = get_category_data(category)
    bench = get_benchmark(bench_name, category)

metrics = eng.compute_metrics(funds, navs, facts, bench, p)
res, funnel = eng.apply_caaresa(metrics, p)

bnav = (1 + bench).cumprod()
asof = bnav.index[-1]
b3, b5 = eng.cagr(bnav, 3, asof), eng.cagr(bnav, 5, asof)
passers = res[res["PASS"]]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Funds scanned", funnel[0][1])
c2.metric("Pass all CAARESA gates", len(passers))
c3.metric(f"Benchmark 3Y / 5Y CAGR", f"{fmt(b3, '%', 1)} / {fmt(b5, '%', 1)}", help=bench_name)
c4.metric("NAV data as of", asof.strftime("%d %b %Y"))

tab_picks, tab_table, tab_exit, tab_entry, tab_method = st.tabs(
    ["🏆 Top picks", "📋 Full scorecard", "🚪 Exit check (held funds)",
     "📈 Entry timing (NAV z-score)", "ℹ️ Method"])

# ------------------------------------------------------------------ top picks
with tab_picks:
    top = res.head(top_n)
    if len(passers) == 0:
        st.warning("No fund clears **all** gates right now. Showing the closest candidates, ranked by "
                   "gates passed and then composite score — check which gate each one misses.")
    elif len(passers) < top_n:
        st.info(f"Only {len(passers)} fund(s) clear every gate; the rest shown are near-misses.")

    cols = st.columns(len(top))
    gate_names = {"G_age": "Age", "G_aum": "AUM", "G_er": "TER", "G_ret": "Return",
                  "G_sortino": "Sortino", "G_alpha": "Alpha"}
    for i, (col, r) in enumerate(zip(cols, top.itertuples())):
        missed = [n for g, n in gate_names.items() if not bool(getattr(r, g))]
        badge = ('<span class="badge pass">All gates passed</span>' if r.PASS else
                 f'<span class="badge near">Misses: {", ".join(missed)}</span>')
        col.markdown(f"""
<div class="pick">
 <div class="rank">#{i+1} · score {r.score:.0f}</div>
 <h4>{r.name}</h4>{badge}
 <div class="kv">
  <span>3Y / 5Y CAGR</span><span>{fmt(r.ret_3y,'%',1)} / {fmt(r.ret_5y,'%',1)}</span>
  <span>3Y / 5Y SIP XIRR</span><span>{fmt(r.sip_3y,'%',1)} / {fmt(r.sip_5y,'%',1)}</span>
  <span>Alpha</span><span>{fmt(r.alpha,'%',2)}</span>
  <span>Sortino</span><span>{fmt(r.sortino)}</span>
  <span>Expense ratio</span><span>{fmt(r.expense_ratio,'%')}</span>
  <span>AUM</span><span>₹{fmt(r.aum_cr,' cr',0)}</span>
  <span>Age</span><span>{fmt(r.age_yrs,' yrs',1)}</span>
  <span>Std-Dev</span><span>{fmt(r.std_dev,'%',1)}</span>
  <span>Max DD (5Y)</span><span>{fmt(r.mdd_5y,'%',1)}</span>
 </div>
</div>""", unsafe_allow_html=True)

    st.markdown("#### Growth of ₹10,000 over 5 years")
    start = asof - pd.DateOffset(years=5)
    fig = go.Figure()
    for r in top.itertuples():
        s = navs[r.code].loc[start:]
        if len(s):
            fig.add_scatter(x=s.index, y=10000 * s / s.iloc[0], name=r.name, mode="lines")
    bb = bnav.loc[start:]
    fig.add_scatter(x=bb.index, y=10000 * bb / bb.iloc[0], name=f"Benchmark ({bench_name})",
                    mode="lines", line=dict(dash="dash", color="gray"))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=-0.15), yaxis_title="₹", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Elimination funnel")
    ff = go.Figure(go.Funnel(y=[f[0] for f in funnel], x=[f[1] for f in funnel],
                             textinfo="value", marker=dict(color="#4c78a8")))
    ff.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(ff, width="stretch")

# ------------------------------------------------------------------ full table
with tab_table:
    tick = lambda b: "✅" if b else "❌"  # noqa: E731
    view = pd.DataFrame({
        "Fund": res["name"],
        "Pass": res["PASS"].map(tick),
        "Gates": res["gates_passed"].astype(int).astype(str) + "/6",
        "Score": res["score"].round(1),
        "Age (y)": res["age_yrs"].round(1), "Age✓": res["G_age"].map(tick),
        "AUM ₹cr": res["aum_cr"].round(0), "AUM✓": res["G_aum"].map(tick),
        "TER %": res["expense_ratio"], "TER✓": res["G_er"].map(tick),
        "3Y CAGR": res["ret_3y"].round(2), "5Y CAGR": res["ret_5y"].round(2),
        "3Y SIP": res["sip_3y"].round(2), "5Y SIP": res["sip_5y"].round(2),
        "Top-Q hits": res["ret_hits"].astype(int), "Ret✓": res["G_ret"].map(tick),
        "Sortino": res["sortino"].round(2), "Sortino✓": res["G_sortino"].map(tick),
        "Alpha": res["alpha"].round(2), "Alpha✓": res["G_alpha"].map(tick),
        "Beta": res["beta"].round(2), "Std-Dev": res["std_dev"].round(2),
        "Max DD 5Y": res["mdd_5y"].round(1),
        "Fund manager": res["fund_manager"],
    })
    st.dataframe(view, width="stretch", hide_index=True, height=560)
    st.caption(f"Return / risk figures as of {asof:%d %b %Y}. Alpha & Sortino over the last "
               f"{p.risk_years} years vs **{bench_name}**. Returns in % (CAGR / SIP XIRR).")
    st.download_button("⬇ Download scorecard (CSV)", res.to_csv(index=False).encode(),
                       file_name=f"caaresa_{category.replace(' ', '_').replace('/', '-')}.csv",
                       mime="text/csv")

# ------------------------------------------------------------------ exit check
with tab_exit:
    st.markdown("Already holding a fund? These are the **exit (“divorce”) criteria** — "
                "not the stricter entry checklist.")
    pick = st.selectbox("Fund you hold", res["name"].tolist())
    r = res[res["name"] == pick].iloc[0]
    neg2 = (r["alpha_y1"] < 0) and (r["alpha_y2"] < 0)
    er_up = r["expense_ratio"] > p.max_er
    under = (r["ret_3y"] < b3) and (r["std_dev"] > (bench.loc[asof - pd.DateOffset(years=p.risk_years):]
                                                     .std() * np.sqrt(252) * 100))
    e1, e2, e3 = st.columns(3)
    e1.metric("Alpha: last 12 m / prior 12 m", f"{fmt(r['alpha_y1'])} / {fmt(r['alpha_y2'])}",
              "Negative 2 yrs running ❌" if neg2 else "OK ✅", delta_color="off")
    e2.metric("Expense ratio", fmt(r["expense_ratio"], "%"),
              f"Above {p.max_er}% ❌" if er_up else "OK ✅", delta_color="off")
    e3.metric("3Y CAGR vs benchmark", f"{fmt(r['ret_3y'], '%', 1)} vs {fmt(b3, '%', 1)}",
              "Lagging with higher risk ❌" if under else "OK ✅", delta_color="off")
    flags = int(neg2) + int(er_up) + int(under)
    if flags == 0:
        st.success("No exit trigger. Stay invested and review again next year.")
    else:
        st.error(f"{flags} exit trigger(s) present. Consider switching to one of the top picks "
                 "(mind exit load and capital-gains tax before switching).")

# ------------------------------------------------------------------ method
with tab_method:
    st.markdown(f"""
**Universe.** AMFI's live scheme list → open-ended, Direct Plan, Growth option, within the chosen SEBI
category. Index funds, ETFs, FoFs and segregated portfolios are removed (CAARESA is for *active* funds).

| Gate | Rule used here | Source |
|---|---|---|
| **C** Category | Your sidebar choice | AMFI |
| **A** Age | ≥ {p.min_age:g} years since first NAV / launch | mfapi.in, Kuvera |
| **A** AUM | > ₹{p.min_aum:,.0f} crore (scheme level) | Kuvera |
| **E** Expense ratio | Direct-plan TER < {p.max_er:g}% | Kuvera |
| **R** Return | {p.min_return_hits} of 4 metrics (3Y & 5Y CAGR, 3Y & 5Y monthly-SIP XIRR) in the top {100 - p.return_pctile:g}% of age-eligible peers | computed from NAVs |
| **S** Sortino | (annualised return − {p.rf}% FD rate) ÷ annualised downside deviation, last {p.risk_years} yrs daily data, > {p.min_sortino:g} | computed |
| **A** Alpha | Jensen's alpha vs benchmark, % p.a., last {p.risk_years} yrs, > {p.min_alpha:g} | computed |

**Ranking** among survivors: 30% return-consistency percentile, 25% alpha, 25% Sortino, 10% low TER,
5% low Std-Dev and 5% shallow 5-year max drawdown (the last two are the tie-breakers).
If fewer funds pass than you asked for, near-misses fill the list and show which gate they miss.

**Benchmarks.** Equity categories use long-history index funds as TRI proxies
(UTI Nifty 50, Motilal Oswal Nifty 500 / Midcap 150 / Smallcap 250; Large & Mid uses a 50:50 blend).
Because index funds carry a small TER, alpha is very slightly flattering. Sectoral/thematic, hybrid and
debt funds are compared with the equal-weight average of their own category.

**Caveats.** Ratios are computed from daily NAVs, so they will not exactly match Value Research /
Morningstar (which mostly use monthly data and TRI benchmarks). In a weak 3-year market, very few funds may
clear Sortino > 1 — that is the screen working, not a bug. Debt funds rarely post Alpha > 1 against peers;
relax the thresholds for debt. This is an educational screen, not investment advice.
""")

# ------------------------------------------------------------------ entry timing (add-on)
# Informational only — nothing here feeds the CAARESA gates, score or ranking above.
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_exit_info(code: str, isin: str, name: str):
    return et.fetch_exit_info(code, isin, name)


with tab_entry:
    st.info("**Add-on, not part of CAARESA scoring.** Use it to time *top-ups* into a fund you have "
            "already chosen — keep the base SIP running regardless of these readings.")

    names = res["name"].tolist()
    epick = st.selectbox("Fund", names, index=0, key="entry_fund",
                         help="Defaults to the #1 CAARESA pick. Any fund in this category can be checked.")
    er = res[res["name"] == epick].iloc[0]
    enav = navs.get(er["code"], pd.Series(dtype=float))

    if len(enav) < et.MA_LEN + 5:
        st.warning("Not enough NAV history for this fund (needs at least ~200 trading days).")
    else:
        ez = et.compute(enav)
        last = ez.iloc[-1]

        # ---- exit load & lock-in
        with st.spinner("Looking up exit load and lock-in…"):
            xi = get_exit_info(str(er["code"]), str(er["isin"]), epick)
        f1, f2, f3 = st.columns([2, 1, 1])
        f1.metric("Exit load", xi["exit_load"] or "Not available")
        f2.metric("Lock-in period", xi["lock_in"] or "Not available")
        f3.metric("NAV as of", ez.index[-1].strftime("%d %b %Y"), f"₹{last.nav:,.2f}", delta_color="off")
        src = f"Source: {xi['source']}." if xi["source"] else ""
        if xi["url"]:
            src += f" [Scheme page]({xi['url']})"
        st.caption(f"{src} Always confirm in the scheme's SID/KIM before investing — "
                   "exit loads can change, and each SIP instalment has its own exit-load and lock-in clock.")

        # ---- the two readings
        lab_a, col_a = et.verdict(last.ma_dev_z)
        lab_b, col_b = et.verdict(last.log_gap_z)
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("##### A · Deviation from 200-day average")
            a1, a2 = st.columns(2)
            a1.metric("NAV vs 200-DMA", fmt(last.ma_dev * 100, "%", 1))
            a2.metric("Z-score (3-yr window)", fmt(last.ma_dev_z))
            st.markdown(f"<span class='badge' style='background:{col_a}22;color:{col_a};"
                        f"border:1px solid {col_a}66'>{lab_a}</span>", unsafe_allow_html=True)
        with m2:
            st.markdown("##### B · Gap from 5-year log trend")
            b1, b2, b3_ = st.columns(3)
            gap_pct = np.expm1(last.log_gap) * 100 if np.isfinite(last.log_gap) else np.nan
            b1.metric("NAV vs trend", fmt(gap_pct, "%", 1))
            b2.metric("Z-score", fmt(last.log_gap_z))
            b3_.metric("Trend growth", fmt(last.trend_cagr, "% p.a.", 1))
            st.markdown(f"<span class='badge' style='background:{col_b}22;color:{col_b};"
                        f"border:1px solid {col_b}66'>{lab_b}</span>", unsafe_allow_html=True)

        za, zb = last.ma_dev_z, last.log_gap_z
        if np.isfinite(za) and np.isfinite(zb):
            if za <= -1 and zb <= -1:
                st.success("Both measures show a discount — a reasonable window for an extra top-up.")
            elif za <= -1 or zb <= -1:
                st.info("One measure shows a discount, the other does not — a partial / smaller top-up zone.")
            elif za >= 2 or zb >= 2:
                st.warning("NAV is well above its normal range — keep the SIP, hold back lump-sum top-ups.")
            else:
                st.caption("No timing signal — NAV is within its normal range. Continue the regular SIP.")
        elif not np.isfinite(zb):
            st.caption("Method B needs ~5 years of NAV history; only Method A is available for this fund.")

        # ---- charts
        yrs = st.radio("Chart period", ["3Y", "5Y", "10Y", "All"], index=1, horizontal=True, key="entry_period")
        cut = ez.index[0] if yrs == "All" else ez.index[-1] - pd.DateOffset(years=int(yrs[:-1]))
        v = ez.loc[cut:]

        fig_n = go.Figure()
        fig_n.add_scatter(x=v.index, y=v.nav, name="NAV", mode="lines", line=dict(width=1.6))
        fig_n.add_scatter(x=v.index, y=v.sma, name=f"{et.MA_LEN}-day average", mode="lines",
                          line=dict(width=1.2, color="#d9822b"))
        fig_n.add_scatter(x=v.index, y=v.nav / np.exp(v.log_gap), name="5-yr log trend", mode="lines",
                          line=dict(width=1.2, dash="dash", color="gray"))
        fig_n.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis_type="log",
                            title=dict(text="NAV (log scale)", font=dict(size=14)),
                            legend=dict(orientation="h", y=-0.15), hovermode="x unified")
        st.plotly_chart(fig_n, width="stretch")

        fig_z = go.Figure()
        for y0, y1, c in [(-4, -2, "#1f9d55"), (-2, -1, "#5cb85c"), (1, 2, "#d9822b"), (2, 4, "#d64545")]:
            fig_z.add_hrect(y0=y0, y1=y1, fillcolor=c, opacity=0.08, line_width=0)
        fig_z.add_hline(y=0, line=dict(color="gray", width=1))
        fig_z.add_scatter(x=v.index, y=v.ma_dev_z, name="A · 200-DMA deviation z", mode="lines",
                          line=dict(width=1.4))
        fig_z.add_scatter(x=v.index, y=v.log_gap_z, name="B · Log-trend gap z", mode="lines",
                          line=dict(width=1.4))
        fig_z.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                            title=dict(text="Z-scores (green = discount, red = stretched)", font=dict(size=14)),
                            yaxis=dict(range=[-3.5, 3.5]), legend=dict(orientation="h", y=-0.15),
                            hovermode="x unified")
        st.plotly_chart(fig_z, width="stretch")

    # ---- snapshot across funds (uses NAVs already loaded for the scan — no extra downloads)
    st.markdown("#### Snapshot")
    show_all = st.toggle("Show every fund in this category (default: top picks only)", key="entry_all")
    rows = res if show_all else res.head(top_n)
    snap = []
    for r in rows.itertuples():
        s = et.snapshot(navs.get(r.code))
        if s:
            snap.append({"Fund": r.name, "CAARESA pass": "✅" if r.PASS else "❌",
                         "vs 200-DMA %": round(s["ma_dev"], 1), "A z": round(s["ma_dev_z"], 2),
                         "A reading": et.verdict(s["ma_dev_z"])[0],
                         "vs trend %": round(s["log_gap"], 1), "B z": round(s["log_gap_z"], 2),
                         "B reading": et.verdict(s["log_gap_z"])[0]})
    if snap:
        st.dataframe(pd.DataFrame(snap), width="stretch", hide_index=True)

    with st.expander("How these are calculated and how to read them"):
        st.markdown(f"""
**A · Deviation from the {et.MA_LEN}-day average**
`dev = NAV ÷ SMA{et.MA_LEN} − 1`, then `z = (dev − mean) ÷ std` over the trailing {et.Z_LEN} trading days (~3 years).
Reacts faster — good for tactical top-ups.

**B · Gap from the log trend**
Fit `ln(NAV) = a + b·t` over the trailing {et.TREND_LEN} trading days (~5 years).
`gap = ln(NAV) − fitted value`; `z = gap ÷ std of the window's residuals`.
`e^gap − 1` is the % above/below trend; `e^(252·b) − 1` is the trend growth rate.
Slower and steadier — tells you where the NAV sits against its own long-run compounding path.

Both use only data available on each day (no look-ahead).

| Z-score | Reading | Suggested use |
|---|---|---|
| ≤ −2 | Deep discount (~1 day in 40) | Larger top-up |
| −2 to −1 | Discount (~1 day in 6) | Normal top-up |
| −1 to +1 | Around normal | SIP only |
| +1 to +2 | Stretched | SIP only, no lump sum |
| ≥ +2 | Very stretched | SIP only |

**Limits.** NAV z-scores say a fund is cheap relative to *its own recent path*, not that the underlying
stocks are cheap. Mean reversion in diversified funds is weak, so use this to size extra top-ups —
not to stop or start the SIP. Mind exit load and capital-gains tax on any switch. Educational only.
""")
