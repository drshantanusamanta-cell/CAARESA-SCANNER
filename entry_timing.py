"""
Entry-timing add-on for the CAARESA app (informational only — NOT part of CAARESA scoring).

Two NAV trend-deviation measures, each turned into a z-score:
  A) MA deviation : dev = NAV / SMA(200) - 1, z-scored over a trailing window (~3 yrs)
  B) Log-trend gap: residual of ln(NAV) from a rolling log-linear fit (~5 yrs),
                    divided by the std of that window's residuals
All values use trailing data only (no look-ahead).

Also fetches exit load (Groww scheme page) and lock-in period (Kuvera, Groww fallback).
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import requests

MA_LEN = 200       # trading days for the moving average
Z_LEN = 750        # ~3 years: window for z-scoring the MA deviation
TREND_LEN = 1250   # ~5 years: window for the log-linear trend fit

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
_KUVERA = "https://mf.captnemo.in/kuvera/{isin}"
_GROWW_SEARCH = "https://groww.in/v1/api/search/v1/entity"
_GROWW_PAGE = "https://groww.in/mutual-funds/{slug}"


# ---------------------------------------------------------------- z-scores
def ma_deviation(nav: pd.Series, ma_len: int = MA_LEN, z_len: int = Z_LEN) -> pd.DataFrame:
    sma = nav.rolling(ma_len).mean()
    dev = nav / sma - 1
    z = (dev - dev.rolling(z_len).mean()) / dev.rolling(z_len).std()
    return pd.DataFrame({"sma": sma, "ma_dev": dev, "ma_dev_z": z})


def log_trend_gap(nav: pd.Series, n: int = TREND_LEN) -> pd.DataFrame:
    """
    Rolling OLS of ln(NAV) on time over the trailing n days, in closed form
    (identical to np.polyfit per window, but vectorised).
      gap   = ln(NAV_t) - fitted trend value at t
      gap_z = gap / sample std of the window's residuals
      trend_cagr = annualised slope of the fitted trend
    """
    y = pd.Series(np.log(nav.values), index=nav.index)
    x = pd.Series(np.arange(len(y), dtype=float), index=nav.index)
    mx, my = x.rolling(n).mean(), y.rolling(n).mean()
    cov = (x * y).rolling(n).mean() - mx * my
    var_x = (n * n - 1) / 12.0                      # population variance of n consecutive integers
    var_y = (y * y).rolling(n).mean() - my * my
    b = cov / var_x
    gap = y - (my + b * (x - mx))
    ssr = (n * (var_y - b * b * var_x)).clip(lower=0)
    resid_sd = np.sqrt(ssr / (n - 1))
    return pd.DataFrame({"log_gap": gap,
                         "log_gap_z": gap / resid_sd,
                         "trend_cagr": (np.exp(b * 252) - 1) * 100})


def compute(nav: pd.Series) -> pd.DataFrame:
    nav = nav.dropna()
    nav = nav[nav > 0]
    return pd.concat([nav.rename("nav"), ma_deviation(nav), log_trend_gap(nav)], axis=1)


def verdict(z: float) -> tuple[str, str]:
    """(label, colour) for a z-score reading."""
    if z is None or not np.isfinite(z):
        return "Not enough history", "gray"
    if z <= -2:
        return "Deep discount", "#1f9d55"
    if z <= -1:
        return "Discount", "#5cb85c"
    if z < 1:
        return "Around normal", "#8a8a8a"
    if z < 2:
        return "Stretched", "#d9822b"
    return "Very stretched", "#d64545"


def snapshot(nav: pd.Series) -> dict:
    """Latest readings only (for the multi-fund table)."""
    if nav is None or len(nav) < MA_LEN + 5:
        return {}
    d = compute(nav).iloc[-1]
    return dict(ma_dev=d.ma_dev * 100, ma_dev_z=d.ma_dev_z,
                log_gap=np.expm1(d.log_gap) * 100 if np.isfinite(d.log_gap) else np.nan,
                log_gap_z=d.log_gap_z)


# ---------------------------------------------------------------- exit load / lock-in
def _find_key(o, key):
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def _groww_slug(code: str, name: str) -> str | None:
    short = re.split(r"\s+-\s+", name)[0]
    for q in (code, name, short):
        try:
            js = requests.get(_GROWW_SEARCH, headers=_UA, timeout=20,
                              params=dict(app="false", entity_type="scheme", page=0, q=q, size=10)).json()
        except Exception:  # noqa: BLE001
            continue
        for c in js.get("content", []) or []:
            if str(c.get("scheme_code")) == str(code):
                return c.get("id")
    return None


def _fmt_lock(d) -> str | None:
    if not isinstance(d, dict):
        return None
    parts = [f"{int(d[k])} {k if int(d[k]) != 1 else k[:-1]}"
             for k in ("years", "months", "days") if d.get(k)]
    return ", ".join(parts) if parts else None


def fetch_exit_info(code: str, isin: str, name: str) -> dict:
    """
    Returns dict(exit_load, lock_in, source, url). Missing pieces are None —
    the caller should then point the user to the scheme's SID/KIM.
    """
    out = dict(exit_load=None, lock_in=None, source=[], url=None)

    # Lock-in from Kuvera (years; 0 = none)
    try:
        k = requests.get(_KUVERA.format(isin=isin), headers=_UA, timeout=20).json()
        k = k[0] if isinstance(k, list) else k
        lp = k.get("lock_in_period")
        if lp is not None:
            lp = float(lp)
            out["lock_in"] = "None" if lp == 0 else f"{lp:g} year{'s' if lp != 1 else ''}"
            out["source"].append("Kuvera")
    except Exception:  # noqa: BLE001
        pass

    # Exit load (and lock-in fallback) from the Groww scheme page
    slug = _groww_slug(code, name)
    if slug:
        url = _GROWW_PAGE.format(slug=slug)
        try:
            html = requests.get(url, headers=_UA, timeout=25).text
            m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
            data = json.loads(m.group(1)) if m else {}
            el = _find_key(data, "exit_load")
            if isinstance(el, str) and el.strip():
                out["exit_load"] = el.strip()
                out["url"] = url
                out["source"].append("Groww")
            if out["lock_in"] is None:
                lk = _fmt_lock(_find_key(data, "lock_in"))
                out["lock_in"] = lk or ("None" if el else None)
        except Exception:  # noqa: BLE001
            pass
    out["source"] = " · ".join(out["source"]) or None
    return out
