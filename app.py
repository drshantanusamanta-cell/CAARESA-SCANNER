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

tab_picks, tab_table, tab_exit, tab_method = st.tabs(
    ["🏆 Top picks", "📋 Full scorecard", "🚪 Exit check (held funds)", "ℹ️ Method"])

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
