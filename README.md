# CAARESA Mutual Fund Scanner

Streamlit app that screens Indian actively managed mutual funds with Dr. Ram's **CAARESA** criteria
(Category, Age, AUM, Return, Expense ratio, Sortino, Alpha) and returns the top 3–5 funds in any
SEBI category. Everything is fetched live — no uploads.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Deploys as-is to Streamlit Community Cloud or Hugging Face Spaces (Streamlit SDK).

## Data sources
| Data | Source |
|---|---|
| Scheme list, SEBI category, plan/option | AMFI `NAVAll.txt` |
| Daily NAV history (age, returns, SIP XIRR, Sortino, Alpha, Std-Dev, drawdown) | api.mfapi.in |
| Direct-plan expense ratio, AUM, fund manager | Kuvera public API (ISIN lookup via mf.captnemo.in) |

## Files
- `app.py` — the Streamlit UI (top picks, full scorecard, exit check, method)
- `caaresa_engine.py` — data fetching and all metric calculations (usable without Streamlit)

Educational screen only — not investment advice.
