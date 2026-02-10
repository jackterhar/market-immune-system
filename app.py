# Market Immune System Dashboard v4 (Daily Close)
# Fully web-based via Streamlit Community Cloud

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# CONFIG
# -----------------------------
SPX_TICKER = "^GSPC"
AI_SEMI_TICKERS = ["NVDA", "AVGO", "TSM", "ASML", "AMD", "AMAT", "LRCX", "KLAC"]
LOOKBACK_DAYS = 252
WARNING_PCTL = 0.95
EXTREME_PCTL = 0.99
MA_WINDOW = 50
FWD_WINDOWS = [10, 20, 60]

st.set_page_config(page_title="Market Immune System v4", layout="wide")

# -----------------------------
# DATA
# -----------------------------
@st.cache_data

def load_prices(tickers):
    data = yf.download(list(tickers), period="4y", interval="1d", auto_adjust=True)["Close"]
    return data.dropna()

@st.cache_data

def load_market_caps(tickers):
    caps = {}
    for t in tickers:
        info = yf.Ticker(t).info
        caps[t] = info.get("marketCap", np.nan)
    return pd.Series(caps).dropna()

spx = load_prices([SPX_TICKER])
ai_prices = load_prices(AI_SEMI_TICKERS)
market_caps = load_market_caps(AI_SEMI_TICKERS)
weights = market_caps / market_caps.sum()

# -----------------------------
# TURBULENCE ENGINE
# -----------------------------

def calc_turbulence(prices, window=LOOKBACK_DAYS):
    rets = np.log(prices / prices.shift(1)).dropna()
    mu = rets.rolling(window).mean()
    cov = rets.rolling(window).cov()
    scores = []

    for d in rets.index[window:]:
        r = rets.loc[d] - mu.loc[d]
        c = cov.loc[d]
        try:
            scores.append(np.dot(np.dot(r.T, np.linalg.pinv(c)), r))
        except:
            scores.append(np.nan)

    return pd.Series(scores, index=rets.index[window:])

market_turb = calc_turbulence(spx)
ai_turb_raw = calc_turbulence(ai_prices)
ai_turb = ai_turb_raw.mul(weights, axis=1).sum(axis=1)

# -----------------------------
# REGIME LOGIC
# -----------------------------
warn = market_turb.quantile(WARNING_PCTL)
extreme = market_turb.quantile(EXTREME_PCTL)

spx_px = spx[SPX_TICKER].loc[market_turb.index]
spx_ma = spx_px.rolling(MA_WINDOW).mean()

soft_regime = market_turb > warn
hard_regime = soft_regime & (spx_px > spx_ma)

# -----------------------------
# REGIME PERSISTENCE
# -----------------------------
persistence = soft_regime.groupby((~soft_regime).cumsum()).cumcount() + 1
persistence = persistence.where(soft_regime, 0)

# -----------------------------
# FORWARD RETURN STATS
# -----------------------------
stats = {}
for w in FWD_WINDOWS:
    fwd = spx_px.pct_change(w).shift(-w)
    stats[w] = {
        "hit": (fwd[hard_regime] < 0).mean() * 100,
        "avg": fwd[hard_regime].mean() * 100,
        "median": fwd[hard_regime].median() * 100,
        "worst": fwd[hard_regime].min() * 100
    }

# -----------------------------
# CONFIDENCE SCORE
# -----------------------------
conf_score = (
    0.4 * (market_turb.iloc[-1] / warn) +
    0.3 * (ai_turb.iloc[-1] / market_turb.iloc[-1]) +
    0.3 * (persistence.iloc[-1] / 10)
)
conf_score = min(conf_score, 1.5)

# -----------------------------
# DAILY VERDICT
# -----------------------------
if hard_regime.iloc[-1] and conf_score > 1:
    verdict = "RISK OFF"
elif soft_regime.iloc[-1]:
    verdict = "CAUTION"
else:
    verdict = "RISK ON"

# -----------------------------
# DASHBOARD
# -----------------------------
st.title("Market Immune System – v4")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Verdict", verdict)
c2.metric("Status", "HARD" if hard_regime.iloc[-1] else "SOFT/NORMAL")
c3.metric("Persistence (days)", int(persistence.iloc[-1]))
c4.metric("AI / Market Stress", f"{(ai_turb.iloc[-1]/market_turb.iloc[-1]):.2f}x")
c5.metric("Confidence", f"{conf_score:.2f}")

# -----------------------------
# CHART
# -----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=spx_px.index, y=spx_px, name="SPX", yaxis="y2"))
fig.add_trace(go.Scatter(x=spx_ma.index, y=spx_ma, name="SPX 50DMA", yaxis="y2", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=market_turb.index, y=market_turb, name="Market Turbulence"))

for d in hard_regime[hard_regime].index:
    fig.add_vrect(x0=d, x1=d + pd.Timedelta(days=1), fillcolor="green", opacity=0.15, line_width=0)

fig.add_hline(y=warn, line_dash="dash", annotation_text="95% Warning")
fig.add_hline(y=extreme, line_dash="dot", annotation_text="99% Extreme")

fig.update_layout(height=650, yaxis=dict(title="Turbulence"), yaxis2=dict(title="SPX", overlaying="y", side="right"))
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# STATS TABLE
# -----------------------------
st.subheader("Historical Outcomes After HARD Warnings")
rows = []
for w, s in stats.items():
    rows.append([f"{w}D", f"{s['hit']:.1f}%", f"{s['avg']:.2f}%", f"{s['median']:.2f}%", f"{s['worst']:.2f}%"])

st.table(pd.DataFrame(rows, columns=["Window", "Downside Hit Rate", "Avg", "Median", "Worst"]))

# -----------------------------
# INTERPRETATION
# -----------------------------
st.subheader("System Interpretation")

if verdict == "RISK OFF":
    st.error("High-confidence internal stress regime. Historical outcomes skew materially negative over the next 2–8 weeks.")
elif verdict == "CAUTION":
    st.warning("Internal stress elevated. Trend support weakening — risk rising.")
else:
    st.success("Internal conditions healthy. No systemic stress detected.")
