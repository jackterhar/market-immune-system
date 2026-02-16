import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🧬 Market Immune System (Test Version)")

RESAMPLE_RULE = "W"
LOOKBACK_YEARS = 10


# =====================================================
# SAFE DOWNLOAD HELPERS
# =====================================================

@st.cache_data
def download_spy(start, end):
    spy = yf.download(
        "SPY",
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )

    if spy.empty:
        st.error("SPY download failed.")
        st.stop()

    # Flatten MultiIndex if necessary
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    spy.columns = [c.capitalize() for c in spy.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in spy.columns:
            st.error(f"Missing SPY column: {col}")
            st.stop()

    return spy


@st.cache_data
def download_close_series(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)

    if data.empty:
        st.error(f"{ticker} download failed.")
        st.stop()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" in data.columns:
        series = data["Close"]
    else:
        series = data.iloc[:, 0]

    return series


# =====================================================
# LOAD DATA
# =====================================================

end = datetime.today()
start = datetime(end.year - LOOKBACK_YEARS, end.month, end.day)

spy = download_spy(start, end)

vix = download_close_series("^VIX", start, end)
hyg = download_close_series("HYG", start, end)
lqd = download_close_series("LQD", start, end)
tnx = download_close_series("^TNX", start, end) / 10
irx = download_close_series("^IRX", start, end) / 10


# =====================================================
# RESAMPLE (FORCE SERIES SHAPE)
# =====================================================

spy = spy.resample(RESAMPLE_RULE).agg({
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum"
}).dropna()


def resample_to_series(s):
    s = s.resample(RESAMPLE_RULE).last()
    if isinstance(s, pd.DataFrame):
        if "Close" in s.columns:
            return s["Close"]
        return s.iloc[:, 0]
    return s


vix = resample_to_series(vix)
hyg = resample_to_series(hyg)
lqd = resample_to_series(lqd)
tnx = resample_to_series(tnx)
irx = resample_to_series(irx)


# =====================================================
# ALIGN ALL DATA TO SPY INDEX
# =====================================================

common_index = spy.index

vix = vix.reindex(common_index)
hyg = hyg.reindex(common_index)
lqd = lqd.reindex(common_index)
tnx = tnx.reindex(common_index)
irx = irx.reindex(common_index)


# =====================================================
# INDICATORS (STRUCTURALLY GUARANTEED)
# =====================================================

spy["Return"] = spy["Close"].pct_change()

df = pd.DataFrame(index=common_index)

df["YieldCurve"] = tnx - irx
df["VIX"] = vix
df["CreditRatio"] = hyg / lqd
df["Vol20"] = spy["Return"].rolling(4).std() * np.sqrt(52)

df = df.dropna()

df["Score"] = (
    (df["YieldCurve"] > 0).astype(int)
    + (df["VIX"] < 25).astype(int)
    + (df["CreditRatio"] > df["CreditRatio"].rolling(20).mean()).astype(int)
    + (df["Vol20"] < df["Vol20"].rolling(20).mean()).astype(int)
)

df["Regime"] = np.select(
    [
        df["Score"] >= 3,
        df["Score"] == 2
    ],
    [
        "Risk On",
        "Neutral"
    ],
    default="Risk Off"
)

regime = df["Regime"]


# =====================================================
# PLOT
# =====================================================

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=spy.index,
    open=spy["Open"],
    high=spy["High"],
    low=spy["Low"],
    close=spy["Close"],
    increasing=dict(line=dict(color="lime", width=2), fillcolor="rgba(0,0,0,0)"),
    decreasing=dict(line=dict(color="red", width=2), fillcolor="rgba(0,0,0,0)"),
    name="SPY"
))

colors = {
    "Risk On": "rgba(0,200,0,0.35)",
    "Neutral": "rgba(255,200,0,0.35)",
    "Risk Off": "rgba(200,0,0,0.35)"
}

if not regime.empty:

    last_regime = regime.iloc[0]
    start_date = regime.index[0]

    for i in range(1, len(regime)):
        current = regime.iloc[i]
        date = regime.index[i]

        if current != last_regime:
            fig.add_vrect(
                x0=start_date,
                x1=date,
                fillcolor=colors[last_regime],
                layer="below",
                line_width=0
            )
            start_date = date
            last_regime = current

    fig.add_vrect(
        x0=start_date,
        x1=regime.index[-1],
        fillcolor=colors[last_regime],
        layer="below",
        line_width=0
    )

fig.update_layout(
    template="plotly_dark",
    height=750,
    xaxis_rangeslider_visible=False,
    title="SPY with Market Regime Overlay"
)

st.plotly_chart(fig, use_container_width=True)


# =====================================================
# INTERPRETATION GUIDE
# =====================================================

st.markdown("---")
st.header("📊 Interpretation Guide")

st.markdown("""
### 🧬 Immune Score
0–1 → Defensive  
2 → Transitional / Mixed  
3–4 → Risk-On environment  

### 📉 VIX
<20 → Calm  
20–30 → Stress building  
>30 → Crisis regime  

### 🏦 Credit Ratio (HYG / LQD)
Rising → Risk appetite strong  
Falling → Credit stress increasing  

### 📈 Yield Curve (10Y − 2Y)
Positive → Expansionary  
Flat → Late cycle  
Negative → Recession probability elevated  

### 📊 SPY 20D Volatility
Low → Stable regime  
Moderate → Normal fluctuations  
High → Instability / drawdown risk  

### 🎯 Regimes
Risk On → Favor equities & cyclicals  
Neutral → Balanced allocation  
Risk Off → Defensive positioning  
""")
