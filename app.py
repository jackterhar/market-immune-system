import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(layout="wide")
st.title("Market Immune System — Tactical Regime Engine")

st.markdown("""
This model classifies macro environment into **Risk-On**, **Neutral**, or **Risk-Off**
using trend, volatility, and yield-curve dynamics.
""")

# =====================================================
# DATA LOADER
# =====================================================

@st.cache_data
def load_data():

    tickers = {
        "SPY": "SPY",
        "VIX": "^VIX",
        "BTC": "BTC-USD",
        "TNX": "^TNX",
        "IRX": "^IRX"
    }

    data = {}

    for name, ticker in tickers.items():
        raw = yf.download(
            ticker,
            period="5y",
            auto_adjust=True,
            progress=False
        )

        close = raw["Close"].squeeze()
        close.name = name
        data[name] = close

    df = pd.concat(data.values(), axis=1)
    df["YieldCurve"] = df["TNX"] - df["IRX"]
    df = df.drop(columns=["TNX", "IRX"])
    df = df.dropna()

    return df


df = load_data()

if len(df) < 300:
    st.error("Insufficient data history.")
    st.stop()

# =====================================================
# FACTOR ENGINE
# =====================================================

df['Return'] = df['SPY'].pct_change()

df['Vol20'] = df['Return'].rolling(20).std()
df['Vol50'] = df['Return'].rolling(50).std()

df['Momentum63'] = df['SPY'].pct_change(63)

df['Trend_z'] = (
    df['Momentum63'] - df['Momentum63'].rolling(252).mean()
) / df['Momentum63'].rolling(252).std()

df['VIX_z'] = (
    df['VIX'] - df['VIX'].rolling(252).mean()
) / df['VIX'].rolling(252).std()

df['YC_z'] = (
    df['YieldCurve'] - df['YieldCurve'].rolling(252).mean()
) / df['YieldCurve'].rolling(252).std()

df['VolSpread'] = df['Vol20'] - df['Vol50']

df['Vol_z'] = (
    df['VolSpread'] - df['VolSpread'].rolling(252).mean()
) / df['VolSpread'].rolling(252).std()

df = df.dropna()

# =====================================================
# COMPOSITE SCORE
# =====================================================

weights = {
    "Trend_z": 0.40,
    "VIX_z": -0.30,
    "YC_z": 0.15,
    "Vol_z": -0.15
}

df["MacroScore_raw"] = (
    weights["Trend_z"] * df["Trend_z"] +
    weights["VIX_z"] * df["VIX_z"] +
    weights["YC_z"] * df["YC_z"] +
    weights["Vol_z"] * df["Vol_z"]
)

df["MacroScore"] = df["MacroScore_raw"].ewm(span=10).mean()

upper = 0.4
lower = -0.4

df["Regime"] = np.select(
    [df["MacroScore"] > upper,
     df["MacroScore"] < lower],
    ["Risk-On", "Risk-Off"],
    default="Neutral"
)

# =====================================================
# REGIME DURATION CALCULATION
# =====================================================

df["RegimeShift"] = df["Regime"] != df["Regime"].shift()
df["RegimeBlock"] = df["RegimeShift"].cumsum()

durations = df.groupby("RegimeBlock").size()
current_duration = durations.iloc[-1]
current_regime = df["Regime"].iloc[-1]
current_score = df["MacroScore"].iloc[-1]

# =====================================================
# BUILD FIGURE
# =====================================================

fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.35, 0.35, 0.2, 0.1],
    subplot_titles=[
        "SPY Price (USD)",
        "Bitcoin Price (USD)",
        "Composite Macro Score (Z-Scaled)",
        "Component Contributions"
    ]
)

# SPY
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["SPY"],
        line=dict(width=2),
        name="SPY"
    ),
    row=1, col=1
)

# BTC
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["BTC"],
        line=dict(width=2, color="orange"),
        name="BTC"
    ),
    row=2, col=1
)

# MacroScore
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MacroScore"],
        line=dict(width=2, color="blue"),
        name="MacroScore"
    ),
    row=3, col=1
)

fig.add_hline(y=upper, line_dash="dash", row=3, col=1)
fig.add_hline(y=lower, line_dash="dash", row=3, col=1)

# Component bars (latest values)
latest_components = [
    df["Trend_z"].iloc[-1] * weights["Trend_z"],
    df["VIX_z"].iloc[-1] * weights["VIX_z"],
    df["YC_z"].iloc[-1] * weights["YC_z"],
    df["Vol_z"].iloc[-1] * weights["Vol_z"]
]

fig.add_trace(
    go.Bar(
        x=["Trend", "VIX", "YieldCurve", "VolSpread"],
        y=latest_components,
        name="Contribution"
    ),
    row=4, col=1
)

# =====================================================
# REGIME SHADING (LIGHT)
# =====================================================

colors = {
    "Risk-On": "rgba(0,200,0,0.08)",
    "Neutral": "rgba(200,180,0,0.08)",
    "Risk-Off": "rgba(200,0,0,0.08)"
}

current = df["Regime"].iloc[0]
start = df.index[0]

for i in range(1, len(df)):
    regime = df["Regime"].iloc[i]
    if regime != current:
        fig.add_vrect(
            x0=start,
            x1=df.index[i],
            fillcolor=colors[current],
            line_width=0,
            layer="below"
        )
        start = df.index[i]
        current = regime

fig.add_vrect(
    x0=start,
    x1=df.index[-1],
    fillcolor=colors[current],
    line_width=0,
    layer="below"
)

# =====================================================
# LAYOUT IMPROVEMENTS
# =====================================================

fig.update_layout(
    height=1100,
    template="plotly_white",
    showlegend=True
)

fig.update_yaxes(title_text="USD", row=1, col=1)
fig.update_yaxes(title_text="USD", row=2, col=1)
fig.update_yaxes(title_text="Z-Score", row=3, col=1)
fig.update_yaxes(title_text="Weighted Value", row=4, col=1)

fig.update_xaxes(title_text="Date")

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# CURRENT STATE PANEL
# =====================================================

st.markdown("### Current Regime Status")

col1, col2, col3 = st.columns(3)

col1.metric("Regime", current_regime)
col2.metric("Composite Score", round(current_score, 2))
col3.metric("Days in Regime", int(current_duration))

st.markdown("""
**Score Interpretation**

• Above +0.4 → Risk-On  
• Between -0.4 and +0.4 → Neutral  
• Below -0.4 → Risk-Off  

Z-score normalization makes factors comparable across regimes.
""")
