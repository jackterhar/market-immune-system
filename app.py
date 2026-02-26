import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
st.title("Market Immune System — Tactical")

# =====================================================
# 1️⃣ LOAD DATA (HARDENED FOR STREAMLIT CLOUD)
# =====================================================

@st.cache_data
def load_data():

    tickers = {
        "SPY": "SPY",
        "VIX": "^VIX",
        "BTC": "BTC-USD",
        "TNX": "^TNX",   # 10Y
        "IRX": "^IRX"    # 3M
    }

    data = {}

    for name, ticker in tickers.items():
        df = yf.download(
            ticker,
            period="5y",
            auto_adjust=True,
            progress=False
        )

        # Force single Series (prevents MultiIndex shape errors)
        close = df["Close"].squeeze()
        close.name = name
        data[name] = close

    # Align and combine
    df = pd.concat(data.values(), axis=1)

    # Yield curve (10Y - 3M)
    df["YieldCurve"] = df["TNX"] - df["IRX"]

    df = df.drop(columns=["TNX", "IRX"])
    df = df.dropna()

    return df


df = load_data()

# Safety check
if len(df) < 300:
    st.error("Not enough data to compute rolling factors.")
    st.stop()

# =====================================================
# 2️⃣ FACTORS
# =====================================================

df['SPY_20d_vol'] = df['SPY'].pct_change().rolling(20).std()
df['SPY_50d_vol'] = df['SPY'].pct_change().rolling(50).std()

# 3-month momentum
df['Trend'] = df['SPY'].pct_change(63)

df['Trend_z'] = (
    df['Trend'] - df['Trend'].rolling(252).mean()
) / df['Trend'].rolling(252).std()

df['VIX_z'] = (
    df['VIX'] - df['VIX'].rolling(252).mean()
) / df['VIX'].rolling(252).std()

df['YC_z'] = (
    df['YieldCurve'] - df['YieldCurve'].rolling(252).mean()
) / df['YieldCurve'].rolling(252).std()

df['VolSpread'] = df['SPY_20d_vol'] - df['SPY_50d_vol']

df['Vol_z'] = (
    df['VolSpread'] - df['VolSpread'].rolling(252).mean()
) / df['VolSpread'].rolling(252).std()

df = df.dropna()

# =====================================================
# 3️⃣ TACTICAL MACRO SCORE
# =====================================================

df['MacroScore_raw'] = (
    0.40 * df['Trend_z'] +
    -0.30 * df['VIX_z'] +
    0.15 * df['YC_z'] +
    -0.15 * df['Vol_z']
)

df['MacroScore'] = df['MacroScore_raw'].ewm(span=8).mean()

upper = 0.4
lower = -0.4

df['MacroRegime'] = np.select(
    [
        df['MacroScore'] > upper,
        df['MacroScore'] < lower
    ],
    ['Risk-On', 'Risk-Off'],
    default='Neutral'
)

# =====================================================
# 4️⃣ REGIME BLOCK FUNCTION (STRIPE-FREE)
# =====================================================

def add_regime_blocks(fig, df, regime_col, row):

    colors = {
        'Risk-On': 'rgba(0,120,0,0.18)',
        'Neutral': 'rgba(180,150,0,0.18)',
        'Risk-Off': 'rgba(150,0,0,0.18)'
    }

    if df.empty:
        return

    current_regime = df[regime_col].iloc[0]
    start_date = df.index[0]

    for i in range(1, len(df)):
        regime = df[regime_col].iloc[i]

        if regime != current_regime:
            fig.add_vrect(
                x0=start_date,
                x1=df.index[i],
                fillcolor=colors[current_regime],
                line_width=0,
                layer="below",
                row=row,
                col=1
            )
            start_date = df.index[i]
            current_regime = regime

    # Final block
    fig.add_vrect(
        x0=start_date,
        x1=df.index[-1],
        fillcolor=colors[current_regime],
        line_width=0,
        layer="below",
        row=row,
        col=1
    )

# =====================================================
# 5️⃣ BUILD CHART
# =====================================================

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.07,
    row_heights=[0.4, 0.4, 0.2]
)

# SPY
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['SPY'],
        line=dict(color='white', width=2),
        name="SPY"
    ),
    row=1, col=1
)

# BTC
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['BTC'],
        line=dict(color='#f7931a', width=2),
        name="BTC"
    ),
    row=2, col=1
)

# MacroScore
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['MacroScore'],
        line=dict(color='cyan', width=2),
        name="MacroScore"
    ),
    row=3, col=1
)

fig.add_hline(y=upper, line_dash="dot", row=3, col=1)
fig.add_hline(y=lower, line_dash="dot", row=3, col=1)

# Regime shading
add_regime_blocks(fig, df, 'MacroRegime', 1)
add_regime_blocks(fig, df, 'MacroRegime', 2)

fig.update_layout(
    template='plotly_dark',
    height=950,
    showlegend=False
)

fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')

st.plotly_chart(fig, use_container_width=True)
