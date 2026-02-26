import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

st.title("Market Immune System (Tactical)")

# =========================
# 1️⃣ LOAD DATA
# =========================

@st.cache_data
def load_data():

    spy = yf.download("SPY", period="3y")
    vix = yf.download("^VIX", period="3y")
    btc = yf.download("BTC-USD", period="3y")
    t10 = yf.download("^TNX", period="3y")
    t2 = yf.download("^IRX", period="3y")

    df = pd.DataFrame(index=spy.index)
    df["SPY"] = spy["Close"]
    df["VIX"] = vix["Close"]
    df["BTC"] = btc["Close"]

    # Approximate yield curve proxy (10y - 3m)
    df["YieldCurve"] = t10["Close"] - t2["Close"]

    return df.dropna()

df = load_data()

# =========================
# 2️⃣ FACTORS
# =========================

df['SPY_20d_vol'] = df['SPY'].pct_change().rolling(20).std()
df['SPY_50d_vol'] = df['SPY'].pct_change().rolling(50).std()

df['Trend'] = df['SPY'].pct_change(63)
df['Trend_z'] = (df['Trend'] - df['Trend'].rolling(252).mean()) / df['Trend'].rolling(252).std()

df['VIX_z'] = (df['VIX'] - df['VIX'].rolling(252).mean()) / df['VIX'].rolling(252).std()
df['YC_z'] = (df['YieldCurve'] - df['YieldCurve'].rolling(252).mean()) / df['YieldCurve'].rolling(252).std()

df['VolSpread'] = df['SPY_20d_vol'] - df['SPY_50d_vol']
df['Vol_z'] = (df['VolSpread'] - df['VolSpread'].rolling(252).mean()) / df['VolSpread'].rolling(252).std()

# =========================
# 3️⃣ TACTICAL MACRO SCORE
# =========================

df['MacroScore_raw'] = (
    0.40 * df['Trend_z'] +
    -0.30 * df['VIX_z'] +
    0.15 * df['YC_z'] +
    -0.15 * df['Vol_z']
)

df['MacroScore'] = df['MacroScore_raw'].ewm(span=8).mean()

upper = 0.4
lower = -0.4

conditions = [
    df['MacroScore'] > upper,
    df['MacroScore'] < lower
]

choices = ['Risk-On', 'Risk-Off']

df['MacroRegime'] = np.select(conditions, choices, default='Neutral')

# =========================
# 4️⃣ REGIME BLOCK FUNCTION
# =========================

def add_regime_blocks(fig, df, regime_col, row):

    colors = {
        'Risk-On': 'rgba(0,120,0,0.18)',
        'Neutral': 'rgba(180,150,0,0.18)',
        'Risk-Off': 'rgba(150,0,0,0.18)'
    }

    current_regime = None
    start_date = None

    for date, regime in df[regime_col].items():

        if regime != current_regime:

            if current_regime is not None:
                fig.add_vrect(
                    x0=start_date,
                    x1=prev_date,
                    fillcolor=colors[current_regime],
                    layer="below",
                    line_width=0,
                    row=row,
                    col=1
                )

            start_date = date
            current_regime = regime

        prev_date = date

    fig.add_vrect(
        x0=start_date,
        x1=prev_date,
        fillcolor=colors[current_regime],
        layer="below",
        line_width=0,
        row=row,
        col=1
    )

# =========================
# 5️⃣ BUILD FIGURE
# =========================

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.4, 0.4, 0.2]
)

fig.add_trace(
    go.Scatter(x=df.index, y=df['SPY'],
               line=dict(color='white', width=2)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=df.index, y=df['BTC'],
               line=dict(color='#f7931a', width=2)),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=df.index, y=df['MacroScore'],
               line=dict(color='cyan', width=2)),
    row=3, col=1
)

fig.add_hline(y=upper, line_dash="dot", row=3, col=1)
fig.add_hline(y=lower, line_dash="dot", row=3, col=1)

add_regime_blocks(fig, df, 'MacroRegime', 1)
add_regime_blocks(fig, df, 'MacroRegime', 2)

fig.update_layout(
    template='plotly_dark',
    height=950,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)
