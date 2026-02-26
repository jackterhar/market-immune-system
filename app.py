import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# 1️⃣ PREP
# =========================

df = df.copy()
df = df.dropna()

# Volatility measures
df['SPY_20d_vol'] = df['SPY'].pct_change().rolling(20).std()
df['SPY_50d_vol'] = df['SPY'].pct_change().rolling(50).std()

# =========================
# 2️⃣ TACTICAL FACTORS (Z-SCORED)
# =========================

# --- Trend (3 month momentum) ---
df['Trend'] = df['SPY'].pct_change(63)
df['Trend_z'] = (df['Trend'] - df['Trend'].rolling(252).mean()) / df['Trend'].rolling(252).std()

# --- VIX ---
df['VIX_z'] = (df['VIX'] - df['VIX'].rolling(252).mean()) / df['VIX'].rolling(252).std()

# --- Yield Curve ---
df['YC_z'] = (df['YieldCurve'] - df['YieldCurve'].rolling(252).mean()) / df['YieldCurve'].rolling(252).std()

# --- Volatility Regime ---
df['VolSpread'] = df['SPY_20d_vol'] - df['SPY_50d_vol']
df['Vol_z'] = (df['VolSpread'] - df['VolSpread'].rolling(252).mean()) / df['VolSpread'].rolling(252).std()

# =========================
# 3️⃣ WEIGHTED MACRO SCORE (TACTICAL)
# =========================

df['MacroScore_raw'] = (
    0.40 * df['Trend_z'] +
    -0.30 * df['VIX_z'] +
    0.15 * df['YC_z'] +
    -0.15 * df['Vol_z']
)

# Tactical smoothing (fast but not noisy)
df['MacroScore'] = df['MacroScore_raw'].ewm(span=8).mean()

# =========================
# 4️⃣ 3-STATE REGIME
# =========================

upper = 0.4
lower = -0.4

conditions = [
    df['MacroScore'] > upper,
    df['MacroScore'] < lower
]

choices = ['Risk-On', 'Risk-Off']

df['MacroRegime'] = np.select(conditions, choices, default='Neutral')

# =========================
# 5️⃣ REGIME BLOCK RENDERING (NO STRIPES)
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
                    opacity=1,
                    layer="below",
                    line_width=0,
                    row=row,
                    col=1
                )

            start_date = date
            current_regime = regime

        prev_date = date

    # final block
    fig.add_vrect(
        x0=start_date,
        x1=prev_date,
        fillcolor=colors[current_regime],
        opacity=1,
        layer="below",
        line_width=0,
        row=row,
        col=1
    )

# =========================
# 6️⃣ BUILD CHART
# =========================

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.4, 0.4, 0.2]
)

# --- SPY ---
fig.add_trace(
    go.Scatter(x=df.index, y=df['SPY'],
               mode='lines',
               line=dict(color='white', width=2),
               name='SPY'),
    row=1, col=1
)

# --- BTC ---
fig.add_trace(
    go.Scatter(x=df.index, y=df['BTC'],
               mode='lines',
               line=dict(color='#f7931a', width=2),
               name='BTC'),
    row=2, col=1
)

# --- MacroScore Panel ---
fig.add_trace(
    go.Scatter(x=df.index, y=df['MacroScore'],
               mode='lines',
               line=dict(color='cyan', width=2),
               name='MacroScore'),
    row=3, col=1
)

fig.add_hline(y=upper, line_dash="dot", line_color="green", row=3, col=1)
fig.add_hline(y=lower, line_dash="dot", line_color="red", row=3, col=1)

# Add regime shading to SPY + BTC panels
add_regime_blocks(fig, df, 'MacroRegime', row=1)
add_regime_blocks(fig, df, 'MacroRegime', row=2)

# =========================
# 7️⃣ LAYOUT CLEANUP
# =========================

fig.update_layout(
    template='plotly_dark',
    height=1000,
    title="Market Immune System (Tactical)",
    showlegend=False
)

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')

fig.show()
