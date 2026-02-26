"""
Market Immune System — Tactical Regime Engine v2.0
===================================================
Classifies macro environment into Risk-On / Neutral / Risk-Off using
trend, volatility, yield-curve, and technical indicators.

Enhancements over v1:
  • RSI, MACD, Bollinger Bands for SPY & BTC
  • Regime-change alerting panel with historical log
  • Dark theme, gauge widgets, tabbed layout
  • Fixed: pandas deprecations, data alignment, error handling
  • Optimized caching and modular code structure

Usage:
  pip install streamlit pandas numpy yfinance plotly
  streamlit run market_immune_system.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import json

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    layout="wide",
    page_title="Market Immune System v2",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ──────────────────────────────────
st.markdown("""
<style>
    /* Global dark overrides */
    .stApp { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1d24;
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
        color: #e0e0e0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262b36;
        border-bottom: 2px solid #6366f1;
        color: #ffffff;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1d24 0%, #21252e 100%);
        border: 1px solid #2d3344;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-label { color: #8b95a5; font-size: 0.85rem; margin-bottom: 4px; }
    .metric-value { color: #ffffff; font-size: 1.8rem; font-weight: 700; }
    .alert-card {
        background: #1a1d24;
        border-left: 4px solid;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .alert-risk-on  { border-left-color: #22c55e; }
    .alert-neutral   { border-left-color: #eab308; }
    .alert-risk-off  { border-left-color: #ef4444; }
    div[data-testid="stExpander"] {
        background-color: #1a1d24;
        border: 1px solid #2d3344;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="color:#e0e0e0; margin-bottom:0;">🛡️ Market Immune System</h1>
<p style="color:#8b95a5; font-size:1.05rem; margin-top:4px;">
Tactical Regime Engine — SPY & BTC macro classification with technical overlays
</p>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    lookback = st.selectbox("History", ["2y", "3y", "5y", "10y"], index=2)
    zscore_window = st.slider("Z-score window (days)", 126, 504, 252, step=21,
                               help="Rolling window for z-score normalization")
    ewm_span = st.slider("EWM smoothing span", 5, 30, 10,
                           help="Exponential smoothing applied before regime thresholding")
    upper_thresh = st.slider("Risk-On threshold", 0.1, 1.0, 0.4, step=0.05)
    lower_thresh = st.slider("Risk-Off threshold", -1.0, -0.1, -0.4, step=0.05)

    st.markdown("---")
    st.markdown("### 📊 Technical Indicators")
    show_rsi = st.checkbox("RSI (14-day)", value=True)
    show_macd = st.checkbox("MACD", value=True)
    show_bbands = st.checkbox("Bollinger Bands", value=True)

    st.markdown("---")
    st.markdown("### 🔄 Auto-Refresh")
    auto_refresh = st.checkbox("Enable auto-refresh", value=False)
    refresh_interval = st.selectbox("Refresh interval", [60, 300, 600, 1800, 3600],
                                     format_func=lambda x: {60: "1 min", 300: "5 min", 600: "10 min",
                                                             1800: "30 min", 3600: "1 hour"}[x],
                                     index=2)

    st.markdown("---")
    st.markdown("### 🔔 Alerts")
    alert_on_regime_change = st.checkbox("Show regime change alerts", value=True)
    alert_score_proximity = st.slider(
        "Score proximity warning",
        0.05, 0.20, 0.10, step=0.01,
        help="Warn when score is within this distance of a threshold"
    )


# =====================================================
# DATA LOADER — with error handling & alignment fixes
# =====================================================
@st.cache_data(ttl=14400, show_spinner="Fetching market data…")
def load_data(period: str) -> pd.DataFrame:
    tickers = {
        "SPY": "SPY",
        "VIX": "^VIX",
        "BTC": "BTC-USD",
        "TNX": "^TNX",
        "IRX": "^IRX",
    }
    data = {}
    errors = []

    for name, ticker in tickers.items():
        try:
            raw = yf.download(
                ticker,
                period=period,
                auto_adjust=False,
                progress=False,
            )
            if raw.empty:
                errors.append(f"{name} ({ticker}): no data returned")
                continue

            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.squeeze()
            close.name = name
            data[name] = close
        except Exception as e:
            errors.append(f"{name} ({ticker}): {str(e)[:80]}")

    if errors:
        for err in errors:
            st.warning(f"⚠️ Data issue — {err}")

    required = {"SPY", "VIX", "BTC"}
    if not required.issubset(data.keys()):
        st.error("Cannot load required tickers (SPY, VIX, BTC). Check your connection.")
        st.stop()

    # Align on trading days (SPY calendar) to avoid BTC weekend misalignment
    df = pd.concat(data.values(), axis=1)
    df.columns = list(data.keys())

    # Yield curve: 10yr - 3mo
    if "TNX" in df.columns and "IRX" in df.columns:
        df["YieldCurve"] = df["TNX"] - df["IRX"]
        df = df.drop(columns=["TNX", "IRX"])
    else:
        df["YieldCurve"] = 0.0

    # Forward-fill then back-fill to handle edges
    df["YieldCurve"] = df["YieldCurve"].ffill().bfill()
    df["VIX"] = df["VIX"].ffill()
    df["BTC"] = df["BTC"].ffill()

    # Only drop rows where SPY is missing (our calendar anchor)
    df = df.dropna(subset=["SPY"])
    df = df.dropna()

    return df


df = load_data(lookback)


# =====================================================
# ON-CHAIN / SENTIMENT DATA
# =====================================================
@st.cache_data(ttl=14400, show_spinner="Fetching on-chain data…")
def load_onchain_data():
    """Fetch BTC on-chain & sentiment metrics from free public APIs."""
    result = {}

    # 1. Crypto Fear & Greed Index (alternative.me — free, no key)
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=365&format=json",
            timeout=10,
        )
        if resp.status_code == 200:
            fng_raw = resp.json().get("data", [])
            fng_df = pd.DataFrame(fng_raw)
            fng_df["date"] = pd.to_datetime(fng_df["timestamp"].astype(int), unit="s")
            fng_df["value"] = fng_df["value"].astype(int)
            fng_df = fng_df.set_index("date").sort_index()
            result["fear_greed"] = fng_df
    except Exception:
        pass

    # 2. BTC volume data from yfinance for exchange-volume metrics
    try:
        btc_raw = yf.download("BTC-USD", period="2y", auto_adjust=False, progress=False)
        if not btc_raw.empty:
            vol = btc_raw["Volume"]
            if isinstance(vol, pd.DataFrame):
                vol = vol.iloc[:, 0]
            close = btc_raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            high = btc_raw["High"]
            if isinstance(high, pd.DataFrame):
                high = high.iloc[:, 0]
            low = btc_raw["Low"]
            if isinstance(low, pd.DataFrame):
                low = low.iloc[:, 0]

            vol_df = pd.DataFrame({
                "BTC_Volume": vol,
                "BTC_Close": close,
                "BTC_High": high,
                "BTC_Low": low,
            })
            # Volume SMA ratio (current vol vs 50d avg — proxy for exchange activity)
            vol_df["Vol_SMA50"] = vol_df["BTC_Volume"].rolling(50).mean()
            vol_df["Volume_Ratio"] = vol_df["BTC_Volume"] / vol_df["Vol_SMA50"]

            # Realized volatility: 30d annualized
            vol_df["BTC_LogRet"] = np.log(vol_df["BTC_Close"] / vol_df["BTC_Close"].shift(1))
            vol_df["Realized_Vol_30d"] = vol_df["BTC_LogRet"].rolling(30).std() * np.sqrt(365) * 100

            # NVT-lite proxy: market cap / volume (using price * vol as rough proxy)
            vol_df["Dollar_Volume"] = vol_df["BTC_Close"] * vol_df["BTC_Volume"]
            vol_df["NVT_Proxy"] = vol_df["BTC_Close"] / (
                vol_df["Dollar_Volume"].rolling(28).mean() / vol_df["BTC_Close"].rolling(28).mean()
            )

            # MVRV proxy: price / 200-day SMA (rough realized value proxy)
            vol_df["SMA200"] = vol_df["BTC_Close"].rolling(200).mean()
            vol_df["MVRV_Proxy"] = vol_df["BTC_Close"] / vol_df["SMA200"]

            result["volume_metrics"] = vol_df.dropna()
    except Exception:
        pass

    return result


onchain = load_onchain_data()

if len(df) < 300:
    st.error("Insufficient data history. Need at least 300 trading days.")
    st.stop()


# =====================================================
# TECHNICAL INDICATOR FUNCTIONS
# =====================================================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return sma, upper, lower


# =====================================================
# FACTOR ENGINE
# =====================================================
df["SPY_Return"] = df["SPY"].pct_change()
df["BTC_Return"] = df["BTC"].pct_change()
df["Vol20"] = df["SPY_Return"].rolling(20).std()
df["Vol50"] = df["SPY_Return"].rolling(50).std()
df["Momentum63"] = df["SPY"].pct_change(63)


def zscore(series: pd.Series, window: int = 252) -> pd.Series:
    mu = series.rolling(window).mean()
    sig = series.rolling(window).std()
    return (series - mu) / sig


df["Trend_z"] = zscore(df["Momentum63"], zscore_window)
df["VIX_z"] = zscore(df["VIX"], zscore_window)
df["YC_z"] = zscore(df["YieldCurve"], zscore_window)
df["VolSpread"] = df["Vol20"] - df["Vol50"]
df["Vol_z"] = zscore(df["VolSpread"], zscore_window)

# Technical indicators
df["SPY_RSI"] = compute_rsi(df["SPY"])
df["BTC_RSI"] = compute_rsi(df["BTC"])
df["SPY_MACD"], df["SPY_Signal"], df["SPY_MACD_Hist"] = compute_macd(df["SPY"])
df["BTC_MACD"], df["BTC_Signal"], df["BTC_MACD_Hist"] = compute_macd(df["BTC"])
df["SPY_BB_Mid"], df["SPY_BB_Upper"], df["SPY_BB_Lower"] = compute_bollinger(df["SPY"])
df["BTC_BB_Mid"], df["BTC_BB_Upper"], df["BTC_BB_Lower"] = compute_bollinger(df["BTC"])

df = df.dropna()

# =====================================================
# COMPOSITE SCORE
# =====================================================
weights = {
    "Trend_z": 0.40,
    "VIX_z": -0.30,
    "YC_z": 0.15,
    "Vol_z": -0.15,
}

df["MacroScore_raw"] = sum(w * df[k] for k, w in weights.items())
df["MacroScore"] = df["MacroScore_raw"].ewm(span=ewm_span).mean()

df["Regime"] = np.select(
    [df["MacroScore"] > upper_thresh, df["MacroScore"] < lower_thresh],
    ["Risk-On", "Risk-Off"],
    default="Neutral",
)

# =====================================================
# REGIME BLOCKS & ALERTS
# =====================================================
df["RegimeShift"] = (df["Regime"] != df["Regime"].shift()).astype(int)
df["RegimeBlock"] = df["RegimeShift"].cumsum()

current_duration = df.groupby("RegimeBlock").size().iloc[-1]
current_regime = df["Regime"].iloc[-1]
current_score = df["MacroScore"].iloc[-1]
prev_score = df["MacroScore"].iloc[-2] if len(df) > 1 else current_score
score_delta = current_score - prev_score

# Build alert log
alerts = []
shift_dates = df[df["RegimeShift"] == 1].index
for dt in shift_dates:
    idx = df.index.get_loc(dt)
    new_regime = df["Regime"].iloc[idx]
    old_regime = df["Regime"].iloc[idx - 1] if idx > 0 else "Unknown"
    score_at = df["MacroScore"].iloc[idx]
    alerts.append({
        "date": dt,
        "from": old_regime,
        "to": new_regime,
        "score": score_at,
    })

# Proximity warnings
proximity_upper = abs(current_score - upper_thresh) < alert_score_proximity
proximity_lower = abs(current_score - lower_thresh) < alert_score_proximity


# =====================================================
# REGIME TABLE & FORWARD RETURNS
# =====================================================
def build_regime_table(df):
    rows = []
    for _, group in df.groupby("RegimeBlock"):
        regime = group["Regime"].iloc[0]
        start_date = group.index[0]
        end_date = group.index[-1]
        duration = len(group)
        spy_ret = (group["SPY"].iloc[-1] / group["SPY"].iloc[0] - 1) * 100
        btc_ret = (group["BTC"].iloc[-1] / group["BTC"].iloc[0] - 1) * 100
        rows.append({
            "Regime": regime,
            "Start": start_date.strftime("%Y-%m-%d"),
            "End": end_date.strftime("%Y-%m-%d"),
            "Duration (days)": duration,
            "SPY Return (%)": round(spy_ret, 1),
            "BTC Return (%)": round(btc_ret, 1),
        })
    return pd.DataFrame(rows)


def forward_return_analysis(df, horizons=[21, 63, 126]):
    results = []
    for regime in ["Risk-On", "Neutral", "Risk-Off"]:
        mask = df["Regime"] == regime
        for h in horizons:
            spy_fwd = df["SPY"].pct_change(h).shift(-h)
            btc_fwd = df["BTC"].pct_change(h).shift(-h)
            results.append({
                "Regime": regime,
                "Horizon": f"{h}d (~{h // 21}mo)",
                "SPY Avg %": round(spy_fwd[mask].mean() * 100, 1),
                "BTC Avg %": round(btc_fwd[mask].mean() * 100, 1),
                "N": int(mask.sum()),
            })
    return pd.DataFrame(results)


regime_table = build_regime_table(df)
fwd_returns = forward_return_analysis(df)

# =====================================================
# COMPONENT CONTRIBUTIONS
# =====================================================
df["Contrib_Trend"] = weights["Trend_z"] * df["Trend_z"]
df["Contrib_VIX"] = weights["VIX_z"] * df["VIX_z"]
df["Contrib_YC"] = weights["YC_z"] * df["YC_z"]
df["Contrib_Vol"] = weights["Vol_z"] * df["Vol_z"]

# =====================================================
# COLORS
# =====================================================
REGIME_COLORS = {
    "Risk-On": "rgba(34, 197, 94, 0.08)",
    "Neutral": "rgba(234, 179, 8, 0.06)",
    "Risk-Off": "rgba(239, 68, 68, 0.08)",
}
BADGE_COLORS = {
    "Risk-On": "#22c55e",
    "Neutral": "#eab308",
    "Risk-Off": "#ef4444",
}
DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#c8cdd5"),
    margin=dict(t=40, b=30, l=50, r=20),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    ),
)


def add_regime_shading(fig, df, rows):
    for _, group in df.groupby("RegimeBlock"):
        regime = group["Regime"].iloc[0]
        x0 = group.index[0]
        x1 = group.index[-1]
        color = REGIME_COLORS[regime]
        for row in rows:
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor=color,
                line_width=0,
                layer="below",
                row=row, col=1,
            )


# =====================================================
# CURRENT STATUS PANEL (top of page)
# =====================================================
st.markdown("")

# Score gauge
def make_gauge(score, upper, lower):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": prev_score, "valueformat": ".3f"},
        number={"font": {"size": 38, "color": "#ffffff"}, "valueformat": ".3f"},
        gauge={
            "axis": {"range": [-2, 2], "tickcolor": "#555", "tickwidth": 1},
            "bar": {"color": BADGE_COLORS[current_regime], "thickness": 0.3},
            "bgcolor": "#1a1d24",
            "bordercolor": "#2d3344",
            "steps": [
                {"range": [-2, lower], "color": "rgba(239,68,68,0.15)"},
                {"range": [lower, upper], "color": "rgba(234,179,8,0.10)"},
                {"range": [upper, 2], "color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 2},
                "thickness": 0.8,
                "value": score,
            },
        },
        title={"text": "Composite Score", "font": {"size": 14, "color": "#8b95a5"}},
    ))
    fig.update_layout(
        height=220,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#c8cdd5"),
        margin=dict(t=50, b=10, l=30, r=30),
    )
    return fig


c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])

with c1:
    st.plotly_chart(make_gauge(current_score, upper_thresh, lower_thresh),
                    use_container_width=True, key="gauge")

with c2:
    badge_color = BADGE_COLORS[current_regime]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current Regime</div>
        <div class="metric-value" style="color:{badge_color};">{current_regime}</div>
        <div style="color:#8b95a5; margin-top:8px; font-size:0.85rem;">
            {int(current_duration)} days in regime
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    spy_price = df["SPY"].iloc[-1]
    spy_1d = df["SPY_Return"].iloc[-1] * 100
    spy_color = "#22c55e" if spy_1d >= 0 else "#ef4444"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">SPY</div>
        <div class="metric-value">${spy_price:,.2f}</div>
        <div style="color:{spy_color}; margin-top:8px; font-size:0.95rem; font-weight:600;">
            {"+" if spy_1d >= 0 else ""}{spy_1d:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    btc_price = df["BTC"].iloc[-1]
    btc_1d = df["BTC_Return"].iloc[-1] * 100
    btc_color = "#22c55e" if btc_1d >= 0 else "#ef4444"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">BTC</div>
        <div class="metric-value">${btc_price:,.0f}</div>
        <div style="color:{btc_color}; margin-top:8px; font-size:0.95rem; font-weight:600;">
            {"+" if btc_1d >= 0 else ""}{btc_1d:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Proximity warnings ──────────────────────────────
if proximity_upper and current_regime != "Risk-On":
    st.warning(f"⚡ Score ({current_score:.3f}) approaching Risk-On threshold ({upper_thresh})")
if proximity_lower and current_regime != "Risk-Off":
    st.warning(f"⚡ Score ({current_score:.3f}) approaching Risk-Off threshold ({lower_thresh})")

st.markdown("")

# =====================================================
# TABBED LAYOUT
# =====================================================
tab_regime, tab_technicals, tab_onchain, tab_alerts, tab_report, tab_analysis, tab_methodology = st.tabs([
    "📈 Regime Dashboard",
    "🔬 Technical Indicators",
    "⛓️ BTC On-Chain",
    "🔔 Alerts",
    "📋 Daily Report",
    "📊 Analysis",
    "📖 Methodology",
])

# ─────────────────────────────────────────────────────
# TAB 1: REGIME DASHBOARD
# ─────────────────────────────────────────────────────
with tab_regime:
    # Main chart: SPY, BTC, MacroScore, Factor Contributions
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.30, 0.30, 0.22, 0.18],
        subplot_titles=["SPY", "Bitcoin", "Composite Macro Score", "Factor Contributions"],
    )

    # SPY
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["SPY"],
            line=dict(width=1.5, color="#3b82f6"),
            name="SPY",
            hovertemplate="%{x|%b %d %Y}<br>$%{y:.2f}<extra>SPY</extra>",
        ),
        row=1, col=1,
    )

    # Bollinger Bands on SPY
    if show_bbands:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["SPY_BB_Upper"],
                line=dict(width=0.8, color="rgba(99,102,241,0.3)"),
                name="BB Upper", showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["SPY_BB_Lower"],
                line=dict(width=0.8, color="rgba(99,102,241,0.3)"),
                name="BB Lower", showlegend=False,
                fill="tonexty", fillcolor="rgba(99,102,241,0.05)",
            ),
            row=1, col=1,
        )

    # BTC
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["BTC"],
            line=dict(width=1.5, color="#f97316"),
            name="BTC",
            hovertemplate="%{x|%b %d %Y}<br>$%{y:,.0f}<extra>BTC</extra>",
        ),
        row=2, col=1,
    )

    if show_bbands:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BTC_BB_Upper"],
                line=dict(width=0.8, color="rgba(249,115,22,0.3)"),
                name="BB Upper (BTC)", showlegend=False,
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BTC_BB_Lower"],
                line=dict(width=0.8, color="rgba(249,115,22,0.3)"),
                name="BB Lower (BTC)", showlegend=False,
                fill="tonexty", fillcolor="rgba(249,115,22,0.05)",
            ),
            row=2, col=1,
        )

    # MacroScore
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["MacroScore"],
            line=dict(width=2, color="#818cf8"),
            name="MacroScore",
            hovertemplate="%{x|%b %d %Y}<br>Score: %{y:.3f}<extra></extra>",
        ),
        row=3, col=1,
    )
    fig.add_hline(y=upper_thresh, line_dash="dot", line_color="rgba(34,197,94,0.5)", row=3, col=1)
    fig.add_hline(y=lower_thresh, line_dash="dot", line_color="rgba(239,68,68,0.5)", row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(150,150,150,0.3)", row=3, col=1)

    # Factor contributions
    contrib_meta = {
        "Contrib_Trend": ("#3b82f6", "Trend (40%)"),
        "Contrib_VIX": ("#ef4444", "VIX (−30%)"),
        "Contrib_YC": ("#10b981", "Yield Curve (15%)"),
        "Contrib_Vol": ("#f59e0b", "Vol Spread (−15%)"),
    }
    for col_name, (color, label) in contrib_meta.items():
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df[col_name],
                line=dict(width=1.2, color=color),
                name=label,
                hovertemplate="%{x|%b %d %Y}<br>%{y:.3f}<extra>" + label + "</extra>",
            ),
            row=4, col=1,
        )
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(150,150,150,0.3)", row=4, col=1)

    # Regime shading
    add_regime_shading(fig, df, rows=[1, 2, 3, 4])

    fig.update_layout(
        height=1100,
        **DARK_LAYOUT,
    )
    fig.update_yaxes(title_text="USD", row=1, col=1, gridcolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(title_text="USD", row=2, col=1, gridcolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(title_text="Score", row=3, col=1, gridcolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(title_text="Contribution", row=4, col=1, gridcolor="rgba(255,255,255,0.04)")
    for i in range(1, 5):
        fig.update_xaxes(
            tickformat="%b '%y", tickangle=-30,
            showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            row=i, col=1,
        )

    st.plotly_chart(fig, use_container_width=True, key="regime_chart")


# ─────────────────────────────────────────────────────
# TAB 2: TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────
with tab_technicals:
    asset_choice = st.radio("Asset", ["SPY", "BTC"], horizontal=True)
    prefix = asset_choice

    tech_rows = 1 + int(show_rsi) + int(show_macd)
    row_heights = [0.4]
    titles = [f"{asset_choice} Price"]
    if show_bbands:
        titles[0] += " + Bollinger Bands"
    if show_rsi:
        row_heights.append(0.25)
        titles.append("RSI (14)")
    if show_macd:
        row_heights.append(0.35)
        titles.append("MACD")

    # Normalize heights
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    tech_fig = make_subplots(
        rows=tech_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=row_heights,
        subplot_titles=titles,
    )

    # Price + BB
    price_col = asset_choice
    tech_fig.add_trace(
        go.Scatter(
            x=df.index, y=df[price_col],
            line=dict(width=1.5, color="#3b82f6" if asset_choice == "SPY" else "#f97316"),
            name=asset_choice,
        ),
        row=1, col=1,
    )

    if show_bbands:
        bb_mid = f"{prefix}_BB_Mid"
        bb_up = f"{prefix}_BB_Upper"
        bb_lo = f"{prefix}_BB_Lower"
        tech_fig.add_trace(
            go.Scatter(x=df.index, y=df[bb_mid], line=dict(width=1, color="#818cf8", dash="dash"),
                        name="BB Mid", showlegend=True), row=1, col=1)
        tech_fig.add_trace(
            go.Scatter(x=df.index, y=df[bb_up], line=dict(width=0.8, color="rgba(129,140,248,0.4)"),
                        name="BB Upper", showlegend=False), row=1, col=1)
        tech_fig.add_trace(
            go.Scatter(x=df.index, y=df[bb_lo], line=dict(width=0.8, color="rgba(129,140,248,0.4)"),
                        name="BB Lower", showlegend=False,
                        fill="tonexty", fillcolor="rgba(129,140,248,0.07)"), row=1, col=1)

    current_row = 2

    # RSI
    if show_rsi:
        rsi_col = f"{prefix}_RSI"
        tech_fig.add_trace(
            go.Scatter(
                x=df.index, y=df[rsi_col],
                line=dict(width=1.5, color="#a78bfa"),
                name="RSI",
            ),
            row=current_row, col=1,
        )
        tech_fig.add_hline(y=70, line_dash="dot", line_color="rgba(239,68,68,0.5)",
                            row=current_row, col=1)
        tech_fig.add_hline(y=30, line_dash="dot", line_color="rgba(34,197,94,0.5)",
                            row=current_row, col=1)
        tech_fig.add_hline(y=50, line_dash="dot", line_color="rgba(150,150,150,0.3)",
                            row=current_row, col=1)
        tech_fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.05)",
                            line_width=0, row=current_row, col=1)
        tech_fig.add_hrect(y0=0, y1=30, fillcolor="rgba(34,197,94,0.05)",
                            line_width=0, row=current_row, col=1)
        current_row += 1

    # MACD
    if show_macd:
        macd_col = f"{prefix}_MACD"
        sig_col = f"{prefix}_Signal"
        hist_col = f"{prefix}_MACD_Hist"
        hist_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in df[hist_col]]

        tech_fig.add_trace(
            go.Bar(
                x=df.index, y=df[hist_col],
                marker_color=hist_colors,
                name="MACD Hist",
                opacity=0.5,
            ),
            row=current_row, col=1,
        )
        tech_fig.add_trace(
            go.Scatter(
                x=df.index, y=df[macd_col],
                line=dict(width=1.5, color="#3b82f6"),
                name="MACD",
            ),
            row=current_row, col=1,
        )
        tech_fig.add_trace(
            go.Scatter(
                x=df.index, y=df[sig_col],
                line=dict(width=1.5, color="#f97316"),
                name="Signal",
            ),
            row=current_row, col=1,
        )
        tech_fig.add_hline(y=0, line_dash="dot", line_color="rgba(150,150,150,0.3)",
                            row=current_row, col=1)

    add_regime_shading(tech_fig, df, rows=list(range(1, tech_rows + 1)))

    tech_fig.update_layout(
        height=250 + tech_rows * 220,
        **DARK_LAYOUT,
    )
    for i in range(1, tech_rows + 1):
        tech_fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", row=i, col=1)
        tech_fig.update_xaxes(
            tickformat="%b '%y", showgrid=True,
            gridcolor="rgba(255,255,255,0.04)", row=i, col=1,
        )

    st.plotly_chart(tech_fig, use_container_width=True, key="tech_chart")

    # Current indicator values
    st.markdown("#### Current Readings")
    ic1, ic2, ic3 = st.columns(3)
    rsi_val = df[f"{prefix}_RSI"].iloc[-1]
    macd_val = df[f"{prefix}_MACD"].iloc[-1]
    macd_sig = df[f"{prefix}_Signal"].iloc[-1]
    bb_pct = ((df[price_col].iloc[-1] - df[f"{prefix}_BB_Lower"].iloc[-1]) /
              (df[f"{prefix}_BB_Upper"].iloc[-1] - df[f"{prefix}_BB_Lower"].iloc[-1])) * 100

    rsi_status = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
    rsi_color = "#ef4444" if rsi_val > 70 else ("#22c55e" if rsi_val < 30 else "#eab308")

    with ic1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RSI (14)</div>
            <div class="metric-value" style="color:{rsi_color};">{rsi_val:.1f}</div>
            <div style="color:{rsi_color}; margin-top:4px; font-size:0.85rem;">{rsi_status}</div>
        </div>
        """, unsafe_allow_html=True)

    macd_cross = "Bullish" if macd_val > macd_sig else "Bearish"
    macd_color = "#22c55e" if macd_val > macd_sig else "#ef4444"
    with ic2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MACD</div>
            <div class="metric-value" style="color:{macd_color};">{macd_val:.2f}</div>
            <div style="color:{macd_color}; margin-top:4px; font-size:0.85rem;">{macd_cross} crossover</div>
        </div>
        """, unsafe_allow_html=True)

    bb_color = "#ef4444" if bb_pct > 90 else ("#22c55e" if bb_pct < 10 else "#818cf8")
    with ic3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Bollinger %B</div>
            <div class="metric-value" style="color:{bb_color};">{bb_pct:.0f}%</div>
            <div style="color:#8b95a5; margin-top:4px; font-size:0.85rem;">Position within bands</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# TAB 3: BTC ON-CHAIN
# ─────────────────────────────────────────────────────
with tab_onchain:
    st.markdown("### Bitcoin On-Chain & Sentiment Metrics")
    st.caption("Derived from public APIs and volume analysis. MVRV and NVT are proxy approximations.")

    # Fear & Greed Index
    if "fear_greed" in onchain:
        fng = onchain["fear_greed"]
        current_fng = int(fng["value"].iloc[-1])
        fng_class = fng["value_classification"].iloc[-1] if "value_classification" in fng.columns else ""

        fng_color = "#ef4444" if current_fng < 25 else (
            "#f59e0b" if current_fng < 45 else (
                "#eab308" if current_fng < 55 else (
                    "#22c55e" if current_fng < 75 else "#16a34a"
                )
            )
        )

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            # Fear & Greed gauge
            fng_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=current_fng,
                number={"font": {"size": 42, "color": "#ffffff"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#555"},
                    "bar": {"color": fng_color, "thickness": 0.3},
                    "bgcolor": "#1a1d24",
                    "bordercolor": "#2d3344",
                    "steps": [
                        {"range": [0, 25], "color": "rgba(239,68,68,0.15)"},
                        {"range": [25, 45], "color": "rgba(245,158,11,0.10)"},
                        {"range": [45, 55], "color": "rgba(234,179,8,0.10)"},
                        {"range": [55, 75], "color": "rgba(34,197,94,0.10)"},
                        {"range": [75, 100], "color": "rgba(22,163,74,0.15)"},
                    ],
                },
                title={"text": "Fear & Greed Index", "font": {"size": 14, "color": "#8b95a5"}},
            ))
            fng_gauge.update_layout(
                height=220,
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="#c8cdd5"),
                margin=dict(t=50, b=10, l=30, r=30),
            )
            st.plotly_chart(fng_gauge, use_container_width=True, key="fng_gauge")
            st.markdown(f"<p style='text-align:center; color:{fng_color}; font-weight:600;'>{fng_class}</p>",
                        unsafe_allow_html=True)

        # Fear & Greed time series
        fng_fig = go.Figure()
        fng_colors_ts = ["#ef4444" if v < 25 else "#f59e0b" if v < 45 else "#eab308" if v < 55
                         else "#22c55e" if v < 75 else "#16a34a" for v in fng["value"]]
        fng_fig.add_trace(go.Scatter(
            x=fng.index, y=fng["value"],
            mode="lines",
            line=dict(width=1.5, color="#a78bfa"),
            name="Fear & Greed",
            hovertemplate="%{x|%b %d %Y}<br>Score: %{y}<extra></extra>",
        ))
        fng_fig.add_hline(y=25, line_dash="dot", line_color="rgba(239,68,68,0.4)")
        fng_fig.add_hline(y=75, line_dash="dot", line_color="rgba(34,197,94,0.4)")
        fng_fig.add_hline(y=50, line_dash="dot", line_color="rgba(150,150,150,0.3)")
        fng_fig.add_hrect(y0=0, y1=25, fillcolor="rgba(239,68,68,0.05)", line_width=0)
        fng_fig.add_hrect(y0=75, y1=100, fillcolor="rgba(34,197,94,0.05)", line_width=0)
        fng_fig.update_layout(
            height=300,
            **DARK_LAYOUT,
            yaxis_title="Score",
            yaxis_range=[0, 100],
        )
        with fc2:
            st.plotly_chart(fng_fig, use_container_width=True, key="fng_ts")

        # Fear & Greed distribution
        fng_hist = go.Figure()
        fng_hist.add_trace(go.Histogram(
            x=fng["value"],
            nbinsx=20,
            marker_color="#818cf8",
            opacity=0.7,
        ))
        fng_hist.update_layout(
            height=300,
            **DARK_LAYOUT,
            xaxis_title="Score",
            yaxis_title="Frequency",
        )
        with fc3:
            st.plotly_chart(fng_hist, use_container_width=True, key="fng_hist")
    else:
        st.info("Fear & Greed data unavailable. API may be down.")

    st.markdown("---")

    # Volume & On-Chain Proxy Metrics
    if "volume_metrics" in onchain:
        vm = onchain["volume_metrics"]

        # Current values
        oc1, oc2, oc3, oc4 = st.columns(4)
        with oc1:
            mvrv_val = vm["MVRV_Proxy"].iloc[-1]
            mvrv_color = "#ef4444" if mvrv_val > 2.5 else ("#22c55e" if mvrv_val < 1.0 else "#818cf8")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MVRV Proxy (P/200SMA)</div>
                <div class="metric-value" style="color:{mvrv_color};">{mvrv_val:.2f}x</div>
                <div style="color:#8b95a5; margin-top:4px; font-size:0.8rem;">
                    {"Overheated" if mvrv_val > 2.5 else "Undervalued" if mvrv_val < 1.0 else "Fair value range"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with oc2:
            vol_ratio = vm["Volume_Ratio"].iloc[-1]
            vr_color = "#22c55e" if vol_ratio > 1.5 else ("#ef4444" if vol_ratio < 0.5 else "#eab308")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Volume Ratio (vs 50d)</div>
                <div class="metric-value" style="color:{vr_color};">{vol_ratio:.2f}x</div>
                <div style="color:#8b95a5; margin-top:4px; font-size:0.8rem;">
                    {"High activity" if vol_ratio > 1.5 else "Low activity" if vol_ratio < 0.5 else "Normal"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with oc3:
            rvol = vm["Realized_Vol_30d"].iloc[-1]
            rv_color = "#ef4444" if rvol > 80 else ("#22c55e" if rvol < 40 else "#eab308")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Realized Vol (30d ann.)</div>
                <div class="metric-value" style="color:{rv_color};">{rvol:.1f}%</div>
                <div style="color:#8b95a5; margin-top:4px; font-size:0.8rem;">
                    {"Extreme" if rvol > 80 else "Calm" if rvol < 40 else "Moderate"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with oc4:
            nvt = vm["NVT_Proxy"].iloc[-1]
            nvt_color = "#ef4444" if nvt > 1.5 else ("#22c55e" if nvt < 0.5 else "#818cf8")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">NVT Proxy</div>
                <div class="metric-value" style="color:{nvt_color};">{nvt:.3f}</div>
                <div style="color:#8b95a5; margin-top:4px; font-size:0.8rem;">
                    Network value / transaction proxy
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Charts: MVRV + Volume Ratio + Realized Vol
        onchain_fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.35, 0.35, 0.30],
            subplot_titles=["MVRV Proxy (Price / 200-SMA)", "Volume Ratio (Daily / 50d Avg)", "Realized Volatility (30d Annualized)"],
        )

        onchain_fig.add_trace(
            go.Scatter(x=vm.index, y=vm["MVRV_Proxy"], line=dict(width=1.5, color="#a78bfa"),
                       name="MVRV Proxy",
                       hovertemplate="%{x|%b %d %Y}<br>MVRV: %{y:.2f}x<extra></extra>"),
            row=1, col=1,
        )
        onchain_fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(150,150,150,0.4)", row=1, col=1)
        onchain_fig.add_hline(y=2.5, line_dash="dot", line_color="rgba(239,68,68,0.5)", row=1, col=1)
        onchain_fig.add_hrect(y0=2.5, y1=4.0, fillcolor="rgba(239,68,68,0.05)", line_width=0, row=1, col=1)
        onchain_fig.add_hrect(y0=0, y1=1.0, fillcolor="rgba(34,197,94,0.05)", line_width=0, row=1, col=1)

        vol_colors = ["#22c55e" if v > 1.5 else "#ef4444" if v < 0.5 else "#818cf8"
                      for v in vm["Volume_Ratio"]]
        onchain_fig.add_trace(
            go.Bar(x=vm.index, y=vm["Volume_Ratio"], marker_color=vol_colors,
                   name="Vol Ratio", opacity=0.6,
                   hovertemplate="%{x|%b %d %Y}<br>Ratio: %{y:.2f}x<extra></extra>"),
            row=2, col=1,
        )
        onchain_fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(150,150,150,0.4)", row=2, col=1)

        onchain_fig.add_trace(
            go.Scatter(x=vm.index, y=vm["Realized_Vol_30d"],
                       line=dict(width=1.5, color="#f59e0b"),
                       name="Realized Vol",
                       fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
                       hovertemplate="%{x|%b %d %Y}<br>RVol: %{y:.1f}%<extra></extra>"),
            row=3, col=1,
        )
        onchain_fig.add_hline(y=80, line_dash="dot", line_color="rgba(239,68,68,0.5)", row=3, col=1)
        onchain_fig.add_hline(y=40, line_dash="dot", line_color="rgba(34,197,94,0.4)", row=3, col=1)

        onchain_fig.update_layout(height=850, **DARK_LAYOUT)
        for i in range(1, 4):
            onchain_fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", row=i, col=1)
            onchain_fig.update_xaxes(tickformat="%b '%y", showgrid=True,
                                      gridcolor="rgba(255,255,255,0.04)", row=i, col=1)

        st.plotly_chart(onchain_fig, use_container_width=True, key="onchain_charts")
    else:
        st.info("Volume metrics unavailable.")


# ─────────────────────────────────────────────────────
# TAB 4: ALERTS
# ─────────────────────────────────────────────────────
with tab_alerts:
    if not alert_on_regime_change:
        st.info("Enable alerts in the sidebar to view regime change notifications.")
    else:
        st.markdown("### Regime Change Log")
        st.caption("Every time the composite score crossed a threshold, triggering a regime shift.")

        if proximity_upper and current_regime != "Risk-On":
            st.markdown(f"""
            <div class="alert-card alert-risk-on" style="border-left-color:#eab308;">
                <strong style="color:#eab308;">⚡ PROXIMITY WARNING</strong><br>
                <span style="color:#c8cdd5;">Score ({current_score:.3f}) is within
                {alert_score_proximity} of Risk-On threshold ({upper_thresh}).
                Possible regime shift imminent.</span>
            </div>
            """, unsafe_allow_html=True)

        if proximity_lower and current_regime != "Risk-Off":
            st.markdown(f"""
            <div class="alert-card alert-risk-off" style="border-left-color:#eab308;">
                <strong style="color:#eab308;">⚡ PROXIMITY WARNING</strong><br>
                <span style="color:#c8cdd5;">Score ({current_score:.3f}) is within
                {alert_score_proximity} of Risk-Off threshold ({lower_thresh}).
                Possible regime shift imminent.</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Show recent alerts first
        recent_alerts = sorted(alerts, key=lambda a: a["date"], reverse=True)
        num_show = st.slider("Show last N regime changes", 5, min(50, len(recent_alerts)),
                              min(15, len(recent_alerts)))

        for a in recent_alerts[:num_show]:
            css_class = f"alert-{a['to'].lower().replace('-', '-')}"
            arrow_color = BADGE_COLORS.get(a["to"], "#888")
            from_color = BADGE_COLORS.get(a["from"], "#888")
            st.markdown(f"""
            <div class="alert-card {css_class}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <strong style="color:#e0e0e0;">{a['date'].strftime('%b %d, %Y')}</strong>
                        &nbsp;&nbsp;
                        <span style="color:{from_color}; font-weight:600;">{a['from']}</span>
                        <span style="color:#8b95a5;"> → </span>
                        <span style="color:{arrow_color}; font-weight:600;">{a['to']}</span>
                    </div>
                    <div style="color:#8b95a5; font-size:0.85rem;">
                        Score: {a['score']:.3f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Regime change frequency chart
        st.markdown("### Regime Change Frequency")
        if len(alerts) > 0:
            alert_df = pd.DataFrame(alerts)
            alert_df["month"] = alert_df["date"].dt.to_period("M").astype(str)
            freq = alert_df.groupby("month").size().reset_index(name="changes")
            freq_fig = go.Figure(
                go.Bar(
                    x=freq["month"], y=freq["changes"],
                    marker_color="#818cf8",
                    hovertemplate="%{x}<br>%{y} regime changes<extra></extra>",
                )
            )
            freq_fig.update_layout(
                height=250,
                **DARK_LAYOUT,
                yaxis_title="# Changes",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(freq_fig, use_container_width=True, key="freq_chart")


# ─────────────────────────────────────────────────────
# TAB 5: DAILY REPORT
# ─────────────────────────────────────────────────────
with tab_report:
    st.markdown("### 📋 Daily Regime Report")
    st.caption("Auto-generated summary — equivalent to a morning briefing. "
               "Saves a persistent log to track regime changes over time.")

    # ── Regime state persistence (uses Streamlit session + optional file) ──
    import os as _os

    REPORT_DIR = _os.path.dirname(_os.path.abspath(__file__))
    STATE_FILE = _os.path.join(REPORT_DIR, "regime_state.json")
    LOG_FILE = _os.path.join(REPORT_DIR, "regime_alert_log.csv")

    # Load previous state
    prev_state = None
    regime_changed = False
    prev_regime_saved = None
    try:
        if _os.path.exists(STATE_FILE):
            with open(STATE_FILE) as _f:
                prev_state = json.load(_f)
                prev_regime_saved = prev_state.get("regime")
                if prev_regime_saved and prev_regime_saved != current_regime:
                    regime_changed = True
    except Exception:
        pass

    # Save current state
    try:
        with open(STATE_FILE, "w") as _f:
            json.dump({
                "regime": current_regime,
                "score": round(float(current_score), 4),
                "date": df.index[-1].strftime("%Y-%m-%d"),
                "updated_at": datetime.now().isoformat(),
            }, _f, indent=2)
    except Exception:
        pass

    # Append to persistent log
    try:
        log_row = pd.DataFrame([{
            "date": df.index[-1].strftime("%Y-%m-%d"),
            "checked_at": datetime.now().isoformat(),
            "regime": current_regime,
            "score": round(float(current_score), 4),
            "spy_price": round(float(df["SPY"].iloc[-1]), 2),
            "btc_price": round(float(df["BTC"].iloc[-1]), 2),
            "vix": round(float(df["VIX"].iloc[-1]), 2),
            "spy_rsi": round(float(df["SPY_RSI"].iloc[-1]), 1),
            "btc_rsi": round(float(df["BTC_RSI"].iloc[-1]), 1),
            "regime_changed": regime_changed,
        }])
        if _os.path.exists(LOG_FILE):
            existing_log = pd.read_csv(LOG_FILE)
            # Only append if date changed
            if existing_log["date"].iloc[-1] != log_row["date"].iloc[0]:
                combined_log = pd.concat([existing_log, log_row], ignore_index=True)
                combined_log.to_csv(LOG_FILE, index=False)
        else:
            log_row.to_csv(LOG_FILE, index=False)
    except Exception:
        pass

    # ── Regime change banner ──
    if regime_changed:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1d24, #2d1a1a); border: 2px solid #ef4444;
                    border-radius: 12px; padding: 20px; margin-bottom: 20px; text-align:center;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444; margin-bottom: 8px;">
                🚨 REGIME CHANGE DETECTED
            </div>
            <div style="font-size: 1.1rem; color: #e0e0e0;">
                <span style="color:{BADGE_COLORS.get(prev_regime_saved, '#888')}; font-weight:600;">{prev_regime_saved}</span>
                <span style="color:#8b95a5;"> → </span>
                <span style="color:{BADGE_COLORS[current_regime]}; font-weight:600;">{current_regime}</span>
            </div>
            <div style="color:#8b95a5; margin-top:8px; font-size:0.9rem;">
                Score: {current_score:.3f} | Detected: {datetime.now().strftime('%b %d, %Y %H:%M')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        status_color = BADGE_COLORS[current_regime]
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1d24, #21252e); border: 1px solid #2d3344;
                    border-radius: 12px; padding: 20px; margin-bottom: 20px; text-align:center;">
            <div style="font-size: 1.1rem; color: #8b95a5; margin-bottom: 4px;">Current Regime</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {status_color};">{current_regime}</div>
            <div style="color:#8b95a5; margin-top:8px; font-size:0.9rem;">
                No change since last check | Score: {current_score:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Report body ──
    # Score trend
    scores_5d = df["MacroScore"].iloc[-5:]
    score_trend = "📈 Rising" if scores_5d.iloc[-1] > scores_5d.iloc[0] else "📉 Falling"
    spy_5d = (df["SPY"].iloc[-1] / df["SPY"].iloc[-5] - 1) * 100
    btc_5d = (df["BTC"].iloc[-1] / df["BTC"].iloc[-5] - 1) * 100
    spy_1d_val = df["SPY_Return"].iloc[-1] * 100
    btc_1d_val = df["BTC_Return"].iloc[-1] * 100

    rc1, rc2 = st.columns(2)

    with rc1:
        st.markdown("#### Prices & Returns")
        report_price_data = pd.DataFrame({
            "Asset": ["SPY", "BTC"],
            "Price": [f"${df['SPY'].iloc[-1]:,.2f}", f"${df['BTC'].iloc[-1]:,.0f}"],
            "1-Day": [f"{spy_1d_val:+.2f}%", f"{btc_1d_val:+.2f}%"],
            "5-Day": [f"{spy_5d:+.2f}%", f"{btc_5d:+.2f}%"],
            "RSI": [f"{df['SPY_RSI'].iloc[-1]:.1f}", f"{df['BTC_RSI'].iloc[-1]:.1f}"],
        })
        st.dataframe(report_price_data, use_container_width=True, hide_index=True, height=115)

    with rc2:
        st.markdown("#### Factor Breakdown")
        contrib_trend_val = weights["Trend_z"] * df["Trend_z"].iloc[-1]
        contrib_vix_val = weights["VIX_z"] * df["VIX_z"].iloc[-1]
        contrib_yc_val = weights["YC_z"] * df["YC_z"].iloc[-1]
        contrib_vol_val = weights["Vol_z"] * df["Vol_z"].iloc[-1]
        factor_data = pd.DataFrame({
            "Factor": ["Trend (40%)", "VIX (−30%)", "Yield Curve (15%)", "Vol Spread (−15%)"],
            "Z-Score": [f"{df['Trend_z'].iloc[-1]:.3f}", f"{df['VIX_z'].iloc[-1]:.3f}",
                        f"{df['YC_z'].iloc[-1]:.3f}", f"{df['Vol_z'].iloc[-1]:.3f}"],
            "Contribution": [f"{contrib_trend_val:+.3f}", f"{contrib_vix_val:+.3f}",
                             f"{contrib_yc_val:+.3f}", f"{contrib_vol_val:+.3f}"],
        })
        st.dataframe(factor_data, use_container_width=True, hide_index=True, height=180)

    st.markdown(f"""
    <div style="background:#1a1d24; border:1px solid #2d3344; border-radius:8px; padding:16px; margin-top:12px;">
        <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:16px;">
            <div><span style="color:#8b95a5;">Score Trend (5d):</span> <span style="color:#e0e0e0;">{score_trend}</span></div>
            <div><span style="color:#8b95a5;">VIX:</span> <span style="color:#e0e0e0;">{df['VIX'].iloc[-1]:.2f}</span></div>
            <div><span style="color:#8b95a5;">Days in Regime:</span> <span style="color:#e0e0e0;">{int(current_duration)}</span></div>
            <div><span style="color:#8b95a5;">Data Through:</span> <span style="color:#e0e0e0;">{df.index[-1].strftime('%b %d, %Y')}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Proximity warnings in report
    if abs(current_score - upper_thresh) < alert_score_proximity and current_regime != "Risk-On":
        st.warning(f"⚡ Score ({current_score:.3f}) approaching Risk-On threshold ({upper_thresh})")
    if abs(current_score - lower_thresh) < alert_score_proximity and current_regime != "Risk-Off":
        st.warning(f"⚡ Score ({current_score:.3f}) approaching Risk-Off threshold ({lower_thresh})")

    # ── Historical log viewer ──
    st.markdown("---")
    st.markdown("#### 📜 Regime Check Log")
    st.caption("Every time this dashboard loads, it logs the current state. Regime changes are flagged.")
    try:
        if _os.path.exists(LOG_FILE):
            hist_log = pd.read_csv(LOG_FILE)
            hist_log = hist_log.sort_values("date", ascending=False)

            def style_log(tbl):
                def _regime_bg(val):
                    c = {"Risk-On": "#1a3a1a", "Neutral": "#3a3a1a", "Risk-Off": "#3a1a1a"}.get(val, "")
                    return f"background-color: {c}; color: #e0e0e0"
                def _changed(val):
                    if val == True or val == "True":
                        return "background-color: #3a1a1a; color: #ef4444; font-weight: 600"
                    return ""
                return (tbl.style
                        .map(_regime_bg, subset=["regime"])
                        .map(_changed, subset=["regime_changed"]))

            st.dataframe(style_log(hist_log), use_container_width=True, hide_index=True, height=300)
        else:
            st.info("No log entries yet. The log builds up each time the dashboard loads.")
    except Exception as e:
        st.info(f"Log not available yet — will populate on next reload.")


# ─────────────────────────────────────────────────────
# TAB 6: ANALYSIS
# ─────────────────────────────────────────────────────
with tab_analysis:
    st.markdown("### Regime History")
    st.caption("Each row is a continuous regime period. Returns are buy-and-hold within that window.")

    def style_regime_table(tbl):
        def color_regime(val):
            c = {"Risk-On": "#1a3a1a", "Neutral": "#3a3a1a", "Risk-Off": "#3a1a1a"}.get(val, "")
            return f"background-color: {c}; color: #e0e0e0"

        def color_ret(val):
            try:
                v = float(val)
                return "color: #22c55e" if v > 0 else "color: #ef4444"
            except (ValueError, TypeError):
                return ""

        return (tbl.style
                .map(color_regime, subset=["Regime"])
                .map(color_ret, subset=["SPY Return (%)", "BTC Return (%)"]))

    st.dataframe(style_regime_table(regime_table), use_container_width=True, height=320)

    st.markdown("---")
    st.markdown("### Forward Return Analysis")
    st.caption(
        "Average SPY and BTC returns measured N trading days *after* each regime classification. "
        "21d ≈ 1mo, 63d ≈ 3mo, 126d ≈ 6mo."
    )

    def style_fwd(tbl):
        def color_val(val):
            try:
                v = float(val)
                return "color: #22c55e; font-weight:600" if v > 0 else "color: #ef4444; font-weight:600"
            except (ValueError, TypeError):
                return ""

        def color_regime(val):
            c = {"Risk-On": "#1a3a1a", "Neutral": "#3a3a1a", "Risk-Off": "#3a1a1a"}.get(val, "")
            return f"background-color: {c}; color: #e0e0e0"

        return (tbl.style
                .map(color_regime, subset=["Regime"])
                .map(color_val, subset=["SPY Avg %", "BTC Avg %"]))

    st.dataframe(style_fwd(fwd_returns), use_container_width=True, hide_index=True, height=250)

    st.markdown("---")
    st.markdown("### BTC / SPY Return Correlation by Regime")
    st.caption("Daily return correlation within each regime classification.")

    corr_rows = []
    for regime in ["Risk-On", "Neutral", "Risk-Off"]:
        mask = df["Regime"] == regime
        subset = df[mask]
        spy_ret = subset["SPY"].pct_change().dropna()
        btc_ret = subset["BTC"].pct_change().dropna()
        aligned = pd.concat([spy_ret, btc_ret], axis=1).dropna()
        corr = aligned.corr().iloc[0, 1] if len(aligned) > 10 else np.nan
        corr_rows.append({
            "Regime": regime,
            "Correlation": round(corr, 3),
            "Trading Days": int(mask.sum()),
        })

    corr_df = pd.DataFrame(corr_rows)

    # Correlation heatmap-style chart
    corr_fig = go.Figure()
    for _, row in corr_df.iterrows():
        color = "#ef4444" if row["Correlation"] > 0.5 else (
            "#22c55e" if row["Correlation"] < 0.2 else "#eab308"
        )
        corr_fig.add_trace(go.Bar(
            x=[row["Regime"]],
            y=[row["Correlation"]],
            marker_color=color,
            text=f"{row['Correlation']:.3f}",
            textposition="outside",
            textfont=dict(color="#e0e0e0"),
            name=row["Regime"],
            showlegend=False,
        ))

    corr_fig.update_layout(
        height=300,
        **DARK_LAYOUT,
        yaxis_title="Correlation",
        yaxis_range=[0, 1],
    )
    st.plotly_chart(corr_fig, use_container_width=True, key="corr_chart")

    # Rolling correlation
    st.markdown("### Rolling 63-day BTC/SPY Correlation")
    rolling_corr = df["SPY_Return"].rolling(63).corr(df["BTC_Return"])
    rc_fig = go.Figure()
    rc_fig.add_trace(go.Scatter(
        x=df.index, y=rolling_corr,
        line=dict(width=1.5, color="#a78bfa"),
        name="63d Rolling Corr",
        hovertemplate="%{x|%b %d %Y}<br>Corr: %{y:.3f}<extra></extra>",
    ))
    rc_fig.add_hline(y=0, line_dash="dot", line_color="rgba(150,150,150,0.3)")
    rc_fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(239,68,68,0.4)")
    add_regime_shading(rc_fig, df, rows=[1])
    rc_fig.update_layout(height=300, **DARK_LAYOUT, yaxis_title="Correlation")
    st.plotly_chart(rc_fig, use_container_width=True, key="rolling_corr")


# ─────────────────────────────────────────────────────
# TAB 5: METHODOLOGY
# ─────────────────────────────────────────────────────
with tab_methodology:
    st.markdown("""
### Factor Model

The composite macro score is a weighted combination of four z-score normalized factors,
each computed over a rolling window (default 252 trading days = 1 year).

| Factor | Calculation | Weight | Rationale |
|--------|-------------|--------|-----------|
| **Trend** | 63-day SPY return, z-scored | +40% | Strongest single predictor of regime persistence |
| **VIX** | Raw VIX level, z-scored | −30% | Elevated fear = risk-off pressure |
| **Yield Curve** | 10yr − 3mo spread, z-scored | +15% | Steeper curve = growth optimism |
| **Vol Spread** | 20d − 50d rolling vol, z-scored | −15% | Rising near-term vol = stress |

### Score Construction

1. Weighted sum of z-scored factors
2. EWM smoothing (span=10) to reduce whipsawing
3. Threshold at ±0.4 → regime classification

### Technical Indicators

- **RSI (14):** Relative Strength Index using EWM-smoothed gains/losses
- **MACD:** 12/26/9 exponential moving average crossover system
- **Bollinger Bands:** 20-day SMA ± 2 standard deviations

### Alerting System

Alerts trigger on two conditions:
1. **Regime change:** score crosses a threshold boundary
2. **Proximity warning:** score is within configurable distance of a threshold

### Key Assumptions

- 63 trading days ≈ 3 calendar months
- 252 trading days ≈ 1 calendar year
- `^IRX` (13-week T-bill) proxies the short-end of the curve
- BTC data forward-filled to align with equity trading calendar
- All returns are price returns (not total return)
- Forward returns may include overlap within the same regime block
    """)


# =====================================================
# AUTO-REFRESH
# =====================================================
if auto_refresh:
    st.markdown(
        f"""<meta http-equiv="refresh" content="{refresh_interval}">
        <p style="text-align:center; color:#4ade80; font-size:0.8rem;">
        🔄 Auto-refresh active — every {refresh_interval // 60} min{'s' if refresh_interval > 60 else ''}
        </p>""",
        unsafe_allow_html=True,
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
last_update = df.index[-1].strftime("%b %d, %Y")
st.markdown(
    f"<p style='text-align:center; color:#555; font-size:0.8rem;'>"
    f"Market Immune System v3.0 — Data through {last_update} via Yahoo Finance — Not financial advice"
    f"</p>",
    unsafe_allow_html=True,
)
