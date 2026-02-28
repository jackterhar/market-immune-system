"""
Market Immune System — Dual Regime Engine v5.0
================================================
Independent regime classification for SPY and BTC:
  • SPY regime: equity momentum, VIX, yield curve, vol spread
  • BTC regime: crypto momentum, realized vol, Fear & Greed, volume ratio
  • BTC price data sourced from Coinbase Exchange API

Usage:
  pip install streamlit pandas numpy yfinance plotly requests
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
import os
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    layout="wide",
    page_title="Market Immune System v4",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ──────────────────────────────────
st.markdown("""
<style>
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
Dual Regime Engine — Independent SPY &amp; BTC macro classification
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
# DATA LOADER
# =====================================================
@st.cache_data(ttl=14400, show_spinner="Fetching market data…")
def _fetch_coinbase_btc(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch BTC-USD daily candles from Coinbase Exchange API.

    The public endpoint returns max 300 candles per request,
    so we paginate in 300-day windows and stitch together.
    Returns DataFrame with columns: BTC, BTC_High, BTC_Volume indexed by date.
    """
    CB_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    GRANULARITY = 86400  # 1 day
    MAX_CANDLES = 300
    all_candles = []

    window_start = start
    while window_start < end:
        window_end = min(window_start + timedelta(days=MAX_CANDLES), end)
        params = {
            "start": window_start.strftime("%Y-%m-%dT00:00:00Z"),
            "end": window_end.strftime("%Y-%m-%dT00:00:00Z"),
            "granularity": GRANULARITY,
        }
        resp = requests.get(CB_URL, params=params, timeout=15)
        resp.raise_for_status()
        candles = resp.json()
        # Each candle: [timestamp, low, high, open, close, volume]
        all_candles.extend(candles)
        window_start = window_end
        time.sleep(0.15)  # rate-limit courtesy

    if not all_candles:
        return pd.DataFrame()

    cb = pd.DataFrame(all_candles, columns=["ts", "low", "high", "open", "close", "volume"])
    cb["date"] = pd.to_datetime(cb["ts"], unit="s")
    cb = cb.sort_values("date").drop_duplicates(subset="date", keep="last")
    cb = cb.set_index("date")
    result = pd.DataFrame({"BTC": cb["close"], "BTC_High": cb["high"], "BTC_Volume": cb["volume"]})
    result.index.name = None
    return result


def load_data(period: str) -> pd.DataFrame:
    # --- Equity / Macro tickers from yfinance ---
    tickers = {
        "SPY": "SPY",
        "VIX": "^VIX",
        "TNX": "^TNX",
        "IRX": "^IRX",
        "DXY": "DX-Y.NYB",
        "GOLD": "GC=F",
    }
    data = {}
    errors = []

    for name, ticker in tickers.items():
        try:
            raw = yf.download(ticker, period=period, auto_adjust=False, progress=False)
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

    # --- BTC from Coinbase Exchange API (with yfinance fallback) ---
    period_days = {"2y": 730, "3y": 1095, "5y": 1825, "10y": 3650}
    days = period_days.get(period, 1825)
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)

    btc_df = pd.DataFrame()
    try:
        btc_df = _fetch_coinbase_btc(start_dt, end_dt)
        if not btc_df.empty:
            st.caption("🟢 BTC price data: Coinbase Exchange")
    except Exception as e:
        errors.append(f"Coinbase API: {str(e)[:80]} — falling back to Yahoo Finance")

    if btc_df.empty:
        # Fallback to yfinance
        try:
            btc_raw = yf.download("BTC-USD", period=period, auto_adjust=False, progress=False)
            if not btc_raw.empty:
                close = btc_raw["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                high = btc_raw["High"]
                if isinstance(high, pd.DataFrame):
                    high = high.iloc[:, 0]
                vol = btc_raw["Volume"]
                if isinstance(vol, pd.DataFrame):
                    vol = vol.iloc[:, 0]
                btc_df = pd.DataFrame({"BTC": close.squeeze(), "BTC_High": high.squeeze(), "BTC_Volume": vol.squeeze()})
                st.caption("🟡 BTC price data: Yahoo Finance (Coinbase unavailable)")
            else:
                errors.append("BTC: no data from either Coinbase or Yahoo Finance")
        except Exception as e:
            errors.append(f"BTC fallback: {str(e)[:80]}")

    if not btc_df.empty:
        data["BTC"] = btc_df["BTC"]

    if errors:
        for err in errors:
            st.warning(f"⚠️ Data issue — {err}")

    required = {"SPY", "VIX", "BTC"}
    if not required.issubset(data.keys()):
        st.error("Cannot load required tickers (SPY, VIX, BTC). Check your connection.")
        st.stop()

    df = pd.concat(data.values(), axis=1)
    df.columns = list(data.keys())

    if not btc_df.empty and "BTC_Volume" in btc_df.columns:
        df["BTC_Volume"] = btc_df["BTC_Volume"]

    if "TNX" in df.columns and "IRX" in df.columns:
        df["YieldCurve"] = df["TNX"] - df["IRX"]
        df = df.drop(columns=["TNX", "IRX"])
    else:
        df["YieldCurve"] = 0.0

    df["YieldCurve"] = df["YieldCurve"].ffill().bfill()
    df["VIX"] = df["VIX"].ffill()
    df["BTC"] = df["BTC"].ffill()
    if "BTC_Volume" in df.columns:
        df["BTC_Volume"] = df["BTC_Volume"].ffill()
    if "DXY" in df.columns:
        df["DXY"] = df["DXY"].ffill().bfill()
    if "GOLD" in df.columns:
        df["GOLD"] = df["GOLD"].ffill().bfill()

    # --- Filter to equity trading days first ---
    df = df.dropna(subset=["SPY"])
    core_cols = [c for c in df.columns if c not in ("BTC_Peak_Full", "BTC_DD_Full")]
    df = df.dropna(subset=core_cols)

    # --- BTC drawdown from full 7-day series using intraday HIGH for peak ---
    # This captures true ATHs (e.g. Oct 2025 ~$126k) that the daily close misses.
    # Drawdown = (current close / highest high ever) - 1
    if not btc_df.empty:
        btc_close_full = btc_df["BTC"].dropna().sort_index()
        btc_high_full = btc_df["BTC_High"].dropna().sort_index() if "BTC_High" in btc_df.columns else btc_close_full
        btc_peak_full = btc_high_full.cummax()
        btc_dd_full = (btc_close_full / btc_peak_full.reindex(btc_close_full.index, method="ffill") - 1) * 100
        # Map onto equity-day index; ffill carries weekend peaks to Monday
        df["BTC_Peak_Full"] = btc_peak_full.reindex(df.index, method="ffill")
        df["BTC_DD_Full"] = btc_dd_full.reindex(df.index, method="ffill")
        # bfill for any leading NaNs if BTC series starts after SPY
        df["BTC_Peak_Full"] = df["BTC_Peak_Full"].bfill()
        df["BTC_DD_Full"] = df["BTC_DD_Full"].fillna(0.0)

    return df


@st.cache_data(ttl=14400, show_spinner="Fetching on-chain data…")
def load_fear_greed():
    """Fetch Crypto Fear & Greed Index."""
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=365&format=json", timeout=10)
        if resp.status_code == 200:
            fng_raw = resp.json().get("data", [])
            fng_df = pd.DataFrame(fng_raw)
            fng_df["date"] = pd.to_datetime(fng_df["timestamp"].astype(int), unit="s")
            fng_df["value"] = fng_df["value"].astype(int)
            fng_df = fng_df.set_index("date").sort_index()
            return fng_df
    except Exception:
        pass
    return None


df = load_data(lookback)
fng_data = load_fear_greed()

if len(df) < 300:
    st.error("Insufficient data history. Need at least 300 trading days.")
    st.stop()


# =====================================================
# SHARED HELPERS
# =====================================================
def zscore(series: pd.Series, window: int = 252) -> pd.Series:
    mu = series.rolling(window).mean()
    sig = series.rolling(window).std()
    return (series - mu) / sig


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
    return sma, sma + num_std * std, sma - num_std * std


def classify_regime(score_series, upper, lower):
    return np.select(
        [score_series > upper, score_series < lower],
        ["Risk-On", "Risk-Off"],
        default="Neutral",
    )


def regime_blocks(regime_series):
    shift = (regime_series != regime_series.shift()).astype(int)
    return shift, shift.cumsum()


# =====================================================
# SPY REGIME ENGINE — equity factors
# =====================================================
df["SPY_Return"] = df["SPY"].pct_change()
df["BTC_Return"] = df["BTC"].pct_change()

# SPY factors
df["SPY_Vol20"] = df["SPY_Return"].rolling(20).std()
df["SPY_Vol50"] = df["SPY_Return"].rolling(50).std()
df["SPY_Momentum63"] = df["SPY"].pct_change(63)

df["SPY_Trend_z"] = zscore(df["SPY_Momentum63"], zscore_window)
df["SPY_VIX_z"] = zscore(df["VIX"], zscore_window)
df["SPY_YC_z"] = zscore(df["YieldCurve"], zscore_window)
df["SPY_VolSpread"] = df["SPY_Vol20"] - df["SPY_Vol50"]
df["SPY_Vol_z"] = zscore(df["SPY_VolSpread"], zscore_window)

spy_weights = {
    "SPY_Trend_z": 0.40,
    "SPY_VIX_z": -0.30,
    "SPY_YC_z": 0.15,
    "SPY_Vol_z": -0.15,
}

# =====================================================
# BTC REGIME ENGINE — crypto factors
# =====================================================
df["BTC_Momentum63"] = df["BTC"].pct_change(63)
df["BTC_LogRet"] = np.log(df["BTC"] / df["BTC"].shift(1))
df["BTC_RealizedVol30"] = df["BTC_LogRet"].rolling(30).std() * np.sqrt(365)

# Volume ratio: daily volume vs 50d SMA (exchange activity proxy)
if "BTC_Volume" in df.columns:
    df["BTC_VolSMA50"] = df["BTC_Volume"].rolling(50).mean()
    df["BTC_VolumeRatio"] = df["BTC_Volume"] / df["BTC_VolSMA50"]
else:
    df["BTC_VolumeRatio"] = 1.0

# MVRV proxy: price / 200-day SMA
df["BTC_SMA200"] = df["BTC"].rolling(200).mean()
df["BTC_MVRV"] = df["BTC"] / df["BTC_SMA200"]

# Z-score the BTC factors
df["BTC_Trend_z"] = zscore(df["BTC_Momentum63"], zscore_window)
df["BTC_RVol_z"] = zscore(df["BTC_RealizedVol30"], zscore_window)
df["BTC_VolRatio_z"] = zscore(df["BTC_VolumeRatio"], zscore_window)
df["BTC_MVRV_z"] = zscore(df["BTC_MVRV"], zscore_window)

btc_weights = {
    "BTC_Trend_z": 0.35,       # crypto momentum
    "BTC_RVol_z": -0.25,       # high realized vol = risk-off
    "BTC_VolRatio_z": 0.15,    # high volume = conviction
    "BTC_MVRV_z": 0.25,        # overvalued/undervalued signal
}

# =====================================================
# TECHNICAL INDICATORS
# =====================================================
df["SPY_RSI"] = compute_rsi(df["SPY"])
df["BTC_RSI"] = compute_rsi(df["BTC"])
df["SPY_MACD"], df["SPY_Signal"], df["SPY_MACD_Hist"] = compute_macd(df["SPY"])
df["BTC_MACD"], df["BTC_Signal"], df["BTC_MACD_Hist"] = compute_macd(df["BTC"])
df["SPY_BB_Mid"], df["SPY_BB_Upper"], df["SPY_BB_Lower"] = compute_bollinger(df["SPY"])
df["BTC_BB_Mid"], df["BTC_BB_Upper"], df["BTC_BB_Lower"] = compute_bollinger(df["BTC"])

df = df.dropna()

# =====================================================
# COMPOSITE SCORES
# =====================================================
df["SPY_Score_raw"] = sum(w * df[k] for k, w in spy_weights.items())
df["SPY_Score"] = df["SPY_Score_raw"].ewm(span=ewm_span).mean()
df["SPY_Regime"] = classify_regime(df["SPY_Score"], upper_thresh, lower_thresh)
df["SPY_RegimeShift"], df["SPY_RegimeBlock"] = regime_blocks(df["SPY_Regime"])

df["BTC_Score_raw"] = sum(w * df[k] for k, w in btc_weights.items())
df["BTC_Score"] = df["BTC_Score_raw"].ewm(span=ewm_span).mean()
df["BTC_Regime"] = classify_regime(df["BTC_Score"], upper_thresh, lower_thresh)
df["BTC_RegimeShift"], df["BTC_RegimeBlock"] = regime_blocks(df["BTC_Regime"])

# Factor contributions
for k, w in spy_weights.items():
    df[f"SPY_Contrib_{k.replace('SPY_','')}"] = w * df[k]
for k, w in btc_weights.items():
    df[f"BTC_Contrib_{k.replace('BTC_','')}"] = w * df[k]

# =====================================================
# SCORE VELOCITY (5d rate of change)
# =====================================================
df["SPY_Score_Velocity"] = df["SPY_Score"].diff(5)
df["BTC_Score_Velocity"] = df["BTC_Score"].diff(5)

# =====================================================
# DRAWDOWNS
# =====================================================
df["SPY_Peak"] = df["SPY"].cummax()
df["SPY_Drawdown"] = (df["SPY"] / df["SPY_Peak"] - 1) * 100
# BTC drawdown: use full 7-day series (captures weekend/holiday ATHs)
if "BTC_DD_Full" in df.columns:
    df["BTC_Peak"] = df["BTC_Peak_Full"]
    df["BTC_Drawdown"] = df["BTC_DD_Full"]
else:
    df["BTC_Peak"] = df["BTC"].cummax()
    df["BTC_Drawdown"] = (df["BTC"] / df["BTC_Peak"] - 1) * 100

# =====================================================
# REGIME DIVERGENCE
# =====================================================
df["Divergent"] = df["SPY_Regime"] != df["BTC_Regime"]
df["Regime_Agreement"] = np.where(
    df["SPY_Regime"] == df["BTC_Regime"], df["SPY_Regime"], "Divergent"
)

# Current state
spy_regime = df["SPY_Regime"].iloc[-1]
spy_score = df["SPY_Score"].iloc[-1]
spy_score_prev = df["SPY_Score"].iloc[-2] if len(df) > 1 else spy_score
spy_duration = df.groupby("SPY_RegimeBlock").size().iloc[-1]

btc_regime = df["BTC_Regime"].iloc[-1]
btc_score = df["BTC_Score"].iloc[-1]
btc_score_prev = df["BTC_Score"].iloc[-2] if len(df) > 1 else btc_score
btc_duration = df.groupby("BTC_RegimeBlock").size().iloc[-1]


# =====================================================
# COLORS & LAYOUT HELPERS
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
    plot_bgcolor="#161b22",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#c8cdd5"),
    margin=dict(t=40, b=30, l=50, r=20),
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        bgcolor="rgba(0,0,0,0)", font=dict(size=11), tracegroupgap=10,
        itemwidth=30,
    ),
)


def add_regime_shading(fig, df, regime_col, block_col, rows):
    for _, group in df.groupby(block_col):
        regime = group[regime_col].iloc[0]
        x0, x1 = group.index[0], group.index[-1]
        color = REGIME_COLORS[regime]
        for row in rows:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=color, line_width=0, layer="below", row=row, col=1)


def make_gauge(score, prev, regime, upper, lower, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": prev, "valueformat": ".3f"},
        number={"font": {"size": 34, "color": "#ffffff"}, "valueformat": ".3f"},
        gauge={
            "axis": {"range": [-2, 2], "tickcolor": "#555", "tickwidth": 1},
            "bar": {"color": BADGE_COLORS[regime], "thickness": 0.3},
            "bgcolor": "#1a1d24",
            "bordercolor": "#2d3344",
            "steps": [
                {"range": [-2, lower], "color": "rgba(239,68,68,0.15)"},
                {"range": [lower, upper], "color": "rgba(234,179,8,0.10)"},
                {"range": [upper, 2], "color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {"line": {"color": "#ffffff", "width": 2}, "thickness": 0.8, "value": score},
        },
        title={"text": title, "font": {"size": 13, "color": "#8b95a5"}},
    ))
    fig.update_layout(
        height=200, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#c8cdd5"), margin=dict(t=50, b=10, l=20, r=20),
    )
    return fig


def build_alerts(df, regime_col, shift_col):
    alerts = []
    shift_dates = df[df[shift_col] == 1].index
    for dt in shift_dates:
        idx = df.index.get_loc(dt)
        new_r = df[regime_col].iloc[idx]
        old_r = df[regime_col].iloc[idx - 1] if idx > 0 else "Unknown"
        alerts.append({"date": dt, "from": old_r, "to": new_r})
    return alerts


spy_alerts = build_alerts(df, "SPY_Regime", "SPY_RegimeShift")
btc_alerts = build_alerts(df, "BTC_Regime", "BTC_RegimeShift")


# =====================================================
# TOP PANEL — dual gauges + prices
# =====================================================
st.markdown("")
g1, g2, g3, g4 = st.columns([1.2, 1.2, 1, 1])

spy_vel = df["SPY_Score_Velocity"].iloc[-1]
btc_vel = df["BTC_Score_Velocity"].iloc[-1]

with g1:
    st.plotly_chart(make_gauge(spy_score, spy_score_prev, spy_regime, upper_thresh, lower_thresh, "SPY Score"),
                    use_container_width=True, key="spy_gauge")
    badge_c = BADGE_COLORS[spy_regime]
    vel_arrow = "▲" if spy_vel > 0.02 else ("▼" if spy_vel < -0.02 else "►")
    vel_c = "#4ade80" if spy_vel > 0.02 else ("#f87171" if spy_vel < -0.02 else "#8b95a5")
    st.markdown(f"<p style='text-align:center;margin-top:-10px;'>"
                f"<span style='color:{badge_c};font-weight:700;font-size:1.1rem;'>{spy_regime}</span> "
                f"<span style='color:#8b95a5;font-size:0.85rem;'>({int(spy_duration)}d)</span><br>"
                f"<span style='color:{vel_c};font-size:0.85rem;'>{vel_arrow} {spy_vel:+.3f}/5d</span></p>",
                unsafe_allow_html=True)

with g2:
    st.plotly_chart(make_gauge(btc_score, btc_score_prev, btc_regime, upper_thresh, lower_thresh, "BTC Score"),
                    use_container_width=True, key="btc_gauge")
    badge_c = BADGE_COLORS[btc_regime]
    vel_arrow = "▲" if btc_vel > 0.02 else ("▼" if btc_vel < -0.02 else "►")
    vel_c = "#4ade80" if btc_vel > 0.02 else ("#f87171" if btc_vel < -0.02 else "#8b95a5")
    st.markdown(f"<p style='text-align:center;margin-top:-10px;'>"
                f"<span style='color:{badge_c};font-weight:700;font-size:1.1rem;'>{btc_regime}</span> "
                f"<span style='color:#8b95a5;font-size:0.85rem;'>({int(btc_duration)}d)</span><br>"
                f"<span style='color:{vel_c};font-size:0.85rem;'>{vel_arrow} {btc_vel:+.3f}/5d</span></p>",
                unsafe_allow_html=True)

with g3:
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

with g4:
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

# Proximity warnings
for label, score, thresh_u, thresh_l, regime in [
    ("SPY", spy_score, upper_thresh, lower_thresh, spy_regime),
    ("BTC", btc_score, upper_thresh, lower_thresh, btc_regime),
]:
    if abs(score - thresh_u) < alert_score_proximity and regime != "Risk-On":
        st.warning(f"⚡ {label} score ({score:.3f}) approaching Risk-On threshold ({thresh_u})")
    if abs(score - thresh_l) < alert_score_proximity and regime != "Risk-Off":
        st.warning(f"⚡ {label} score ({score:.3f}) approaching Risk-Off threshold ({thresh_l})")

# Divergence banner
is_divergent = spy_regime != btc_regime
if is_divergent:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1d24,#2a1a2d);border:1px solid #a78bfa;
                border-radius:10px;padding:12px 20px;margin:8px 0;display:flex;justify-content:center;align-items:center;gap:20px;">
        <span style="color:#c4b5fd;font-weight:600;font-size:0.95rem;">⚡ REGIME DIVERGENCE</span>
        <span style="color:#e0e0e0;">
            SPY: <span style="color:{BADGE_COLORS[spy_regime]};font-weight:600;">{spy_regime}</span>
            &nbsp;|&nbsp;
            BTC: <span style="color:{BADGE_COLORS[btc_regime]};font-weight:600;">{btc_regime}</span>
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")


# =====================================================
# TABS
# =====================================================
tab_spy, tab_btc, tab_technicals, tab_backtest, tab_drawdown, tab_intermarket, tab_alerts, tab_report, tab_analysis, tab_methodology = st.tabs([
    "📈 SPY",
    "🟠 BTC",
    "🔬 Technicals",
    "🧪 Backtest",
    "📉 Drawdowns",
    "🌐 Intermarket",
    "🔔 Alerts",
    "📋 Report",
    "📊 Analysis",
    "📖 Method",
])


# ─────────────────────────────────────────────────────
# TAB 1: SPY REGIME
# ─────────────────────────────────────────────────────
with tab_spy:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.40, 0.30, 0.30],
        subplot_titles=["SPY Price", "SPY Macro Score", "SPY Factor Contributions"],
    )

    fig.add_trace(go.Scatter(
        x=df.index, y=df["SPY"], line=dict(width=2, color="#60a5fa"),
        name="SPY", hovertemplate="%{x|%b %d %Y}<br>$%{y:.2f}<extra>SPY</extra>",
    ), row=1, col=1)

    if show_bbands:
        fig.add_trace(go.Scatter(x=df.index, y=df["SPY_BB_Upper"],
                      line=dict(width=0.8, color="rgba(165,180,252,0.5)"),
                      name="BB Upper", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SPY_BB_Lower"],
                      line=dict(width=0.8, color="rgba(165,180,252,0.5)"),
                      name="BB Lower", showlegend=False,
                      fill="tonexty", fillcolor="rgba(99,102,241,0.05)"), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["SPY_Score"], line=dict(width=2.5, color="#a5b4fc"),
        name="SPY Score", hovertemplate="%{x|%b %d %Y}<br>Score: %{y:.3f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=upper_thresh, line_dash="dot", line_color="rgba(74,222,128,0.7)", row=2, col=1)
    fig.add_hline(y=lower_thresh, line_dash="dot", line_color="rgba(248,113,113,0.7)", row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(200,200,200,0.4)", row=2, col=1)

    spy_contrib_meta = {
        "SPY_Contrib_Trend_z": ("#60a5fa", "Trend 40%"),
        "SPY_Contrib_VIX_z": ("#f87171", "VIX −30%"),
        "SPY_Contrib_YC_z": ("#34d399", "YC 15%"),
        "SPY_Contrib_Vol_z": ("#fbbf24", "VolSp −15%"),
    }
    for col_name, (color, label) in spy_contrib_meta.items():
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col_name], line=dict(width=1.8, color=color),
            name=label, hovertemplate="%{x|%b %d %Y}<br>%{y:.3f}<extra>" + label + "</extra>",
        ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(200,200,200,0.4)", row=3, col=1)

    add_regime_shading(fig, df, "SPY_Regime", "SPY_RegimeBlock", [1, 2, 3])
    fig.update_layout(height=900, **DARK_LAYOUT)
    for i in range(1, 4):
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", row=i, col=1)
        fig.update_xaxes(tickformat="%b '%y", showgrid=True, gridcolor="rgba(255,255,255,0.08)", row=i, col=1)

    st.plotly_chart(fig, use_container_width=True, key="spy_regime_chart")


# ─────────────────────────────────────────────────────
# TAB 2: BTC REGIME
# ─────────────────────────────────────────────────────
with tab_btc:
    fig2 = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.40, 0.30, 0.30],
        subplot_titles=["Bitcoin Price", "BTC Crypto Score", "BTC Factor Contributions"],
    )

    fig2.add_trace(go.Scatter(
        x=df.index, y=df["BTC"], line=dict(width=2, color="#fb923c"),
        name="BTC", hovertemplate="%{x|%b %d %Y}<br>$%{y:,.0f}<extra>BTC</extra>",
    ), row=1, col=1)

    if show_bbands:
        fig2.add_trace(go.Scatter(x=df.index, y=df["BTC_BB_Upper"],
                       line=dict(width=0.8, color="rgba(251,191,36,0.5)"),
                       name="BB Upper", showlegend=False), row=1, col=1)
        fig2.add_trace(go.Scatter(x=df.index, y=df["BTC_BB_Lower"],
                       line=dict(width=0.8, color="rgba(251,191,36,0.5)"),
                       name="BB Lower", showlegend=False,
                       fill="tonexty", fillcolor="rgba(249,115,22,0.05)"), row=1, col=1)

    fig2.add_trace(go.Scatter(
        x=df.index, y=df["BTC_Score"], line=dict(width=2.5, color="#fdba74"),
        name="BTC Score", hovertemplate="%{x|%b %d %Y}<br>Score: %{y:.3f}<extra></extra>",
    ), row=2, col=1)
    fig2.add_hline(y=upper_thresh, line_dash="dot", line_color="rgba(74,222,128,0.7)", row=2, col=1)
    fig2.add_hline(y=lower_thresh, line_dash="dot", line_color="rgba(248,113,113,0.7)", row=2, col=1)
    fig2.add_hline(y=0, line_dash="dot", line_color="rgba(200,200,200,0.4)", row=2, col=1)

    btc_contrib_meta = {
        "BTC_Contrib_Trend_z": ("#fb923c", "Mom 35%"),
        "BTC_Contrib_RVol_z": ("#f87171", "RVol −25%"),
        "BTC_Contrib_VolRatio_z": ("#60a5fa", "VolR 15%"),
        "BTC_Contrib_MVRV_z": ("#c4b5fd", "MVRV 25%"),
    }
    for col_name, (color, label) in btc_contrib_meta.items():
        fig2.add_trace(go.Scatter(
            x=df.index, y=df[col_name], line=dict(width=1.8, color=color),
            name=label, hovertemplate="%{x|%b %d %Y}<br>%{y:.3f}<extra>" + label + "</extra>",
        ), row=3, col=1)
    fig2.add_hline(y=0, line_dash="dot", line_color="rgba(200,200,200,0.4)", row=3, col=1)

    add_regime_shading(fig2, df, "BTC_Regime", "BTC_RegimeBlock", [1, 2, 3])
    fig2.update_layout(height=900, **DARK_LAYOUT)
    for i in range(1, 4):
        fig2.update_yaxes(gridcolor="rgba(255,255,255,0.08)", row=i, col=1)
        fig2.update_xaxes(tickformat="%b '%y", showgrid=True, gridcolor="rgba(255,255,255,0.08)", row=i, col=1)

    st.plotly_chart(fig2, use_container_width=True, key="btc_regime_chart")

    # Fear & Greed section
    if fng_data is not None:
        st.markdown("---")
        st.markdown("### Crypto Fear & Greed Index")
        current_fng = int(fng_data["value"].iloc[-1])
        fng_class = fng_data["value_classification"].iloc[-1] if "value_classification" in fng_data.columns else ""
        fng_color = "#ef4444" if current_fng < 25 else ("#f59e0b" if current_fng < 45 else (
            "#eab308" if current_fng < 55 else ("#22c55e" if current_fng < 75 else "#16a34a")))

        fc1, fc2 = st.columns([1, 2])
        with fc1:
            fng_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=current_fng,
                number={"font": {"size": 42, "color": "#ffffff"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#555"},
                    "bar": {"color": fng_color, "thickness": 0.3},
                    "bgcolor": "#1a1d24", "bordercolor": "#2d3344",
                    "steps": [
                        {"range": [0, 25], "color": "rgba(239,68,68,0.15)"},
                        {"range": [25, 45], "color": "rgba(245,158,11,0.10)"},
                        {"range": [45, 55], "color": "rgba(234,179,8,0.10)"},
                        {"range": [55, 75], "color": "rgba(34,197,94,0.10)"},
                        {"range": [75, 100], "color": "rgba(22,163,74,0.15)"},
                    ],
                },
                title={"text": "Fear & Greed", "font": {"size": 14, "color": "#8b95a5"}},
            ))
            fng_gauge.update_layout(height=220, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                     font=dict(color="#c8cdd5"), margin=dict(t=50, b=10, l=30, r=30))
            st.plotly_chart(fng_gauge, use_container_width=True, key="fng_gauge")
            st.markdown(f"<p style='text-align:center;color:{fng_color};font-weight:600;'>{fng_class}</p>",
                        unsafe_allow_html=True)

        with fc2:
            fng_fig = go.Figure()
            fng_fig.add_trace(go.Scatter(
                x=fng_data.index, y=fng_data["value"], mode="lines",
                line=dict(width=1.5, color="#a78bfa"), name="Fear & Greed",
            ))
            fng_fig.add_hline(y=25, line_dash="dot", line_color="rgba(239,68,68,0.4)")
            fng_fig.add_hline(y=75, line_dash="dot", line_color="rgba(34,197,94,0.4)")
            fng_fig.add_hrect(y0=0, y1=25, fillcolor="rgba(239,68,68,0.05)", line_width=0)
            fng_fig.add_hrect(y0=75, y1=100, fillcolor="rgba(34,197,94,0.05)", line_width=0)
            fng_fig.update_layout(height=300, **DARK_LAYOUT, yaxis_title="Score", yaxis_range=[0, 100])
            st.plotly_chart(fng_fig, use_container_width=True, key="fng_ts")

    # On-chain proxy metrics
    st.markdown("---")
    st.markdown("### On-Chain Proxy Metrics")
    oc1, oc2, oc3 = st.columns(3)

    mvrv_val = df["BTC_MVRV"].iloc[-1]
    mvrv_color = "#ef4444" if mvrv_val > 2.5 else ("#22c55e" if mvrv_val < 1.0 else "#818cf8")
    with oc1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MVRV Proxy (P / 200-SMA)</div>
            <div class="metric-value" style="color:{mvrv_color};">{mvrv_val:.2f}x</div>
            <div style="color:#8b95a5;margin-top:4px;font-size:0.8rem;">
                {"Overheated" if mvrv_val > 2.5 else "Undervalued" if mvrv_val < 1.0 else "Fair value"}
            </div>
        </div>""", unsafe_allow_html=True)

    rvol = df["BTC_RealizedVol30"].iloc[-1] * 100
    rv_color = "#ef4444" if rvol > 80 else ("#22c55e" if rvol < 40 else "#eab308")
    with oc2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Realized Vol (30d ann.)</div>
            <div class="metric-value" style="color:{rv_color};">{rvol:.1f}%</div>
            <div style="color:#8b95a5;margin-top:4px;font-size:0.8rem;">
                {"Extreme" if rvol > 80 else "Calm" if rvol < 40 else "Moderate"}
            </div>
        </div>""", unsafe_allow_html=True)

    if "BTC_Volume" in df.columns:
        vol_ratio = df["BTC_VolumeRatio"].iloc[-1]
        vr_color = "#22c55e" if vol_ratio > 1.5 else ("#ef4444" if vol_ratio < 0.5 else "#eab308")
        with oc3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Volume Ratio (vs 50d)</div>
                <div class="metric-value" style="color:{vr_color};">{vol_ratio:.2f}x</div>
                <div style="color:#8b95a5;margin-top:4px;font-size:0.8rem;">
                    {"High activity" if vol_ratio > 1.5 else "Low activity" if vol_ratio < 0.5 else "Normal"}
                </div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# TAB 3: TECHNICALS
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
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    tech_fig = make_subplots(rows=tech_rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                              row_heights=row_heights, subplot_titles=titles)

    price_color = "#3b82f6" if asset_choice == "SPY" else "#f97316"
    regime_col = f"{prefix}_Regime"
    block_col = f"{prefix}_RegimeBlock"

    tech_fig.add_trace(go.Scatter(
        x=df.index, y=df[asset_choice], line=dict(width=1.5, color=price_color), name=asset_choice,
    ), row=1, col=1)

    if show_bbands:
        tech_fig.add_trace(go.Scatter(x=df.index, y=df[f"{prefix}_BB_Mid"],
                           line=dict(width=1, color="#818cf8", dash="dash"), name="BB Mid"), row=1, col=1)
        tech_fig.add_trace(go.Scatter(x=df.index, y=df[f"{prefix}_BB_Upper"],
                           line=dict(width=0.8, color="rgba(165,180,252,0.6)"),
                           name="BB Upper", showlegend=False), row=1, col=1)
        tech_fig.add_trace(go.Scatter(x=df.index, y=df[f"{prefix}_BB_Lower"],
                           line=dict(width=0.8, color="rgba(165,180,252,0.6)"),
                           name="BB Lower", showlegend=False,
                           fill="tonexty", fillcolor="rgba(129,140,248,0.07)"), row=1, col=1)

    current_row = 2
    if show_rsi:
        tech_fig.add_trace(go.Scatter(
            x=df.index, y=df[f"{prefix}_RSI"], line=dict(width=1.5, color="#a78bfa"), name="RSI",
        ), row=current_row, col=1)
        tech_fig.add_hline(y=70, line_dash="dot", line_color="rgba(248,113,113,0.7)", row=current_row, col=1)
        tech_fig.add_hline(y=30, line_dash="dot", line_color="rgba(74,222,128,0.7)", row=current_row, col=1)
        tech_fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.05)", line_width=0, row=current_row, col=1)
        tech_fig.add_hrect(y0=0, y1=30, fillcolor="rgba(34,197,94,0.05)", line_width=0, row=current_row, col=1)
        current_row += 1

    if show_macd:
        hist_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in df[f"{prefix}_MACD_Hist"]]
        tech_fig.add_trace(go.Bar(x=df.index, y=df[f"{prefix}_MACD_Hist"],
                           marker_color=hist_colors, name="MACD Hist", opacity=0.5), row=current_row, col=1)
        tech_fig.add_trace(go.Scatter(x=df.index, y=df[f"{prefix}_MACD"],
                           line=dict(width=1.5, color="#3b82f6"), name="MACD"), row=current_row, col=1)
        tech_fig.add_trace(go.Scatter(x=df.index, y=df[f"{prefix}_Signal"],
                           line=dict(width=1.5, color="#f97316"), name="Signal"), row=current_row, col=1)
        tech_fig.add_hline(y=0, line_dash="dot", line_color="rgba(200,200,200,0.4)", row=current_row, col=1)

    add_regime_shading(tech_fig, df, regime_col, block_col, list(range(1, tech_rows + 1)))
    tech_fig.update_layout(height=250 + tech_rows * 220, **DARK_LAYOUT)
    for i in range(1, tech_rows + 1):
        tech_fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", row=i, col=1)
        tech_fig.update_xaxes(tickformat="%b '%y", showgrid=True, gridcolor="rgba(255,255,255,0.08)", row=i, col=1)

    st.plotly_chart(tech_fig, use_container_width=True, key="tech_chart")

    # Current readings
    st.markdown("#### Current Readings")
    ic1, ic2, ic3 = st.columns(3)
    rsi_val = df[f"{prefix}_RSI"].iloc[-1]
    macd_val = df[f"{prefix}_MACD"].iloc[-1]
    macd_sig = df[f"{prefix}_Signal"].iloc[-1]
    bb_pct = ((df[asset_choice].iloc[-1] - df[f"{prefix}_BB_Lower"].iloc[-1]) /
              (df[f"{prefix}_BB_Upper"].iloc[-1] - df[f"{prefix}_BB_Lower"].iloc[-1])) * 100

    rsi_status = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
    rsi_color = "#ef4444" if rsi_val > 70 else ("#22c55e" if rsi_val < 30 else "#eab308")
    with ic1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">RSI (14)</div>
        <div class="metric-value" style="color:{rsi_color};">{rsi_val:.1f}</div>
        <div style="color:{rsi_color};margin-top:4px;font-size:0.85rem;">{rsi_status}</div></div>""", unsafe_allow_html=True)

    macd_cross = "Bullish" if macd_val > macd_sig else "Bearish"
    macd_color = "#22c55e" if macd_val > macd_sig else "#ef4444"
    with ic2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">MACD</div>
        <div class="metric-value" style="color:{macd_color};">{macd_val:.2f}</div>
        <div style="color:{macd_color};margin-top:4px;font-size:0.85rem;">{macd_cross} crossover</div></div>""", unsafe_allow_html=True)

    bb_color = "#ef4444" if bb_pct > 90 else ("#22c55e" if bb_pct < 10 else "#818cf8")
    with ic3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Bollinger %B</div>
        <div class="metric-value" style="color:{bb_color};">{bb_pct:.0f}%</div>
        <div style="color:#8b95a5;margin-top:4px;font-size:0.85rem;">Position within bands</div></div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# TAB 4: BACKTEST
# ─────────────────────────────────────────────────────
with tab_backtest:
    st.markdown("### Strategy Backtester")
    st.caption("Compares a regime-following strategy vs buy-and-hold. "
               "Risk-On = fully invested, Neutral = 50% invested, Risk-Off = 100% cash.")

    bt_asset = st.radio("Backtest asset", ["SPY", "BTC"], horizontal=True, key="bt_asset")
    bt_regime_col = f"{bt_asset}_Regime"
    bt_return_col = f"{bt_asset}_Return"

    # Strategy: position sizing by regime
    regime_exposure = {"Risk-On": 1.0, "Neutral": 0.5, "Risk-Off": 0.0}
    df["BT_Exposure"] = df[bt_regime_col].map(regime_exposure)
    # Use previous day's regime to avoid lookahead
    df["BT_Exposure_Lag"] = df["BT_Exposure"].shift(1).fillna(0)
    df["BT_StratReturn"] = df[bt_return_col] * df["BT_Exposure_Lag"]

    # Equity curves
    df["BT_BuyHold"] = (1 + df[bt_return_col]).cumprod()
    df["BT_Strategy"] = (1 + df["BT_StratReturn"]).cumprod()

    # Stats
    bh_total = (df["BT_BuyHold"].iloc[-1] / df["BT_BuyHold"].iloc[0] - 1) * 100
    st_total = (df["BT_Strategy"].iloc[-1] / df["BT_Strategy"].iloc[0] - 1) * 100

    bh_peak = df["BT_BuyHold"].cummax()
    bh_dd = ((df["BT_BuyHold"] / bh_peak) - 1) * 100
    bh_maxdd = bh_dd.min()

    st_peak = df["BT_Strategy"].cummax()
    st_dd = ((df["BT_Strategy"] / st_peak) - 1) * 100
    st_maxdd = st_dd.min()

    ann_factor = 252
    bh_vol = df[bt_return_col].std() * np.sqrt(ann_factor) * 100
    st_vol = df["BT_StratReturn"].std() * np.sqrt(ann_factor) * 100
    years = len(df) / ann_factor
    bh_cagr = ((df["BT_BuyHold"].iloc[-1] / df["BT_BuyHold"].iloc[0]) ** (1 / years) - 1) * 100
    st_cagr = ((df["BT_Strategy"].iloc[-1] / df["BT_Strategy"].iloc[0]) ** (1 / years) - 1) * 100
    bh_sharpe = bh_cagr / bh_vol if bh_vol > 0 else 0
    st_sharpe = st_cagr / st_vol if st_vol > 0 else 0

    # Stats cards
    bs1, bs2, bs3, bs4 = st.columns(4)
    for col, label, bh_val, st_val, fmt in [
        (bs1, "Total Return", bh_total, st_total, "+.1f"),
        (bs2, "CAGR", bh_cagr, st_cagr, "+.1f"),
        (bs3, "Max Drawdown", bh_maxdd, st_maxdd, ".1f"),
        (bs4, "Sharpe Ratio", bh_sharpe, st_sharpe, ".2f"),
    ]:
        winner = "#4ade80" if (st_val > bh_val if "Drawdown" not in label else st_val > bh_val) else "#f87171"
        with col:
            st.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div style="display:flex;justify-content:space-around;margin-top:8px;">
                <div><span style="color:#8b95a5;font-size:0.75rem;">Buy&Hold</span><br>
                <span style="color:#60a5fa;font-weight:600;font-size:1.1rem;">{bh_val:{fmt}}{'%' if 'Ratio' not in label else ''}</span></div>
                <div><span style="color:#8b95a5;font-size:0.75rem;">Strategy</span><br>
                <span style="color:{winner};font-weight:600;font-size:1.1rem;">{st_val:{fmt}}{'%' if 'Ratio' not in label else ''}</span></div>
            </div></div>""", unsafe_allow_html=True)

    # Equity curve chart
    bt_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            row_heights=[0.65, 0.35],
                            subplot_titles=[f"{bt_asset} Equity Curves (Normalized)", "Drawdown"])

    bt_fig.add_trace(go.Scatter(
        x=df.index, y=df["BT_BuyHold"], line=dict(width=2, color="#60a5fa"),
        name="Buy & Hold", hovertemplate="%{x|%b %d %Y}<br>%{y:.2f}x<extra>B&H</extra>",
    ), row=1, col=1)
    bt_fig.add_trace(go.Scatter(
        x=df.index, y=df["BT_Strategy"], line=dict(width=2, color="#4ade80"),
        name="Regime Strategy", hovertemplate="%{x|%b %d %Y}<br>%{y:.2f}x<extra>Strategy</extra>",
    ), row=1, col=1)

    bt_fig.add_trace(go.Scatter(
        x=df.index, y=bh_dd, line=dict(width=1.2, color="#60a5fa"),
        name="B&H DD", fill="tozeroy", fillcolor="rgba(96,165,250,0.1)",
        hovertemplate="%{x|%b %d %Y}<br>%{y:.1f}%<extra>B&H DD</extra>",
    ), row=2, col=1)
    bt_fig.add_trace(go.Scatter(
        x=df.index, y=st_dd, line=dict(width=1.2, color="#4ade80"),
        name="Strat DD", fill="tozeroy", fillcolor="rgba(74,222,128,0.1)",
        hovertemplate="%{x|%b %d %Y}<br>%{y:.1f}%<extra>Strat DD</extra>",
    ), row=2, col=1)

    add_regime_shading(bt_fig, df, bt_regime_col, f"{bt_asset}_RegimeBlock", [1, 2])
    bt_fig.update_layout(height=600, **DARK_LAYOUT)
    for i in range(1, 3):
        bt_fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", row=i, col=1)
        bt_fig.update_xaxes(tickformat="%b '%y", showgrid=True, gridcolor="rgba(255,255,255,0.08)", row=i, col=1)
    bt_fig.update_yaxes(title_text="Growth of $1", row=1, col=1)
    bt_fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

    st.plotly_chart(bt_fig, use_container_width=True, key="bt_chart")

    # Time in each regime
    regime_pcts = df[bt_regime_col].value_counts(normalize=True) * 100
    st.markdown("#### Time in Each Regime")
    rp1, rp2, rp3 = st.columns(3)
    for col, regime in [(rp1, "Risk-On"), (rp2, "Neutral"), (rp3, "Risk-Off")]:
        pct = regime_pcts.get(regime, 0)
        with col:
            st.markdown(f"""<div class="metric-card">
            <div class="metric-label">{regime}</div>
            <div class="metric-value" style="color:{BADGE_COLORS[regime]};">{pct:.0f}%</div>
            <div style="color:#8b95a5;margin-top:4px;font-size:0.8rem;">of trading days</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# TAB 5: DRAWDOWNS
# ─────────────────────────────────────────────────────
with tab_drawdown:
    st.markdown("### Drawdown Analysis")

    # Current drawdowns
    spy_dd_now = df["SPY_Drawdown"].iloc[-1]
    btc_dd_now = df["BTC_Drawdown"].iloc[-1]
    spy_dd_max = df["SPY_Drawdown"].min()
    btc_dd_max = df["BTC_Drawdown"].min()

    dd1, dd2, dd3, dd4 = st.columns(4)
    with dd1:
        c = "#f87171" if spy_dd_now < -5 else ("#fbbf24" if spy_dd_now < -2 else "#4ade80")
        st.markdown(f"""<div class="metric-card"><div class="metric-label">SPY Current DD</div>
        <div class="metric-value" style="color:{c};">{spy_dd_now:.1f}%</div></div>""", unsafe_allow_html=True)
    with dd2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">SPY Max DD</div>
        <div class="metric-value" style="color:#f87171;">{spy_dd_max:.1f}%</div></div>""", unsafe_allow_html=True)
    with dd3:
        c = "#f87171" if btc_dd_now < -15 else ("#fbbf24" if btc_dd_now < -5 else "#4ade80")
        st.markdown(f"""<div class="metric-card"><div class="metric-label">BTC Current DD</div>
        <div class="metric-value" style="color:{c};">{btc_dd_now:.1f}%</div></div>""", unsafe_allow_html=True)
    with dd4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">BTC Max DD</div>
        <div class="metric-value" style="color:#f87171;">{btc_dd_max:.1f}%</div></div>""", unsafe_allow_html=True)

    # Drawdown time series
    dd_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            subplot_titles=["SPY Drawdown from ATH", "BTC Drawdown from ATH"])

    dd_fig.add_trace(go.Scatter(
        x=df.index, y=df["SPY_Drawdown"], line=dict(width=1.5, color="#60a5fa"),
        fill="tozeroy", fillcolor="rgba(96,165,250,0.12)", name="SPY DD",
        hovertemplate="%{x|%b %d %Y}<br>%{y:.1f}%<extra>SPY</extra>",
    ), row=1, col=1)
    dd_fig.add_trace(go.Scatter(
        x=df.index, y=df["BTC_Drawdown"], line=dict(width=1.5, color="#fb923c"),
        fill="tozeroy", fillcolor="rgba(251,146,60,0.12)", name="BTC DD",
        hovertemplate="%{x|%b %d %Y}<br>%{y:.1f}%<extra>BTC</extra>",
    ), row=2, col=1)

    add_regime_shading(dd_fig, df, "SPY_Regime", "SPY_RegimeBlock", [1])
    add_regime_shading(dd_fig, df, "BTC_Regime", "BTC_RegimeBlock", [2])
    dd_fig.update_layout(height=550, **DARK_LAYOUT)
    for i in range(1, 3):
        dd_fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", title_text="DD %", row=i, col=1)
        dd_fig.update_xaxes(tickformat="%b '%y", showgrid=True, gridcolor="rgba(255,255,255,0.08)", row=i, col=1)

    st.plotly_chart(dd_fig, use_container_width=True, key="dd_chart")

    # Max drawdown by regime
    st.markdown("### Max Drawdown by Regime")
    st.caption("Worst drawdown experienced within each regime classification.")
    dd_by_regime = []
    for asset in ["SPY", "BTC"]:
        r_col = f"{asset}_Regime"
        dd_col = f"{asset}_Drawdown"
        for regime in ["Risk-On", "Neutral", "Risk-Off"]:
            mask = df[r_col] == regime
            if mask.sum() > 0:
                dd_by_regime.append({
                    "Asset": asset,
                    "Regime": regime,
                    "Max DD %": round(df.loc[mask, dd_col].min(), 1),
                    "Avg DD %": round(df.loc[mask, dd_col].mean(), 1),
                    "Days": int(mask.sum()),
                })
    dd_regime_df = pd.DataFrame(dd_by_regime)

    def style_dd(tbl):
        def _bg(val):
            c = {"Risk-On": "#1a3a1a", "Neutral": "#3a3a1a", "Risk-Off": "#3a1a1a"}.get(val, "")
            return f"background-color:{c};color:#e0e0e0"
        def _dd(val):
            try:
                v = float(val)
                return "color:#f87171;font-weight:600" if v < -10 else ("color:#fbbf24" if v < -3 else "color:#4ade80")
            except (ValueError, TypeError):
                return ""
        return tbl.style.map(_bg, subset=["Regime"]).map(_dd, subset=["Max DD %", "Avg DD %"])

    st.dataframe(style_dd(dd_regime_df), use_container_width=True, hide_index=True, height=250)


# ─────────────────────────────────────────────────────
# TAB 6: INTERMARKET
# ─────────────────────────────────────────────────────
with tab_intermarket:
    st.markdown("### Intermarket Context")
    st.caption("DXY (US Dollar Index) and Gold as macro context alongside regime classification.")

    has_dxy = "DXY" in df.columns and df["DXY"].notna().sum() > 100
    has_gold = "GOLD" in df.columns and df["GOLD"].notna().sum() > 100

    if has_dxy or has_gold:
        # Current values
        im_cols = st.columns(4)
        if has_dxy:
            dxy_now = df["DXY"].iloc[-1]
            dxy_1d = (df["DXY"].iloc[-1] / df["DXY"].iloc[-2] - 1) * 100 if len(df) > 1 else 0
            dxy_c = "#f87171" if dxy_1d > 0 else "#4ade80"  # strong dollar generally risk-off
            with im_cols[0]:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">DXY (Dollar)</div>
                <div class="metric-value">{dxy_now:.2f}</div>
                <div style="color:{dxy_c};margin-top:4px;font-size:0.9rem;font-weight:600;">{dxy_1d:+.2f}%</div>
                </div>""", unsafe_allow_html=True)

            dxy_50 = df["DXY"].rolling(50).mean().iloc[-1]
            dxy_trend = "Above 50d" if dxy_now > dxy_50 else "Below 50d"
            dxy_trend_c = "#f87171" if dxy_now > dxy_50 else "#4ade80"
            with im_cols[1]:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">DXY vs 50-SMA</div>
                <div class="metric-value" style="color:{dxy_trend_c};font-size:1.2rem;">{dxy_trend}</div>
                <div style="color:#8b95a5;margin-top:4px;font-size:0.8rem;">50-SMA: {dxy_50:.2f}</div>
                </div>""", unsafe_allow_html=True)

        if has_gold:
            gold_now = df["GOLD"].iloc[-1]
            gold_1d = (df["GOLD"].iloc[-1] / df["GOLD"].iloc[-2] - 1) * 100 if len(df) > 1 else 0
            gold_c = "#4ade80" if gold_1d > 0 else "#f87171"
            with im_cols[2]:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">Gold</div>
                <div class="metric-value">${gold_now:,.0f}</div>
                <div style="color:{gold_c};margin-top:4px;font-size:0.9rem;font-weight:600;">{gold_1d:+.2f}%</div>
                </div>""", unsafe_allow_html=True)

            gold_50 = df["GOLD"].rolling(50).mean().iloc[-1]
            gold_trend = "Above 50d" if gold_now > gold_50 else "Below 50d"
            gold_trend_c = "#4ade80" if gold_now > gold_50 else "#f87171"
            with im_cols[3]:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">Gold vs 50-SMA</div>
                <div class="metric-value" style="color:{gold_trend_c};font-size:1.2rem;">{gold_trend}</div>
                <div style="color:#8b95a5;margin-top:4px;font-size:0.8rem;">50-SMA: ${gold_50:,.0f}</div>
                </div>""", unsafe_allow_html=True)

        # Intermarket charts
        im_rows = int(has_dxy) + int(has_gold)
        im_titles = []
        if has_dxy:
            im_titles.append("DXY (US Dollar Index)")
        if has_gold:
            im_titles.append("Gold (GC=F)")

        im_fig = make_subplots(rows=im_rows, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=im_titles)

        row_i = 1
        if has_dxy:
            im_fig.add_trace(go.Scatter(
                x=df.index, y=df["DXY"], line=dict(width=2, color="#fbbf24"),
                name="DXY", hovertemplate="%{x|%b %d %Y}<br>%{y:.2f}<extra>DXY</extra>",
            ), row=row_i, col=1)
            dxy_sma = df["DXY"].rolling(50).mean()
            im_fig.add_trace(go.Scatter(
                x=df.index, y=dxy_sma, line=dict(width=1.2, color="#fbbf24", dash="dash"),
                name="DXY 50-SMA", opacity=0.6,
            ), row=row_i, col=1)
            add_regime_shading(im_fig, df, "SPY_Regime", "SPY_RegimeBlock", [row_i])
            row_i += 1

        if has_gold:
            im_fig.add_trace(go.Scatter(
                x=df.index, y=df["GOLD"], line=dict(width=2, color="#fbbf24"),
                name="Gold", hovertemplate="%{x|%b %d %Y}<br>$%{y:,.0f}<extra>Gold</extra>",
            ), row=row_i, col=1)
            gold_sma = df["GOLD"].rolling(50).mean()
            im_fig.add_trace(go.Scatter(
                x=df.index, y=gold_sma, line=dict(width=1.2, color="#fbbf24", dash="dash"),
                name="Gold 50-SMA", opacity=0.6,
            ), row=row_i, col=1)
            add_regime_shading(im_fig, df, "BTC_Regime", "BTC_RegimeBlock", [row_i])

        im_fig.update_layout(height=250 + im_rows * 250, **DARK_LAYOUT)
        for i in range(1, im_rows + 1):
            im_fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", row=i, col=1)
            im_fig.update_xaxes(tickformat="%b '%y", showgrid=True, gridcolor="rgba(255,255,255,0.08)", row=i, col=1)

        st.plotly_chart(im_fig, use_container_width=True, key="im_chart")

        # Correlation matrix
        st.markdown("### Cross-Asset Correlations (63-day rolling)")
        corr_assets = ["SPY", "BTC"]
        if has_dxy:
            corr_assets.append("DXY")
        if has_gold:
            corr_assets.append("GOLD")

        corr_data = df[corr_assets].pct_change()
        corr_matrix = corr_data.tail(63).corr()

        corr_heat = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale=[[0, "#f87171"], [0.5, "#1a1d24"], [1, "#4ade80"]],
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont=dict(color="#e0e0e0", size=14),
        ))
        corr_heat.update_layout(height=350, **DARK_LAYOUT)
        st.plotly_chart(corr_heat, use_container_width=True, key="corr_heat")

    else:
        st.info("DXY and Gold data unavailable. Check your connection.")


# ─────────────────────────────────────────────────────
# TAB 7: ALERTS
# ─────────────────────────────────────────────────────
with tab_alerts:
    if not alert_on_regime_change:
        st.info("Enable alerts in the sidebar to view regime change notifications.")
    else:
        alert_asset = st.radio("View alerts for", ["SPY", "BTC", "Both"], horizontal=True)
        alert_list = []
        if alert_asset in ("SPY", "Both"):
            for a in spy_alerts:
                a["asset"] = "SPY"
                alert_list.append(a)
        if alert_asset in ("BTC", "Both"):
            for a in btc_alerts:
                a["asset"] = "BTC"
                alert_list.append(a)

        alert_list = sorted(alert_list, key=lambda a: a["date"], reverse=True)
        num_show = st.slider("Show last N", 5, min(50, max(len(alert_list), 5)), min(20, len(alert_list)))

        for a in alert_list[:num_show]:
            arrow_color = BADGE_COLORS.get(a["to"], "#888")
            from_color = BADGE_COLORS.get(a["from"], "#888")
            asset_tag = f"<span style='color:#818cf8;font-weight:600;'>[{a['asset']}]</span> " if alert_asset == "Both" else ""
            css_class = f"alert-{a['to'].lower().replace('-', '-')}"
            st.markdown(f"""
            <div class="alert-card {css_class}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        {asset_tag}
                        <strong style="color:#e0e0e0;">{a['date'].strftime('%b %d, %Y')}</strong>
                        &nbsp;
                        <span style="color:{from_color};font-weight:600;">{a['from']}</span>
                        <span style="color:#8b95a5;"> → </span>
                        <span style="color:{arrow_color};font-weight:600;">{a['to']}</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# TAB 5: DAILY REPORT
# ─────────────────────────────────────────────────────
with tab_report:
    st.markdown("### 📋 Daily Regime Report")

    REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
    STATE_FILE = os.path.join(REPORT_DIR, "regime_state.json")
    LOG_FILE = os.path.join(REPORT_DIR, "regime_alert_log.csv")

    prev_state = None
    spy_changed = btc_changed = False
    prev_spy = prev_btc = None
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                prev_state = json.load(f)
                prev_spy = prev_state.get("spy_regime")
                prev_btc = prev_state.get("btc_regime")
                if prev_spy and prev_spy != spy_regime:
                    spy_changed = True
                if prev_btc and prev_btc != btc_regime:
                    btc_changed = True
    except Exception:
        pass

    try:
        with open(STATE_FILE, "w") as f:
            json.dump({
                "spy_regime": spy_regime, "spy_score": round(float(spy_score), 4),
                "btc_regime": btc_regime, "btc_score": round(float(btc_score), 4),
                "date": df.index[-1].strftime("%Y-%m-%d"),
                "updated_at": datetime.now().isoformat(),
            }, f, indent=2)
    except Exception:
        pass

    try:
        log_row = pd.DataFrame([{
            "date": df.index[-1].strftime("%Y-%m-%d"),
            "spy_regime": spy_regime, "spy_score": round(float(spy_score), 4),
            "btc_regime": btc_regime, "btc_score": round(float(btc_score), 4),
            "spy_price": round(float(df["SPY"].iloc[-1]), 2),
            "btc_price": round(float(df["BTC"].iloc[-1]), 2),
            "spy_changed": spy_changed, "btc_changed": btc_changed,
        }])
        if os.path.exists(LOG_FILE):
            existing = pd.read_csv(LOG_FILE)
            if existing["date"].iloc[-1] != log_row["date"].iloc[0]:
                pd.concat([existing, log_row], ignore_index=True).to_csv(LOG_FILE, index=False)
        else:
            log_row.to_csv(LOG_FILE, index=False)
    except Exception:
        pass

    # Change banners
    for label, changed, prev_r, curr_r, score in [
        ("SPY", spy_changed, prev_spy, spy_regime, spy_score),
        ("BTC", btc_changed, prev_btc, btc_regime, btc_score),
    ]:
        if changed:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a1d24,#2d1a1a);border:2px solid #ef4444;
                        border-radius:12px;padding:16px;margin-bottom:12px;text-align:center;">
                <div style="font-size:1.3rem;font-weight:700;color:#ef4444;">🚨 {label} REGIME CHANGE</div>
                <div style="color:#e0e0e0;margin-top:4px;">
                    <span style="color:{BADGE_COLORS.get(prev_r,'#888')};font-weight:600;">{prev_r}</span>
                    → <span style="color:{BADGE_COLORS[curr_r]};font-weight:600;">{curr_r}</span>
                    &nbsp;(score: {score:.3f})
                </div>
            </div>""", unsafe_allow_html=True)

    if not spy_changed and not btc_changed:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a1d24,#21252e);border:1px solid #2d3344;
                    border-radius:12px;padding:16px;margin-bottom:12px;text-align:center;">
            <span style="color:#8b95a5;">No regime changes since last check</span>
        </div>""", unsafe_allow_html=True)

    # Summary tables
    rc1, rc2 = st.columns(2)
    spy_5d = (df["SPY"].iloc[-1] / df["SPY"].iloc[-5] - 1) * 100
    btc_5d = (df["BTC"].iloc[-1] / df["BTC"].iloc[-5] - 1) * 100

    with rc1:
        st.markdown("#### SPY")
        st.dataframe(pd.DataFrame({
            "Metric": ["Regime", "Score", "Price", "1-Day", "5-Day", "RSI", "Days in Regime"],
            "Value": [spy_regime, f"{spy_score:.3f}", f"${df['SPY'].iloc[-1]:,.2f}",
                      f"{df['SPY_Return'].iloc[-1]*100:+.2f}%", f"{spy_5d:+.2f}%",
                      f"{df['SPY_RSI'].iloc[-1]:.1f}", str(int(spy_duration))],
        }), use_container_width=True, hide_index=True, height=290)

    with rc2:
        st.markdown("#### BTC")
        st.dataframe(pd.DataFrame({
            "Metric": ["Regime", "Score", "Price", "1-Day", "5-Day", "RSI", "Days in Regime"],
            "Value": [btc_regime, f"{btc_score:.3f}", f"${df['BTC'].iloc[-1]:,.0f}",
                      f"{df['BTC_Return'].iloc[-1]*100:+.2f}%", f"{btc_5d:+.2f}%",
                      f"{df['BTC_RSI'].iloc[-1]:.1f}", str(int(btc_duration))],
        }), use_container_width=True, hide_index=True, height=290)

    # Log viewer
    st.markdown("---")
    st.markdown("#### 📜 Regime Check Log")
    try:
        if os.path.exists(LOG_FILE):
            hist_log = pd.read_csv(LOG_FILE).sort_values("date", ascending=False)
            st.dataframe(hist_log, use_container_width=True, hide_index=True, height=250)
        else:
            st.info("Log builds up each time the dashboard loads.")
    except Exception:
        st.info("Log not available yet.")


# ─────────────────────────────────────────────────────
# TAB 6: ANALYSIS
# ─────────────────────────────────────────────────────
with tab_analysis:
    analysis_asset = st.radio("Asset", ["SPY", "BTC"], horizontal=True, key="analysis_asset")
    regime_col = f"{analysis_asset}_Regime"
    block_col = f"{analysis_asset}_RegimeBlock"
    price_col = analysis_asset

    # Regime history table
    st.markdown(f"### {analysis_asset} Regime History")
    rows = []
    for _, group in df.groupby(block_col):
        regime = group[regime_col].iloc[0]
        ret = (group[price_col].iloc[-1] / group[price_col].iloc[0] - 1) * 100
        rows.append({
            "Regime": regime,
            "Start": group.index[0].strftime("%Y-%m-%d"),
            "End": group.index[-1].strftime("%Y-%m-%d"),
            "Duration": len(group),
            f"{analysis_asset} Return (%)": round(ret, 1),
        })
    regime_hist = pd.DataFrame(rows)

    def style_hist(tbl):
        def _bg(val):
            c = {"Risk-On": "#1a3a1a", "Neutral": "#3a3a1a", "Risk-Off": "#3a1a1a"}.get(val, "")
            return f"background-color:{c};color:#e0e0e0"
        def _ret(val):
            try:
                return "color:#22c55e" if float(val) > 0 else "color:#ef4444"
            except (ValueError, TypeError):
                return ""
        return tbl.style.map(_bg, subset=["Regime"]).map(_ret, subset=[f"{analysis_asset} Return (%)"])

    st.dataframe(style_hist(regime_hist), use_container_width=True, height=320)

    # Forward returns
    st.markdown("---")
    st.markdown(f"### {analysis_asset} Forward Return Analysis")
    fwd_rows = []
    for regime in ["Risk-On", "Neutral", "Risk-Off"]:
        mask = df[regime_col] == regime
        for h in [21, 63, 126]:
            fwd = df[price_col].pct_change(h).shift(-h)
            fwd_rows.append({
                "Regime": regime,
                "Horizon": f"{h}d (~{h//21}mo)",
                f"Avg Return %": round(fwd[mask].mean() * 100, 1),
                "N": int(mask.sum()),
            })
    st.dataframe(pd.DataFrame(fwd_rows), use_container_width=True, hide_index=True, height=250)

    # Rolling correlation
    st.markdown("---")
    st.markdown("### Rolling 63-day BTC/SPY Correlation")
    rolling_corr = df["SPY_Return"].rolling(63).corr(df["BTC_Return"])
    rc_fig = go.Figure()
    rc_fig.add_trace(go.Scatter(
        x=df.index, y=rolling_corr, line=dict(width=1.5, color="#a78bfa"),
        name="63d Corr", hovertemplate="%{x|%b %d %Y}<br>Corr: %{y:.3f}<extra></extra>",
    ))
    rc_fig.add_hline(y=0, line_dash="dot", line_color="rgba(200,200,200,0.4)")
    rc_fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(239,68,68,0.4)")
    rc_fig.update_layout(height=300, **DARK_LAYOUT, yaxis_title="Correlation")
    st.plotly_chart(rc_fig, use_container_width=True, key="rolling_corr")


# ─────────────────────────────────────────────────────
# TAB 7: METHODOLOGY
# ─────────────────────────────────────────────────────
with tab_methodology:
    st.markdown("""
### Dual Regime Architecture (v5.0)

SPY and BTC are classified independently using asset-specific factor models.

#### SPY Factors (Equity)
| Factor | Calculation | Weight |
|--------|-------------|--------|
| **Trend** | 63-day SPY return, z-scored | +40% |
| **VIX** | Raw VIX level, z-scored | −30% |
| **Yield Curve** | 10yr − 3mo spread, z-scored | +15% |
| **Vol Spread** | 20d − 50d rolling vol, z-scored | −15% |

#### BTC Factors (Crypto)
| Factor | Calculation | Weight |
|--------|-------------|--------|
| **Momentum** | 63-day BTC return, z-scored | +35% |
| **Realized Vol** | 30-day log-return vol (annualized), z-scored | −25% |
| **Volume Ratio** | Daily volume / 50d avg, z-scored | +15% |
| **MVRV Proxy** | Price / 200-SMA, z-scored | +25% |

#### Why separate?

BTC and SPY respond to different macro dynamics. Equity markets are driven by interest rates,
corporate earnings expectations, and risk premia. Crypto markets are driven by liquidity cycles,
on-chain activity, and speculative momentum. A unified score would dilute both signals.

#### Score Construction
1. Weighted sum of z-scored factors (per asset)
2. EWM smoothing (span=10) to reduce whipsawing
3. Threshold at ±0.4 → regime classification

#### Score Velocity
The 5-day rate of change of each composite score, indicating whether macro conditions
are improving or deteriorating. Displayed as directional arrows (▲/▼/►) beneath each gauge.
Useful for anticipating regime transitions before they trigger.

#### Regime Divergence
When SPY and BTC regimes disagree (e.g., SPY Risk-On but BTC Risk-Off), a divergence
banner appears. Persistent divergences often signal rotational shifts or decorrelation events
worth monitoring for position sizing.

#### Strategy Backtester
Compares a regime-following strategy against buy-and-hold. The regime strategy allocates
100% in Risk-On, 50% in Neutral, and 0% in Risk-Off. Exposure uses a 1-day lag to avoid
lookahead bias. Metrics include total return, CAGR, max drawdown, Sharpe ratio, and
time-in-regime breakdowns.

#### Drawdown Analysis
Tracks peak-to-trough drawdowns from all-time highs for both assets. Per-regime max
drawdown tables reveal which environments produce the deepest losses. Current drawdown
cards provide at-a-glance risk context.

#### Intermarket Context
DXY (US Dollar Index) and Gold provide macro backdrop. A strong dollar often pressures
risk assets; gold strength may signal flight-to-safety flows. The 63-day rolling cross-asset
correlation heatmap reveals changing intermarket relationships.

#### On-Chain Proxies
- **MVRV Proxy:** Price / 200-day SMA approximates market value / realized value
- **Volume Ratio:** Daily volume vs 50-day average measures exchange activity
- **Fear & Greed:** External sentiment index from alternative.me (0=extreme fear, 100=extreme greed)
    """)


# =====================================================
# AUTO-REFRESH
# =====================================================
if auto_refresh:
    st.markdown(
        f"""<meta http-equiv="refresh" content="{refresh_interval}">
        <p style="text-align:center;color:#4ade80;font-size:0.8rem;">
        🔄 Auto-refresh every {refresh_interval // 60} min{'s' if refresh_interval > 60 else ''}
        </p>""", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(
    f"<p style='text-align:center;color:#555;font-size:0.8rem;'>"
    f"Market Immune System v5.0 — Data through {df.index[-1].strftime('%b %d, %Y')} via Yahoo Finance — Not financial advice"
    f"</p>", unsafe_allow_html=True)
