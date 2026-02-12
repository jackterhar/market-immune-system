import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Market Immune System — Dual Regime Engine")

# -----------------------------
# DATA LOADING
# -----------------------------
@st.cache_data
def load_data():
    spy = yf.download("SPY", period="5y")
    btc = yf.download("BTC-USD", period="5y")
    vix = yf.download("^VIX", period="5y")
    hyg = yf.download("HYG", period="5y")
    ief = yf.download("IEF", period="5y")
    t10 = yf.download("^TNX", period="5y")
    t2 = yf.download("^IRX", period="5y")

    df = pd.DataFrame(index=spy.index)

    df["SPY"] = spy["Close"]
    df["BTC"] = btc["Close"]
    df["VIX"] = vix["Close"]
    df["HYG"] = hyg["Close"]
    df["IEF"] = ief["Close"]
    df["10Y"] = t10["Close"]
    df["2Y"] = t2["Close"]

    df = df.dropna()

    # Credit ratio
    df["Credit_Ratio"] = df["HYG"] / df["IEF"]

    # Yield curve
    df["Curve"] = df["10Y"] - df["2Y"]

    return df

df = load_data()

# -----------------------------
# INDICATORS
# -----------------------------
def add_indicators(df):

    # SPY MAs
    df["SPY_50"] = df["SPY"].rolling(50).mean()
    df["SPY_200"] = df["SPY"].rolling(200).mean()

    # BTC MAs
    df["BTC_50"] = df["BTC"].rolling(50).mean()
    df["BTC_200"] = df["BTC"].rolling(200).mean()

    # Volatility
    df["SPY_vol"] = df["SPY"].pct_change().rolling(20).std() * np.sqrt(252)
    df["BTC_vol"] = df["BTC"].pct_change().rolling(20).std() * np.sqrt(252)

    df["SPY_vol_med"] = df["SPY_vol"].rolling(252).median()
    df["BTC_vol_med"] = df["BTC_vol"].rolling(252).median()

    # Credit trend
    df["Credit_Trend"] = df["Credit_Ratio"].pct_change(60)

    return df

df = add_indicators(df)

# -----------------------------
# REGIME SCORING
# -----------------------------
def score_spy(row):
    score = 0

    # Trend
    if row["SPY"] < row["SPY_200"]:
        score += 1
    if row["SPY_50"] < row["SPY_200"]:
        score += 1

    # Volatility
    if row["SPY_vol"] > row["SPY_vol_med"]:
        score += 1

    # Credit
    if row["Credit_Trend"] < 0:
        score += 1

    # Yield curve
    if row["Curve"] < 0:
        score += 1

    return score  # 0–5


def score_btc(row):
    score = 0

    # Trend
    if row["BTC"] < row["BTC_200"]:
        score += 1
    if row["BTC_50"] < row["BTC_200"]:
        score += 1

    # Volatility
    if row["BTC_vol"] > row["BTC_vol_med"]:
        score += 1

    # VIX stress
    if row["VIX"] > 25:
        score += 1

    return score  # 0–4


def score_macro(row):
    score = 0

    if row["Curve"] < 0:
        score += 1
    if row["Credit_Trend"] < 0:
        score += 1
    if row["VIX"] > 25:
        score += 1

    return score  # 0–3


df["SPY_score"] = df.apply(score_spy, axis=1)
df["BTC_score"] = df.apply(score_btc, axis=1)
df["Macro_score"] = df.apply(score_macro, axis=1)

df["Composite_score"] = (
    df["SPY_score"] + df["BTC_score"] + df["Macro_score"]
)

# -----------------------------
# REGIME LABELS
# -----------------------------
def label_regime(score, max_score):

    if score <= max_score * 0.33:
        return "RISK-ON"
    elif score <= max_score * 0.66:
        return "CAUTION"
    else:
        return "RISK-OFF"

latest = df.iloc[-1]

spy_regime = label_regime(latest["SPY_score"], 5)
btc_regime = label_regime(latest["BTC_score"], 4)
macro_regime = label_regime(latest["Macro_score"], 3)
composite_regime = label_regime(latest["Composite_score"], 12)

# -----------------------------
# DISPLAY METRICS
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("SPY Regime", spy_regime)
col2.metric("BTC Regime", btc_regime)
col3.metric("Macro Overlay", macro_regime)
col4.metric("Composite", composite_regime)

# -----------------------------
# CHART FUNCTION
# -----------------------------
def plot_chart(asset, score_col, max_score):

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df[asset], label=asset)

    for i in range(len(df)):
        score = df.iloc[i][score_col]
        regime = label_regime(score, max_score)

        if regime == "RISK-OFF":
            ax.axvspan(df.index[i], df.index[i], color="red", alpha=0.2)
        elif regime == "CAUTION":
            ax.axvspan(df.index[i], df.index[i], color="yellow", alpha=0.2)

    ax.set_title(f"{asset} with Independent Regime Shading")
    ax.legend()
    st.pyplot(fig)


# -----------------------------
# PLOTS
# -----------------------------
st.subheader("SPY Regime Chart")
plot_chart("SPY", "SPY_score", 5)

st.subheader("BTC Regime Chart")
plot_chart("BTC", "BTC_score", 4)
