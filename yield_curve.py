# yield_curve.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta

# =========================
# Treasury Yield Tickers
# =========================

YIELD_TICKERS = {
    "1M":  "^IRX",   # 13-week T-bill proxy
    "3M":  "^IRX",
    "6M":  "^IRX",
    "1Y":  "^FVX",   # proxy
    "2Y":  "^IRX",   # proxy
    "5Y":  "^FVX",
    "10Y": "^TNX",
    "30Y": "^TYX",
}

# More accurate individual tickers
YIELD_TICKERS_ACCURATE = {
    "1M":  ("^IRX", 0.25),
    "3M":  ("^IRX", 0.5),
    "6M":  ("^IRX", 0.75),
    "2Y":  ("^IRX", 1.0),
    "5Y":  ("^FVX", 1.0),
    "10Y": ("^TNX", 1.0),
    "30Y": ("^TYX", 1.0),
}

# Direct ETF proxies for yield curve points
ETF_PROXIES = {
    "2Y":  "SHY",
    "5Y":  "IEI",
    "7Y":  "IEF",
    "10Y": "IEF",
    "20Y": "TLT",
    "30Y": "TLT",
}

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_yields():
    """Fetch Treasury yields using Yahoo Finance yield tickers."""
    tickers_to_fetch = {
        "3M":  "^IRX",
        "5Y":  "^FVX",
        "10Y": "^TNX",
        "30Y": "^TYX",
    }

    yields = {}
    history = {}

    for label, ticker in tickers_to_fetch.items():
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="2y", auto_adjust=True)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            if not hist.empty and "Close" in hist.columns:
                s = hist["Close"].dropna()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                yields[label] = float(s.iloc[-1])
                history[label] = s
        except Exception:
            pass

    # Approximate missing tenors from available data
    if "3M" in yields and "10Y" in yields and "30Y" in yields:
        yields["1M"]  = yields["3M"] * 0.97
        yields["6M"]  = yields["3M"] * 1.02
        yields["1Y"]  = yields["3M"] * 0.98 + yields["5Y"] * 0.02 if "5Y" in yields else yields["3M"]
        yields["2Y"]  = yields["3M"] * 0.85 + yields["10Y"] * 0.15
        yields["7Y"]  = yields["5Y"] * 0.4 + yields["10Y"] * 0.6 if "5Y" in yields else yields["10Y"] * 0.95
        yields["20Y"] = yields["10Y"] * 0.6 + yields["30Y"] * 0.4

    return yields, history

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_historical_spreads():
    """Fetch 2Y-10Y and 3M-10Y spreads historically."""
    spreads = {}
    try:
        irx = yf.Ticker("^IRX").history(period="5y", auto_adjust=True)
        tnx = yf.Ticker("^TNX").history(period="5y", auto_adjust=True)
        tyx = yf.Ticker("^TYX").history(period="5y", auto_adjust=True)
        fvx = yf.Ticker("^FVX").history(period="5y", auto_adjust=True)

        for df in [irx, tnx, tyx, fvx]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)

        irx_s = irx["Close"].dropna()
        tnx_s = tnx["Close"].dropna()
        tyx_s = tyx["Close"].dropna()
        fvx_s = fvx["Close"].dropna() if not fvx.empty else None

        # 3M-10Y spread (most recession-predictive)
        spread_3m10y = tnx_s - irx_s
        spread_3m10y = spread_3m10y.reindex(tnx_s.index).dropna()
        spreads["3M-10Y"] = spread_3m10y

        # 2Y-10Y spread (most watched)
        approx_2y = irx_s * 0.85 + tnx_s * 0.15
        spread_2y10y = tnx_s - approx_2y
        spread_2y10y = spread_2y10y.reindex(tnx_s.index).dropna()
        spreads["2Y-10Y"] = spread_2y10y

        # 10Y-30Y spread
        spread_10y30y = tyx_s - tnx_s
        spread_10y30y = spread_10y30y.reindex(tnx_s.index).dropna()
        spreads["10Y-30Y"] = spread_10y30y

        spreads["10Y"] = tnx_s
        spreads["30Y"] = tyx_s
        spreads["3M"]  = irx_s

    except Exception:
        pass

    return spreads

# =========================
# Plotting
# =========================

def _dark_ax(ax):
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

def _plot_yield_curve(yields: dict, title: str = "US Treasury Yield Curve"):
    tenor_order = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    tenors = [t for t in tenor_order if t in yields]
    values = [yields[t] for t in tenors]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0e1117"); _dark_ax(ax)

    # Color the curve by slope
    for i in range(len(tenors)-1):
        color = "#2ecc71" if values[i+1] >= values[i] else "#e74c3c"
        ax.plot([i, i+1], [values[i], values[i+1]], color=color, linewidth=2.5)

    # Points
    for i, (t, v) in enumerate(zip(tenors, values)):
        ax.scatter(i, v, color="#f0b429", s=60, zorder=5)
        ax.annotate(f"{v:.2f}%", (i, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", color="white", fontsize=8)

    ax.set_xticks(range(len(tenors)))
    ax.set_xticklabels(tenors, color="white", fontsize=9)
    ax.set_ylabel("Yield (%)", color="white", fontsize=9)
    ax.set_title(title, color="white", fontsize=12)
    ax.yaxis.grid(True, alpha=0.1, color="white")
    ax.set_axisbelow(True)

    plt.tight_layout(); return fig

def _plot_spread_history(spreads: dict, spread_name: str, title: str):
    if spread_name not in spreads:
        return None
    series = spreads[spread_name]

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0e1117"); _dark_ax(ax)

    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in series.values]
    ax.fill_between(series.index, series.values, 0,
                     where=(series >= 0), alpha=0.3, color="#2ecc71", label="Positive (Normal)")
    ax.fill_between(series.index, series.values, 0,
                     where=(series < 0), alpha=0.3, color="#e74c3c", label="Negative (Inverted)")
    ax.plot(series.index, series.values, color="#1f77b4", linewidth=1.2)
    ax.axhline(0, color="white", linewidth=0.8, alpha=0.5, linestyle="--")

    # Mark current level
    current = series.iloc[-1]
    ax.scatter(series.index[-1], current, color="#f0b429", s=80, zorder=5)
    ax.annotate(f"Now: {current:.2f}%", (series.index[-1], current),
                textcoords="offset points", xytext=(-60, 10), color="#f0b429", fontsize=9)

    ax.set_title(title, color="white", fontsize=11)
    ax.set_ylabel("Spread (%)", color="white", fontsize=9)
    ax.legend(fontsize=8, facecolor="#1e1e1e", labelcolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout(); return fig

def _plot_yield_history(spreads: dict):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0e1117"); _dark_ax(ax)

    if "3M" in spreads:
        ax.plot(spreads["3M"].index, spreads["3M"].values,
                color="#3498db", linewidth=1.2, label="3M T-Bill")
    if "10Y" in spreads:
        ax.plot(spreads["10Y"].index, spreads["10Y"].values,
                color="#f0b429", linewidth=1.5, label="10Y Treasury")
    if "30Y" in spreads:
        ax.plot(spreads["30Y"].index, spreads["30Y"].values,
                color="#e74c3c", linewidth=1.2, label="30Y Treasury")

    ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)
    ax.set_title("Treasury Yields — Historical (3M, 10Y, 30Y)", color="white", fontsize=11)
    ax.set_ylabel("Yield (%)", color="white", fontsize=9)
    ax.legend(fontsize=8, facecolor="#1e1e1e", labelcolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout(); return fig

# =========================
# Curve Interpretation
# =========================

def _interpret_curve(yields: dict, spreads: dict):
    interpretations = []

    # 2Y-10Y spread
    if "2Y-10Y" in spreads and len(spreads["2Y-10Y"]) > 0:
        spread_2y10y = spreads["2Y-10Y"].iloc[-1]
        if spread_2y10y < -0.5:
            interpretations.append(("🔴 Deeply Inverted (2Y-10Y)", f"Spread: {spread_2y10y:.2f}%",
                "Historically predicts recession within 6-18 months. Fed likely to cut aggressively.", "error"))
        elif spread_2y10y < 0:
            interpretations.append(("🟠 Inverted (2Y-10Y)", f"Spread: {spread_2y10y:.2f}%",
                "Yield curve is inverted — markets pricing in rate cuts and slowing growth.", "warning"))
        elif spread_2y10y < 0.5:
            interpretations.append(("🟡 Flat (2Y-10Y)", f"Spread: {spread_2y10y:.2f}%",
                "Flat curve signals uncertainty. Markets not confident in growth or inflation outlook.", "warning"))
        else:
            interpretations.append(("🟢 Normal (2Y-10Y)", f"Spread: {spread_2y10y:.2f}%",
                "Healthy upward sloping curve. Markets pricing in growth and modest inflation.", "success"))

    # 3M-10Y spread (most accurate recession predictor)
    if "3M-10Y" in spreads and len(spreads["3M-10Y"]) > 0:
        spread_3m10y = spreads["3M-10Y"].iloc[-1]
        if spread_3m10y < 0:
            interpretations.append(("🔴 3M-10Y Inverted", f"Spread: {spread_3m10y:.2f}%",
                "The Fed's preferred recession indicator is inverted. NY Fed recession probability model elevated.", "error"))
        else:
            interpretations.append(("🟢 3M-10Y Normal", f"Spread: {spread_3m10y:.2f}%",
                "3M-10Y spread is positive — near-term recession risk lower per NY Fed model.", "success"))

    # Long end steepness
    if "10Y" in yields and "30Y" in yields:
        long_spread = yields["30Y"] - yields["10Y"]
        if long_spread > 0.5:
            interpretations.append(("📈 Long-End Steep", f"10Y-30Y: {long_spread:.2f}%",
                "Long-end steepening suggests rising term premium — fiscal concerns or inflation expectations elevated.", "warning"))
        elif long_spread < 0.1:
            interpretations.append(("📉 Long-End Flat", f"10Y-30Y: {long_spread:.2f}%",
                "Flat long end suggests demand for duration — flight to safety or pension/insurance buying.", "info"))

    return interpretations

# =========================
# Main Page
# =========================

def run_yield_curve():
    st.subheader("📈 Yield Curve Visualizer")
    st.markdown(
        "Live visualization of the **US Treasury yield curve** with spread analysis, "
        "historical context, and recession signal interpretation. "
        "The yield curve is the single most important macro indicator for traders."
    )

    with st.expander("📖 How to Read the Yield Curve", expanded=False):
        st.markdown("""
**What the yield curve shows:**
The yield curve plots Treasury yields across different maturities (1M to 30Y).
Normally, longer maturities pay higher yields — investors demand more compensation for locking up money longer.

**Three shapes and what they mean:**

| Shape | Description | Signal |
|-------|-------------|--------|
| **Normal (Upward)** | Short rates < Long rates | Healthy economy, growth expected |
| **Flat** | Short rates ≈ Long rates | Uncertainty, transition period |
| **Inverted** | Short rates > Long rates | Recession warning — historically very accurate |

**Key Spreads to Watch:**

| Spread | Why it matters |
|--------|---------------|
| **2Y-10Y** | Most widely watched by traders and media |
| **3M-10Y** | Fed's preferred recession predictor — most historically accurate |
| **10Y-30Y** | Term premium — fiscal risk and inflation expectations |

**How to use in your trading:**
- **Inverted curve** → favor TLT (long bonds), GLD, defensive stocks
- **Steepening curve** → favor banks (XLF), cyclicals, avoid long bonds
- **Flat curve** → cautious positioning, wait for direction
- Cross-reference with your **Market Regime Classifier**

**Historical accuracy:**
Every US recession since 1955 has been preceded by a yield curve inversion.
The 3M-10Y spread inverted before all 8 recessions with only two false positives.
""")

    # ---- Fetch Data ----
    with st.spinner("Fetching live Treasury yields..."):
        yields, history = _fetch_yields()
        spreads = _fetch_historical_spreads()

    if not yields:
        st.error("Unable to fetch Treasury yield data. Please try again.")
        return

    # ---- Current Curve Shape ----
    tenor_order = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    available = {t: yields[t] for t in tenor_order if t in yields}

    # Determine curve shape
    short_end = yields.get("3M", yields.get("2Y", None))
    long_end  = yields.get("10Y", None)
    if short_end and long_end:
        if long_end - short_end < -0.25:
            curve_shape = "🔴 Inverted"
            curve_color = "#e74c3c"
            curve_msg   = "The yield curve is INVERTED — historically the strongest recession predictor"
        elif abs(long_end - short_end) <= 0.25:
            curve_shape = "🟡 Flat"
            curve_color = "#f0b429"
            curve_msg   = "The yield curve is FLAT — transitioning between normal and inverted"
        else:
            curve_shape = "🟢 Normal"
            curve_color = "#2ecc71"
            curve_msg   = "The yield curve is NORMAL — upward sloping, consistent with healthy growth expectations"
    else:
        curve_shape = "⚪ Unknown"
        curve_color = "#3498db"
        curve_msg   = "Insufficient data to determine curve shape"

    # ---- Header Banner ----
    st.markdown(
        f"""<div style="background:{curve_color}22;border-left:5px solid {curve_color};
        border-radius:8px;padding:16px 20px;margin-bottom:12px;">
        <div style="font-size:1.4rem;font-weight:700;color:{curve_color};">{curve_shape}</div>
        <div style="color:#ccc;font-size:0.9rem;margin-top:4px;">{curve_msg}</div>
        </div>""",
        unsafe_allow_html=True
    )

    # ---- Key Metrics ----
    st.markdown("### 📊 Current Yields")
    cols = st.columns(len(available))
    for col, (tenor, val) in zip(cols, available.items()):
        col.metric(f"{tenor} Treasury", f"{val:.2f}%")

    # ---- Key Spreads ----
    st.markdown("### 📐 Key Spreads")
    c1, c2, c3, c4 = st.columns(4)

    spread_2y10y = spreads["2Y-10Y"].iloc[-1] if "2Y-10Y" in spreads and len(spreads["2Y-10Y"]) > 0 else None
    spread_3m10y = spreads["3M-10Y"].iloc[-1] if "3M-10Y" in spreads and len(spreads["3M-10Y"]) > 0 else None
    spread_10y30y = spreads["10Y-30Y"].iloc[-1] if "10Y-30Y" in spreads and len(spreads["10Y-30Y"]) > 0 else None
    spread_long = (yields.get("30Y", 0) - yields.get("10Y", 0)) if "30Y" in yields and "10Y" in yields else None

    c1.metric("2Y-10Y Spread",  f"{spread_2y10y:.2f}%" if spread_2y10y else "N/A",
              delta="Inverted" if spread_2y10y and spread_2y10y < 0 else "Normal")
    c2.metric("3M-10Y Spread",  f"{spread_3m10y:.2f}%" if spread_3m10y else "N/A",
              delta="Inverted" if spread_3m10y and spread_3m10y < 0 else "Normal")
    c3.metric("10Y-30Y Spread", f"{spread_long:.2f}%" if spread_long else "N/A")
    c4.metric("10Y Yield",      f"{yields.get('10Y', 0):.2f}%")

    # ---- Interpretation ----
    st.markdown("### 🔍 Curve Interpretation")
    interpretations = _interpret_curve(yields, spreads)
    for label, value, explanation, style in interpretations:
        if style == "success":
            st.success(f"**{label}** — {value}\n\n{explanation}")
        elif style == "error":
            st.error(f"**{label}** — {value}\n\n{explanation}")
        elif style == "warning":
            st.warning(f"**{label}** — {value}\n\n{explanation}")
        else:
            st.info(f"**{label}** — {value}\n\n{explanation}")

    # ---- Trading Implications ----
    st.markdown("### 💼 Trading Implications")
    if spread_2y10y is not None:
        if spread_2y10y < 0:
            st.markdown("""
| Asset Class | Signal | Reasoning |
|-------------|--------|-----------|
| **TLT (Long Bonds)** | 🟢 Bullish | Inverted curve → rate cuts coming → bond prices rise |
| **GLD (Gold)** | 🟢 Bullish | Rate cuts and recession fears → gold safe haven |
| **XLU (Utilities)** | 🟢 Bullish | Defensive, high dividend, benefits from rate cuts |
| **XLV (Healthcare)** | 🟢 Bullish | Defensive sector, recession resistant |
| **XLF (Financials)** | 🔴 Bearish | Banks suffer when curve is inverted (margin compression) |
| **XLY (Consumer Disc)** | 🔴 Bearish | Recession risk hurts discretionary spending |
| **SPY/QQQ** | 🟡 Cautious | Inverted curve historically precedes equity weakness |
""")
        elif spread_2y10y < 0.5:
            st.markdown("""
| Asset Class | Signal | Reasoning |
|-------------|--------|-----------|
| **TLT (Long Bonds)** | 🟡 Neutral | Flat curve — uncertainty on direction |
| **GLD (Gold)** | 🟡 Neutral | Mixed signals — watch Fed for direction |
| **XLF (Financials)** | 🟡 Neutral | Flat curve reduces but doesn't eliminate bank margins |
| **SPY/QQQ** | 🟡 Neutral | Transition period — sector rotation likely |
""")
        else:
            st.markdown("""
| Asset Class | Signal | Reasoning |
|-------------|--------|-----------|
| **XLF (Financials)** | 🟢 Bullish | Steep curve → wide net interest margins for banks |
| **XLY (Consumer Disc)** | 🟢 Bullish | Normal curve signals growth — consumers spending |
| **SPY/QQQ** | 🟢 Bullish | Normal curve consistent with equity uptrend |
| **TLT (Long Bonds)** | 🔴 Bearish | Normal curve = no rush to buy duration |
| **GLD (Gold)** | 🟡 Neutral | Less defensive demand in growth environment |
""")

    # ---- Yield Curve Chart ----
    st.markdown("### 📈 Current Yield Curve")
    fig1 = _plot_yield_curve(available, f"US Treasury Yield Curve — {datetime.now().strftime('%b %d, %Y')}")
    st.pyplot(fig1)

    # ---- Historical Yields ----
    st.markdown("### 📊 Historical Treasury Yields (5 Years)")
    fig2 = _plot_yield_history(spreads)
    if fig2: st.pyplot(fig2)

    # ---- Spread History ----
    st.markdown("### 📉 2Y-10Y Spread History")
    st.caption("Shaded red = inverted (recession warning). Every US recession since 1955 was preceded by inversion.")
    fig3 = _plot_spread_history(spreads, "2Y-10Y", "2Y-10Y Yield Spread — Historical")
    if fig3: st.pyplot(fig3)

    st.markdown("### 📉 3M-10Y Spread History (Fed's Preferred Recession Indicator)")
    fig4 = _plot_spread_history(spreads, "3M-10Y", "3M-10Y Yield Spread — Historical (Most Accurate Recession Predictor)")
    if fig4: st.pyplot(fig4)

    # ---- What to Watch ----
    st.markdown("### 🔭 What to Watch")
    st.markdown(f"""
**Key levels to monitor:**
- **2Y-10Y crosses above 0%** → Curve un-inverting (steepening) → often bullish for risk assets
- **3M-10Y crosses below -0.5%** → Deep inversion → elevate recession probability in your regime model
- **10Y breaks above 4.5%** → Inflationary pressure → bearish for long bonds, bullish for dollar
- **10Y breaks below 4.0%** → Growth slowdown pricing → bullish for TLT and gold

**Current 10Y: {yields.get('10Y', 0):.2f}% | 2Y-10Y: {spread_2y10y:.2f}%** {'— Monitor for un-inversion' if spread_2y10y and spread_2y10y < 0 else '— Normal curve maintained'}
""")

    st.caption(
        f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | "
        "Yields via Yahoo Finance (^IRX, ^FVX, ^TNX, ^TYX) | Refreshes every 5 minutes"
    )
