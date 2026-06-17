#cd /workspaces/apolloquant && git add . && git commit -m "Add AI Market Regime Classifier" && git push
np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timezone, timedelta

# =========================
# Data Fetching
# =========================

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_regime_data(period="1y"):
    """Fetch all cross-asset data needed for regime classification."""
    tickers = {
        "SPY":  "SPY",       # S&P 500
        "VIX":  "^VIX",      # Volatility
        "DXY":  "DX-Y.NYB",  # US Dollar
        "OIL":  "CL=F",      # WTI Crude
        "GOLD": "GLD",       # Gold
        "HYG":  "HYG",       # High Yield Bonds (credit proxy)
        "LQD":  "LQD",       # Investment Grade Bonds
        "TLT":  "TLT",       # 20Y Treasury (rates proxy)
        "TNX":  "^TNX",      # 10Y Yield
        "IRX":  "^IRX",      # 2Y Yield proxy
        "IVV":  "IVV",       # S&P 500 ETF (backup)
    }

    frames = {}
    for name, ticker in tickers.items():
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period=period, auto_adjust=True)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            if not hist.empty and "Close" in hist.columns:
                s = hist["Close"].dropna()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                frames[name] = s
        except Exception:
            pass

    if not frames:
        return None

    df = pd.DataFrame(frames)
    df = df.sort_index()
    return df


# =========================
# Regime Scoring Engine
# =========================

def _compute_scores(df: pd.DataFrame):
    """
    Compute daily regime scores using rule-based cross-asset signals.
    Returns a DataFrame with scores and regime labels for each day.
    """
    results = []

    # Need at least 60 days of history for momentum signals
    if len(df) < 60:
        return None

    for i in range(60, len(df)):
        row = df.iloc[i]
        window_20  = df.iloc[max(0, i-20):i]
        window_60  = df.iloc[max(0, i-60):i]
        date = df.index[i]

        scores = {
            "risk_on":       0.0,
            "risk_off":      0.0,
            "inflationary":  0.0,
            "recessionary":  0.0,
            "neutral":       0.0,
        }

        # ---- SPY Momentum ----
        if "SPY" in df.columns and "SPY" in window_20.columns:
            spy_ret_20 = (row.get("SPY", np.nan) / window_20["SPY"].iloc[0] - 1) if len(window_20) > 0 else np.nan
            spy_ret_60 = (row.get("SPY", np.nan) / window_60["SPY"].iloc[0] - 1) if len(window_60) > 0 else np.nan
            if pd.notna(spy_ret_20):
                if spy_ret_20 > 0.03:
                    scores["risk_on"] += 2.0
                elif spy_ret_20 > 0.0:
                    scores["risk_on"] += 1.0
                elif spy_ret_20 < -0.05:
                    scores["risk_off"] += 2.0
                    scores["recessionary"] += 1.0
                elif spy_ret_20 < 0.0:
                    scores["risk_off"] += 1.0
            if pd.notna(spy_ret_60):
                if spy_ret_60 < -0.10:
                    scores["recessionary"] += 2.0
                elif spy_ret_60 > 0.08:
                    scores["risk_on"] += 1.0

        # ---- VIX ----
        if "VIX" in df.columns:
            vix = row.get("VIX", np.nan)
            if pd.notna(vix):
                if vix < 15:
                    scores["risk_on"] += 2.0
                elif vix < 20:
                    scores["risk_on"] += 1.0
                elif vix > 30:
                    scores["risk_off"] += 2.5
                    scores["recessionary"] += 1.0
                elif vix > 20:
                    scores["risk_off"] += 1.5

        # ---- DXY ----
        if "DXY" in df.columns and len(window_20) > 0 and "DXY" in window_20.columns:
            dxy_ret = (row.get("DXY", np.nan) / window_20["DXY"].iloc[0] - 1) if pd.notna(row.get("DXY")) else np.nan
            if pd.notna(dxy_ret):
                if dxy_ret > 0.02:
                    scores["risk_off"] += 1.5   # strong dollar = risk-off / tightening
                    scores["inflationary"] += 0.5
                elif dxy_ret < -0.02:
                    scores["risk_on"] += 1.0    # weak dollar = risk-on / reflationary

        # ---- Oil ----
        if "OIL" in df.columns and len(window_20) > 0 and "OIL" in window_20.columns:
            oil_ret = (row.get("OIL", np.nan) / window_20["OIL"].iloc[0] - 1) if pd.notna(row.get("OIL")) else np.nan
            if pd.notna(oil_ret):
                if oil_ret > 0.05:
                    scores["inflationary"] += 2.0
                elif oil_ret > 0.02:
                    scores["inflationary"] += 1.0
                elif oil_ret < -0.05:
                    scores["recessionary"] += 1.5
                    scores["risk_off"] += 0.5

        # ---- Gold ----
        if "GOLD" in df.columns and len(window_20) > 0 and "GOLD" in window_20.columns:
            gold_ret = (row.get("GOLD", np.nan) / window_20["GOLD"].iloc[0] - 1) if pd.notna(row.get("GOLD")) else np.nan
            if pd.notna(gold_ret):
                if gold_ret > 0.03:
                    scores["risk_off"] += 1.5   # gold rallying = fear / inflation hedge
                    scores["inflationary"] += 1.0
                elif gold_ret < -0.02:
                    scores["risk_on"] += 0.5

        # ---- Yield Curve (2Y/10Y spread proxy via IRX/TNX) ----
        if "TNX" in df.columns and "IRX" in df.columns:
            tnx = row.get("TNX", np.nan)
            irx = row.get("IRX", np.nan)
            if pd.notna(tnx) and pd.notna(irx):
                spread = tnx - irx
                if spread < 0:
                    scores["recessionary"] += 2.0   # inverted yield curve
                elif spread < 0.5:
                    scores["recessionary"] += 0.5
                elif spread > 1.5:
                    scores["risk_on"] += 1.0        # steep curve = growth

        # ---- 10Y Yield Level ----
        if "TNX" in df.columns:
            tnx = row.get("TNX", np.nan)
            if pd.notna(tnx):
                if tnx > 4.5:
                    scores["inflationary"] += 1.5
                    scores["risk_off"] += 0.5
                elif tnx < 3.0:
                    scores["recessionary"] += 1.0

        # ---- Credit Spreads (HYG/LQD ratio as proxy) ----
        if "HYG" in df.columns and "LQD" in df.columns and len(window_20) > 0:
            if "HYG" in window_20.columns and "LQD" in window_20.columns:
                hyg_now = row.get("HYG", np.nan)
                lqd_now = row.get("LQD", np.nan)
                hyg_prev = window_20["HYG"].iloc[0]
                lqd_prev = window_20["LQD"].iloc[0]
                if all(pd.notna(x) for x in [hyg_now, lqd_now, hyg_prev, lqd_prev]) and lqd_prev > 0 and hyg_prev > 0:
                    ratio_now  = hyg_now / lqd_now
                    ratio_prev = hyg_prev / lqd_prev
                    ratio_chg  = (ratio_now / ratio_prev) - 1
                    if ratio_chg < -0.02:
                        scores["risk_off"] += 2.0
                        scores["recessionary"] += 1.0
                    elif ratio_chg > 0.02:
                        scores["risk_on"] += 1.5

        # ---- TLT (Treasury bond price — risk-off = TLT rallies) ----
        if "TLT" in df.columns and len(window_20) > 0 and "TLT" in window_20.columns:
            tlt_ret = (row.get("TLT", np.nan) / window_20["TLT"].iloc[0] - 1) if pd.notna(row.get("TLT")) else np.nan
            if pd.notna(tlt_ret):
                if tlt_ret > 0.03:
                    scores["risk_off"] += 1.5
                    scores["recessionary"] += 0.5
                elif tlt_ret < -0.03:
                    scores["inflationary"] += 1.0

        # ---- Classify ----
        total = sum(scores.values())
        if total == 0:
            regime = "Neutral"
            confidence = 0.0
        else:
            regime_key = max(scores, key=scores.get)
            top_score = scores[regime_key]
            second_score = sorted(scores.values())[-2]
            confidence = min(100.0, (top_score / total) * 100 * 1.5)

            regime_map = {
                "risk_on":      "Risk-On 🟢",
                "risk_off":     "Risk-Off 🔴",
                "inflationary": "Inflationary 🟠",
                "recessionary": "Recessionary ⚫",
                "neutral":      "Neutral 🔵",
            }
            regime = regime_map.get(regime_key, "Neutral 🔵")

        results.append({
            "date":          date,
            "regime":        regime,
            "confidence":    round(confidence, 1),
            "risk_on":       round(scores["risk_on"], 2),
            "risk_off":      round(scores["risk_off"], 2),
            "inflationary":  round(scores["inflationary"], 2),
            "recessionary":  round(scores["recessionary"], 2),
            "neutral":       round(scores["neutral"], 2),
        })

    return pd.DataFrame(results).set_index("date")


# =========================
# Color Mapping
# =========================

REGIME_COLORS = {
    "Risk-On 🟢":      "#2ecc71",
    "Risk-Off 🔴":     "#e74c3c",
    "Inflationary 🟠": "#e67e22",
    "Recessionary ⚫": "#7f8c8d",
    "Neutral 🔵":      "#3498db",
}

REGIME_DESCRIPTIONS = {
    "Risk-On 🟢": {
        "description": "Markets are in a risk-on environment. Equities are trending higher, volatility is low, credit spreads are tight, and investors are rotating into growth and cyclical assets.",
        "favored":     "Equities (SPY, QQQ), High Yield Bonds (HYG), EM Assets, Cyclicals (XLI, XLF)",
        "avoid":       "Treasuries (TLT), Gold, Defensive sectors (XLU, XLP)",
        "fed_watch":   "Fed likely on hold or in a cutting cycle. Watch for strong NFP and CPI above target.",
    },
    "Risk-Off 🔴": {
        "description": "Markets are in a risk-off environment. Volatility is elevated, equities are selling off, and investors are rotating into safe-haven assets like Treasuries and Gold.",
        "favored":     "Treasuries (TLT, IEF), Gold (GLD), US Dollar (UUP), Defensive sectors (XLU, XLP, XLV)",
        "avoid":       "Equities, High Yield Bonds, EM Assets, Cyclicals",
        "fed_watch":   "Watch for potential Fed pivot or emergency cuts. VIX above 25 is the key signal.",
    },
    "Inflationary 🟠": {
        "description": "Markets are pricing in elevated inflation. Oil and commodities are rising, the yield curve is steep, and the Fed is likely hawkish. Real assets outperform.",
        "favored":     "Commodities (GLD, OIL, XLE), TIPS, Short-duration bonds, Value stocks",
        "avoid":       "Long-duration Treasuries (TLT), Growth stocks, Unprofitable tech",
        "fed_watch":   "Fed likely hiking or holding at elevated rates. Watch CPI, PCE, and oil prices closely.",
    },
    "Recessionary ⚫": {
        "description": "Markets are pricing in economic contraction. The yield curve is inverted, equities are falling, credit spreads are widening, and defensive assets are outperforming.",
        "favored":     "Long-duration Treasuries (TLT), Gold (GLD), Cash, Defensive sectors (XLU, XLP)",
        "avoid":       "Cyclicals, High Yield Bonds, Small Caps, Commodities",
        "fed_watch":   "Fed likely cutting aggressively. Watch ISM PMI, NFP, and unemployment for confirmation.",
    },
    "Neutral 🔵": {
        "description": "Mixed signals across asset classes. No dominant regime is clearly in control. Markets are in a transition or consolidation phase.",
        "favored":     "Balanced portfolio. Equal weight across asset classes.",
        "avoid":       "Concentrated positions in any single regime trade.",
        "fed_watch":   "Watch for the next catalyst — CPI, NFP, Fed speech — to determine regime direction.",
    },
}


# =========================
# Chart Functions
# =========================

def _plot_regime_history(regime_df: pd.DataFrame, spy_df: pd.Series):
    """Plot regime history with S&P 500 overlay."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 2]})
    fig.patch.set_facecolor("#0e1117")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # Top panel: regime bands
    regime_map_color = {
        "Risk-On 🟢":      "#2ecc71",
        "Risk-Off 🔴":     "#e74c3c",
        "Inflationary 🟠": "#e67e22",
        "Recessionary ⚫": "#7f8c8d",
        "Neutral 🔵":      "#3498db",
    }

    dates = regime_df.index
    regimes = regime_df["regime"]

    for i in range(len(dates) - 1):
        color = regime_map_color.get(regimes.iloc[i], "#3498db")
        ax1.axvspan(dates[i], dates[i+1], alpha=0.8, color=color, linewidth=0)

    ax1.set_ylabel("Regime", color="white", fontsize=9)
    ax1.set_yticks([])
    ax1.set_title("Market Regime History (Past 12 Months)", color="white", fontsize=12, pad=8)

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=r.replace(" 🟢","").replace(" 🔴","").replace(" 🟠","").replace(" ⚫","").replace(" 🔵",""))
        for r, c in regime_map_color.items()
    ]
    ax1.legend(handles=legend_patches, loc="upper left", fontsize=8,
               facecolor="#1e1e1e", labelcolor="white", framealpha=0.8)

    # Bottom panel: S&P 500
    aligned_spy = spy_df.reindex(dates, method="ffill").dropna()
    if not aligned_spy.empty:
        ax2.plot(aligned_spy.index, aligned_spy.values, color="#1f77b4",
                 linewidth=1.5, label="S&P 500 (IVV)")
        ax2.fill_between(aligned_spy.index, aligned_spy.values,
                         alpha=0.1, color="#1f77b4")
        ax2.set_ylabel("S&P 500 (IVV)", color="white", fontsize=9)
        ax2.legend(fontsize=8, facecolor="#1e1e1e", labelcolor="white")

    plt.tight_layout()
    return fig


def _plot_score_breakdown(latest_scores: dict):
    """Bar chart of current regime scores."""
    labels = ["Risk-On", "Risk-Off", "Inflationary", "Recessionary", "Neutral"]
    values = [
        latest_scores.get("risk_on", 0),
        latest_scores.get("risk_off", 0),
        latest_scores.get("inflationary", 0),
        latest_scores.get("recessionary", 0),
        latest_scores.get("neutral", 0),
    ]
    colors = ["#2ecc71", "#e74c3c", "#e67e22", "#7f8c8d", "#3498db"]

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    bars = ax.barh(labels, values, color=colors, alpha=0.85, height=0.6)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}", va="center", ha="left",
                    color="white", fontsize=9)

    ax.set_xlabel("Signal Score", color="white", fontsize=9)
    ax.set_title("Current Signal Scores by Regime", color="white", fontsize=11)
    ax.yaxis.label.set_color("white")
    plt.tight_layout()
    return fig


# =========================
# Main Page
# =========================

def run_regime_classifier():
    st.subheader("🧠 AI Market Regime Classifier")
    st.markdown(
        "Classifies the current market environment using **cross-asset signals** — "
        "equity momentum, volatility, credit spreads, yield curve, oil, gold, and the dollar. "
        "Updated every 5 minutes with live market data."
    )

    # Methodology expander
    with st.expander("📖 How It Works", expanded=False):
        st.markdown("""
**Methodology: Rule-Based Cross-Asset Scoring**

The classifier scores five regime states in real time using nine cross-asset signals:

| Signal | Indicator | Logic |
|--------|-----------|-------|
| Equity Momentum | SPY 20d/60d return | Rising = Risk-On; Falling = Risk-Off/Recessionary |
| Volatility | VIX level | VIX < 15 = Risk-On; VIX > 30 = Risk-Off |
| US Dollar | DXY 20d return | Dollar strengthening = Risk-Off/Inflationary |
| Oil | WTI 20d return | Oil rising = Inflationary; Falling = Recessionary |
| Gold | GLD 20d return | Gold rising = Risk-Off/Inflationary |
| Yield Curve | 10Y minus 2Y spread | Inverted = Recessionary; Steep = Risk-On |
| Rate Level | 10Y yield | Above 4.5% = Inflationary pressure |
| Credit Spreads | HYG/LQD ratio | Ratio falling = Risk-Off/Recessionary |
| Duration | TLT 20d return | TLT rallying = Risk-Off/Recessionary |

Each signal contributes weighted scores to each regime category. The regime with the highest total score wins. Confidence reflects how dominant the winning regime is relative to the others.

**Why rule-based?** Rule-based models are more explainable in interviews, more robust without labeled training data, and more interpretable for cross-asset analysis.
""")

    # Fetch data
    with st.spinner("Fetching cross-asset data and classifying regime..."):
        df = _fetch_regime_data(period="1y")

    if df is None or df.empty:
        st.error("Unable to fetch market data. Please try again in a few minutes.")
        return

    # Compute scores
    regime_df = _compute_scores(df)
    if regime_df is None or regime_df.empty:
        st.error("Insufficient data to classify regime.")
        return

    # Current regime
    latest = regime_df.iloc[-1]
    current_regime = latest["regime"]
    confidence = latest["confidence"]
    regime_color = REGIME_COLORS.get(current_regime, "#3498db")
    regime_info = REGIME_DESCRIPTIONS.get(current_regime, REGIME_DESCRIPTIONS["Neutral 🔵"])

    # ---- Current Regime Display ----
    st.markdown("---")
    st.markdown("### 📡 Current Market Regime")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(
            f"""
            <div style="
                background-color: {regime_color}22;
                border-left: 5px solid {regime_color};
                border-radius: 8px;
                padding: 20px 24px;
                margin-bottom: 8px;
            ">
                <div style="font-size: 2.2rem; font-weight: 700; color: {regime_color};">
                    {current_regime}
                </div>
                <div style="font-size: 0.95rem; color: #ccc; margin-top: 6px;">
                    {regime_info['description']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.metric("Confidence", f"{confidence:.0f}%")
        st.metric("As of", regime_df.index[-1].strftime("%b %d, %Y"))

    with col3:
        # Regime distribution last 30 days
        last_30 = regime_df.tail(30)["regime"].value_counts()
        most_common = last_30.index[0] if not last_30.empty else "N/A"
        days_in_current = int((regime_df["regime"] == current_regime).tail(30).sum())
        st.metric("30D Dominant Regime", most_common.split(" ")[0] if most_common != "N/A" else "N/A")
        st.metric("Days in Current Regime (30D)", f"{days_in_current}")

    # ---- Asset Allocation Implications ----
    st.markdown("### 💼 Asset Allocation Implications")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"**✅ Favored Assets**")
        st.info(regime_info["favored"])
    with col_b:
        st.markdown(f"**❌ Assets to Avoid**")
        st.error(regime_info["avoid"])
    with col_c:
        st.markdown(f"**🏦 Fed Watch**")
        st.warning(regime_info["fed_watch"])

    # ---- Score Breakdown ----
    st.markdown("### 📊 Signal Score Breakdown")
    st.caption("Current scores across all five regime categories — the highest score determines the active regime.")
    latest_scores = {
        "risk_on":      latest["risk_on"],
        "risk_off":     latest["risk_off"],
        "inflationary": latest["inflationary"],
        "recessionary": latest["recessionary"],
        "neutral":      latest["neutral"],
    }
    fig_scores = _plot_score_breakdown(latest_scores)
    st.pyplot(fig_scores)

    # ---- Historical Regime Chart ----
    st.markdown("### 📈 Regime History (Past 12 Months)")
    st.caption("Top panel shows the classified regime for each trading day. Bottom panel shows the S&P 500 (IVV) for context.")

    spy_series = df["IVV"] if "IVV" in df.columns else (df["SPY"] if "SPY" in df.columns else None)
    if spy_series is not None:
        fig_history = _plot_regime_history(regime_df, spy_series)
        st.pyplot(fig_history)

    # ---- Regime Distribution ----
    st.markdown("### 🥧 Regime Distribution (Past 12 Months)")
    regime_counts = regime_df["regime"].value_counts()
    total_days = len(regime_df)

    cols = st.columns(len(regime_counts))
    for col, (regime, count) in zip(cols, regime_counts.items()):
        pct = count / total_days * 100
        color = REGIME_COLORS.get(regime, "#3498db")
        col.markdown(
            f"""
            <div style="
                background-color: {color}22;
                border: 1px solid {color}55;
                border-radius: 8px;
                padding: 12px;
                text-align: center;
            ">
                <div style="font-size: 1.1rem; font-weight: 600; color: {color};">
                    {regime.split()[0]}
                </div>
                <div style="font-size: 1.6rem; font-weight: 700; color: white;">
                    {pct:.0f}%
                </div>
                <div style="font-size: 0.8rem; color: #999;">
                    {count} days
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Recent Regime Log ----
    st.markdown("### 📋 Recent Regime Log (Last 20 Trading Days)")
    recent = regime_df.tail(20)[["regime", "confidence"]].copy()
    recent.index = recent.index.strftime("%Y-%m-%d")
    recent.columns = ["Regime", "Confidence (%)"]
    recent = recent.iloc[::-1]
    st.dataframe(recent, use_container_width=True)

    st.caption(
        f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | "
        "Data via Yahoo Finance | Refreshes every 5 minutes"
    )