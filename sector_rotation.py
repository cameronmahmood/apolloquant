# sector_rotation.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timezone

SECTORS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLY": "Consumer Disc",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Comm Services",
}

PERIODS = {
    "1 Week": 5,
    "1 Month": 22,
    "3 Months": 66,
    "6 Months": 126,
    "1 Year": 252,
}

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_sector_data():
    tickers = list(SECTORS.keys()) + ["SPY"]
    frames = {}
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="1y", auto_adjust=True)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            if not hist.empty and "Close" in hist.columns:
                s = hist["Close"].dropna()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                frames[ticker] = s
        except Exception:
            pass
    if not frames:
        return None
    return pd.DataFrame(frames).sort_index()

def _dark_ax(ax):
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

def run_sector_rotation():
    st.subheader("🌡️ Sector Rotation Heatmap")
    st.markdown(
        "Tracks **money flow across S&P 500 sectors** using performance heatmaps, "
        "relative strength vs SPY, and rotation cycle analysis. "
        "Identify which sectors are leading and lagging the market in real time."
    )

    with st.expander("📖 Sector Rotation Theory", expanded=False):
        st.markdown("""
**The Sector Rotation Cycle** follows the economic cycle:

| Economic Phase | Leading Sectors | Lagging Sectors |
|---------------|----------------|----------------|
| **Early Recovery** | Financials, Consumer Disc, Tech | Utilities, Staples |
| **Mid Cycle** | Tech, Industrials, Materials | Healthcare, Staples |
| **Late Cycle** | Energy, Materials, Industrials | Tech, Consumer Disc |
| **Recession** | Utilities, Healthcare, Staples | Energy, Financials, Tech |

**How to use with your Market Regime:**
- **Risk-On regime** → favor Tech (XLK), Financials (XLF), Consumer Disc (XLY)
- **Risk-Off regime** → favor Utilities (XLU), Healthcare (XLV), Staples (XLP)
- **Inflationary regime** → favor Energy (XLE), Materials (XLB)
- **Recessionary regime** → favor Utilities (XLU), Healthcare (XLV), short Energy/Financials
""")

    with st.spinner("Fetching sector data..."):
        df = _fetch_sector_data()

    if df is None:
        st.error("Unable to fetch sector data.")
        return

    # ---- Compute Returns ----
    rows = []
    spy = df["SPY"] if "SPY" in df.columns else None

    for ticker, name in SECTORS.items():
        if ticker not in df.columns:
            continue
        series = df[ticker].dropna()
        if len(series) < 10:
            continue

        rets = {}
        for label, days in PERIODS.items():
            if len(series) > days:
                rets[label] = round((series.iloc[-1] / series.iloc[-days] - 1) * 100, 2)
            else:
                rets[label] = None

        # vs SPY
        vs_spy = {}
        if spy is not None:
            for label, days in PERIODS.items():
                if len(series) > days and len(spy) > days:
                    sec_ret = (series.iloc[-1] / series.iloc[-days] - 1) * 100
                    spy_ret = (spy.iloc[-1] / spy.iloc[-days] - 1) * 100
                    vs_spy[f"vs SPY {label}"] = round(sec_ret - spy_ret, 2)
                else:
                    vs_spy[f"vs SPY {label}"] = None

        rows.append({"Ticker": ticker, "Sector": name, **rets, **vs_spy})

    results = pd.DataFrame(rows)

    # ---- Summary Metrics ----
    st.markdown("### 📊 Sector Performance Summary")
    col1, col2 = st.columns(2)

    with col1:
        period_show = st.selectbox("Performance Period", list(PERIODS.keys()), index=1, key="sr_period")

    with col2:
        view = st.radio("View", ["Absolute Return", "vs SPY"], horizontal=True, key="sr_view")

    col_to_show = period_show if view == "Absolute Return" else f"vs SPY {period_show}"
    sorted_results = results.dropna(subset=[col_to_show]).sort_values(col_to_show, ascending=False)

    # ---- Heatmap ----
    st.markdown("### 🌡️ Performance Heatmap")
    heatmap_cols = list(PERIODS.keys())
    heatmap_data = results.set_index("Sector")[heatmap_cols].dropna()

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0e1117"); _dark_ax(ax)

    data = heatmap_data.values.astype(float)
    data_clean = data[~np.isnan(data)]
    vmax = max(abs(data_clean.max()), abs(data_clean.min()), 5) if len(data_clean) > 0 else 10
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(heatmap_cols)))
    ax.set_xticklabels(heatmap_cols, color="white", fontsize=9)
    ax.set_yticks(range(len(heatmap_data)))
    ax.set_yticklabels(heatmap_data.index.tolist(), color="white", fontsize=9)

    for i in range(len(heatmap_data)):
        for j in range(len(heatmap_cols)):
            val = data[i, j]
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                    color="white" if abs(val) > vmax * 0.5 else "black", fontsize=8, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Return %", shrink=0.8)
    ax.set_title("S&P 500 Sector Performance Heatmap", color="white", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    # ---- Bar Chart ----
    st.markdown(f"### 📈 {period_show} — {view}")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    fig2.patch.set_facecolor("#0e1117"); _dark_ax(ax2)

    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sorted_results[col_to_show]]
    bars = ax2.barh(sorted_results["Sector"], sorted_results[col_to_show],
                    color=colors, alpha=0.85, height=0.6)
    ax2.axvline(0, color="white", linewidth=0.5, alpha=0.3)
    for bar, val in zip(bars, sorted_results[col_to_show]):
        ax2.text(val + (0.1 if val >= 0 else -0.1), bar.get_y() + bar.get_height()/2,
                 f"{val:+.1f}%", va="center", ha="left" if val >= 0 else "right",
                 color="white", fontsize=8)
    ax2.set_title(f"Sector {view} — {period_show}", color="white", fontsize=11)
    ax2.set_xlabel("Return %", color="white", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig2)

    # ---- Normalized Performance Chart ----
    st.markdown("### 📊 Normalized Sector Performance (1 Year)")
    fig3, ax3 = plt.subplots(figsize=(13, 5))
    fig3.patch.set_facecolor("#0e1117"); _dark_ax(ax3)

    colors_line = plt.cm.tab20(np.linspace(0, 1, len(SECTORS)))
    for (ticker, name), color in zip(SECTORS.items(), colors_line):
        if ticker in df.columns:
            series = df[ticker].dropna()
            normalized = series / series.iloc[0] * 100
            ax3.plot(normalized.index, normalized.values, linewidth=1.3, label=name, color=color)

    if "SPY" in df.columns:
        spy_norm = df["SPY"].dropna() / df["SPY"].dropna().iloc[0] * 100
        ax3.plot(spy_norm.index, spy_norm.values, linewidth=2, label="SPY", color="white",
                 linestyle="--", alpha=0.7)

    ax3.axhline(100, color="white", linewidth=0.5, alpha=0.2, linestyle=":")
    ax3.set_title("Normalized Sector Performance vs SPY (Base=100)", color="white", fontsize=11)
    ax3.set_ylabel("Normalized Return", color="white", fontsize=9)
    ax3.legend(fontsize=7, facecolor="#1e1e1e", labelcolor="white", ncol=4, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig3)

    # ---- Full Table ----
    st.markdown("### 📋 Full Sector Data Table")
    display_cols = ["Sector", "Ticker"] + list(PERIODS.keys()) + [f"vs SPY {p}" for p in list(PERIODS.keys())[:3]]
    display_cols = [c for c in display_cols if c in results.columns]
    st.dataframe(sorted_results[display_cols], use_container_width=True)

    # ---- Rotation Signal ----
    st.markdown("### 🔄 Current Rotation Signal")
    if not sorted_results.empty:
        top3 = sorted_results.head(3)["Sector"].tolist()
        bot3 = sorted_results.tail(3)["Sector"].tolist()

        defensive = ["Utilities", "Healthcare", "Consumer Staples"]
        growth    = ["Technology", "Consumer Disc", "Financials"]
        cyclical  = ["Energy", "Materials", "Industrials"]

        top_def  = sum(1 for s in top3 if s in defensive)
        top_grow = sum(1 for s in top3 if s in growth)
        top_cyc  = sum(1 for s in top3 if s in cyclical)

        if top_def >= 2:
            signal = "🔴 Defensive Rotation — Risk-Off"
            signal_detail = "Money flowing into defensive sectors. Consistent with Risk-Off or Recessionary regime."
        elif top_grow >= 2:
            signal = "🟢 Growth Rotation — Risk-On"
            signal_detail = "Money flowing into growth sectors. Consistent with Risk-On regime."
        elif top_cyc >= 2:
            signal = "🟠 Cyclical Rotation — Inflationary"
            signal_detail = "Money flowing into cyclical sectors. Consistent with Inflationary regime."
        else:
            signal = "⚪ Mixed Rotation — No Clear Signal"
            signal_detail = "No dominant sector theme. Market in transition or consolidation."

        st.info(f"""
**{signal}**

{signal_detail}

**Leading:** {', '.join(top3)}
**Lagging:** {', '.join(bot3)}
""")

    st.caption(
        f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | "
        "Data via Yahoo Finance | Refreshes every 5 minutes"
    )
