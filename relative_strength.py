# relative_strength.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timezone

# =========================
# Default Universe
# =========================

DEFAULT_TICKERS = {
    "📊 Broad Market": ["SPY", "QQQ", "IWM", "DIA", "VTI"],
    "🏦 Financials":   ["XLF", "JPM", "BAC", "GS", "MS", "V", "MA"],
    "💻 Technology":   ["XLK", "AAPL", "MSFT", "NVDA", "AMD", "META", "GOOGL"],
    "⚡ Energy":        ["XLE", "USO", "XOM", "CVX", "COP", "SLB"],
    "🏥 Healthcare":   ["XLV", "JNJ", "PFE", "UNH", "ABBV", "MRK"],
    "🛒 Consumer":     ["XLP", "XLY", "WMT", "AMZN", "KO", "PEP", "MCD"],
    "🏭 Industrials":  ["XLI", "BA", "CAT", "UPS", "HON", "GE"],
    "🔧 Materials":    ["XLB", "GLD", "SLV", "CORN", "USO"],
    "🏢 Real Estate":  ["XLRE", "AMT", "SPG", "O"],
    "⚡ Utilities":    ["XLU", "NEE", "DUK", "SO"],
    "📡 Comm Svcs":    ["XLC", "NFLX", "DIS", "GOOGL", "META"],
    "🌍 International":["EEM", "EFA", "FXI", "EWJ", "EWZ"],
    "💰 Fixed Income": ["TLT", "IEF", "HYG", "LQD", "BND", "SHY"],
    "🥇 Commodities":  ["GLD", "SLV", "USO", "UNG", "CORN", "WEAT", "PALL"],
}

ALL_TICKERS = list(dict.fromkeys([t for tickers in DEFAULT_TICKERS.values() for t in tickers]))

# =========================
# Data Fetching
# =========================

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_prices(tickers: list, period: str = "6mo"):
    frames = {}
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period=period, auto_adjust=True)
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
    df = pd.DataFrame(frames).sort_index()
    return df

# =========================
# Relative Strength Calculations
# =========================

def _compute_rs_metrics(df: pd.DataFrame, benchmark: str = "SPY"):
    results = []
    bench = df[benchmark] if benchmark in df.columns else None

    for ticker in df.columns:
        series = df[ticker].dropna()
        if len(series) < 20:
            continue

        price = series.iloc[-1]
        ret_1d  = (series.iloc[-1] / series.iloc[-2]  - 1) * 100 if len(series) > 1  else np.nan
        ret_1w  = (series.iloc[-1] / series.iloc[-5]  - 1) * 100 if len(series) > 5  else np.nan
        ret_1m  = (series.iloc[-1] / series.iloc[-22] - 1) * 100 if len(series) > 22 else np.nan
        ret_3m  = (series.iloc[-1] / series.iloc[-66] - 1) * 100 if len(series) > 66 else np.nan
        ret_6m  = (series.iloc[-1] / series.iloc[0]   - 1) * 100

        # 52-week high/low
        ret_52w_high = (series.iloc[-1] / series.max() - 1) * 100
        ret_52w_low  = (series.iloc[-1] / series.min() - 1) * 100

        # Relative strength vs benchmark
        rs_1m = np.nan
        rs_3m = np.nan
        if bench is not None and ticker != benchmark:
            bench_clean = bench.reindex(series.index).dropna()
            if len(bench_clean) > 22:
                bench_1m = (bench_clean.iloc[-1] / bench_clean.iloc[-22] - 1) * 100
                rs_1m = ret_1m - bench_1m if pd.notna(ret_1m) else np.nan
            if len(bench_clean) > 66:
                bench_3m = (bench_clean.iloc[-1] / bench_clean.iloc[-66] - 1) * 100
                rs_3m = ret_3m - bench_3m if pd.notna(ret_3m) else np.nan

        # Momentum score (weighted average of returns)
        mom_score = 0
        count = 0
        for ret, weight in [(ret_1w, 1), (ret_1m, 2), (ret_3m, 3), (ret_6m, 4)]:
            if pd.notna(ret):
                mom_score += ret * weight
                count += weight
        mom_score = mom_score / count if count > 0 else np.nan

        # Volatility
        daily_ret = series.pct_change().dropna()
        vol_20d = daily_ret.tail(20).std() * np.sqrt(252) * 100 if len(daily_ret) >= 20 else np.nan

        # Trend score
        sma20  = series.rolling(20).mean().iloc[-1]
        sma50  = series.rolling(50).mean().iloc[-1] if len(series) >= 50 else np.nan
        sma200 = series.rolling(200).mean().iloc[-1] if len(series) >= 200 else np.nan

        trend = 0
        if pd.notna(sma20)  and price > sma20:  trend += 1
        if pd.notna(sma50)  and price > sma50:  trend += 1
        if pd.notna(sma200) and price > sma200: trend += 1

        results.append({
            "Ticker":        ticker,
            "Price":         round(price, 2),
            "1D %":          round(ret_1d, 2)  if pd.notna(ret_1d)  else None,
            "1W %":          round(ret_1w, 2)  if pd.notna(ret_1w)  else None,
            "1M %":          round(ret_1m, 2)  if pd.notna(ret_1m)  else None,
            "3M %":          round(ret_3m, 2)  if pd.notna(ret_3m)  else None,
            "6M %":          round(ret_6m, 2),
            "vs SPY (1M)":   round(rs_1m, 2)   if pd.notna(rs_1m)   else None,
            "vs SPY (3M)":   round(rs_3m, 2)   if pd.notna(rs_3m)   else None,
            "52W High %":    round(ret_52w_high, 2),
            "Trend (0-3)":   trend,
            "Mom Score":     round(mom_score, 2) if pd.notna(mom_score) else None,
            "Vol (Ann)":     round(vol_20d, 1)   if pd.notna(vol_20d)   else None,
        })

    return pd.DataFrame(results)

def _rank_signal(row):
    score = row.get("Mom Score", 0) or 0
    trend = row.get("Trend (0-3)", 0) or 0
    rs    = row.get("vs SPY (1M)", 0) or 0
    total = score * 0.5 + trend * 2 + rs * 0.3
    if total >= 8:   return "🟢 Strong Leader"
    elif total >= 3: return "🟡 Leader"
    elif total >= -3: return "⚪ Neutral"
    elif total >= -8: return "🟠 Laggard"
    else:            return "🔴 Strong Laggard"

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

def _plot_momentum_bar(results: pd.DataFrame, col: str, title: str, n=20):
    df = results.dropna(subset=[col]).sort_values(col, ascending=True).tail(n)
    if df.empty:
        return None
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df[col]]
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)))
    fig.patch.set_facecolor("#0e1117"); _dark_ax(ax)
    bars = ax.barh(df["Ticker"], df[col], color=colors, alpha=0.85, height=0.6)
    ax.axvline(0, color="white", linewidth=0.5, alpha=0.3)
    for bar, val in zip(bars, df[col]):
        ax.text(val + (0.1 if val >= 0 else -0.1), bar.get_y() + bar.get_height()/2,
                f"{val:+.1f}%", va="center", ha="left" if val >= 0 else "right",
                color="white", fontsize=7)
    ax.set_title(title, color="white", fontsize=11)
    ax.set_xlabel("Return %", color="white", fontsize=9)
    plt.tight_layout(); return fig

def _plot_normalized_performance(df: pd.DataFrame, tickers: list, title: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0e1117"); _dark_ax(ax)
    colors = plt.cm.tab20(np.linspace(0, 1, len(tickers)))
    for ticker, color in zip(tickers, colors):
        if ticker in df.columns:
            series = df[ticker].dropna()
            normalized = series / series.iloc[0] * 100
            ax.plot(normalized.index, normalized.values, linewidth=1.5,
                    label=ticker, color=color)
    ax.axhline(100, color="white", linewidth=0.5, alpha=0.3, linestyle="--")
    ax.set_title(title, color="white", fontsize=11)
    ax.set_ylabel("Normalized Return (Base=100)", color="white", fontsize=9)
    ax.legend(fontsize=7, facecolor="#1e1e1e", labelcolor="white",
              loc="upper left", ncol=3)
    plt.tight_layout(); return fig

def _plot_heatmap(results: pd.DataFrame, cols: list):
    df = results.set_index("Ticker")[cols].dropna()
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(len(cols)*1.8, max(4, len(df)*0.4)))
    fig.patch.set_facecolor("#0e1117"); _dark_ax(ax)
    data = df.values
    vmax = max(abs(data.max()), abs(data.min()))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, color="white", fontsize=8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index.tolist(), color="white", fontsize=8)
    for i in range(len(df)):
        for j in range(len(cols)):
            val = data[i, j]
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                    color="white" if abs(val) > vmax*0.5 else "black", fontsize=7)
    plt.colorbar(im, ax=ax, label="Return %")
    ax.set_title("Relative Strength Heatmap", color="white", fontsize=11)
    plt.tight_layout(); return fig

# =========================
# Main Page
# =========================

def run_relative_strength():
    st.subheader("📡 Relative Strength Scanner")
    st.markdown(
        "Ranks assets by **momentum and relative strength** versus the S&P 500. "
        "Identifies market leaders and laggards across sectors, ETFs, and individual stocks. "
        "Use this to find **where money is flowing** before placing trades on Investopedia."
    )

    with st.expander("📖 How It Works", expanded=False):
        st.markdown("""
**What Relative Strength Tells You:**

Relative strength measures how an asset is performing **compared to the overall market (SPY)**. 
Assets with positive relative strength are outperforming — money is flowing in. 
Assets with negative relative strength are underperforming — money is flowing out.

**Momentum Score** is a weighted average of returns across timeframes:
- 1W return × 1 weight
- 1M return × 2 weight  
- 3M return × 3 weight
- 6M return × 4 weight

Longer timeframes get more weight because they reflect stronger, more persistent trends.

**Trend Score (0-3):** How many of the three SMAs (20, 50, 200) the price is above.
- 3/3 = Strong uptrend
- 0/3 = Strong downtrend

**Signal Labels:**
- 🟢 Strong Leader = Top momentum + outperforming SPY + above all SMAs
- 🟡 Leader = Positive momentum + outperforming SPY
- ⚪ Neutral = Mixed signals
- 🟠 Laggard = Underperforming SPY
- 🔴 Strong Laggard = Worst momentum + underperforming SPY

**How to use with your trading workflow:**
- In Risk-On regime → buy Strong Leaders in growth sectors (XLK, QQQ)
- In Risk-Off regime → buy Strong Leaders in defensive sectors (XLU, XLV, TLT)
- Never buy a Strong Laggard expecting a bounce — use Mean Reversion Scanner for that
- Combine with MACD & Technical Signals for entry timing
""")

    # ---- Settings ----
    st.markdown("### ⚙️ Universe Selection")
    col1, col2, col3 = st.columns(3)

    with col1:
        universe_choice = st.multiselect(
            "Select Groups",
            list(DEFAULT_TICKERS.keys()),
            default=["📊 Broad Market", "🏦 Financials", "💻 Technology",
                     "⚡ Energy", "🏥 Healthcare", "💰 Fixed Income"],
            key="rs_groups"
        )

    with col2:
        period = st.selectbox("Lookback Period", ["3mo", "6mo", "1y"], index=1, key="rs_period")
        benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM"], index=0, key="rs_bench")

    with col3:
        custom_input = st.text_input("Add Custom Tickers (comma-separated)", value="", key="rs_custom")
        sort_by = st.selectbox("Sort By", ["Mom Score", "1M %", "3M %", "vs SPY (1M)", "6M %"], key="rs_sort")

    # Build ticker list
    selected_tickers = list(dict.fromkeys(
        [t for group in universe_choice for t in DEFAULT_TICKERS.get(group, [])]
    ))
    if custom_input:
        custom_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
        selected_tickers = list(dict.fromkeys(selected_tickers + custom_tickers))
    if benchmark not in selected_tickers:
        selected_tickers.insert(0, benchmark)

    if not selected_tickers:
        st.warning("Please select at least one group.")
        return

    # ---- Fetch ----
    with st.spinner(f"Fetching {len(selected_tickers)} tickers and computing relative strength..."):
        df = _fetch_prices(selected_tickers, period=period)

    if df is None or df.empty:
        st.error("Unable to fetch data.")
        return

    # ---- Compute ----
    results = _compute_rs_metrics(df, benchmark=benchmark)
    results["Signal"] = results.apply(_rank_signal, axis=1)
    results_sorted = results.sort_values(sort_by, ascending=False).reset_index(drop=True)

    # ---- Summary ----
    st.markdown("### 📊 Market Summary")
    leaders    = len(results[results["Signal"].isin(["🟢 Strong Leader", "🟡 Leader"])])
    neutral    = len(results[results["Signal"] == "⚪ Neutral"])
    laggards   = len(results[results["Signal"].isin(["🔴 Strong Laggard", "🟠 Laggard"])])
    total      = len(results)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🟢 Leaders",        leaders)
    c2.metric("⚪ Neutral",         neutral)
    c3.metric("🔴 Laggards",       laggards)
    c4.metric("Total Tickers",     total)
    c5.metric("Market Breadth",    f"{leaders/total*100:.0f}% leading" if total > 0 else "N/A")

    # ---- Top Leaders & Laggards ----
    st.markdown("### 🏆 Top Leaders & Laggards")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 🟢 Top 5 Leaders (Strongest Momentum)")
        top5 = results_sorted.head(5)
        for _, row in top5.iterrows():
            st.markdown(
                f"**{row['Ticker']}** — {row['Signal']} | "
                f"1M: {row['1M %']:+.1f}% | 3M: {row['3M %']:+.1f}% | "
                f"vs SPY: {row['vs SPY (1M)']:+.1f}%" if pd.notna(row['vs SPY (1M)']) else
                f"**{row['Ticker']}** — {row['Signal']} | 1M: {row['1M %']:+.1f}%"
            )

    with col_r:
        st.markdown("#### 🔴 Bottom 5 Laggards (Weakest Momentum)")
        bot5 = results_sorted.tail(5).iloc[::-1]
        for _, row in bot5.iterrows():
            st.markdown(
                f"**{row['Ticker']}** — {row['Signal']} | "
                f"1M: {row['1M %']:+.1f}% | 3M: {row['3M %']:+.1f}% | "
                f"vs SPY: {row['vs SPY (1M)']:+.1f}%" if pd.notna(row['vs SPY (1M)']) else
                f"**{row['Ticker']}** — {row['Signal']} | 1M: {row['1M %']:+.1f}%"
            )

    # ---- Full Table ----
    st.markdown("### 📋 Full Relative Strength Table")
    st.caption(f"Sorted by {sort_by} — highest momentum at top.")
    display_cols = ["Ticker", "Price", "1D %", "1W %", "1M %", "3M %", "6M %",
                    "vs SPY (1M)", "vs SPY (3M)", "Trend (0-3)", "Mom Score", "Signal"]
    st.dataframe(results_sorted[display_cols], use_container_width=True)

    # ---- Charts ----
    st.markdown("### 📈 1-Month Return Ranking")
    fig1 = _plot_momentum_bar(results_sorted, "1M %", "1-Month Returns — All Tickers", n=30)
    if fig1: st.pyplot(fig1)

    st.markdown("### 📈 Relative Strength vs SPY (1M)")
    fig2 = _plot_momentum_bar(results_sorted, "vs SPY (1M)", "1-Month Return vs SPY Benchmark", n=30)
    if fig2: st.pyplot(fig2)

    # ---- Normalized Performance Chart ----
    st.markdown("### 📊 Normalized Performance (Select Tickers)")
    available = [t for t in results_sorted["Ticker"].tolist() if t in df.columns]
    chart_tickers = st.multiselect(
        "Select tickers to compare",
        available,
        default=available[:8] if len(available) >= 8 else available,
        key="rs_chart_tickers"
    )
    if chart_tickers:
        fig3 = _plot_normalized_performance(df, chart_tickers, "Normalized Price Performance (Base=100)")
        st.pyplot(fig3)

    # ---- Heatmap ----
    st.markdown("### 🌡️ Relative Strength Heatmap")
    heatmap_cols = ["1D %", "1W %", "1M %", "3M %", "6M %"]
    fig4 = _plot_heatmap(results_sorted.head(25), heatmap_cols)
    if fig4: st.pyplot(fig4)

    # ---- Rotation Insight ----
    st.markdown("### 🔄 Sector Rotation Insight")
    sector_etfs = {
        "XLK": "Tech", "XLF": "Financials", "XLE": "Energy",
        "XLV": "Healthcare", "XLY": "Cons Disc", "XLP": "Cons Staples",
        "XLI": "Industrials", "XLB": "Materials", "XLRE": "Real Estate",
        "XLU": "Utilities", "XLC": "Comm Svcs"
    }
    sector_data = results[results["Ticker"].isin(sector_etfs.keys())].copy()
    sector_data["Sector"] = sector_data["Ticker"].map(sector_etfs)

    if not sector_data.empty:
        sector_data = sector_data.sort_values("1M %", ascending=False)
        top_sector    = sector_data.iloc[0]
        bottom_sector = sector_data.iloc[-1]

        col_a, col_b = st.columns(2)
        with col_a:
            st.success(f"""
**🟢 Strongest Sector: {top_sector['Sector']} ({top_sector['Ticker']})**
- 1M Return: {top_sector['1M %']:+.1f}%
- 3M Return: {top_sector['3M %']:+.1f}%
- vs SPY: {top_sector['vs SPY (1M)']:+.1f}%
- Trend Score: {int(top_sector['Trend (0-3)'])}/3
""")
        with col_b:
            st.error(f"""
**🔴 Weakest Sector: {bottom_sector['Sector']} ({bottom_sector['Ticker']})**
- 1M Return: {bottom_sector['1M %']:+.1f}%
- 3M Return: {bottom_sector['3M %']:+.1f}%
- vs SPY: {bottom_sector['vs SPY (1M)']:+.1f}%
- Trend Score: {int(bottom_sector['Trend (0-3)'])}/3
""")

        st.markdown("**Sector Rankings (1-Month):**")
        for _, row in sector_data.iterrows():
            bar_color = "🟢" if row["1M %"] > 0 else "🔴"
            st.markdown(f"{bar_color} **{row['Sector']}** ({row['Ticker']}): {row['1M %']:+.1f}% | vs SPY: {row['vs SPY (1M)']:+.1f}%")

    st.caption(
        f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | "
        "Data via Yahoo Finance | Refreshes every 5 minutes"
    )