# correlation_matrix.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone

DEFAULT_TICKERS = ["SPY", "QQQ", "TLT", "GLD", "USO", "HYG", "UUP", "XLK", "XLF", "XLE", "XLV", "XLP"]

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_returns(tickers, period="1y"):
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
                frames[ticker] = s.pct_change().dropna()
        except Exception:
            pass
    if not frames:
        return None
    return pd.DataFrame(frames).dropna()

def _dark_ax(ax):
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

def run_correlation_matrix():
    st.subheader("🔗 Correlation Matrix")
    st.markdown(
        "Visualizes **how assets move together**. Use before building a portfolio to avoid "
        "accidentally concentrating risk in correlated positions. "
        "A well-diversified portfolio should have low average correlation between holdings."
    )

    with st.expander("📖 How to Read the Correlation Matrix", expanded=False):
        st.markdown("""
**Correlation ranges from -1 to +1:**

| Range | Meaning | Portfolio Impact |
|-------|---------|-----------------|
| **+0.7 to +1.0** | Highly correlated | Same bet — no diversification benefit |
| **+0.3 to +0.7** | Moderately correlated | Some diversification |
| **-0.3 to +0.3** | Low correlation | Good diversification |
| **-0.7 to -0.3** | Negatively correlated | Hedge — one zigs when other zags |
| **-1.0 to -0.7** | Highly negative | Perfect hedge |

**Key relationships to know:**
- SPY / QQQ: ~0.95 (almost identical) → don't hold both
- SPY / TLT: ~-0.3 (stocks/bonds hedge) → classic 60/40
- GLD / SPY: ~0.1 (uncorrelated) → gold is a true diversifier
- USO / XLE: ~0.85 (oil price drives energy stocks)
- HYG / SPY: ~0.7 (high yield bonds act like stocks in stress)

**Rule of thumb:** Avoid adding positions with correlation > 0.7 to existing holdings.
""")

    # ---- Settings ----
    st.markdown("### ⚙️ Settings")
    col1, col2 = st.columns(2)
    with col1:
        tickers_input = st.text_input("Tickers (comma-separated)",
                                       value=", ".join(DEFAULT_TICKERS), key="corr_tickers")
    with col2:
        period = st.selectbox("Lookback Period", ["6mo", "1y", "2y"], index=1, key="corr_period")
        corr_method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"], key="corr_method")

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        st.warning("Please enter at least one ticker.")
        return

    with st.spinner("Fetching data and computing correlations..."):
        returns = _fetch_returns(tickers, period=period)

    if returns is None or returns.empty:
        st.error("Could not fetch return data.")
        return

    available = [t for t in tickers if t in returns.columns]
    returns = returns[available]
    corr = returns.corr(method=corr_method)

    # ---- Summary Stats ----
    st.markdown("### 📊 Portfolio Correlation Summary")
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    avg_corr = upper.stack().mean()
    max_corr = upper.stack().max()
    min_corr = upper.stack().min()
    high_corr_pairs = [(i, j, upper.loc[i,j]) for i in upper.index
                       for j in upper.columns if pd.notna(upper.loc[i,j]) and upper.loc[i,j] > 0.7]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Pairwise Correlation", f"{avg_corr:.2f}",
              delta="Well diversified" if avg_corr < 0.4 else ("Moderate" if avg_corr < 0.6 else "Concentrated"))
    c2.metric("Max Correlation", f"{max_corr:.2f}")
    c3.metric("Min Correlation", f"{min_corr:.2f}")
    c4.metric("High Corr Pairs (>0.7)", len(high_corr_pairs))

    if high_corr_pairs:
        st.warning(f"⚠️ {len(high_corr_pairs)} highly correlated pairs found — consider removing one from each pair:")
        for i, j, v in sorted(high_corr_pairs, key=lambda x: -x[2])[:5]:
            st.markdown(f"- **{i} / {j}**: {v:.2f}")

    # ---- Correlation Heatmap ----
    st.markdown("### 🌡️ Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(max(8, len(available)*0.8), max(6, len(available)*0.7)))
    fig.patch.set_facecolor("#0e1117"); _dark_ax(ax)

    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(available, rotation=45, ha="right", color="white", fontsize=9)
    ax.set_yticks(range(len(available)))
    ax.set_yticklabels(available, color="white", fontsize=9)

    for i in range(len(available)):
        for j in range(len(available)):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if abs(val) > 0.5 else "black", fontsize=7, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Correlation", shrink=0.8)
    ax.set_title(f"Asset Correlation Matrix ({corr_method.capitalize()}, {period})", color="white", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)

    # ---- Rolling Correlation ----
    st.markdown("### 📈 Rolling Correlation (60-Day)")
    if len(available) >= 2:
        col_a, col_b = st.columns(2)
        with col_a:
            asset1 = st.selectbox("Asset 1", available, index=0, key="corr_a1")
        with col_b:
            asset2 = st.selectbox("Asset 2", available,
                                   index=min(1, len(available)-1), key="corr_a2")

        if asset1 != asset2 and asset1 in returns.columns and asset2 in returns.columns:
            rolling_corr = returns[asset1].rolling(60).corr(returns[asset2])
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            fig2.patch.set_facecolor("#0e1117"); _dark_ax(ax2)
            ax2.plot(rolling_corr.index, rolling_corr.values, color="#1f77b4", linewidth=1.2)
            ax2.axhline(0, color="white", linewidth=0.5, alpha=0.3, linestyle="--")
            ax2.axhline(0.7, color="#e74c3c", linewidth=0.8, linestyle=":", alpha=0.6, label="0.7 threshold")
            ax2.axhline(-0.7, color="#2ecc71", linewidth=0.8, linestyle=":", alpha=0.6, label="-0.7 threshold")
            ax2.fill_between(rolling_corr.index, rolling_corr.values, 0,
                              where=(rolling_corr >= 0), alpha=0.15, color="#2ecc71")
            ax2.fill_between(rolling_corr.index, rolling_corr.values, 0,
                              where=(rolling_corr < 0), alpha=0.15, color="#e74c3c")
            ax2.set_ylim(-1, 1)
            ax2.set_title(f"60-Day Rolling Correlation: {asset1} vs {asset2}", color="white", fontsize=11)
            ax2.set_ylabel("Correlation", color="white", fontsize=9)
            ax2.legend(fontsize=8, facecolor="#1e1e1e", labelcolor="white")
            plt.tight_layout()
            st.pyplot(fig2)

    # ---- Full Correlation Table ----
    st.markdown("### 📋 Full Correlation Table")
    st.dataframe(corr.round(3), use_container_width=True)

    st.caption(
        f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | "
        "Data via Yahoo Finance"
    )
