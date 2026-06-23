# factor_model.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime, timezone

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_returns(tickers, period="2y"):
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

def run_factor_model():
    st.subheader("🧮 Factor Model (Fama-French Inspired)")
    st.markdown(
        "Decomposes stock returns into **systematic risk factors**: market (beta), "
        "size, value, momentum, and quality. "
        "Understand WHY a stock moved — not just that it moved."
    )

    with st.expander("📖 Factor Model Theory", expanded=False):
        st.markdown("""
**The Fama-French Factor Model** explains stock returns using systematic factors:

| Factor | Description | High Exposure → |
|--------|-------------|----------------|
| **Market (Beta)** | Sensitivity to overall market moves | Amplified market returns |
| **Size (SMB)** | Small-cap vs large-cap premium | Small caps outperform long-term |
| **Value (HML)** | Value vs growth premium | Value stocks outperform long-term |
| **Momentum (MOM)** | Past winners vs losers | Winners keep winning (short-term) |
| **Quality** | Profitable vs unprofitable | High quality firms outperform |

**Alpha (α):**
The return NOT explained by any factor. Positive alpha = manager skill or unique edge.
A stock with 5% annual alpha is outperforming even after accounting for all risk factors.

**How to use this tool:**
- High beta (>1.5) + Risk-Off regime = risky position to hold
- Positive momentum factor loading = your stock benefits from trend-following
- Positive alpha = the stock has an edge beyond pure market exposure
- Use to understand if your Investopedia returns come from skill or just market beta

**Proxies used (since actual FF data requires special access):**
- Market: SPY (S&P 500)
- Size: IWM - SPY (small caps minus large caps)
- Value: IVE - IVW (value ETF minus growth ETF)
- Momentum: MTUM (momentum factor ETF)
- Quality: QUAL (quality factor ETF)
""")

    # ---- Input ----
    st.markdown("### ⚙️ Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Stock to Analyze", value="AAPL", key="fm_ticker").upper().strip()
    with col2:
        period = st.selectbox("Lookback Period", ["1y", "2y", "3y"], index=1, key="fm_period")
    with col3:
        rf_rate = st.number_input("Risk-Free Rate (Annual %)", value=4.5, min_value=0.0,
                                   max_value=10.0, step=0.1, key="fm_rf") / 100 / 252

    run = st.button("▶ Run Factor Analysis", key="fm_run")
    if not run:
        st.info("Enter a ticker and click Run Factor Analysis.")
        return

    factor_tickers = [ticker, "SPY", "IWM", "IVE", "IVW", "MTUM", "QUAL"]

    with st.spinner(f"Fetching data for {ticker} and factor proxies..."):
        returns = _fetch_returns(factor_tickers, period=period)

    if returns is None or ticker not in returns.columns:
        st.error(f"Could not fetch data for {ticker}.")
        return

    available_factors = [t for t in ["SPY", "IWM", "IVE", "IVW", "MTUM", "QUAL"] if t in returns.columns]
    if not available_factors:
        st.error("Could not fetch factor data.")
        return

    # ---- Build Factor Returns ----
    stock_ret = returns[ticker] - rf_rate
    mkt = returns["SPY"] - rf_rate if "SPY" in returns.columns else None

    factors = pd.DataFrame(index=returns.index)
    if mkt is not None:
        factors["Market (MKT)"] = mkt
    if "IWM" in returns.columns and "SPY" in returns.columns:
        factors["Size (SMB)"] = returns["IWM"] - returns["SPY"]
    if "IVE" in returns.columns and "IVW" in returns.columns:
        factors["Value (HML)"] = returns["IVE"] - returns["IVW"]
    if "MTUM" in returns.columns:
        factors["Momentum (MOM)"] = returns["MTUM"] - rf_rate
    if "QUAL" in returns.columns:
        factors["Quality (QMJ)"] = returns["QUAL"] - rf_rate

    common_idx = stock_ret.index.intersection(factors.index)
    y = stock_ret.loc[common_idx]
    X = factors.loc[common_idx]
    X = sm.add_constant(X)
    X = X.dropna()
    y = y.loc[X.index]

    if len(y) < 30:
        st.error("Not enough data for factor regression.")
        return

    model = sm.OLS(y, X).fit()

    # ---- Results ----
    st.markdown("### 📊 Factor Exposure Results")
    st.success(f"Analyzed **{ticker}** using {len(y)} trading days of data")

    alpha_daily = model.params.get("const", 0)
    alpha_annual = alpha_daily * 252 * 100
    r_squared = model.rsquared

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alpha (Annual)", f"{alpha_annual:.2f}%",
              delta="Positive edge ✅" if alpha_annual > 0 else "Negative edge ❌")
    c2.metric("Market Beta", f"{model.params.get('Market (MKT)', np.nan):.3f}" if "Market (MKT)" in model.params else "N/A")
    c3.metric("R² (% explained)", f"{r_squared*100:.1f}%")
    c4.metric("Observations", len(y))

    # ---- Factor Loadings ----
    st.markdown("### 📋 Factor Loadings")
    factor_names = [f for f in model.params.index if f != "const"]
    loading_data = []
    for fname in factor_names:
        coef = model.params[fname]
        pval = model.pvalues[fname]
        tstat = model.tvalues[fname]
        sig = "✅ Significant" if pval < 0.05 else ("⚠️ Marginal" if pval < 0.10 else "❌ Not significant")
        interp = ""
        if "Market" in fname:
            interp = f"{'High risk — amplifies market moves' if coef > 1.2 else ('Defensive — less than market' if coef < 0.8 else 'Market-like exposure')}"
        elif "Size" in fname:
            interp = f"{'Small-cap characteristics' if coef > 0.2 else ('Large-cap characteristics' if coef < -0.2 else 'Neutral size')}"
        elif "Value" in fname:
            interp = f"{'Value stock characteristics' if coef > 0.2 else ('Growth stock characteristics' if coef < -0.2 else 'Neutral style')}"
        elif "Momentum" in fname:
            interp = f"{'Benefits from momentum trends' if coef > 0.3 else ('Anti-momentum' if coef < -0.3 else 'Neutral momentum')}"
        elif "Quality" in fname:
            interp = f"{'High quality characteristics' if coef > 0.2 else ('Lower quality' if coef < -0.2 else 'Neutral quality')}"

        loading_data.append({
            "Factor": fname,
            "Loading (β)": round(coef, 4),
            "t-stat": round(tstat, 2),
            "p-value": round(pval, 4),
            "Significance": sig,
            "Interpretation": interp,
        })

    st.dataframe(pd.DataFrame(loading_data), use_container_width=True)

    # Alpha interpretation
    if alpha_annual > 2:
        st.success(f"✅ {ticker} generates **{alpha_annual:.2f}% annual alpha** — strong outperformance beyond factor exposure")
    elif alpha_annual > 0:
        st.info(f"ℹ️ {ticker} generates **{alpha_annual:.2f}% annual alpha** — modest positive edge")
    else:
        st.warning(f"⚠️ {ticker} has **{alpha_annual:.2f}% annual alpha** — underperforming after accounting for risk factors")

    # ---- Return Decomposition Chart ----
    st.markdown("### 📊 Return Attribution")
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0e1117"); _dark_ax(ax)

    factor_contribution = {}
    for fname in factor_names:
        if fname in factors.columns and fname in model.params:
            contrib = model.params[fname] * factors[fname].loc[common_idx].mean() * 252 * 100
            factor_contribution[fname.split("(")[0].strip()] = contrib

    factor_contribution["Alpha"] = alpha_annual

    labels = list(factor_contribution.keys())
    values = list(factor_contribution.values())
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]
    colors[-1] = "#f0b429"  # Alpha in gold

    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + (0.1 if val >= 0 else -0.3),
                f"{val:+.2f}%", ha="center", va="bottom" if val >= 0 else "top",
                color="white", fontsize=8)
    ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)
    ax.set_title(f"{ticker} — Annualized Return Attribution by Factor", color="white", fontsize=11)
    ax.set_ylabel("Annual Contribution (%)", color="white", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    # ---- Rolling Beta ----
    st.markdown("### 📈 Rolling 60-Day Market Beta")
    if "Market (MKT)" in factors.columns:
        window = 60
        rolling_beta = []
        idx = []
        for i in range(window, len(y)):
            y_w = y.iloc[i-window:i]
            x_w = factors["Market (MKT)"].loc[y_w.index]
            x_w_c = sm.add_constant(x_w)
            try:
                m = sm.OLS(y_w, x_w_c).fit()
                rolling_beta.append(m.params.get("Market (MKT)", np.nan))
                idx.append(y.index[i])
            except Exception:
                pass

        if rolling_beta:
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            fig2.patch.set_facecolor("#0e1117"); _dark_ax(ax2)
            ax2.plot(idx, rolling_beta, color="#1f77b4", linewidth=1.5)
            ax2.axhline(1.0, color="#f0b429", linewidth=1, linestyle="--", alpha=0.7, label="Beta = 1 (Market)")
            ax2.axhline(0.0, color="white", linewidth=0.5, alpha=0.3)
            ax2.fill_between(idx, rolling_beta, 1.0,
                              where=[b > 1 for b in rolling_beta], alpha=0.15, color="#e74c3c")
            ax2.fill_between(idx, rolling_beta, 1.0,
                              where=[b <= 1 for b in rolling_beta], alpha=0.15, color="#2ecc71")
            ax2.set_title(f"{ticker} — Rolling 60-Day Market Beta", color="white", fontsize=11)
            ax2.set_ylabel("Beta", color="white", fontsize=9)
            ax2.legend(fontsize=8, facecolor="#1e1e1e", labelcolor="white")
            plt.tight_layout()
            st.pyplot(fig2)

    st.caption(
        f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | "
        "Factor proxies: SPY (market), IWM-SPY (size), IVE-IVW (value), MTUM (momentum), QUAL (quality)"
    )
