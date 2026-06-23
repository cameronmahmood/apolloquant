# var_calculator.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from scipy import stats

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_returns(tickers: list, period: str = "2y"):
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

def _compute_var(returns: pd.Series, confidence: float = 0.95, method: str = "Historical"):
    if method == "Historical":
        var = -np.percentile(returns, (1 - confidence) * 100)
    elif method == "Parametric":
        mu = returns.mean()
        sigma = returns.std()
        var = -(mu + sigma * stats.norm.ppf(1 - confidence))
    elif method == "Monte Carlo":
        mu = returns.mean()
        sigma = returns.std()
        simulated = np.random.normal(mu, sigma, 100000)
        var = -np.percentile(simulated, (1 - confidence) * 100)
    return var

def _compute_cvar(returns: pd.Series, confidence: float = 0.95):
    var = _compute_var(returns, confidence, "Historical")
    cvar = -returns[returns < -var].mean()
    return cvar if not np.isnan(cvar) else var

def run_var_calculator():
    st.subheader("⚠️ Value at Risk (VaR) Calculator")
    st.markdown(
        "Calculate **Value at Risk** for a custom portfolio. VaR answers: "
        "*'What is the maximum loss I can expect over a given time period at a given confidence level?'* "
        "Every risk manager at a bank, hedge fund, and trading desk calculates VaR daily."
    )

    with st.expander("📖 How VaR Works", expanded=False):
        st.markdown("""
**Value at Risk (VaR)** is the maximum expected loss over a given time horizon at a given confidence level.

**Example:** 1-day 95% VaR of $1,000 means there is a 95% chance you will NOT lose more than $1,000 in one day.
Equivalently, there is a 5% chance you WILL lose more than $1,000.

**Three Methods:**
| Method | How it works | Best for |
|--------|-------------|---------|
| **Historical** | Uses actual past returns distribution | Most realistic, no assumptions |
| **Parametric** | Assumes normal distribution | Fast, works for large portfolios |
| **Monte Carlo** | Simulates 100,000 random scenarios | Most flexible, handles non-linearity |

**CVaR (Conditional VaR / Expected Shortfall):**
CVaR answers: *"Given that I DO lose more than VaR, how much should I expect to lose?"*
CVaR is always larger than VaR and is considered a more complete risk measure.

**How to use in your trading:**
- Calculate VaR before entering a position on Investopedia
- Never let 1-day VaR exceed 2-5% of your total portfolio
- If VaR is too high, reduce position size
- Compare VaR across positions to understand where your risk is concentrated
""")

    # ---- Portfolio Input ----
    st.markdown("### 💼 Portfolio Construction")
    st.caption("Enter your Investopedia positions or a hypothetical portfolio.")

    col1, col2 = st.columns(2)
    with col1:
        portfolio_value = st.number_input("Total Portfolio Value ($)", value=100000, min_value=1000, step=1000, key="var_value")
        confidence = st.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1,
                                   format_func=lambda x: f"{x*100:.0f}%", key="var_conf")
    with col2:
        method = st.selectbox("VaR Method", ["Historical", "Parametric", "Monte Carlo"], key="var_method")
        horizon = st.selectbox("Time Horizon", ["1 Day", "1 Week", "1 Month"], key="var_horizon")
        horizon_days = {"1 Day": 1, "1 Week": 5, "1 Month": 22}[horizon]

    st.markdown("#### Enter Holdings")
    st.caption("Enter ticker and portfolio weight (%). Weights must sum to 100%.")

    n_assets = st.number_input("Number of Assets", min_value=1, max_value=15, value=4, key="var_n")

    tickers = []
    weights = []
    cols = st.columns(2)
    for i in range(int(n_assets)):
        with cols[i % 2]:
            t = st.text_input(f"Ticker {i+1}", value=["SPY", "TLT", "GLD", "QQQ"][i] if i < 4 else "", key=f"var_t{i}").upper().strip()
            w = st.number_input(f"Weight {i+1} (%)", min_value=0.0, max_value=100.0,
                                value=[40.0, 30.0, 20.0, 10.0][i] if i < 4 else 0.0, key=f"var_w{i}")
            if t:
                tickers.append(t)
                weights.append(w / 100)

    if not tickers:
        st.warning("Please enter at least one ticker.")
        return

    total_weight = sum(weights)
    if abs(total_weight - 1.0) > 0.01:
        st.error(f"Weights sum to {total_weight*100:.1f}%. They must sum to 100%.")
        return

    run = st.button("▶ Calculate VaR", key="var_run")
    if not run:
        st.info("Enter your portfolio and click Calculate VaR.")
        return

    # ---- Fetch ----
    with st.spinner("Fetching return data..."):
        returns_df = _fetch_returns(tickers)

    if returns_df is None:
        st.error("Could not fetch return data.")
        return

    available = [t for t in tickers if t in returns_df.columns]
    if not available:
        st.error("None of the tickers could be fetched.")
        return

    weights_clean = [w for t, w in zip(tickers, weights) if t in available]
    weights_arr = np.array(weights_clean)
    weights_arr = weights_arr / weights_arr.sum()

    returns_clean = returns_df[available]
    portfolio_returns = returns_clean.dot(weights_arr)

    # ---- Compute VaR ----
    daily_var = _compute_var(portfolio_returns, confidence, method)
    scaled_var = daily_var * np.sqrt(horizon_days)
    daily_cvar = _compute_cvar(portfolio_returns, confidence)
    scaled_cvar = daily_cvar * np.sqrt(horizon_days)

    var_dollar    = scaled_var * portfolio_value
    cvar_dollar   = scaled_cvar * portfolio_value
    var_pct       = scaled_var * 100
    cvar_pct      = scaled_cvar * 100

    # ---- Results ----
    st.markdown("### 📊 VaR Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"VaR ({confidence*100:.0f}%, {horizon})", f"${var_dollar:,.0f}", delta=f"{var_pct:.2f}%")
    c2.metric(f"CVaR ({confidence*100:.0f}%, {horizon})", f"${cvar_dollar:,.0f}", delta=f"{cvar_pct:.2f}%")
    c3.metric("Portfolio Volatility (Ann)", f"{portfolio_returns.std() * np.sqrt(252) * 100:.1f}%")
    c4.metric("Sharpe Ratio (Ann)", f"{(portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)):.2f}")

    if var_pct > 5:
        st.error(f"⚠️ VaR of {var_pct:.1f}% exceeds recommended 5% threshold. Consider reducing position sizes.")
    elif var_pct > 3:
        st.warning(f"⚠️ VaR of {var_pct:.1f}% is elevated. Monitor closely.")
    else:
        st.success(f"✅ VaR of {var_pct:.1f}% is within acceptable range.")

    # ---- Per Asset VaR ----
    st.markdown("### 📋 Position-Level Risk")
    pos_data = []
    for t, w in zip(available, weights_arr):
        ret = returns_clean[t]
        pos_var = _compute_var(ret, confidence, "Historical") * np.sqrt(horizon_days)
        pos_cvar = _compute_cvar(ret, confidence) * np.sqrt(horizon_days)
        pos_dollar = portfolio_value * w
        pos_data.append({
            "Ticker": t,
            "Weight": f"{w*100:.1f}%",
            "Position ($)": f"${pos_dollar:,.0f}",
            f"VaR ({horizon})": f"{pos_var*100:.2f}%",
            f"VaR $ ({horizon})": f"${pos_var*pos_dollar:,.0f}",
            f"CVaR ({horizon})": f"{pos_cvar*100:.2f}%",
            "Ann Volatility": f"{ret.std()*np.sqrt(252)*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(pos_data), use_container_width=True)

    # ---- Return Distribution Chart ----
    st.markdown("### 📈 Portfolio Return Distribution")
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0e1117"); _dark_ax(ax)

    ax.hist(portfolio_returns * 100, bins=80, color="#1f77b4", alpha=0.7, edgecolor="none", label="Daily Returns")
    var_line = -daily_var * 100
    cvar_line = -daily_cvar * 100
    ax.axvline(var_line, color="#f0b429", linewidth=2, linestyle="--",
               label=f"VaR ({confidence*100:.0f}%): {var_line:.2f}%")
    ax.axvline(cvar_line, color="#e74c3c", linewidth=2, linestyle="--",
               label=f"CVaR ({confidence*100:.0f}%): {cvar_line:.2f}%")
    ax.fill_between([portfolio_returns.min()*100, var_line], 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100,
                     alpha=0.15, color="#e74c3c")
    ax.set_xlabel("Daily Return (%)", color="white", fontsize=9)
    ax.set_ylabel("Frequency", color="white", fontsize=9)
    ax.set_title(f"Portfolio Return Distribution | VaR & CVaR at {confidence*100:.0f}% Confidence", color="white", fontsize=11)
    ax.legend(fontsize=8, facecolor="#1e1e1e", labelcolor="white")
    plt.tight_layout()
    st.pyplot(fig)

    # ---- Cumulative Returns ----
    st.markdown("### 📊 Portfolio Cumulative Performance")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    fig2.patch.set_facecolor("#0e1117"); _dark_ax(ax2)
    cumulative = (1 + portfolio_returns).cumprod()
    ax2.plot(cumulative.index, cumulative.values, color="#f0b429", linewidth=1.5)
    ax2.axhline(1.0, color="white", linewidth=0.5, alpha=0.3, linestyle="--")
    ax2.fill_between(cumulative.index, cumulative.values, 1.0,
                      where=(cumulative >= 1.0), alpha=0.15, color="#2ecc71")
    ax2.fill_between(cumulative.index, cumulative.values, 1.0,
                      where=(cumulative < 1.0), alpha=0.15, color="#e74c3c")
    ax2.set_title("Portfolio Cumulative Return", color="white", fontsize=11)
    ax2.set_ylabel("Growth of $1", color="white", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig2)

    st.caption(
        f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | "
        "Data via Yahoo Finance | VaR calculated using historical simulation"
    )
