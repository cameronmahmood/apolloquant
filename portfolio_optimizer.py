# portfolio_optimizer.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from scipy.optimize import minimize

DEFAULT_TICKERS = ["SPY", "TLT", "GLD", "QQQ", "XLE"]

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

def _portfolio_stats(weights, returns, rf=0.045/252):
    port_ret = returns.dot(weights)
    ann_ret  = port_ret.mean() * 252
    ann_vol  = port_ret.std() * np.sqrt(252)
    sharpe   = (ann_ret - rf*252) / ann_vol if ann_vol > 0 else 0
    cumulative = (1 + port_ret).cumprod()
    max_dd = (1 - cumulative / cumulative.cummax()).max()
    return ann_ret, ann_vol, sharpe, max_dd

def _neg_sharpe(weights, returns, rf=0.045/252):
    _, _, sharpe, _ = _portfolio_stats(weights, returns, rf)
    return -sharpe

def _portfolio_vol(weights, returns):
    return returns.dot(weights).std() * np.sqrt(252)

def _efficient_frontier(returns, n_points=100, rf=0.045/252):
    n = len(returns.columns)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((0.01, 0.60) for _ in range(n))
    w0 = np.ones(n) / n

    # Target returns range
    min_ret = returns.mean().min() * 252
    max_ret = returns.mean().max() * 252
    target_rets = np.linspace(min_ret * 0.5, max_ret * 1.2, n_points)

    frontier_vols = []
    frontier_rets = []
    frontier_weights = []

    for target in target_rets:
        cons = constraints + [{"type": "eq", "fun": lambda w, t=target: _portfolio_stats(w, returns)[0] - t}]
        try:
            result = minimize(_portfolio_vol, w0, args=(returns,), method="SLSQP",
                               bounds=bounds, constraints=cons,
                               options={"maxiter": 500, "ftol": 1e-9})
            if result.success:
                r, v, s, d = _portfolio_stats(result.x, returns, rf)
                frontier_vols.append(v)
                frontier_rets.append(r)
                frontier_weights.append(result.x)
        except Exception:
            pass

    return frontier_rets, frontier_vols, frontier_weights

def _dark_ax(ax):
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

def run_portfolio_optimizer():
    st.subheader("⚡ Portfolio Optimizer")
    st.markdown(
        "Finds the **optimal asset allocation** using Modern Portfolio Theory. "
        "Plots the efficient frontier — the set of portfolios that maximize return for a given level of risk. "
        "Used by every institutional portfolio manager and quantitative fund."
    )

    with st.expander("📖 Modern Portfolio Theory", expanded=False):
        st.markdown("""
**Harry Markowitz's Modern Portfolio Theory (1952)** shows that:
1. Investors want to maximize return for a given level of risk
2. Combining assets with low correlation reduces portfolio risk without reducing return
3. There is an "efficient frontier" — the set of optimal portfolios

**Key Portfolios on the Frontier:**

| Portfolio | Description | Best for |
|-----------|-------------|---------|
| **Maximum Sharpe** | Best risk-adjusted return | Most investors |
| **Minimum Variance** | Lowest possible volatility | Risk-averse investors |
| **Equal Weight** | 1/N allocation | Simple baseline |

**Sharpe Ratio = (Return - Risk Free Rate) / Volatility**
The higher the Sharpe ratio, the better the return per unit of risk taken.

**Constraints used:**
- Minimum weight per asset: 1% (no going to zero)
- Maximum weight per asset: 60% (no over-concentration)
- Weights sum to 100% (fully invested)
- Long only (no short positions)

**How to use:**
- Enter your Investopedia holdings or potential portfolio
- Find the Maximum Sharpe portfolio for optimal allocation
- Compare to your current allocation to see if you're leaving return on the table
""")

    # ---- Input ----
    st.markdown("### 💼 Portfolio Construction")
    col1, col2, col3 = st.columns(3)
    with col1:
        tickers_input = st.text_input("Tickers (comma-separated)",
                                       value=", ".join(DEFAULT_TICKERS), key="opt_tickers")
    with col2:
        period = st.selectbox("Lookback Period", ["1y", "2y", "3y"], index=1, key="opt_period")
        rf_rate = st.number_input("Risk-Free Rate (%)", value=4.5, min_value=0.0,
                                   max_value=10.0, step=0.1, key="opt_rf") / 100
    with col3:
        n_simulations = st.selectbox("Monte Carlo Portfolios", [1000, 5000, 10000], index=1, key="opt_sims")
        show_frontier = st.checkbox("Show Efficient Frontier", value=True, key="opt_frontier")

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if len(tickers) < 2:
        st.warning("Please enter at least 2 tickers.")
        return

    run = st.button("▶ Optimize Portfolio", key="opt_run")
    if not run:
        st.info("Enter your tickers and click Optimize Portfolio.")
        return

    with st.spinner(f"Fetching data and optimizing portfolio..."):
        returns = _fetch_returns(tickers, period=period)

    if returns is None:
        st.error("Could not fetch return data.")
        return

    available = [t for t in tickers if t in returns.columns]
    if len(available) < 2:
        st.error("Need at least 2 tickers with available data.")
        return

    returns = returns[available]
    rf_daily = rf_rate / 252
    n = len(available)

    st.success(f"Optimizing portfolio of {len(available)} assets using {len(returns):,} trading days")

    # ---- Optimize ----
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((0.01, 0.60) for _ in range(n))
    w0 = np.ones(n) / n

    # Max Sharpe
    result_sharpe = minimize(_neg_sharpe, w0, args=(returns, rf_daily),
                              method="SLSQP", bounds=bounds, constraints=constraints)
    w_sharpe = result_sharpe.x if result_sharpe.success else w0

    # Min Variance
    result_minvar = minimize(_portfolio_vol, w0, args=(returns,),
                              method="SLSQP", bounds=bounds, constraints=constraints)
    w_minvar = result_minvar.x if result_minvar.success else w0

    # Equal weight
    w_equal = np.ones(n) / n

    # Stats
    r_sharpe, v_sharpe, s_sharpe, d_sharpe = _portfolio_stats(w_sharpe, returns, rf_daily)
    r_minvar, v_minvar, s_minvar, d_minvar = _portfolio_stats(w_minvar, returns, rf_daily)
    r_equal,  v_equal,  s_equal,  d_equal  = _portfolio_stats(w_equal,  returns, rf_daily)

    # ---- Summary ----
    st.markdown("### 🏆 Optimal Portfolio Results")
    tab1, tab2, tab3 = st.tabs(["⭐ Max Sharpe", "🛡️ Min Variance", "⚖️ Equal Weight"])

    for tab, label, weights, ret, vol, sharpe, dd in [
        (tab1, "Maximum Sharpe", w_sharpe, r_sharpe, v_sharpe, s_sharpe, d_sharpe),
        (tab2, "Minimum Variance", w_minvar, r_minvar, v_minvar, s_minvar, d_minvar),
        (tab3, "Equal Weight", w_equal, r_equal, v_equal, s_equal, d_equal),
    ]:
        with tab:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Annual Return", f"{ret*100:.1f}%")
            c2.metric("Annual Volatility", f"{vol*100:.1f}%")
            c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
            c4.metric("Max Drawdown", f"{dd*100:.1f}%")

            st.markdown("**Optimal Weights:**")
            weight_df = pd.DataFrame({
                "Ticker": available,
                "Weight": [f"{w*100:.1f}%" for w in weights],
                "Weight (Raw)": weights
            }).sort_values("Weight (Raw)", ascending=False)
            st.dataframe(weight_df[["Ticker", "Weight"]], use_container_width=True)

    # ---- Monte Carlo Simulation ----
    st.markdown("### 🎯 Efficient Frontier")
    np.random.seed(42)
    mc_rets = []
    mc_vols = []
    mc_sharpes = []

    for _ in range(n_simulations):
        w = np.random.dirichlet(np.ones(n))
        r, v, s, _ = _portfolio_stats(w, returns, rf_daily)
        mc_rets.append(r * 100)
        mc_vols.append(v * 100)
        mc_sharpes.append(s)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0e1117"); _dark_ax(ax)

    sc = ax.scatter(mc_vols, mc_rets, c=mc_sharpes, cmap="RdYlGn",
                     alpha=0.4, s=8, vmin=min(mc_sharpes), vmax=max(mc_sharpes))
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio", shrink=0.8)

    # Plot optimal portfolios
    ax.scatter(v_sharpe*100, r_sharpe*100, color="#f0b429", s=200, zorder=10,
               marker="*", label=f"Max Sharpe (SR={s_sharpe:.2f})")
    ax.scatter(v_minvar*100, r_minvar*100, color="#3498db", s=150, zorder=10,
               marker="D", label=f"Min Variance")
    ax.scatter(v_equal*100, r_equal*100, color="white", s=120, zorder=10,
               marker="^", label=f"Equal Weight")

    # Individual assets
    for ticker in available:
        a_ret = returns[ticker].mean() * 252 * 100
        a_vol = returns[ticker].std() * np.sqrt(252) * 100
        ax.scatter(a_vol, a_ret, color="#e74c3c", s=80, zorder=8, alpha=0.8)
        ax.annotate(ticker, (a_vol, a_ret), textcoords="offset points",
                    xytext=(5, 5), color="white", fontsize=8)

    ax.set_xlabel("Annual Volatility (%)", color="white", fontsize=10)
    ax.set_ylabel("Annual Return (%)", color="white", fontsize=10)
    ax.set_title(f"Efficient Frontier — {', '.join(available)}", color="white", fontsize=12)
    ax.legend(fontsize=8, facecolor="#1e1e1e", labelcolor="white")
    plt.tight_layout()
    st.pyplot(fig)

    # ---- Correlation Matrix ----
    st.markdown("### 🔗 Asset Correlations")
    corr = returns.corr()
    fig2, ax2 = plt.subplots(figsize=(max(6, n*0.8), max(5, n*0.7)))
    fig2.patch.set_facecolor("#0e1117"); _dark_ax(ax2)
    im = ax2.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax2.set_xticks(range(n)); ax2.set_xticklabels(available, rotation=45, ha="right", color="white", fontsize=9)
    ax2.set_yticks(range(n)); ax2.set_yticklabels(available, color="white", fontsize=9)
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                     color="white" if abs(corr.iloc[i,j]) > 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax2, shrink=0.8)
    ax2.set_title("Asset Correlation Matrix", color="white", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig2)

    # ---- Individual Asset Stats ----
    st.markdown("### 📋 Individual Asset Statistics")
    asset_stats = []
    for t in available:
        r = returns[t].mean() * 252 * 100
        v = returns[t].std() * np.sqrt(252) * 100
        s = (returns[t].mean()*252 - rf_rate) / (returns[t].std()*np.sqrt(252))
        cum = (1 + returns[t]).cumprod()
        dd = (1 - cum/cum.cummax()).max() * 100
        asset_stats.append({"Ticker": t, "Ann Return": f"{r:.1f}%",
                             "Ann Volatility": f"{v:.1f}%",
                             "Sharpe": f"{s:.2f}", "Max Drawdown": f"{dd:.1f}%"})
    st.dataframe(pd.DataFrame(asset_stats), use_container_width=True)

    st.caption(
        f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | "
        "Optimization via SciPy SLSQP | Bounds: 1%-60% per asset"
    )
