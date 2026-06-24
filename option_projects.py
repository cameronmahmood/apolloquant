# option_projects.py

import time
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from scipy.stats import norm
from scipy.optimize import brentq

# =========================
# Core Black-Scholes pricing
# =========================

def bs_price_european(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def black_scholes(S, K, T, r, sigma, option_type="call"):
    return bs_price_european(S, K, T, r, sigma, option_type)

# =========================
# Greeks
# =========================

def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """Returns dict of Delta, Gamma, Theta, Vega, Rho."""
    if T <= 0 or sigma <= 0:
        return {"Delta": 0.0, "Gamma": 0.0, "Theta": 0.0, "Vega": 0.0, "Rho": 0.0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    if option_type == "call":
        delta = norm.cdf(d1)
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega  = S * pdf_d1 * np.sqrt(T) / 100
    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}

# =========================
# Streamlit caches
# =========================

@st.cache_data(ttl=300, show_spinner=False)
def _cached_hist(symbol):
    tk = yf.Ticker(symbol)
    return tk.history(period="5d", auto_adjust=True)

@st.cache_data(ttl=300, show_spinner=False)
def _cached_options_list(symbol):
    tk = yf.Ticker(symbol)
    return tk.options or []

@st.cache_data(ttl=300, show_spinner=False)
def _cached_option_chain(symbol, exp):
    tk = yf.Ticker(symbol)
    chain = tk.option_chain(exp)
    return chain.calls.copy(), chain.puts.copy()

# =========================
# Black-Scholes UI
# =========================

def run_black_scholes():
    st.subheader("📈 Black-Scholes Option Pricer")
    st.markdown(
        "Price European options using Black-Scholes, view the P&L heatmap, "
        "and see all **Greeks** for risk analysis."
    )

    st.markdown("### 🔧 Inputs")
    c1, c2, c3 = st.columns(3)
    with c1:
        S = st.number_input("Asset Price (S)", value=100.0, step=1.0, min_value=0.01, key="bs_S")
    with c2:
        K = st.number_input("Strike Price (K)", value=100.0, step=1.0, min_value=0.01, key="bs_K")
    with c3:
        T = st.number_input("Time to Maturity (years)", value=1.0, step=0.1, min_value=0.01, key="bs_T")

    c4, c5, c6 = st.columns(3)
    with c4:
        r = st.number_input("Risk-Free Rate (%)", value=4.5, step=0.1, key="bs_r") / 100
    with c5:
        sigma = st.number_input("Volatility / IV (%)", value=20.0, step=0.1, min_value=0.01, key="bs_sigma") / 100
    with c6:
        option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True, key="bs_type")

    st.markdown("#### 💸 Optional: Market Price (for comparison)")
    c7, c8 = st.columns(2)
    with c7:
        market_price = st.number_input(
            "Market Price of Option ($)",
            value=0.0, min_value=0.0, step=0.01, key="bs_mktprice",
            help="Enter the actual market price to compare against theoretical value"
        )
    with c8:
        plot_type = st.radio("Heatmap Type", ["Option Value", "P&L vs Market Price"], horizontal=True, key="bs_plot")

    opt = option_type.lower()
    call_price = bs_price_european(S, K, T, r, sigma, "call")
    put_price  = bs_price_european(S, K, T, r, sigma, "put")
    theo_price = call_price if opt == "call" else put_price
    greeks     = bs_greeks(S, K, T, r, sigma, opt)

    # Intrinsic and time value
    intrinsic = max(0.0, (S - K) if opt == "call" else (K - S))
    time_val  = theo_price - intrinsic
    breakeven = (K + theo_price) if opt == "call" else (K - theo_price)

    # ---- Prices ----
    st.markdown("### 🧮 Theoretical Prices")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Call Price", f"${call_price:.4f}")
    c2.metric("Put Price",  f"${put_price:.4f}")
    c3.metric("Intrinsic Value", f"${intrinsic:.4f}")
    c4.metric("Time Value",      f"${time_val:.4f}")

    c5, c6 = st.columns(2)
    c5.metric("Breakeven at Expiry", f"${breakeven:.2f}")
    if market_price > 0:
        diff = theo_price - market_price
        if diff > 0:
            c6.metric("vs Market Price", f"${diff:+.4f}", delta="Underpriced ✅ (model > market)")
        else:
            c6.metric("vs Market Price", f"${diff:+.4f}", delta="Overpriced ⚠️ (model < market)")

    if market_price > 0:
        st.info(
            f"**Interpretation:** Theoretical value is ${theo_price:.4f}. "
            f"Market price is ${market_price:.4f}. "
            + ("The option appears **underpriced** relative to the model — potential buy opportunity. "
               if theo_price > market_price else
               "The option appears **overpriced** relative to the model — consider selling or avoiding. ")
            + "⚠️ Model depends heavily on the volatility assumption."
        )

    # ---- Greeks ----
    st.markdown("### 🔬 Option Greeks")
    st.caption("Greeks measure how the option price changes with respect to each input.")

    gc1, gc2, gc3, gc4, gc5 = st.columns(5)
    gc1.metric(
        "Delta (Δ)",
        f"{greeks['Delta']:.4f}",
        help="Change in option price per $1 move in the underlying. "
             "Call delta: 0 to 1. Put delta: -1 to 0."
    )
    gc2.metric(
        "Gamma (Γ)",
        f"{greeks['Gamma']:.4f}",
        help="Rate of change of Delta per $1 move in underlying. "
             "High gamma = delta changes quickly near expiry or ATM."
    )
    gc3.metric(
        "Theta (Θ)",
        f"${greeks['Theta']:.4f}/day",
        help="Daily time decay — how much value the option loses per calendar day. "
             "Negative for long options. Benefits option sellers."
    )
    gc4.metric(
        "Vega (ν)",
        f"${greeks['Vega']:.4f}/%",
        help="Change in option price per 1% change in implied volatility. "
             "Positive for long options — rising IV benefits buyers."
    )
    gc5.metric(
        "Rho (ρ)",
        f"${greeks['Rho']:.4f}/%",
        help="Change in option price per 1% change in risk-free rate. "
             "Usually the smallest Greek in magnitude."
    )

    st.markdown(f"""
**Greeks Interpretation for this {option_type}:**
- **Delta {greeks['Delta']:.2f}**: The option moves ~${abs(greeks['Delta']):.2f} for every $1 move in the underlying
- **Theta {greeks['Theta']:.4f}/day**: This option loses ~${abs(greeks['Theta']):.4f} in value every calendar day
- **Vega {greeks['Vega']:.4f}**: A 1% rise in IV increases the option value by ~${greeks['Vega']:.4f}
- **Gamma {greeks['Gamma']:.4f}**: Delta will change by {greeks['Gamma']:.4f} for every $1 move in the underlying
""")

    # ---- Heatmap ----
    st.subheader("📊 P&L Heatmap")
    S_range     = np.linspace(max(0.01, S * 0.8), S * 1.2, 30)
    sigma_range = np.linspace(max(0.01, sigma * 0.5), sigma * 1.5, 30)
    heatmap = np.zeros((len(S_range), len(sigma_range)))

    for i, s_val in enumerate(S_range):
        for j, sig_val in enumerate(sigma_range):
            price = bs_price_european(s_val, K, T, r, sig_val, opt)
            if plot_type == "P&L vs Market Price" and market_price > 0:
                heatmap[i, j] = price - market_price
            else:
                heatmap[i, j] = price

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap,
        xticklabels=np.round(sigma_range * 100, 1),
        yticklabels=np.round(S_range, 1),
        cmap="RdYlGn" if (plot_type == "P&L vs Market Price" and market_price > 0) else "YlGnBu",
        ax=ax,
        fmt=".2f",
        annot=False,
    )
    ax.set_xlabel("Implied Volatility (%)")
    ax.set_ylabel("Asset Price ($)")
    ax.set_title(f"{option_type} {plot_type} Heatmap — Strike ${K:.0f}, T={T:.1f}yr")
    st.pyplot(fig)
    st.caption(f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | Data: manual inputs")

# =========================
# Monte Carlo
# =========================

def run_monte_carlo():
    st.subheader("🎲 Monte Carlo Option Pricing")
    st.markdown(
        "Simulate thousands of price paths using **Geometric Brownian Motion** to estimate "
        "European option prices and compare against Black-Scholes. "
        "Monte Carlo is useful when you want to simulate many possible future paths "
        "instead of relying only on a closed-form formula."
    )

    st.markdown("### 🔧 Inputs")
    c1, c2, c3 = st.columns(3)
    with c1:
        S = st.number_input("Asset Price (S)", value=100.0, step=1.0, min_value=0.01, key="mc_S")
    with c2:
        K = st.number_input("Strike Price (K)", value=100.0, step=1.0, min_value=0.01, key="mc_K")
    with c3:
        T = st.number_input("Time to Maturity (years)", value=1.0, step=0.1, min_value=0.01, key="mc_T")

    c4, c5, c6 = st.columns(3)
    with c4:
        r = st.number_input("Risk-Free Rate (%)", value=4.5, step=0.1, key="mc_r") / 100
    with c5:
        sigma = st.number_input("Volatility (%)", value=20.0, step=0.1, min_value=0.01, key="mc_sigma") / 100
    with c6:
        n_sims = st.selectbox("Simulations", [1_000, 10_000, 50_000, 100_000], index=1, key="mc_sims")

    c7, c8 = st.columns(2)
    with c7:
        n_steps = st.number_input("Time Steps", value=252, step=1, min_value=10, key="mc_steps")
    with c8:
        option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True, key="mc_type")

    run = st.button("▶ Run Simulation", key="mc_run")
    if not run:
        return

    opt = option_type.lower()
    with st.spinner(f"Running {n_sims:,} simulations..."):
        dt = T / n_steps
        Z = np.random.standard_normal((n_steps, n_sims))
        increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        log_paths = np.vstack([np.zeros(n_sims), np.cumsum(increments, axis=0)])
        price_paths = S * np.exp(log_paths)
        S_T = price_paths[-1]
        payoffs = np.maximum(S_T - K, 0) if opt == "call" else np.maximum(K - S_T, 0)
        mc_price  = np.exp(-r * T) * np.mean(payoffs)
        mc_stderr = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
        mc_ci_low  = mc_price - 1.96 * mc_stderr
        mc_ci_high = mc_price + 1.96 * mc_stderr
        bs_price_val = bs_price_european(S, K, T, r, sigma, opt)
        pct_itm = 100.0 * np.sum(payoffs > 0) / n_sims
        avg_payoff = np.mean(payoffs[payoffs > 0]) if np.sum(payoffs > 0) > 0 else 0

    st.markdown("### 📊 Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MC Price",    f"${mc_price:.4f}")
    col2.metric("BS Price",    f"${bs_price_val:.4f}", delta=f"{mc_price - bs_price_val:+.4f}")
    col3.metric("95% CI",      f"${mc_ci_low:.3f} – ${mc_ci_high:.3f}")
    col4.metric("Std Error",   f"${mc_stderr:.4f}")

    col5, col6 = st.columns(2)
    col5.metric("Prob ITM",    f"{pct_itm:.1f}%")
    col6.metric("Avg Payoff (ITM)", f"${avg_payoff:.2f}")

    st.info(
        f"**Interpretation:** Based on {n_sims:,} simulations, this {option_type} has a "
        f"**{pct_itm:.1f}% probability of expiring in the money**. "
        f"When it does expire ITM, the average payoff is **${avg_payoff:.2f}**. "
        f"The Monte Carlo price of **${mc_price:.4f}** compares to Black-Scholes **${bs_price_val:.4f}** "
        f"({'within normal variance' if abs(mc_price - bs_price_val) < 0.05 else 'slight divergence due to simulation variance'})."
    )

    st.markdown("### 📈 Simulated Price Paths")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    n_display = min(200, n_sims)
    t_axis = np.linspace(0, T, n_steps + 1)
    for i in range(n_display):
        axes[0].plot(t_axis, price_paths[:, i], alpha=0.15, linewidth=0.5, color="#1f77b4")
    axes[0].axhline(K, color="red", linewidth=1.5, linestyle="--", label=f"Strike K=${K}")
    axes[0].axhline(S, color="yellow", linewidth=1.0, linestyle=":", label=f"Spot S=${S}")
    axes[0].set_xlabel("Time (years)"); axes[0].set_ylabel("Asset Price")
    axes[0].set_title(f"{n_display} Sample Paths")
    axes[0].legend(fontsize=8, labelcolor="white", facecolor="#1e1e1e")

    axes[1].hist(S_T, bins=80, color="#1f77b4", edgecolor="none", alpha=0.85, density=True)
    axes[1].axvline(K, color="red", linewidth=1.5, linestyle="--", label=f"Strike K=${K}")
    axes[1].axvline(np.mean(S_T), color="yellow", linewidth=1.2, linestyle=":", label=f"Mean=${np.mean(S_T):.2f}")
    axes[1].set_xlabel("Terminal Price"); axes[1].set_ylabel("Density")
    axes[1].set_title("Terminal Price Distribution")
    axes[1].legend(fontsize=8, labelcolor="white", facecolor="#1e1e1e")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### 🔁 MC Price Convergence")
    checkpoints = np.unique(np.logspace(2, np.log10(n_sims), num=60).astype(int))
    conv_prices = [np.exp(-r * T) * np.mean(payoffs[:n]) for n in checkpoints]

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    fig2.patch.set_facecolor("#0e1117"); ax2.set_facecolor("#0e1117")
    ax2.tick_params(colors="white"); ax2.xaxis.label.set_color("white")
    ax2.yaxis.label.set_color("white"); ax2.title.set_color("white")
    for spine in ax2.spines.values(): spine.set_edgecolor("#444")
    ax2.plot(checkpoints, conv_prices, color="#1f77b4", linewidth=1.5, label="MC Price")
    ax2.axhline(bs_price_val, color="orange", linewidth=1.5, linestyle="--", label=f"BS Price ${bs_price_val:.4f}")
    ax2.set_xlabel("Number of Simulations"); ax2.set_ylabel("Estimated Price")
    ax2.set_title("MC Price Convergence vs Black-Scholes")
    ax2.legend(fontsize=9, labelcolor="white", facecolor="#1e1e1e")
    plt.tight_layout()
    st.pyplot(fig2)

    st.caption(f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | Data: Monte Carlo simulation")

# =========================
# Market-Implied Move
# =========================

_DEFAULT_TICKERS = ["SPY", "QQQ", "TLT", "GLD"]
_TARGETS = {"1W": 7, "1M": 30}

def run_implied_move():
    st.subheader("📈 Market-Implied Move (from ATM IV)")
    st.markdown(
        "Shows the **expected price range** implied by options markets for each ticker. "
        "Use before every options trade on Investopedia to understand what the market is pricing in."
    )
    st.caption(
        "Method: ATM IV × sqrt(DTE/365) × spot. "
        "If IV is missing it is backed out from the ATM call mid via Black-Scholes."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        tickers_text = st.text_input("Tickers (comma-separated)", value=",".join(_DEFAULT_TICKERS), key="im_tickers")
    with col2:
        st.button("🔄 Refresh", key="im_refresh")

    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    if not tickers:
        st.warning("Please enter at least one ticker.")
        return

    with st.spinner("Fetching option chains and computing implied moves..."):
        df, notes = _build_implied_move_table(tickers)

    # ---- Enhanced display with price ranges ----
    st.markdown("### 📊 Implied Move Summary")
    for _, row in df.iterrows():
        ticker = row["Ticker"]
        price_str = row["Price"]
        move1m_str = row["1M Move ($)"]
        move1w_str = row["1W Move ($)"]

        try:
            price = float(price_str)
            move1m = float(move1m_str)
            move1w = float(move1w_str)

            with st.expander(f"**{ticker}** — Price: ${price:.2f} | 1W: ±${move1w:.2f} | 1M: ±${move1m:.2f}", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Price", f"${price:.2f}")
                c2.metric("ATM IV", row["ATM IV"])
                c3.metric("1W Implied Move", row["1W Move (%)"])

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**📅 1-Week Expected Range:**")
                    upper1w = price + move1w
                    lower1w = price - move1w
                    if upper1w > 0 and lower1w > 0:
                        st.success(f"Upper: **${upper1w:.2f}**")
                        st.error(f"Lower: **${lower1w:.2f}**")
                        st.caption(f"Options market expects {ticker} to stay between ${lower1w:.2f} and ${upper1w:.2f} over the next week with ~68% probability.")
                with col_b:
                    st.markdown("**📅 1-Month Expected Range:**")
                    upper1m = price + move1m
                    lower1m = price - move1m
                    if upper1m > 0 and lower1m > 0:
                        st.success(f"Upper: **${upper1m:.2f}**")
                        st.error(f"Lower: **${lower1m:.2f}**")
                        st.caption(f"Options market expects {ticker} to stay between ${lower1m:.2f} and ${upper1m:.2f} over the next month with ~68% probability.")
        except (ValueError, TypeError):
            st.warning(f"**{ticker}** — {price_str} ({row.get('ATM IV', 'N/A')})")

    st.markdown("### 📋 Full Data Table")
    st.dataframe(df, use_container_width=True)

    if notes:
        st.info(" / ".join(notes))

    st.warning(
        "⚠️ **Before earnings, FOMC, or CPI releases:** Implied moves are often elevated. "
        "Check the Economic Calendar before placing options trades around major events."
    )
    st.caption(
        f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | "
        "Data via Yahoo Finance options chains"
    )

def run_implied_move_table():
    run_implied_move()

# ---------- Helpers ----------

def _to_dt(s):
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def _pick_expiration(expirations, days_ahead_target):
    now = datetime.now(timezone.utc)
    exps = [_to_dt(e) for e in expirations if e]
    exps = [e for e in exps if e >= now]
    if not exps:
        return None
    target = now + pd.Timedelta(days=days_ahead_target)
    chosen = min(exps, key=lambda d: abs((d - target).days))
    return chosen.strftime("%Y-%m-%d")

def _bs_price(S, K, T, r, vol, cp="c"):
    if T <= 0 or vol <= 0:
        return max(0.0, (S - K) if cp == "c" else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * vol * vol) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    if cp == "c":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def _implied_vol_from_mid(mkt_price, S, K, T, r=0.0, cp="c"):
    if (mkt_price is None) or (mkt_price <= 0) or (T <= 0):
        return np.nan
    def f(vol):
        return _bs_price(S, K, T, r, vol, cp) - mkt_price
    try:
        return brentq(f, 1e-6, 5.0, maxiter=200)
    except ValueError:
        return np.nan

def _mid(bid, ask, last):
    try:
        if bid is not None and ask is not None and pd.notna(bid) and pd.notna(ask) and (float(bid) > 0 or float(ask) > 0):
            return (float(bid) + float(ask)) / 2.0
    except Exception:
        pass
    try:
        if last is not None and pd.notna(last):
            return float(last)
    except Exception:
        pass
    return np.nan

def _nearest_atm_strike(strikes, spot):
    arr = np.asarray(strikes, dtype=float)
    if arr.size == 0:
        return np.nan
    return float(arr[np.abs(arr - spot).argmin()])

def _flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df

def _get_spot(symbol):
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period="5d", auto_adjust=True)
        hist = _flatten_columns(hist)
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    try:
        raw = yf.download(symbol, period="5d", progress=False, auto_adjust=True)
        raw = _flatten_columns(raw)
        if not raw.empty and "Close" in raw.columns:
            return float(raw["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return np.nan

def _get_option_chain(symbol, exp):
    try:
        tk = yf.Ticker(symbol)
        chain = tk.option_chain(exp)
        calls = _flatten_columns(chain.calls.copy())
        puts  = _flatten_columns(chain.puts.copy())
        return calls, puts
    except Exception:
        return None, None

def _get_atm_iv_for_exp(symbol, spot, exp):
    try:
        exp_dt = _to_dt(exp)
        now = datetime.now(timezone.utc)
        dte = max((exp_dt - now).days, 1)
    except Exception:
        return np.nan, 0
    T_years = dte / 365.0
    try:
        calls, puts = _cached_option_chain(symbol, exp)
    except YFRateLimitError:
        raise
    except Exception:
        return np.nan, dte
    if calls is None or puts is None or calls.empty or puts.empty:
        return np.nan, dte
    calls = _flatten_columns(calls)
    puts  = _flatten_columns(puts)
    if "strike" not in calls.columns:
        return np.nan, dte
    atm = _nearest_atm_strike(calls["strike"].values, spot)
    if np.isnan(atm):
        return np.nan, dte
    call_row = calls[calls["strike"] == atm]
    put_row  = puts[puts["strike"] == atm] if "strike" in puts.columns else pd.DataFrame()
    if call_row.empty:
        return np.nan, dte
    call_row = call_row.iloc[0]
    put_row  = put_row.iloc[0] if not put_row.empty else None
    ivs = []
    for col in ("impliedVolatility", "impliedVol"):
        try:
            v = call_row[col]
            if pd.notna(v) and float(v) > 0.001:
                ivs.append(float(v))
        except Exception:
            pass
        if put_row is not None:
            try:
                v = put_row[col]
                if pd.notna(v) and float(v) > 0.001:
                    ivs.append(float(v))
            except Exception:
                pass
    if ivs:
        return float(np.nanmean(ivs)), int(dte)
    try:
        bid  = call_row.get("bid",       None)
        ask  = call_row.get("ask",       None)
        last = call_row.get("lastPrice", None)
        call_mid = _mid(bid, ask, last)
        atm_iv = _implied_vol_from_mid(call_mid, spot, float(atm), T_years, r=0.0, cp="c")
        if pd.notna(atm_iv) and atm_iv > 0:
            return float(atm_iv), int(dte)
    except Exception:
        pass
    return np.nan, int(dte)

def _expected_move_dollars(spot, iv_ann, dte_calendar):
    if pd.isna(iv_ann) or dte_calendar <= 0 or pd.isna(spot):
        return np.nan
    return float(iv_ann * math.sqrt(dte_calendar / 365.0) * spot)

def _fmt_pct(x):
    return "" if (x is None or pd.isna(x)) else f"{x:.2f}%"

def _fmt_num(x, nd=2):
    return "" if (x is None or pd.isna(x)) else f"{x:.{nd}f}"

def _build_implied_move_table(tickers):
    rows = []
    notes = []
    rate_limited = []
    for symbol in tickers:
        try:
            spot = _get_spot(symbol)
        except YFRateLimitError:
            rate_limited.append(symbol)
            rows.append([symbol, "rate-limited", "", "", "", "", ""])
            continue
        except Exception:
            rows.append([symbol, "error", "", "", "", "", ""])
            continue
        if pd.isna(spot):
            rows.append([symbol, "no data", "", "", "", "", ""])
            continue
        try:
            tk = yf.Ticker(symbol)
            expirations = tk.options or []
        except YFRateLimitError:
            rate_limited.append(symbol)
            rows.append([symbol, _fmt_num(spot), "rate-limited", "", "", "", ""])
            continue
        except Exception:
            rows.append([symbol, _fmt_num(spot), "error", "", "", "", ""])
            continue
        if not expirations:
            rows.append([symbol, _fmt_num(spot), "no options", "", "", "", ""])
            continue
        exp_1w = _pick_expiration(expirations, _TARGETS["1W"])
        exp_1m = _pick_expiration(expirations, _TARGETS["1M"])
        iv_1w, dte_1w = np.nan, 0
        iv_1m, dte_1m = np.nan, 0
        if exp_1w:
            try:
                iv_1w, dte_1w = _get_atm_iv_for_exp(symbol, spot, exp_1w)
                time.sleep(0.2)
            except YFRateLimitError:
                rate_limited.append(symbol)
        if exp_1m:
            try:
                iv_1m, dte_1m = _get_atm_iv_for_exp(symbol, spot, exp_1m)
                time.sleep(0.2)
            except YFRateLimitError:
                rate_limited.append(symbol)
        move1w_d   = _expected_move_dollars(spot, iv_1w, dte_1w)
        move1m_d   = _expected_move_dollars(spot, iv_1m, dte_1m)
        move1w_pct = (move1w_d / spot * 100.0) if pd.notna(move1w_d) else np.nan
        move1m_pct = (move1m_d / spot * 100.0) if pd.notna(move1m_d) else np.nan
        atm_iv_display = iv_1m if pd.notna(iv_1m) else iv_1w
        atm_iv_text = _fmt_num(atm_iv_display, 4) if pd.notna(atm_iv_display) else ""
        if symbol in rate_limited and not atm_iv_text:
            atm_iv_text = "rate-limited"
        rows.append([symbol, _fmt_num(spot, 2), atm_iv_text,
                     _fmt_num(move1w_d, 2), _fmt_pct(move1w_pct),
                     _fmt_num(move1m_d, 2), _fmt_pct(move1m_pct)])
    if rate_limited:
        notes.append(f"Rate-limited: {', '.join(sorted(set(rate_limited)))}. Auto-refresh in ~5 min.")
    df = pd.DataFrame(rows, columns=["Ticker","Price","ATM IV","1W Move ($)","1W Move (%)","1M Move ($)","1M Move (%)"])
    return df, notes