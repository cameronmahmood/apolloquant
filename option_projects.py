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
# Core Black–Scholes pricing (define first to avoid NameError)
# =========================

def bs_price_european(S, K, T, r, sigma, option_type="call"):
    """European option price via Black–Scholes."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Backward-compatible name if other modules call black_scholes()
def black_scholes(S, K, T, r, sigma, option_type="call"):
    return bs_price_european(S, K, T, r, sigma, option_type)

# =========================
# Streamlit caches for Yahoo calls (reduce rate-limit hits)
# =========================

@st.cache_data(ttl=300, show_spinner=False)
def _cached_hist(symbol: str):
    tk = yf.Ticker(symbol)
    return tk.history(period="5d", auto_adjust=True)

@st.cache_data(ttl=300, show_spinner=False)
def _cached_options_list(symbol: str):
    tk = yf.Ticker(symbol)
    return tk.options or []

@st.cache_data(ttl=300, show_spinner=False)
def _cached_option_chain(symbol: str, exp: str):
    tk = yf.Ticker(symbol)
    chain = tk.option_chain(exp)
    # Return plain DataFrames so cache is hashable/stable
    return chain.calls.copy(), chain.puts.copy()

# =========================
# Black–Scholes UI
# =========================

def run_black_scholes():
    st.subheader("📈 Black-Scholes Option Pricer with P&L Heatmap")
    st.markdown(
        "Use this tool to visualize **option prices** and **P&L surfaces** using the Black-Scholes model."
    )

    # ---- Inputs on the page (no sidebar) ----
    st.markdown("### 🔧 Inputs")

    c1, c2, c3 = st.columns(3)
    with c1:
        S = st.number_input("Asset Price (S)", value=100.0, step=1.0, min_value=0.0)
    with c2:
        K = st.number_input("Strike Price (K)", value=100.0, step=1.0, min_value=0.0)
    with c3:
        T = st.number_input("Time to Maturity (years)", value=1.0, step=0.1, min_value=0.0)

    c4, c5, c6 = st.columns(3)
    with c4:
        r = st.number_input("Risk-Free Rate (%)", value=2.0, step=0.1) / 100
    with c5:
        sigma = st.number_input("Volatility (%)", value=20.0, step=0.1, min_value=0.0) / 100
    with c6:
        plot_type = st.radio(
            "Heatmap Type",
            options=["Option Value", "Call P&L"],
            horizontal=True,
        )

    st.markdown("#### 💸 Optional Trade Inputs")
    c7, c8 = st.columns(2)
    with c7:
        call_price_paid = st.number_input("Call Purchase Price", value=0.0, min_value=0.0)
    with c8:
        put_price_paid = st.number_input("Put Purchase Price", value=0.0, min_value=0.0)

    # ---- Pricing ----
    call_price = bs_price_european(S, K, T, r, sigma, "call")
    put_price  = bs_price_european(S, K, T, r, sigma, "put")

    st.markdown(
        f"""
        ### 🧮 Option Prices
        - **Call Price:** ${call_price:.2f}  
        - **Put Price:** ${put_price:.2f}
        """
    )

    if call_price_paid > 0 or put_price_paid > 0:
        st.markdown(
            f"""
            ### 📊 Implied P&L (Based on Purchase Price)
            - Call P&L: ${call_price - call_price_paid:.2f}
            - Put P&L: ${put_price - put_price_paid:.2f}
            """
        )

    # ---- Heatmap ----
    st.subheader("📊 Heatmap Visualization")
    S_range = np.linspace(max(1e-9, S * 0.8), S * 1.2, 30)
    sigma_range = np.linspace(max(1e-6, sigma * 0.5), max(1e-6, sigma * 1.5), 30)
    heatmap = np.zeros((len(S_range), len(sigma_range)))

    for i, s_val in enumerate(S_range):
        for j, sig_val in enumerate(sigma_range):
            price = bs_price_european(s_val, K, T, r, sig_val, "call")
            heatmap[i, j] = (
                price - call_price_paid if (plot_type == "Call P&L" and call_price_paid > 0) else price
            )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap,
        xticklabels=np.round(sigma_range, 2),
        yticklabels=np.round(S_range, 2),
        cmap="RdYlGn" if (plot_type == "Call P&L" and call_price_paid > 0) else "YlGnBu",
        ax=ax,
    )
    plt.xlabel("Volatility")
    plt.ylabel("Asset Price")
    plt.title(f"{plot_type} Heatmap")
    st.pyplot(fig)

# =========================
# Monte Carlo Simulation for Option Pricing
# =========================

def run_monte_carlo():
    st.subheader("🎲 Monte Carlo Simulation for Option Pricing")
    st.markdown(
        "Simulate thousands of price paths using **Geometric Brownian Motion** to estimate "
        "European option prices and compare against Black–Scholes."
    )

    # ---- Inputs ----
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
        r = st.number_input("Risk-Free Rate (%)", value=2.0, step=0.1, key="mc_r") / 100
    with c5:
        sigma = st.number_input("Volatility (%)", value=20.0, step=0.1, min_value=0.01, key="mc_sigma") / 100
    with c6:
        n_sims = st.selectbox("Number of Simulations", options=[1_000, 10_000, 50_000, 100_000], index=1, key="mc_sims")

    c7, c8 = st.columns(2)
    with c7:
        n_steps = st.number_input("Time Steps", value=252, step=1, min_value=10, key="mc_steps")
    with c8:
        option_type = st.radio("Option Type", options=["Call", "Put"], horizontal=True, key="mc_type")

    run = st.button("▶ Run Simulation", key="mc_run")
    if not run:
        return

    opt_type = option_type.lower()

    # ---- Simulation ----
    with st.spinner(f"Running {n_sims:,} simulations..."):
        dt = T / n_steps
        # Shape: (n_steps, n_sims)
        Z = np.random.standard_normal((n_steps, n_sims))
        # Daily log-return increments
        increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        # Cumulative price paths: shape (n_steps+1, n_sims)
        log_paths = np.vstack([np.zeros(n_sims), np.cumsum(increments, axis=0)])
        price_paths = S * np.exp(log_paths)

        # Terminal prices
        S_T = price_paths[-1]

        # Payoffs
        if opt_type == "call":
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)

        # Discounted expected payoff
        mc_price = np.exp(-r * T) * np.mean(payoffs)
        mc_stderr = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
        mc_ci_low  = mc_price - 1.96 * mc_stderr
        mc_ci_high = mc_price + 1.96 * mc_stderr

        # Black–Scholes benchmark
        bs_price = bs_price_european(S, K, T, r, sigma, opt_type)

    # ---- Results ----
    st.markdown("### 📊 Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MC Price", f"${mc_price:.4f}")
    col2.metric("BS Price", f"${bs_price:.4f}", delta=f"{mc_price - bs_price:+.4f}")
    col3.metric("95% CI Low", f"${mc_ci_low:.4f}")
    col4.metric("95% CI High", f"${mc_ci_high:.4f}")

    st.caption(
        f"Std Error: ${mc_stderr:.4f} | "
        f"Simulations: {n_sims:,} | "
        f"Time Steps: {n_steps}"
    )

    # ---- Plots ----
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

    # --- Left: sample paths ---
    n_display = min(200, n_sims)
    t_axis = np.linspace(0, T, n_steps + 1)
    for i in range(n_display):
        axes[0].plot(t_axis, price_paths[:, i], alpha=0.15, linewidth=0.5, color="#1f77b4")
    axes[0].axhline(K, color="red", linewidth=1.5, linestyle="--", label=f"Strike K={K}")
    axes[0].axhline(S, color="yellow", linewidth=1.0, linestyle=":", label=f"Spot S={S}")
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("Asset Price")
    axes[0].set_title(f"{n_display} Sample Paths")
    axes[0].legend(fontsize=8, labelcolor="white", facecolor="#1e1e1e")

    # --- Right: terminal price distribution ---
    axes[1].hist(S_T, bins=80, color="#1f77b4", edgecolor="none", alpha=0.85, density=True)
    axes[1].axvline(K, color="red", linewidth=1.5, linestyle="--", label=f"Strike K={K}")
    axes[1].axvline(np.mean(S_T), color="yellow", linewidth=1.2, linestyle=":", label=f"Mean=${np.mean(S_T):.2f}")
    axes[1].set_xlabel("Terminal Price $S_T$")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Terminal Price Distribution")
    axes[1].legend(fontsize=8, labelcolor="white", facecolor="#1e1e1e")

    plt.tight_layout()
    st.pyplot(fig)

    # ---- Convergence chart ----
    st.markdown("### 🔁 Convergence of MC Price")

    checkpoints = np.unique(np.logspace(2, np.log10(n_sims), num=60).astype(int))
    conv_prices = []
    for n in checkpoints:
        sub_payoffs = payoffs[:n]
        conv_prices.append(np.exp(-r * T) * np.mean(sub_payoffs))

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    fig2.patch.set_facecolor("#0e1117")
    ax2.set_facecolor("#0e1117")
    ax2.tick_params(colors="white")
    ax2.xaxis.label.set_color("white")
    ax2.yaxis.label.set_color("white")
    ax2.title.set_color("white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")

    ax2.plot(checkpoints, conv_prices, color="#1f77b4", linewidth=1.5, label="MC Price")
    ax2.axhline(bs_price, color="orange", linewidth=1.5, linestyle="--", label=f"BS Price ${bs_price:.4f}")
    ax2.set_xlabel("Number of Simulations")
    ax2.set_ylabel("Estimated Price")
    ax2.set_title("MC Price Convergence vs Black–Scholes")
    ax2.legend(fontsize=9, labelcolor="white", facecolor="#1e1e1e")
    plt.tight_layout()
    st.pyplot(fig2)

    # ---- Payoff distribution ----
    st.markdown("### 💰 Payoff Distribution")

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    fig3.patch.set_facecolor("#0e1117")
    ax3.set_facecolor("#0e1117")
    ax3.tick_params(colors="white")
    ax3.xaxis.label.set_color("white")
    ax3.yaxis.label.set_color("white")
    ax3.title.set_color("white")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#444")

    nonzero = payoffs[payoffs > 0]
    pct_itm = 100.0 * len(nonzero) / n_sims
    ax3.hist(nonzero, bins=60, color="#2ca02c", edgecolor="none", alpha=0.85, density=True)
    ax3.set_xlabel("Payoff at Expiry")
    ax3.set_ylabel("Density (ITM only)")
    ax3.set_title(f"ITM Payoff Distribution  |  {pct_itm:.1f}% of paths expire ITM")
    plt.tight_layout()
    st.pyplot(fig3)

# =========================
# Market-Implied Move (ATM IV)
# =========================

_DEFAULT_TICKERS = ["USO", "TLT", "UUP", "QQQ"]
_TARGETS = {"1W": 7, "1M": 30}  # calendar days

def run_implied_move():
    st.subheader("📈 Market-Implied Move (from ATM IV)")
    st.caption(
        "Method: ATM IV (nearest ~7d and ~30d expiries) × sqrt(time in years) × spot. "
        "If IV is missing, it's backed out from the ATM call mid via Black–Scholes. "
        "Calendar days used for DTE."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        tickers_text = st.text_input("Tickers (comma-separated)", value=",".join(_DEFAULT_TICKERS))
    with col2:
        st.button("Refresh")

    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    if not tickers:
        st.warning("Please enter at least one ticker.")
        return

    with st.spinner("Fetching option chains and computing implied moves..."):
        df, notes = _build_implied_move_table(tickers)

    st.dataframe(df, use_container_width=True)

    if notes:
        st.info(" / ".join(notes))

    st.caption(
        "Notes: ATM IV column displays the ~30-day expiry IV when available (fallback to ~7-day). "
        "1W/1M moves use horizon-specific IVs and DTE. Time = DTE/365."
    )
    st.caption(f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

def run_implied_move_table():
    """Backward-compatible alias."""
    run_implied_move()

# ---------- Helpers (implied move) ----------

def _to_dt(s: str):
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def _pick_expiration(expirations, days_ahead_target: int):
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
    idx = np.abs(arr - spot).argmin()
    return float(arr[idx])

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df

def _get_spot(symbol: str) -> float:
    """Get latest close price robustly."""
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

def _get_option_chain(symbol: str, exp: str):
    """Get option chain robustly, returns (calls_df, puts_df) or (None, None)."""
    try:
        tk = yf.Ticker(symbol)
        chain = tk.option_chain(exp)
        calls = _flatten_columns(chain.calls.copy())
        puts  = _flatten_columns(chain.puts.copy())
        return calls, puts
    except Exception:
        return None, None

def _get_atm_iv_for_exp(symbol: str, spot: float, exp: str):
    """Return (ATM IV annualized, DTE calendar days)."""
    # Compute DTE
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

    strikes = calls["strike"].values
    atm = _nearest_atm_strike(strikes, spot)
    if np.isnan(atm):
        return np.nan, dte

    call_row = calls[calls["strike"] == atm]
    put_row  = puts[puts["strike"] == atm] if "strike" in puts.columns else pd.DataFrame()

    if call_row.empty:
        return np.nan, dte

    call_row = call_row.iloc[0]
    put_row  = put_row.iloc[0] if not put_row.empty else None

    # Try reported IV first
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

    # Fall back to backing out IV from call mid
    try:
        bid  = call_row["bid"]  if "bid"       in call_row.index else None
        ask  = call_row["ask"]  if "ask"       in call_row.index else None
        last = call_row["lastPrice"] if "lastPrice" in call_row.index else None
        call_mid = _mid(bid, ask, last)
        atm_iv = _implied_vol_from_mid(call_mid, spot, float(atm), T_years, r=0.0, cp="c")
        if pd.notna(atm_iv) and atm_iv > 0:
            return float(atm_iv), int(dte)
    except Exception:
        pass

    return np.nan, int(dte)

def _expected_move_dollars(spot: float, iv_ann: float, dte_calendar: int):
    if pd.isna(iv_ann) or dte_calendar <= 0 or pd.isna(spot):
        return np.nan
    T_years = dte_calendar / 365.0
    return float(iv_ann * math.sqrt(T_years) * spot)

def _fmt_pct(x):
    return "" if (x is None or pd.isna(x)) else f"{x:.2f}%"

def _fmt_num(x, nd=2):
    return "" if (x is None or pd.isna(x)) else f"{x:.{nd}f}"

def _build_implied_move_table(tickers):
    rows = []
    notes = []
    rate_limited_symbols = []

    for symbol in tickers:
        # --- Spot price ---
        try:
            spot = _get_spot(symbol)
        except YFRateLimitError:
            rate_limited_symbols.append(symbol)
            rows.append([symbol, "rate-limited", "", "", "", "", ""])
            continue
        except Exception:
            rows.append([symbol, "error", "", "", "", "", ""])
            continue

        if pd.isna(spot):
            rows.append([symbol, "no data", "", "", "", "", ""])
            continue

        # --- Expirations ---
        try:
            tk = yf.Ticker(symbol)
            expirations = tk.options or []
        except YFRateLimitError:
            rate_limited_symbols.append(symbol)
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
                rate_limited_symbols.append(symbol)

        if exp_1m:
            try:
                iv_1m, dte_1m = _get_atm_iv_for_exp(symbol, spot, exp_1m)
                time.sleep(0.2)
            except YFRateLimitError:
                rate_limited_symbols.append(symbol)

        # --- Expected moves ---
        move1w_d = _expected_move_dollars(spot, iv_1w, dte_1w)
        move1m_d = _expected_move_dollars(spot, iv_1m, dte_1m)
        move1w_pct = (move1w_d / spot * 100.0) if pd.notna(move1w_d) else np.nan
        move1m_pct = (move1m_d / spot * 100.0) if pd.notna(move1m_d) else np.nan

        atm_iv_display = iv_1m if pd.notna(iv_1m) else iv_1w
        atm_iv_text = _fmt_num(atm_iv_display, 4) if pd.notna(atm_iv_display) else ""
        if symbol in rate_limited_symbols and not atm_iv_text:
            atm_iv_text = "rate-limited"

        rows.append([
            symbol,
            _fmt_num(spot, 2),
            atm_iv_text,
            _fmt_num(move1w_d, 2),
            _fmt_pct(move1w_pct),
            _fmt_num(move1m_d, 2),
            _fmt_pct(move1m_pct),
        ])

    if rate_limited_symbols:
        notes.append(
            f"Yahoo API rate-limited for: {', '.join(sorted(set(rate_limited_symbols)))}. "
            "Cached data will auto-refresh after ~5 minutes."
        )

    df = pd.DataFrame(
        rows,
        columns=["Ticker", "Price", "ATM IV", "1W Move ($)", "1W Move (%)", "1M Move ($)", "1M Move (%)"]
    )
    return df, notes