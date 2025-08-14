# option_projects.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from datetime import datetime, timezone
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq

# =========================
# Existing: Black–Scholes
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
        S = st.number_input("Asset Price (S)", value=100.0, step=1.0)
    with c2:
        K = st.number_input("Strike Price (K)", value=100.0, step=1.0)
    with c3:
        T = st.number_input("Time to Maturity (years)", value=1.0, step=0.1)

    c4, c5, c6 = st.columns(3)
    with c4:
        r = st.number_input("Risk-Free Rate (%)", value=2.0, step=0.1) / 100
    with c5:
        sigma = st.number_input("Volatility (%)", value=20.0, step=0.1) / 100
    with c6:
        plot_type = st.radio(
            "Heatmap Type",
            options=["Option Value", "Call P&L"],
            horizontal=True,
        )

    st.markdown("#### 💸 Optional Trade Inputs")
    c7, c8 = st.columns(2)
    with c7:
        call_price_paid = st.number_input("Call Purchase Price", value=0.0)
    with c8:
        put_price_paid = st.number_input("Put Purchase Price", value=0.0)

    # ---- Pricing ----
    call_price = black_scholes(S, K, T, r, sigma, "call")
    put_price = black_scholes(S, K, T, r, sigma, "put")

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
    S_range = np.linspace(S * 0.8, S * 1.2, 30)
    sigma_range = np.linspace(sigma * 0.5, sigma * 1.5, 30)
    heatmap = np.zeros((len(S_range), len(sigma_range)))

    for i, s_val in enumerate(S_range):
        for j, sig_val in enumerate(sigma_range):
            price = black_scholes(s_val, K, T, r, sig_val, "call")
            heatmap[i, j] = (
                price - call_price_paid if plot_type == "Call P&L" and call_price_paid > 0 else price
            )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap,
        xticklabels=np.round(sigma_range, 2),
        yticklabels=np.round(S_range, 2),
        cmap="RdYlGn" if plot_type == "Call P&L" and call_price_paid > 0 else "YlGnBu",
        ax=ax,
    )
    plt.xlabel("Volatility")
    plt.ylabel("Asset Price")
    plt.title(f"{plot_type} Heatmap")
    st.pyplot(fig)



# =========================
# Existing: Monte Carlo placeholder
# =========================

def run_monte_carlo():
    st.subheader("🎲 Monte Carlo Simulation for Option Pricing")
    st.info("This section is under construction.")


# =========================
# NEW: Market-Implied Move (ATM IV)
# =========================

_DEFAULT_TICKERS = ["USO", "TLT", "UUP", "QQQ"]
_TARGETS = {"1W": 7, "1M": 30}  # calendar days

def run_implied_move():
    st.subheader("📈 Market-Implied Move (from ATM IV)")
    st.caption(
        "Method: ATM IV (nearest ~7d and ~30d expiries) × sqrt(time in years) × spot. "
        "If IV is missing, it’s backed out from the ATM call mid via Black–Scholes. "
        "Calendar days used for DTE."
    )

    # --- Controls
    col1, col2 = st.columns([2, 1])
    with col1:
        tickers_text = st.text_input("Tickers (comma-separated)", value=",".join(_DEFAULT_TICKERS))
    with col2:
        refresh = st.button("Refresh")

    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

    if not tickers:
        st.warning("Please enter at least one ticker.")
        return

    with st.spinner("Fetching option chains and computing implied moves..."):
        df = _build_implied_move_table(tickers)

    st.dataframe(df, use_container_width=True)
    st.caption(
        "Notes: ATM IV column displays the ~30-day expiry IV when available (fallback to ~7-day). "
        "1W/1M moves use horizon-specific IVs and DTE. Time = DTE/365."
    )
    st.caption(f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

# Backward-compatible alias if main app calls this name
def run_implied_move_table():
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
    if pd.notna(bid) and pd.notna(ask) and (bid > 0 or ask > 0):
        return (bid + ask) / 2.0
    return last if pd.notna(last) else np.nan

def _nearest_atm_strike(strikes, spot):
    arr = np.asarray(strikes, dtype=float)
    if arr.size == 0:
        return np.nan
    idx = np.abs(arr - spot).argmin()
    return float(arr[idx])

def _get_atm_iv_for_exp(tk: yf.Ticker, spot: float, exp: str):
    """Return (ATM IV ann, DTE calendar days) for a given expiration."""
    try:
        chain = tk.option_chain(exp)
    except Exception:
        return np.nan, 0

    calls, puts = chain.calls.copy(), chain.puts.copy()
    if calls.empty or puts.empty:
        return np.nan, 0

    strikes = calls["strike"].values
    atm = _nearest_atm_strike(strikes, spot)
    if np.isnan(atm):
        return np.nan, 0

    call_row = calls[calls["strike"] == atm]
    put_row  = puts[puts["strike"] == atm]
    if call_row.empty or put_row.empty:
        return np.nan, 0

    call_row = call_row.iloc[0]
    put_row  = put_row.iloc[0]

    # Prefer reported IVs if present
    ivs = []
    for col in ("impliedVolatility", "impliedVol"):
        if col in call_row and pd.notna(call_row[col]): ivs.append(float(call_row[col]))
        if col in put_row  and pd.notna(put_row[col]):  ivs.append(float(put_row[col]))
    if ivs:
        atm_iv = float(np.nanmean(ivs))
    else:
        # Back out IV from call mid
        call_mid = _mid(call_row.get("bid"), call_row.get("ask"), call_row.get("lastPrice"))
        exp_dt = _to_dt(exp)
        now = datetime.now(timezone.utc)
        dte = max((exp_dt - now).days, 0)
        T_years = max(dte, 1) / 365.0  # guard against 0
        atm_iv = _implied_vol_from_mid(call_mid, spot, float(atm), T_years, r=0.0, cp="c")

    # Compute DTE for return
    exp_dt = _to_dt(exp)
    now = datetime.now(timezone.utc)
    dte = max((exp_dt - now).days, 0)

    return float(atm_iv) if pd.notna(atm_iv) else np.nan, int(dte)

def _expected_move_dollars(spot: float, iv_ann: float, dte_calendar: int):
    if pd.isna(iv_ann) or dte_calendar <= 0:
        return np.nan
    T_years = dte_calendar / 365.0
    return float(iv_ann * math.sqrt(T_years) * spot)

def _fmt_pct(x):
    return "" if pd.isna(x) else f"{x:.2f}%"

def _fmt_num(x, nd=2):
    return "" if pd.isna(x) else f"{x:.{nd}f}"

def _build_implied_move_table(tickers):
    rows = []
    for ticker in tickers:
        tk = yf.Ticker(ticker)

        # Spot price (last close)
        hist = tk.history(period="5d", auto_adjust=True)
        if hist.empty:
            rows.append([ticker, "", "", "", "", "", ""])
            continue
        spot = float(hist["Close"].iloc[-1])

        expirations = tk.options
        if not expirations:
            rows.append([ticker, _fmt_num(spot), "", "", "", "", ""])
            continue

        # Pick expiries nearest ~7d and ~30d
        exp_1w = _pick_expiration(expirations, _TARGETS["1W"])
        exp_1m = _pick_expiration(expirations, _TARGETS["1M"])

        # ATM IVs + DTE
        iv_1w, dte_1w = (np.nan, 0)
        iv_1m, dte_1m = (np.nan, 0)
        if exp_1w:
            iv_1w, dte_1w = _get_atm_iv_for_exp(tk, spot, exp_1w)
        if exp_1m:
            iv_1m, dte_1m = _get_atm_iv_for_exp(tk, spot, exp_1m)

        # Expected moves
        move1w_d = _expected_move_dollars(spot, iv_1w, dte_1w)
        move1m_d = _expected_move_dollars(spot, iv_1m, dte_1m)

        move1w_pct = (move1w_d / spot * 100.0) if pd.notna(move1w_d) else np.nan
        move1m_pct = (move1m_d / spot * 100.0) if pd.notna(move1m_d) else np.nan

        # Display ATM IV as the ~30d IV when available (fallback to ~7d)
        atm_iv_display = iv_1m if pd.notna(iv_1m) else iv_1w

        rows.append([
            ticker,
            _fmt_num(spot, 2),
            "" if pd.isna(atm_iv_display) else f"{atm_iv_display:.4f}",
            _fmt_num(move1w_d, 2),
            _fmt_pct(move1w_pct),
            _fmt_num(move1m_d, 2),
            _fmt_pct(move1m_pct),
        ])

    df = pd.DataFrame(
        rows,
        columns=["Ticker", "Price", "ATM IV", "1W Move ($)", "1W Move (%)", "1M Move ($)", "1M Move (%)"]
    )
    return df
