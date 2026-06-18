# fear_greed.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timezone, timedelta

# =========================
# Data Fetching
# =========================

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_fear_greed_data():
    """Compute a multi-factor Fear & Greed Index from market data."""
    tickers = {
        "SPY":  "^GSPC",
        "VIX":  "^VIX",
        "HYG":  "HYG",
        "LQD":  "LQD",
        "TLT":  "TLT",
        "GLD":  "GLD",
        "UUP":  "UUP",
        "IVV":  "IVV",
        "JNK":  "JNK",
    }

    data = {}
    for name, ticker in tickers.items():
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="1y", auto_adjust=True)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            if not hist.empty and "Close" in hist.columns:
                s = hist["Close"].dropna()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                data[name] = s
        except Exception:
            pass

    return data

def _compute_fear_greed(data: dict):
    """
    Compute Fear & Greed score (0-100) from multiple market signals.
    50 = Neutral, 0 = Extreme Fear, 100 = Extreme Greed
    """
    scores = {}
    details = {}

    # 1. VIX — Volatility (Fear indicator)
    if "VIX" in data and len(data["VIX"]) > 20:
        vix = data["VIX"]
        vix_now = vix.iloc[-1]
        vix_52w_low  = vix.tail(252).min()
        vix_52w_high = vix.tail(252).max()
        # Low VIX = Greed, High VIX = Fear
        vix_score = 100 * (vix_52w_high - vix_now) / (vix_52w_high - vix_52w_low + 1e-9)
        vix_score = max(0, min(100, vix_score))
        scores["VIX (Volatility)"] = vix_score
        details["VIX (Volatility)"] = {
            "value": f"{vix_now:.1f}",
            "score": vix_score,
            "signal": "Fear" if vix_score < 40 else ("Greed" if vix_score > 60 else "Neutral"),
            "description": f"VIX at {vix_now:.1f} | 52W Range: {vix_52w_low:.1f}–{vix_52w_high:.1f}"
        }

    # 2. Market Momentum (S&P 500 vs 125-day MA)
    spy = data.get("SPY") or data.get("IVV")
    if spy is not None and len(spy) > 125:
        ma125 = spy.rolling(125).mean()
        spy_now = spy.iloc[-1]
        ma_now  = ma125.iloc[-1]
        pct_above = (spy_now / ma_now - 1) * 100
        mom_score = 50 + pct_above * 3
        mom_score = max(0, min(100, mom_score))
        scores["Market Momentum"] = mom_score
        details["Market Momentum"] = {
            "value": f"{pct_above:+.1f}% vs 125D MA",
            "score": mom_score,
            "signal": "Fear" if mom_score < 40 else ("Greed" if mom_score > 60 else "Neutral"),
            "description": f"S&P 500 is {abs(pct_above):.1f}% {'above' if pct_above >= 0 else 'below'} its 125-day moving average"
        }

    # 3. Stock Price Breadth (52W highs vs lows proxy via SPY momentum)
    if spy is not None and len(spy) > 252:
        high_52w = spy.tail(252).max()
        low_52w  = spy.tail(252).min()
        breadth_score = 100 * (spy.iloc[-1] - low_52w) / (high_52w - low_52w + 1e-9)
        breadth_score = max(0, min(100, breadth_score))
        scores["Price Breadth"] = breadth_score
        details["Price Breadth"] = {
            "value": f"{breadth_score:.0f}% of 52W range",
            "score": breadth_score,
            "signal": "Fear" if breadth_score < 40 else ("Greed" if breadth_score > 60 else "Neutral"),
            "description": f"S&P 500 at {breadth_score:.0f}% of its 52-week high-low range"
        }

    # 4. Credit Demand (HYG/LQD ratio — high yield vs investment grade)
    if "HYG" in data and "LQD" in data:
        hyg = data["HYG"]; lqd = data["LQD"]
        if len(hyg) > 20 and len(lqd) > 20:
            ratio = (hyg / lqd).dropna()
            ratio_now  = ratio.iloc[-1]
            ratio_ma20 = ratio.tail(20).mean()
            ratio_std  = ratio.tail(60).std() if len(ratio) > 60 else ratio.std()
            z_score    = (ratio_now - ratio_ma20) / (ratio_std + 1e-9)
            credit_score = 50 + z_score * 15
            credit_score = max(0, min(100, credit_score))
            scores["Credit Demand"] = credit_score
            details["Credit Demand"] = {
                "value": f"HYG/LQD ratio: {ratio_now:.4f}",
                "score": credit_score,
                "signal": "Fear" if credit_score < 40 else ("Greed" if credit_score > 60 else "Neutral"),
                "description": "High HYG/LQD = investors prefer risky bonds (Greed). Low = flight to safety (Fear)."
            }

    # 5. Safe Haven Demand (Gold vs Stocks)
    if "GLD" in data and spy is not None:
        gld = data["GLD"]
        if len(gld) > 20 and len(spy) > 20:
            gld_ret_20d = (gld.iloc[-1] / gld.iloc[-20] - 1) * 100
            spy_ret_20d = (spy.iloc[-1] / spy.iloc[-20] - 1) * 100
            diff = spy_ret_20d - gld_ret_20d
            safe_score = 50 + diff * 2
            safe_score = max(0, min(100, safe_score))
            scores["Safe Haven Demand"] = safe_score
            details["Safe Haven Demand"] = {
                "value": f"Stocks vs Gold 20D: {diff:+.1f}%",
                "score": safe_score,
                "signal": "Fear" if safe_score < 40 else ("Greed" if safe_score > 60 else "Neutral"),
                "description": f"Stocks outperforming gold by {diff:+.1f}% over 20 days. Positive = Greed (risk appetite). Negative = Fear."
            }

    # 6. Junk Bond Demand (JNK momentum)
    if "JNK" in data and len(data["JNK"]) > 20:
        jnk = data["JNK"]
        jnk_ret = (jnk.iloc[-1] / jnk.iloc[-20] - 1) * 100
        jnk_score = 50 + jnk_ret * 5
        jnk_score = max(0, min(100, jnk_score))
        scores["Junk Bond Demand"] = jnk_score
        details["Junk Bond Demand"] = {
            "value": f"JNK 20D return: {jnk_ret:+.1f}%",
            "score": jnk_score,
            "signal": "Fear" if jnk_score < 40 else ("Greed" if jnk_score > 60 else "Neutral"),
            "description": "Rising junk bond prices = investors comfortable with risk (Greed). Falling = risk aversion (Fear)."
        }

    # 7. Dollar Demand (UUP — inverse of risk appetite)
    if "UUP" in data and len(data["UUP"]) > 20:
        uup = data["UUP"]
        uup_ret = (uup.iloc[-1] / uup.iloc[-20] - 1) * 100
        dollar_score = 50 - uup_ret * 5
        dollar_score = max(0, min(100, dollar_score))
        scores["Dollar Demand"] = dollar_score
        details["Dollar Demand"] = {
            "value": f"UUP 20D return: {uup_ret:+.1f}%",
            "score": dollar_score,
            "signal": "Fear" if dollar_score < 40 else ("Greed" if dollar_score > 60 else "Neutral"),
            "description": "Rising dollar = flight to safety (Fear). Falling dollar = risk-on (Greed)."
        }

    if not scores:
        return 50, {}, {}

    overall = np.mean(list(scores.values()))
    return overall, scores, details

def _score_to_label(score):
    if score >= 75:   return "Extreme Greed", "#2ecc71"
    elif score >= 60: return "Greed",          "#27ae60"
    elif score >= 45: return "Neutral",         "#3498db"
    elif score >= 25: return "Fear",            "#e67e22"
    else:             return "Extreme Fear",    "#e74c3c"

# =========================
# Plotting
# =========================

def _dark_ax(ax):
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

def _plot_gauge(score: float, label: str, color: str):
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(polar=False))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.axis("off")

    # Background arc zones
    zones = [
        (0,   25,  "#e74c3c", "Extreme Fear"),
        (25,  45,  "#e67e22", "Fear"),
        (45,  55,  "#3498db", "Neutral"),
        (55,  75,  "#27ae60", "Greed"),
        (75,  100, "#2ecc71", "Extreme Greed"),
    ]

    import matplotlib.patches as mpatches
    from matplotlib.patches import Arc, FancyArrow

    # Draw color bar
    for i, (start, end, c, lbl) in enumerate(zones):
        x_start = start / 10
        x_end   = end / 10
        ax.barh(2, x_end - x_start, left=x_start, height=1.2, color=c, alpha=0.7)
        ax.text((x_start + x_end) / 2, 1.2, lbl, ha="center", va="top",
                color="white", fontsize=7, fontweight="bold")

    # Needle
    needle_x = score / 10
    ax.annotate("", xy=(needle_x, 3.2), xytext=(needle_x, 2.6),
                arrowprops=dict(arrowstyle="->", color="white", lw=2.5))

    # Score display
    ax.text(5, 4.2, f"{score:.0f}", ha="center", va="center",
            color=color, fontsize=36, fontweight="bold")
    ax.text(5, 3.6, label, ha="center", va="center",
            color=color, fontsize=14, fontweight="bold")
    ax.text(5, 0.3, "0 = Extreme Fear                                                    100 = Extreme Greed",
            ha="center", va="center", color="#888", fontsize=7)

    plt.tight_layout(); return fig

def _plot_component_bars(scores: dict):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0e1117"); _dark_ax(ax)

    labels = list(scores.keys())
    values = list(scores.values())
    colors = ["#2ecc71" if v >= 55 else ("#e74c3c" if v <= 45 else "#3498db") for v in values]

    bars = ax.barh(labels, values, color=colors, alpha=0.85, height=0.6)
    ax.axvline(50, color="white", linewidth=1, alpha=0.4, linestyle="--")
    ax.axvline(25, color="#e74c3c", linewidth=0.5, alpha=0.3, linestyle=":")
    ax.axvline(75, color="#2ecc71", linewidth=0.5, alpha=0.3, linestyle=":")
    ax.set_xlim(0, 100)

    for bar, val in zip(bars, values):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}", va="center", color="white", fontsize=8)

    ax.set_title("Fear & Greed Components", color="white", fontsize=11)
    ax.set_xlabel("Score (0=Extreme Fear, 100=Extreme Greed)", color="white", fontsize=9)
    plt.tight_layout(); return fig

# =========================
# Main Page
# =========================

def run_fear_greed():
    st.subheader("😨 Fear & Greed Index")
    st.markdown(
        "A **multi-factor market sentiment indicator** built from 7 real market signals. "
        "Inspired by CNN's Fear & Greed Index but calculated live from market data. "
        "Use as a contrarian signal — **Extreme Fear = potential buy, Extreme Greed = potential sell.**"
    )

    with st.expander("📖 How It Works", expanded=False):
        st.markdown("""
**7 Components (equal weighted):**

| Component | Greed Signal | Fear Signal |
|-----------|-------------|------------|
| **VIX (Volatility)** | VIX low (< 15) | VIX high (> 25) |
| **Market Momentum** | S&P 500 above 125D MA | S&P 500 below 125D MA |
| **Price Breadth** | Near 52W highs | Near 52W lows |
| **Credit Demand** | HYG outperforming LQD | LQD outperforming HYG |
| **Safe Haven Demand** | Stocks outperforming Gold | Gold outperforming Stocks |
| **Junk Bond Demand** | JNK rising | JNK falling |
| **Dollar Demand** | Dollar falling (risk-on) | Dollar rising (flight to safety) |

**Score Interpretation:**
- **75–100** = Extreme Greed → markets may be overheated → consider taking profits
- **55–74** = Greed → positive sentiment → momentum favors longs
- **45–54** = Neutral → no clear sentiment signal
- **25–44** = Fear → negative sentiment → contrarian buy opportunity
- **0–24** = Extreme Fear → markets oversold → strong contrarian buy signal

**How to use with your other tools:**
- **Extreme Fear + Risk-Off Regime** = strongest sell signal
- **Extreme Fear + Mean Reversion oversold** = strongest contrarian buy signal
- **Extreme Greed + Overbought RSI** = reduce positions, take profits
- **Greed + Risk-On Regime** = momentum favors staying long

**Historical context:**
- COVID crash (March 2020): Index hit 2 (Extreme Fear) → SPY +65% over next 12 months
- 2021 peak: Index hit 92 (Extreme Greed) → SPY -20% over next 12 months
- SVB crisis (March 2023): Index hit 22 (Extreme Fear) → SPY +30% over next 12 months
""")

    # ---- Fetch & Compute ----
    with st.spinner("Computing Fear & Greed Index from live market data..."):
        market_data = _fetch_fear_greed_data()
        score, component_scores, details = _compute_fear_greed(market_data)

    label, color = _score_to_label(score)

    # ---- Gauge ----
    st.markdown("### 🎯 Current Fear & Greed Score")
    fig_gauge = _plot_gauge(score, label, color)
    st.pyplot(fig_gauge)

    # ---- What it means ----
    if score >= 75:
        st.error(f"""
**⚠️ Extreme Greed ({score:.0f}) — Markets May Be Overheated**
- Sentiment is euphoric — historically a contrarian sell signal
- Consider taking partial profits on long positions
- Avoid adding new longs at elevated prices
- Watch for catalyst that could trigger sentiment reversal
""")
    elif score >= 55:
        st.success(f"""
**🟢 Greed ({score:.0f}) — Positive Market Sentiment**
- Risk appetite is elevated — momentum favors longs
- Consistent with Risk-On market regime
- Good environment for growth stocks and cyclicals
- Stay long but watch for sentiment shift
""")
    elif score >= 45:
        st.info(f"""
**⚪ Neutral ({score:.0f}) — Mixed Sentiment**
- No clear directional bias from sentiment
- Wait for sentiment to shift before making large directional bets
- Good time to rebalance and review positions
""")
    elif score >= 25:
        st.warning(f"""
**🟡 Fear ({score:.0f}) — Negative Sentiment**
- Risk aversion elevated — markets pricing in uncertainty
- Defensive positioning favored
- Watch for oversold conditions in Mean Reversion Scanner
- Contrarian opportunities may be forming
""")
    else:
        st.error(f"""
**🔴 Extreme Fear ({score:.0f}) — Potential Contrarian Buy**
- Markets in panic — historically one of the best buying opportunities
- Check Mean Reversion Scanner for oversold confirmation
- Consider scaling into positions if fundamentals are intact
- High conviction long setup if Market Regime also shows oversold
""")

    # ---- Component Breakdown ----
    st.markdown("### 📊 Component Breakdown")
    if component_scores:
        fig_bars = _plot_component_bars(component_scores)
        st.pyplot(fig_bars)

    # ---- Detail Cards ----
    st.markdown("### 🔍 Component Details")
    if details:
        cols = st.columns(2)
        for i, (name, detail) in enumerate(details.items()):
            with cols[i % 2]:
                sig_color = "#2ecc71" if detail["signal"] == "Greed" else ("#e74c3c" if detail["signal"] == "Fear" else "#3498db")
                st.markdown(
                    f"""<div style="border:1px solid {sig_color}44;border-radius:8px;
                    padding:12px;margin-bottom:8px;background:{sig_color}11;">
                    <div style="font-weight:700;color:white;">{name}</div>
                    <div style="font-size:1.3rem;font-weight:700;color:{sig_color};">{detail['score']:.0f}/100</div>
                    <div style="color:#aaa;font-size:0.8rem;">{detail['value']}</div>
                    <div style="color:{sig_color};font-weight:600;font-size:0.85rem;">{detail['signal']}</div>
                    <div style="color:#888;font-size:0.78rem;margin-top:4px;">{detail['description']}</div>
                    </div>""",
                    unsafe_allow_html=True
                )

    # ---- Historical Context ----
    st.markdown("### 📈 Historical Reference Points")
    hist_data = {
        "Event": ["COVID Crash", "2021 Peak", "2022 Rate Shock", "SVB Crisis", "AI Rally Peak", "Current"],
        "Date": ["Mar 2020", "Nov 2021", "Oct 2022", "Mar 2023", "Jul 2023", "Jun 2026"],
        "Score": [2, 92, 15, 22, 83, f"{score:.0f}"],
        "Label": ["Extreme Fear", "Extreme Greed", "Extreme Fear", "Extreme Fear", "Extreme Greed", label],
        "SPY 12M After": ["+65%", "-20%", "+18%", "+30%", "+8%", "TBD"],
    }
    st.dataframe(pd.DataFrame(hist_data), use_container_width=True)

    st.caption(
        f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | "
        "Computed from live market data via Yahoo Finance | "
        "Inspired by CNN Fear & Greed Index methodology | Refreshes every 5 minutes"
    )
