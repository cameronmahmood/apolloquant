# pairs_trading.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from datetime import datetime, timezone, timedelta

# =========================
# Preset Popular Pairs
# =========================

PRESET_PAIRS = {
    # ── Most Popular ──
    "KO / PEP — Coca-Cola vs Pepsi":                   ("KO",   "PEP"),
    "XOM / CVX — ExxonMobil vs Chevron":               ("XOM",  "CVX"),
    "GLD / SLV — Gold vs Silver":                      ("GLD",  "SLV"),
    "SPY / QQQ — S&P 500 vs Nasdaq":                   ("SPY",  "QQQ"),
    "JPM / BAC — JPMorgan vs Bank of America":         ("JPM",  "BAC"),
    "V / MA — Visa vs Mastercard":                     ("V",    "MA"),
    "NVDA / AMD — NVIDIA vs AMD":                      ("NVDA", "AMD"),
    "GS / MS — Goldman Sachs vs Morgan Stanley":       ("GS",   "MS"),
    "HD / LOW — Home Depot vs Lowe's":                 ("HD",   "LOW"),
    "UPS / FDX — UPS vs FedEx":                       ("UPS",  "FDX"),
    # ── Consumer Staples ──
    "WMT / TGT — Walmart vs Target":                   ("WMT",  "TGT"),
    "PG / CL — Procter & Gamble vs Colgate":           ("PG",   "CL"),
    "MCD / YUM — McDonald's vs Yum Brands":            ("MCD",  "YUM"),
    "SBUX / CMG — Starbucks vs Chipotle":              ("SBUX", "CMG"),
    "PM / MO — Philip Morris vs Altria":               ("PM",   "MO"),
    # ── Financials ──
    "BLK / SCHW — BlackRock vs Charles Schwab":        ("BLK",  "SCHW"),
    "WFC / USB — Wells Fargo vs US Bancorp":           ("WFC",  "USB"),
    "C / BAC — Citigroup vs Bank of America":          ("C",    "BAC"),
    "AXP / DFS — AmEx vs Discover":                   ("AXP",  "DFS"),
    # ── Energy ──
    "USO / XLE — Oil ETF vs Energy Sector":            ("USO",  "XLE"),
    "COP / EOG — ConocoPhillips vs EOG Resources":     ("COP",  "EOG"),
    "SLB / HAL — Schlumberger vs Halliburton":         ("SLB",  "HAL"),
    "BP / SHEL — BP vs Shell":                        ("BP",   "SHEL"),
    # ── Technology ──
    "MSFT / GOOGL — Microsoft vs Alphabet":            ("MSFT", "GOOGL"),
    "AAPL / MSFT — Apple vs Microsoft":                ("AAPL", "MSFT"),
    "META / SNAP — Meta vs Snap":                      ("META", "SNAP"),
    "INTC / AMD — Intel vs AMD":                       ("INTC", "AMD"),
    "ORCL / SAP — Oracle vs SAP":                     ("ORCL", "SAP"),
    "UBER / LYFT — Uber vs Lyft":                      ("UBER", "LYFT"),
    "NFLX / DIS — Netflix vs Disney":                  ("NFLX", "DIS"),
    # ── Healthcare ──
    "JNJ / PFE — Johnson & Johnson vs Pfizer":         ("JNJ",  "PFE"),
    "ABBV / MRK — AbbVie vs Merck":                   ("ABBV", "MRK"),
    "UNH / CVS — UnitedHealth vs CVS":                 ("UNH",  "CVS"),
    "BMY / LLY — Bristol-Myers vs Eli Lilly":          ("BMY",  "LLY"),
    "AMGN / BIIB — Amgen vs Biogen":                   ("AMGN", "BIIB"),
    "MDT / BSX — Medtronic vs Boston Scientific":      ("MDT",  "BSX"),
    # ── Industrials ──
    "BA / RTX — Boeing vs Raytheon":                   ("BA",   "RTX"),
    "CAT / DE — Caterpillar vs Deere":                 ("CAT",  "DE"),
    "HON / MMM — Honeywell vs 3M":                    ("HON",  "MMM"),
    "LMT / NOC — Lockheed vs Northrop":               ("LMT",  "NOC"),
    "GE / ETN — GE vs Eaton":                         ("GE",   "ETN"),
    # ── ETF Pairs ──
    "SPY / IWM — S&P 500 vs Russell 2000":             ("SPY",  "IWM"),
    "QQQ / IWM — Nasdaq vs Russell 2000":              ("QQQ",  "IWM"),
    "XLF / XLK — Financials vs Tech":                  ("XLF",  "XLK"),
    "XLE / XLU — Energy vs Utilities":                 ("XLE",  "XLU"),
    "XLV / XLP — Healthcare vs Staples":               ("XLV",  "XLP"),
    "GLD / GDX — Gold vs Gold Miners":                 ("GLD",  "GDX"),
    "TLT / IEF — 20Y vs 10Y Treasury":                ("TLT",  "IEF"),
    "TLT / HYG — Long Treasuries vs High Yield":       ("TLT",  "HYG"),
    "EEM / EFA — Emerging vs Developed Markets":       ("EEM",  "EFA"),
    "DIA / SPY — Dow vs S&P 500":                      ("DIA",  "SPY"),
    # ── Commodities ──
    "USO / UNG — Oil vs Natural Gas":                  ("USO",  "UNG"),
    "GLD / USO — Gold vs Oil":                         ("GLD",  "USO"),
    "CORN / WEAT — Corn vs Wheat":                    ("CORN", "WEAT"),
    "PALL / PPLT — Palladium vs Platinum":             ("PALL", "PPLT"),
    # ── Retail ──
    "AMZN / EBAY — Amazon vs eBay":                   ("AMZN", "EBAY"),
    "NKE / ADDYY — Nike vs Adidas":                   ("NKE",  "ADDYY"),
    "COST / BJ — Costco vs BJ's Wholesale":           ("COST", "BJ"),
    "ROST / TJX — Ross Stores vs TJX":                ("ROST", "TJX"),
    # ── Autos ──
    "TSLA / GM — Tesla vs GM":                         ("TSLA", "GM"),
    "TSLA / F — Tesla vs Ford":                        ("TSLA", "F"),
    "GM / F — GM vs Ford":                             ("GM",   "F"),
    # ── Real Estate ──
    "SPG / O — Simon Property vs Realty Income":       ("SPG",  "O"),
    "AMT / CCI — American Tower vs Crown Castle":      ("AMT",  "CCI"),
    # ── Airlines ──
    "DAL / UAL — Delta vs United":                     ("DAL",  "UAL"),
    "AAL / LUV — American vs Southwest":               ("AAL",  "LUV"),
    # ── Regional Banks ──
    "RF / KEY — Regions vs KeyCorp":                   ("RF",   "KEY"),
    "FITB / HBAN — Fifth Third vs Huntington":         ("FITB", "HBAN"),
    # ── Custom ──
    "Custom Pair":                                     ("",     ""),
}

PAIR_CATEGORIES = {
    "⭐ Most Popular":   ["KO / PEP — Coca-Cola vs Pepsi","XOM / CVX — ExxonMobil vs Chevron","GLD / SLV — Gold vs Silver","SPY / QQQ — S&P 500 vs Nasdaq","JPM / BAC — JPMorgan vs Bank of America","V / MA — Visa vs Mastercard","NVDA / AMD — NVIDIA vs AMD","GS / MS — Goldman Sachs vs Morgan Stanley","HD / LOW — Home Depot vs Lowe's","UPS / FDX — UPS vs FedEx"],
    "🛒 Consumer":       ["KO / PEP — Coca-Cola vs Pepsi","WMT / TGT — Walmart vs Target","PG / CL — Procter & Gamble vs Colgate","MCD / YUM — McDonald's vs Yum Brands","SBUX / CMG — Starbucks vs Chipotle","PM / MO — Philip Morris vs Altria","AMZN / EBAY — Amazon vs eBay","NKE / ADDYY — Nike vs Adidas","COST / BJ — Costco vs BJ's Wholesale","ROST / TJX — Ross Stores vs TJX"],
    "🏦 Financials":     ["JPM / BAC — JPMorgan vs Bank of America","GS / MS — Goldman Sachs vs Morgan Stanley","V / MA — Visa vs Mastercard","BLK / SCHW — BlackRock vs Charles Schwab","WFC / USB — Wells Fargo vs US Bancorp","C / BAC — Citigroup vs Bank of America","AXP / DFS — AmEx vs Discover","RF / KEY — Regions vs KeyCorp","FITB / HBAN — Fifth Third vs Huntington"],
    "⚡ Energy":          ["XOM / CVX — ExxonMobil vs Chevron","USO / XLE — Oil ETF vs Energy Sector","COP / EOG — ConocoPhillips vs EOG Resources","SLB / HAL — Schlumberger vs Halliburton","BP / SHEL — BP vs Shell"],
    "💻 Technology":     ["NVDA / AMD — NVIDIA vs AMD","MSFT / GOOGL — Microsoft vs Alphabet","AAPL / MSFT — Apple vs Microsoft","META / SNAP — Meta vs Snap","INTC / AMD — Intel vs AMD","ORCL / SAP — Oracle vs SAP","UBER / LYFT — Uber vs Lyft","NFLX / DIS — Netflix vs Disney"],
    "🏥 Healthcare":     ["JNJ / PFE — Johnson & Johnson vs Pfizer","ABBV / MRK — AbbVie vs Merck","UNH / CVS — UnitedHealth vs CVS","BMY / LLY — Bristol-Myers vs Eli Lilly","AMGN / BIIB — Amgen vs Biogen","MDT / BSX — Medtronic vs Boston Scientific"],
    "🏭 Industrials":    ["BA / RTX — Boeing vs Raytheon","CAT / DE — Caterpillar vs Deere","UPS / FDX — UPS vs FedEx","HON / MMM — Honeywell vs 3M","LMT / NOC — Lockheed vs Northrop","GE / ETN — GE vs Eaton"],
    "📊 ETF Pairs":      ["SPY / QQQ — S&P 500 vs Nasdaq","SPY / IWM — S&P 500 vs Russell 2000","QQQ / IWM — Nasdaq vs Russell 2000","XLF / XLK — Financials vs Tech","XLE / XLU — Energy vs Utilities","XLV / XLP — Healthcare vs Staples","GLD / GDX — Gold vs Gold Miners","TLT / IEF — 20Y vs 10Y Treasury","TLT / HYG — Long Treasuries vs High Yield","EEM / EFA — Emerging vs Developed Markets","DIA / SPY — Dow vs S&P 500"],
    "🛢️ Commodities":    ["GLD / SLV — Gold vs Silver","USO / UNG — Oil vs Natural Gas","GLD / USO — Gold vs Oil","CORN / WEAT — Corn vs Wheat","PALL / PPLT — Palladium vs Platinum"],
    "🚗 Autos":          ["TSLA / GM — Tesla vs GM","TSLA / F — Tesla vs Ford","GM / F — GM vs Ford"],
    "🏢 Real Estate":    ["SPG / O — Simon Property vs Realty Income","AMT / CCI — American Tower vs Crown Castle"],
    "✈️ Airlines":       ["DAL / UAL — Delta vs United","AAL / LUV — American vs Southwest"],
    "🔧 Custom Pair":    ["Custom Pair"],
}

# =========================
# Data Fetching
# =========================

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_pair(ticker1: str, ticker2: str, start: str):
    try:
        data = yf.download([ticker1, ticker2], start=start, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data
        close = close[[ticker1, ticker2]].dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close
    except Exception:
        return None

# =========================
# Analysis Functions
# =========================

def _compute_hedge_ratio(price1, price2):
    X = sm.add_constant(price2)
    model = sm.OLS(price1, X).fit()
    return model.params[price2.name], model.params["const"], model.rsquared

def _compute_spread(price1, price2, hedge_ratio, intercept):
    return price1 - (intercept + hedge_ratio * price2)

def _compute_zscore(spread, window=30):
    mean = spread.rolling(window).mean()
    std  = spread.rolling(window).std()
    return (spread - mean) / std.replace(0, np.nan)

def _run_adf(spread):
    try:
        result = adfuller(spread.dropna())
        return result[0], result[1], result[4]
    except Exception:
        return None, None, None

def _run_coint(price1, price2):
    try:
        _, pvalue, _ = coint(price1, price2)
        return pvalue
    except Exception:
        return None

def _compute_signals(zscore, entry=2.0, exit_z=0.5):
    position = pd.Series(0, index=zscore.index)
    position[zscore < -entry] =  1
    position[zscore >  entry] = -1
    position[zscore.abs() < exit_z] = 0
    return position.ffill().fillna(0)

def _compute_returns(price1, price2, hedge_ratio, position):
    spread_ret = price1.pct_change() - hedge_ratio * price2.pct_change()
    strat_ret  = spread_ret * position.shift()
    cumulative = (1 + strat_ret.fillna(0)).cumprod()
    daily = strat_ret.dropna()
    sharpe   = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else np.nan
    max_dd   = (1 - cumulative / cumulative.cummax()).max()
    win_rate = (daily > 0).mean()
    return cumulative, cumulative.iloc[-1] - 1, sharpe, max_dd, win_rate

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

def _plot_prices(p1, p2, t1, t2):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0e1117"); _dark_ax(ax)
    n1 = p1 / p1.iloc[0] * 100
    n2 = p2 / p2.iloc[0] * 100
    ax.plot(n1.index, n1.values, color="#1f77b4", linewidth=1.5, label=t1)
    ax.plot(n2.index, n2.values, color="#f0b429", linewidth=1.5, label=t2)
    corr = p1.pct_change().corr(p2.pct_change())
    ax.set_title(f"{t1} vs {t2} — Normalized Price (Base=100) | Correlation: {corr:.2f}", color="white", fontsize=11)
    ax.set_ylabel("Normalized Price", color="white", fontsize=9)
    ax.legend(fontsize=8, facecolor="#1e1e1e", labelcolor="white")
    plt.tight_layout(); return fig

def _plot_spread(spread, t1, t2):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0e1117"); _dark_ax(ax)
    ax.plot(spread.index, spread.values, color="#9b59b6", linewidth=1.2)
    ax.axhline(spread.mean(), color="#f0b429", linewidth=1, linestyle="--", alpha=0.8, label="Mean")
    ax.axhline(spread.mean() + 2*spread.std(), color="#e74c3c", linewidth=0.8, linestyle=":", alpha=0.7, label="+2σ")
    ax.axhline(spread.mean() - 2*spread.std(), color="#2ecc71", linewidth=0.8, linestyle=":", alpha=0.7, label="-2σ")
    ax.fill_between(spread.index, spread.values, spread.mean(), alpha=0.08, color="#9b59b6")
    ax.set_title(f"Hedged Spread: {t1} - (α + β×{t2})", color="white", fontsize=11)
    ax.legend(fontsize=8, facecolor="#1e1e1e", labelcolor="white")
    plt.tight_layout(); return fig

def _plot_zscore(zscore, position, t1, t2, entry):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#0e1117"); _dark_ax(ax1); _dark_ax(ax2)
    ax1.plot(zscore.index, zscore.values, color="#1abc9c", linewidth=1.2)
    ax1.axhline( entry, color="#e74c3c", linewidth=1, linestyle="--", alpha=0.8, label=f"Short {t1}/Long {t2}")
    ax1.axhline(-entry, color="#2ecc71", linewidth=1, linestyle="--", alpha=0.8, label=f"Long {t1}/Short {t2}")
    ax1.axhline(0, color="white", linewidth=0.5, alpha=0.3)
    ax1.fill_between(zscore.index, zscore.values,  entry, where=(zscore >  entry), alpha=0.15, color="#e74c3c")
    ax1.fill_between(zscore.index, zscore.values, -entry, where=(zscore < -entry), alpha=0.15, color="#2ecc71")
    ax1.set_title(f"Z-Score of Hedged Spread: {t1} vs {t2}", color="white", fontsize=11)
    ax1.set_ylabel("Z-Score", color="white", fontsize=9)
    ax1.legend(fontsize=7, facecolor="#1e1e1e", labelcolor="white")
    colors = position.map({1: "#2ecc71", -1: "#e74c3c", 0: "#555555"})
    ax2.bar(position.index, position.values, color=colors, width=1.0)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["Short", "Flat", "Long"], color="white", fontsize=8)
    ax2.set_title("Position", color="white", fontsize=9)
    plt.tight_layout(); return fig

def _plot_cumulative(cumulative, t1, t2):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0e1117"); _dark_ax(ax)
    ax.plot(cumulative.index, cumulative.values, color="#f0b429", linewidth=1.8)
    ax.axhline(1.0, color="white", linewidth=0.5, alpha=0.3, linestyle="--")
    ax.fill_between(cumulative.index, cumulative.values, 1.0,
                     where=(cumulative >= 1.0), alpha=0.15, color="#2ecc71")
    ax.fill_between(cumulative.index, cumulative.values, 1.0,
                     where=(cumulative < 1.0), alpha=0.15, color="#e74c3c")
    ax.set_title(f"Cumulative Strategy Return: {t1} vs {t2}", color="white", fontsize=11)
    ax.set_ylabel("Cumulative Return", color="white", fontsize=9)
    plt.tight_layout(); return fig

# =========================
# Main Page
# =========================

def run_pairs_trading():
    st.subheader("🔗 Pairs Trading")
    st.markdown(
        "Identifies and trades **mean-reverting spreads** between two historically correlated assets. "
        "Choose between **Strict Mode** (requires cointegration) or **Correlation-Based Mode** "
        "(uses correlation + Z-Score without requiring cointegration)."
    )

    with st.expander("📖 How It Works", expanded=False):
        st.markdown("""
**Two Analysis Modes:**

| | 🔬 Strict Mode | 📈 Correlation-Based Mode |
|--|----------------|--------------------------|
| **Requires cointegration** | Yes (p < 0.05) | No |
| **Signal source** | Spread Z-Score | Spread Z-Score |
| **Best for** | Stable market regimes | Trending/volatile markets |
| **Risk** | Lower — statistically validated | Higher — no mean-reversion guarantee |
| **When to use** | Normal market conditions | When classic pairs fail cointegration |

**The Pairs Trading Process:**

1. **Cointegration Test** — Tests if two prices have a long-run equilibrium (p < 0.05 = valid)
2. **Hedge Ratio (OLS)** — Finds β so Spread = Asset1 - (α + β × Asset2) is stationary
3. **Z-Score** — Measures standard deviations from the rolling mean
4. **Signal** — Z > +2 → Short Asset1/Long Asset2 | Z < -2 → Long Asset1/Short Asset2

**Why cointegration fails in trending markets:**
During major macro regime shifts (rate shock 2022, Middle East conflict 2024-2026), even classic pairs like GLD/SLV and XOM/CVX can fail cointegration tests because prices trend rather than mean-revert. Correlation-Based Mode handles these periods.

**How to use with your other tools:**
- Check **Market Regime** first — pairs trades work best in Neutral/sideways regimes
- Use **Mean Reversion Scanner** to confirm individual asset signals align
- Never enter before a major **Economic Calendar** event
""")

    # ---- Mode Selection ----
    st.markdown("### 🎛️ Analysis Mode")
    mode = st.radio(
        "Select Mode",
        ["🔬 Strict Mode — Cointegration Required", "📈 Correlation-Based Mode — Z-Score Only"],
        horizontal=True,
        key="pt_mode"
    )
    strict_mode = "Strict" in mode

    if strict_mode:
        st.info("**Strict Mode:** Only trade if cointegration p-value < 0.05. Academically rigorous. Best in stable markets.")
    else:
        st.warning("**Correlation-Based Mode:** Cointegration not required. Uses correlation + Z-Score only. Higher risk — use smaller position sizes and tighter stops.")

    # ---- Pair Selection ----
    st.markdown("### ⚙️ Select a Pair")
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Category", list(PAIR_CATEGORIES.keys()), key="pt_cat")
    with col2:
        pairs_in_cat = PAIR_CATEGORIES[category]
        pair_choice = st.selectbox("Pair", pairs_in_cat, key="pt_pair")

    t1_default, t2_default = PRESET_PAIRS[pair_choice]

    col3, col4, col5 = st.columns(3)
    with col3:
        t1 = st.text_input("Ticker 1", value=t1_default if t1_default else "KO", key="pt_t1").upper().strip()
        t2 = st.text_input("Ticker 2", value=t2_default if t2_default else "PEP", key="pt_t2").upper().strip()
    with col4:
        period_map = {"6 Months": 182, "1 Year": 365, "2 Years": 730}
        period_label = st.selectbox("Lookback", list(period_map.keys()), index=1, key="pt_period")
        entry_z = st.number_input("Entry Z-Score", value=2.0, min_value=1.0, max_value=3.0, step=0.5, key="pt_entry")
    with col5:
        window = st.slider("Rolling Window (days)", 10, 60, 30, key="pt_window")
        corr_threshold = st.slider("Min Correlation (Corr Mode)", 0.5, 0.9, 0.6, step=0.05, key="pt_corr_thresh") if not strict_mode else 0.6

    run = st.button("▶ Run Pairs Analysis", key="pt_run")
    if not run:
        st.info("Select a pair and click Run Pairs Analysis to begin.")
        return

    # ---- Fetch ----
    start_date = (datetime.now() - timedelta(days=period_map[period_label])).strftime("%Y-%m-%d")
    with st.spinner(f"Fetching {t1} and {t2}..."):
        prices = _fetch_pair(t1, t2, start_date)

    if prices is None or prices.empty or t1 not in prices.columns or t2 not in prices.columns:
        st.error(f"Could not fetch data for {t1} and {t2}. Please check the tickers.")
        return

    price1 = prices[t1]; price2 = prices[t2]
    if len(prices) < 60:
        st.error("Not enough data. Try a longer lookback or different tickers.")
        return

    st.success(f"Loaded {len(prices):,} days from {prices.index.min().date()} to {prices.index.max().date()}")

    # ---- Compute ----
    correlation     = price1.pct_change().corr(price2.pct_change())
    coint_pvalue    = _run_coint(price1, price2)
    hedge_ratio, intercept, r_sq = _compute_hedge_ratio(price1, price2)
    spread          = _compute_spread(price1, price2, hedge_ratio, intercept)
    adf_stat, adf_p, _ = _run_adf(spread)
    zscore          = _compute_zscore(spread, window=window)
    position        = _compute_signals(zscore, entry=entry_z)
    cumulative, total_ret, sharpe, max_dd, win_rate = _compute_returns(price1, price2, hedge_ratio, position)
    current_z       = zscore.dropna().iloc[-1]

    # ---- Mode Validation ----
    st.markdown("### 📊 Pair Validation")

    if strict_mode:
        if coint_pvalue is not None:
            if coint_pvalue < 0.01:
                st.success(f"✅ Strong cointegration (p={coint_pvalue:.4f}) — High quality pairs trade.")
            elif coint_pvalue < 0.05:
                st.success(f"✅ Cointegration detected (p={coint_pvalue:.4f}) — Statistically valid.")
            elif coint_pvalue < 0.10:
                st.warning(f"⚠️ Weak cointegration (p={coint_pvalue:.4f}) — Trade with caution.")
            else:
                st.error(f"❌ No cointegration (p={coint_pvalue:.4f}) — Switch to Correlation-Based Mode or choose a different pair.")
                st.markdown("""
**Why is cointegration failing?**
- The macro regime may have shifted — try Correlation-Based Mode
- Try a longer lookback period (2 Years)
- Try a structurally similar pair (same industry, same revenue drivers)
- Cointegration is time-varying — even classic pairs fail during regime shifts
""")
    else:
        if correlation >= corr_threshold:
            st.success(f"✅ Correlation-Based Mode: Correlation {correlation:.2f} meets threshold ({corr_threshold:.2f}). Z-Score signals active.")
        elif correlation >= 0.5:
            st.warning(f"⚠️ Correlation {correlation:.2f} is below your threshold ({corr_threshold:.2f}) but above 0.50. Signals less reliable.")
        else:
            st.error(f"❌ Correlation {correlation:.2f} is too low for pairs trading. Choose a more correlated pair.")

    # ---- Stats ----
    st.markdown("### 📊 Pair Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Return Correlation", f"{correlation:.2f}", delta="Strong" if correlation > 0.7 else ("Moderate" if correlation > 0.5 else "Weak"))
    c2.metric("Cointegration p-value", f"{coint_pvalue:.4f}" if coint_pvalue else "N/A",
              delta="✅ Valid" if (coint_pvalue and coint_pvalue < 0.05) else "❌ Invalid")
    c3.metric("Hedge Ratio β", f"{hedge_ratio:.4f}")
    c4.metric("R²", f"{r_sq:.3f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("ADF Statistic", f"{adf_stat:.4f}" if adf_stat else "N/A")
    c6.metric("ADF p-value", f"{adf_p:.4f}" if adf_p else "N/A",
              delta="✅ Stationary" if (adf_p and adf_p < 0.05) else "❌ Non-stationary")
    c7.metric("Current Z-Score", f"{current_z:.2f}")

    if current_z > entry_z:
        signal_now = f"🔴 Short {t1} / Long {t2}"
    elif current_z < -entry_z:
        signal_now = f"🟢 Long {t1} / Short {t2}"
    else:
        signal_now = "⚪ Flat — No Signal"
    c8.metric("Current Signal", signal_now)

    # ---- Mode-specific signal quality ----
    if not strict_mode and abs(current_z) > entry_z:
        if correlation >= corr_threshold:
            st.success(f"✅ Correlation-Based Signal Active: {signal_now} | Z-Score: {current_z:.2f} | Correlation: {correlation:.2f}")
            st.caption("⚠️ Correlation-Based Mode — use smaller position sizes than you would in Strict Mode")
        else:
            st.warning(f"⚠️ Z-Score signal exists but correlation ({correlation:.2f}) is below threshold. Low conviction trade.")

    # ---- Performance ----
    st.markdown("### 📈 Strategy Performance")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Total Return",  f"{total_ret:.2%}")
    p2.metric("Sharpe Ratio",  f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")
    p3.metric("Max Drawdown",  f"{max_dd:.2%}")
    p4.metric("Win Rate",      f"{win_rate:.1%}" if pd.notna(win_rate) else "N/A")

    # ---- Charts ----
    st.markdown("### 📉 Price Comparison")
    st.pyplot(_plot_prices(price1, price2, t1, t2))

    st.markdown("### 📊 Hedged Spread")
    st.pyplot(_plot_spread(spread, t1, t2))

    st.markdown("### 🎯 Z-Score & Trading Signals")
    st.pyplot(_plot_zscore(zscore, position, t1, t2, entry_z))

    st.markdown("### 💰 Cumulative Strategy Return")
    st.pyplot(_plot_cumulative(cumulative, t1, t2))

    # ---- Recent Log ----
    st.markdown("### 📋 Recent Z-Score (Last 10 Days)")
    recent = zscore.dropna().tail(10).reset_index()
    recent.columns = ["Date", "Z-Score"]
    recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")
    recent["Signal"] = recent["Z-Score"].apply(
        lambda z: f"🟢 Long {t1}/Short {t2}" if z < -entry_z
        else (f"🔴 Short {t1}/Long {t2}" if z > entry_z else "⚪ Flat")
    )
    recent["Z-Score"] = recent["Z-Score"].round(3)
    st.dataframe(recent.iloc[::-1], use_container_width=True)

    # ---- Trade Instructions ----
    st.markdown("### 🔢 Trade Instructions")
    st.markdown(f"**Formula:** `Spread = {t1} - ({intercept:.4f} + {hedge_ratio:.4f} × {t2})`")
    st.markdown(f"**Mode:** {'🔬 Strict' if strict_mode else '📈 Correlation-Based'} | **Signal:** {signal_now}")

    if current_z > entry_z:
        st.markdown(f"""
- **Short** {t1} (1 share) + **Long** {hedge_ratio:.2f} shares of {t2}
- {t1} is **{current_z:.2f}σ above** its historical relationship with {t2}
- Expected: spread converges as {t1} falls and/or {t2} rises
{"- ⚠️ Correlation-Based Mode: use 50% normal position size" if not strict_mode else ""}
""")
    elif current_z < -entry_z:
        st.markdown(f"""
- **Long** {t1} (1 share) + **Short** {hedge_ratio:.2f} shares of {t2}
- {t1} is **{abs(current_z):.2f}σ below** its historical relationship with {t2}
- Expected: spread converges as {t1} rises and/or {t2} falls
{"- ⚠️ Correlation-Based Mode: use 50% normal position size" if not strict_mode else ""}
""")
    else:
        st.markdown(f"Z-Score of **{current_z:.2f}** is within ±{entry_z} — wait for a signal before entering.")

    st.caption(
        f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | "
        "Data via Yahoo Finance | statsmodels for cointegration testing"
    )