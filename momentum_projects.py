import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def run_sma():
    st.subheader("\U0001F4C8 Time-Series Momentum with SMA Crossover")
    st.write("Tests whether short-term trends in price predict future returns using SMA crossovers.")
    # You can add a chart or code sample here

def run_cross_sectional():
    st.subheader("\U0001F501 Cross-Sectional Momentum Across Sectors")
    st.write("Ranks selected tickers by past performance and rotates capital monthly into top performers.")

    # --- Parameters ---
    start_date = "2019-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    lookback_months = 3
    holding_period_months = 1
    top_n = 5
    transaction_cost = 0.002
    risk_free_rate = 0.02 / 12

    # ===== Universe selection =====
    # Stocks kept separate from commodities
    STOCKS = [
        'AAPL','MSFT','TSLA','GOOGL','NVDA','META','AMZN','NFLX','JPM','UNH',
        'V','MA','HD','BAC','XOM','WMT','PEP','KO','CSCO','INTC'
    ]

    COMMODITIES = {
        "📊 Broad Commodity Indexes": ["DBC", "PDBC", "COMT", "BCI"],
        "⚡ Energy": ["USO", "BNO", "UNG", "UGA", "XLE"],  # XLE = equities proxy
        "🥇 Metals — Precious": ["GLD", "SLV", "PLTM", "PALL"],
        "🔩 Metals — Industrial": ["CPER", "JJN", "SLX"],
        "⛏️ Miners": ["SGDM"],  # (or SGDJ if you prefer)
        "🌾 Agriculture": ["CORN", "SOYB", "WEAT", "JO", "NIB", "CANE"],
        "🐄 Livestock": ["COW"],
        "🚀 Thematic / Special Commodities": ["URA", "LIT", "KRBN"],
    }

    st.markdown("### Universe Selection")

    # ---- Stocks block ----
    with st.expander("Stocks (separate from commodities)", expanded=True):
        sel_all_stocks = st.checkbox("Select **all stocks**", key="all_stocks", value=False)
        default_stocks = STOCKS if sel_all_stocks else ["AAPL", "MSFT", "NVDA"]
        selected_stocks = st.multiselect(
            "Choose stocks",
            options=STOCKS,
            default=default_stocks,
            key="stocks_multiselect"
        )

    # ---- Commodities block ----
    with st.expander("Commodities (grouped by section)", expanded=True):
        sel_all_comms = st.checkbox("Select **ALL commodities** (every section)", key="all_comms", value=False)

        selected_commodities = []
        if sel_all_comms:
            for _section, tickers in COMMODITIES.items():
                selected_commodities.extend(tickers)
        else:
            for section_name, tickers in COMMODITIES.items():
                with st.expander(section_name, expanded=False):
                    sec_key = section_name.replace(" ", "_")
                    sec_all = st.checkbox(f"Select all in {section_name}", key=f"all_{sec_key}", value=False)
                    default_sec = tickers if sec_all else []
                    chosen = st.multiselect(
                        "Tickers",
                        options=tickers,
                        default=default_sec,
                        key=f"multi_{sec_key}"
                    )
                    selected_commodities.extend(chosen)

    # Final combined universe (unique order-preserving)
    main_tickers = list(dict.fromkeys(selected_stocks + selected_commodities))

    # Safety default so the app still runs
    if not main_tickers:
        st.info("No tickers selected — defaulting to ['AAPL','MSFT','NVDA','GLD','DBC'].")
        main_tickers = ['AAPL', 'MSFT', 'NVDA', 'GLD', 'DBC']

    # Strategy option
    short_bottom = st.checkbox("\U0001F53B Short Bottom 20% of selected assets?", value=False)

    # ===== Data =====
    with st.spinner("Downloading data..."):
        price_data = yf.download(main_tickers, start=start_date, end=end_date, auto_adjust=True)[['Close']]
        price_data.columns = price_data.columns.droplevel(0)  # drop 'Close' multiindex level
        spy_data = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)['Close']
        price_data['SPY'] = spy_data

    monthly_prices = price_data.resample('ME').last()
    momentum = monthly_prices.pct_change(periods=lookback_months)

    st.subheader("Momentum Scores (Last 10 Rows)")
    st.dataframe(momentum.tail(10).round(4))

    last_rebalance = momentum.dropna().index[-1]
    st.subheader(f"Latest Momentum Ranking on {last_rebalance.strftime('%Y-%m-%d')}")
    st.dataframe(momentum.loc[last_rebalance].drop('SPY').sort_values(ascending=False).round(4))

    # ===== Backtest =====
    portfolio_value = 1000.0
    portfolio_values, spy_values, monthly_returns, rebalance_log = [], [], [], []
    selection_matrix = pd.DataFrame(0, index=momentum.index, columns=monthly_prices.columns)
    rebalance_dates = momentum.dropna().index

    for date in rebalance_dates:
        try:
            past_returns = momentum.loc[date].drop('SPY')
            top_names = past_returns.nlargest(top_n)
            entry_prices = monthly_prices.loc[date, top_names.index]

            exit_idx = monthly_prices.index.get_loc(date) + holding_period_months
            if exit_idx >= len(monthly_prices.index):
                break
            exit_date = monthly_prices.index[exit_idx]
            exit_prices = monthly_prices.loc[exit_date, top_names.index]

            long_returns = (exit_prices - entry_prices) / entry_prices
            position_marks = pd.Series(0, index=monthly_prices.columns)

            if short_bottom:
                bottom_names = past_returns.nsmallest(top_n)
                short_entry = monthly_prices.loc[date, bottom_names.index]
                short_exit = monthly_prices.loc[exit_date, bottom_names.index]
                short_returns = (short_entry - short_exit) / short_entry
                net_returns = pd.concat([long_returns, short_returns]) - transaction_cost

                position_marks[top_names.index] = 1
                position_marks[bottom_names.index] = -1
            else:
                net_returns = long_returns - transaction_cost
                position_marks[top_names.index] = 1

            avg_return = net_returns.mean()
            portfolio_value *= (1 + avg_return)

            # SPY benchmark
            spy_entry = monthly_prices.loc[date, 'SPY']
            spy_exit = monthly_prices.loc[exit_date, 'SPY']
            spy_return = (spy_exit - spy_entry) / spy_entry
            spy_value = spy_values[-1] * (1 + spy_return) if spy_values else 1000.0

            portfolio_values.append(portfolio_value)
            spy_values.append(spy_value)
            monthly_returns.append(avg_return)
            selection_matrix.loc[date] = position_marks

            rebalance_log.append({
                'Rebalance Date': date.strftime('%Y-%m-%d'),
                'Exit Date': exit_date.strftime('%Y-%m-%d'),
                'Top Tickers': ', '.join(top_names.index),
                'Monthly Return (%)': round(avg_return * 100, 2)
            })

        except Exception:
            continue

    result_df = pd.DataFrame({
        'Portfolio Value': portfolio_values,
        'SPY Value': spy_values,
        'Monthly Return': monthly_returns
    }, index=rebalance_dates[:len(portfolio_values)])

    # ===== Metrics & Charts =====
    returns = pd.Series(monthly_returns, index=result_df.index)
    excess_returns = returns - risk_free_rate
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(12)
    volatility = returns.std() * np.sqrt(12)
    cumulative = pd.Series(portfolio_values).pct_change().add(1).cumprod()
    drawdown = 1 - cumulative / cumulative.cummax()
    max_drawdown = drawdown.max()
    total_years = (result_df.index[-1] - result_df.index[0]).days / 365.25
    cagr = (portfolio_values[-1] / portfolio_values[0]) ** (1 / total_years) - 1
    downside_returns = returns[returns < risk_free_rate]
    downside_std = downside_returns.std() * np.sqrt(12)
    sortino = (returns.mean() - risk_free_rate) / downside_std if downside_std > 0 else np.nan

    st.subheader("Strategy Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col2.metric("Sortino Ratio", f"{sortino:.2f}")
    col3.metric("CAGR", f"{cagr:.2%}")
    st.metric("Annualized Volatility", f"{volatility:.2%}")
    st.metric("Max Drawdown", f"{max_drawdown:.2%}")

    st.subheader("Last 5 Rebalances")
    st.dataframe(pd.DataFrame(rebalance_log).tail())

    st.subheader("\U0001F4C8 Strategy vs SPY")
    st.line_chart(result_df[['Portfolio Value', 'SPY Value']])

    st.subheader("\U0001F4CA Selection Heatmap (Blue = Long, Red = Short)")
    fig, ax = plt.subplots(figsize=(14, 8))
    heatmap_data = selection_matrix.loc[rebalance_dates[:len(portfolio_values)]].T
    heatmap_data.columns = heatmap_data.columns.strftime('%Y-%m')
    sns.heatmap(
        heatmap_data,
        cmap=sns.color_palette(["red", "white", "blue"], as_cmap=True),
        center=0,
        cbar=True,
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("\U0001F4C9 Rolling 12-Month Sharpe Ratio")
    rolling_sharpe = (
        (returns.rolling(window=12).mean() - risk_free_rate) /
        returns.rolling(window=12).std()
    ) * np.sqrt(12)
    st.line_chart(rolling_sharpe)


def run_dual_momentum():
    st.subheader("\U0001F9EE Dual Momentum Strategy")
    st.write("Combines time-series and cross-sectional filters to build a robust long-only ETF strategy.")
    # Add your logic or visuals
