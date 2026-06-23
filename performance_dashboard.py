# performance_dashboard.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os
from datetime import datetime, timezone, date

TRADES_FILE = "trades.json"

# =========================
# Trade Storage
# =========================

def _load_trades():
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _save_trades(trades):
    try:
        with open(TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=2)
        return True
    except Exception:
        return False

def _init_trades():
    if "trades" not in st.session_state:
        st.session_state.trades = _load_trades()

# =========================
# Price Fetching
# =========================

@st.cache_data(ttl=300, show_spinner=False)
def _get_current_price(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="2d", auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None

@st.cache_data(ttl=300, show_spinner=False)
def _get_spy_return(start_date: str):
    try:
        tk = yf.Ticker("IVV")
        hist = tk.history(start=start_date, auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        if not hist.empty and "Close" in hist.columns:
            s = hist["Close"].dropna()
            return (s.iloc[-1] / s.iloc[0] - 1) * 100
    except Exception:
        pass
    return None

@st.cache_data(ttl=300, show_spinner=False)
def _get_spy_history(start_date: str):
    try:
        tk = yf.Ticker("IVV")
        hist = tk.history(start=start_date, auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        if not hist.empty and "Close" in hist.columns:
            s = hist["Close"].dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            return s
    except Exception:
        pass
    return None

# =========================
# Sector Mapping
# =========================

SECTOR_MAP = {
    "SPY": "Broad Market", "QQQ": "Technology", "IWM": "Small Cap",
    "TLT": "Fixed Income", "IEF": "Fixed Income", "SHY": "Fixed Income",
    "GLD": "Commodities", "SLV": "Commodities", "USO": "Commodities",
    "HYG": "Fixed Income", "LQD": "Fixed Income", "JNK": "Fixed Income",
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Healthcare", "XLY": "Consumer Disc", "XLP": "Consumer Staples",
    "XLI": "Industrials", "XLB": "Materials", "XLRE": "Real Estate",
    "XLU": "Utilities", "XLC": "Comm Services",
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "META": "Technology", "AMD": "Technology",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "XOM": "Energy", "CVX": "Energy",
    "TSLA": "Consumer Disc", "AMZN": "Consumer Disc",
    "UNH": "Healthcare", "JNJ": "Healthcare",
    "V": "Financials", "MA": "Financials",
}

def _get_sector(ticker: str):
    return SECTOR_MAP.get(ticker.upper(), "Other")

# =========================
# Analytics
# =========================

def _compute_analytics(trades: list):
    if not trades:
        return None

    rows = []
    for t in trades:
        current_price = _get_current_price(t["ticker"])
        if current_price is None:
            current_price = t["entry_price"]

        if t["action"] == "BUY":
            pnl_pct = (current_price - t["entry_price"]) / t["entry_price"] * 100
            pnl_dollar = (current_price - t["entry_price"]) * t["shares"]
        else:  # SHORT
            pnl_pct = (t["entry_price"] - current_price) / t["entry_price"] * 100
            pnl_dollar = (t["entry_price"] - current_price) * t["shares"]

        if t.get("status") == "Closed" and t.get("exit_price"):
            if t["action"] == "BUY":
                pnl_pct = (t["exit_price"] - t["entry_price"]) / t["entry_price"] * 100
                pnl_dollar = (t["exit_price"] - t["entry_price"]) * t["shares"]
            else:
                pnl_pct = (t["entry_price"] - t["exit_price"]) / t["entry_price"] * 100
                pnl_dollar = (t["entry_price"] - t["exit_price"]) * t["shares"]
            current_price = t["exit_price"]

        rows.append({
            "ticker":        t["ticker"],
            "action":        t["action"],
            "entry_date":    t["entry_date"],
            "entry_price":   t["entry_price"],
            "shares":        t["shares"],
            "current_price": current_price,
            "exit_price":    t.get("exit_price"),
            "status":        t.get("status", "Open"),
            "reason":        t.get("reason", ""),
            "regime":        t.get("regime", ""),
            "sector":        _get_sector(t["ticker"]),
            "pnl_pct":       pnl_pct,
            "pnl_dollar":    pnl_dollar,
            "position_value": t["entry_price"] * t["shares"],
        })

    df = pd.DataFrame(rows)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    return df

def _dark_ax(ax):
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

# =========================
# Main Page
# =========================

def run_performance_dashboard():
    _init_trades()

    st.subheader("📊 Trading Performance Dashboard")
    st.markdown(
        "Track your **Investopedia paper trades** vs S&P 500 benchmark. "
        "Log every trade with your reasoning. Build a quantitative track record for interviews."
    )

    # ---- Tabs ----
    tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "➕ Log Trade", "📋 Trade Journal"])

    # =========================
    # TAB 1 — DASHBOARD
    # =========================
    with tab1:
        trades = st.session_state.trades
        if not trades:
            st.info("No trades logged yet. Go to the **Log Trade** tab to add your first trade.")
            return

        df = _compute_analytics(trades)
        if df is None or df.empty:
            st.info("No trade data available.")
            return

        # ---- Key Metrics ----
        st.markdown("### 🎯 Portfolio Overview")

        total_invested = df["position_value"].sum()
        total_pnl = df["pnl_dollar"].sum()
        total_return = total_pnl / total_invested * 100 if total_invested > 0 else 0
        win_rate = (df["pnl_dollar"] > 0).mean() * 100
        n_trades = len(df)
        n_open = len(df[df["status"] == "Open"])
        n_closed = len(df[df["status"] == "Closed"])
        avg_win = df[df["pnl_dollar"] > 0]["pnl_pct"].mean() if len(df[df["pnl_dollar"] > 0]) > 0 else 0
        avg_loss = df[df["pnl_dollar"] < 0]["pnl_pct"].mean() if len(df[df["pnl_dollar"] < 0]) > 0 else 0

        # SPY benchmark
        if len(df) > 0:
            first_trade_date = df["entry_date"].min().strftime("%Y-%m-%d")
            spy_return = _get_spy_return(first_trade_date)
        else:
            spy_return = None

        # Sharpe
        if len(df) >= 3:
            daily_returns = df["pnl_pct"] / 100
            sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else np.nan
        else:
            sharpe = np.nan

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total P&L", f"${total_pnl:+,.0f}", delta=f"{total_return:+.1f}%")
        c2.metric("vs S&P 500", f"{total_return - spy_return:+.1f}%" if spy_return else "N/A",
                  delta="Outperforming ✅" if spy_return and total_return > spy_return else "Underperforming")
        c3.metric("Win Rate", f"{win_rate:.0f}%")
        c4.metric("Sharpe Ratio", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Total Trades", n_trades)
        c6.metric("Open Positions", n_open)
        c7.metric("Avg Win", f"{avg_win:+.1f}%")
        c8.metric("Avg Loss", f"{avg_loss:+.1f}%")

        if spy_return:
            if total_return > spy_return:
                st.success(f"✅ Outperforming S&P 500 by {total_return - spy_return:.1f}% since first trade")
            else:
                st.warning(f"📉 Underperforming S&P 500 by {spy_return - total_return:.1f}% since first trade")

        # ---- P&L by Position ----
        st.markdown("### 💼 Current Positions")
        display_df = df[["ticker", "action", "entry_date", "entry_price", "shares",
                          "current_price", "pnl_pct", "pnl_dollar", "status", "sector"]].copy()
        display_df["entry_date"] = display_df["entry_date"].dt.strftime("%Y-%m-%d")
        display_df["pnl_pct"]    = display_df["pnl_pct"].round(2)
        display_df["pnl_dollar"] = display_df["pnl_dollar"].round(0)
        display_df.columns = ["Ticker", "Action", "Entry Date", "Entry $",
                               "Shares", "Current $", "P&L %", "P&L $", "Status", "Sector"]
        st.dataframe(display_df, use_container_width=True)

        # ---- P&L Waterfall Chart ----
        st.markdown("### 📊 Trade P&L Waterfall")
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0e1117"); _dark_ax(ax)
        df_sorted = df.sort_values("entry_date")
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df_sorted["pnl_dollar"]]
        bars = ax.bar(range(len(df_sorted)), df_sorted["pnl_dollar"], color=colors, alpha=0.85, width=0.7)
        ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars, df_sorted["pnl_dollar"])):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + (20 if val >= 0 else -20),
                    f"${val:,.0f}", ha="center", va="bottom" if val >= 0 else "top",
                    color="white", fontsize=7)
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels(df_sorted["ticker"].tolist(), rotation=45, ha="right", color="white", fontsize=8)
        ax.set_title("P&L by Trade ($)", color="white", fontsize=11)
        ax.set_ylabel("P&L ($)", color="white", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

        # ---- Cumulative P&L vs SPY ----
        st.markdown("### 📈 Cumulative P&L vs S&P 500")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        fig2.patch.set_facecolor("#0e1117"); _dark_ax(ax2)

        df_cum = df_sorted.copy()
        df_cum["cum_pnl"] = df_cum["pnl_dollar"].cumsum()
        df_cum["cum_invested"] = df_cum["position_value"].cumsum()
        df_cum["cum_return"] = df_cum["cum_pnl"] / df_cum["cum_invested"] * 100

        ax2.plot(range(len(df_cum)), df_cum["cum_return"], color="#f0b429",
                 linewidth=2, label="My Portfolio", marker="o", markersize=4)
        ax2.axhline(0, color="white", linewidth=0.5, alpha=0.3, linestyle="--")
        ax2.fill_between(range(len(df_cum)), df_cum["cum_return"], 0,
                          where=(df_cum["cum_return"] >= 0), alpha=0.15, color="#2ecc71")
        ax2.fill_between(range(len(df_cum)), df_cum["cum_return"], 0,
                          where=(df_cum["cum_return"] < 0), alpha=0.15, color="#e74c3c")
        if spy_return is not None:
            ax2.axhline(spy_return, color="#3498db", linewidth=1.5,
                        linestyle="--", label=f"SPY: {spy_return:+.1f}%", alpha=0.8)
        ax2.set_xticks(range(len(df_cum)))
        ax2.set_xticklabels(df_cum["ticker"].tolist(), rotation=45, ha="right", color="white", fontsize=8)
        ax2.set_title("Cumulative Portfolio Return vs S&P 500", color="white", fontsize=11)
        ax2.set_ylabel("Return (%)", color="white", fontsize=9)
        ax2.legend(fontsize=9, facecolor="#1e1e1e", labelcolor="white")
        plt.tight_layout()
        st.pyplot(fig2)

        # ---- Sector Attribution ----
        st.markdown("### 🥧 Sector Attribution")
        sector_pnl = df.groupby("sector")["pnl_dollar"].sum().sort_values(ascending=False)
        col_pie, col_bar = st.columns(2)

        with col_pie:
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            fig3.patch.set_facecolor("#0e1117")
            ax3.set_facecolor("#0e1117")
            pos_sectors = sector_pnl[sector_pnl > 0]
            if not pos_sectors.empty:
                ax3.pie(pos_sectors, labels=pos_sectors.index, autopct="%1.1f%%",
                        colors=plt.cm.Set3(np.linspace(0, 1, len(pos_sectors))),
                        textprops={"color": "white", "fontsize": 9})
                ax3.set_title("P&L by Sector (Winners)", color="white", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig3)

        with col_bar:
            fig4, ax4 = plt.subplots(figsize=(6, 6))
            fig4.patch.set_facecolor("#0e1117"); _dark_ax(ax4)
            colors_s = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sector_pnl.values]
            ax4.barh(sector_pnl.index, sector_pnl.values, color=colors_s, alpha=0.85)
            ax4.axvline(0, color="white", linewidth=0.5, alpha=0.3)
            for i, v in enumerate(sector_pnl.values):
                ax4.text(v + (5 if v >= 0 else -5), i, f"${v:,.0f}",
                         va="center", ha="left" if v >= 0 else "right", color="white", fontsize=8)
            ax4.set_title("P&L by Sector ($)", color="white", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig4)

        # ---- Best & Worst Trades ----
        st.markdown("### 🏆 Best & Worst Trades")
        col_b, col_w = st.columns(2)
        with col_b:
            st.markdown("#### 🟢 Best Trades")
            best = df.nlargest(3, "pnl_pct")
            for _, row in best.iterrows():
                st.success(f"**{row['ticker']}** {row['action']} — {row['pnl_pct']:+.1f}% (${row['pnl_dollar']:+,.0f})\n\n_{row['reason']}_")
        with col_w:
            st.markdown("#### 🔴 Worst Trades")
            worst = df.nsmallest(3, "pnl_pct")
            for _, row in worst.iterrows():
                st.error(f"**{row['ticker']}** {row['action']} — {row['pnl_pct']:+.1f}% (${row['pnl_dollar']:+,.0f})\n\n_{row['reason']}_")

        st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC | Prices via Yahoo Finance")

    # =========================
    # TAB 2 — LOG TRADE
    # =========================
    with tab2:
        st.markdown("### ➕ Log a New Trade")
        st.caption("Log your Investopedia trade immediately after placing it.")

        with st.form("log_trade_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                ticker = st.text_input("Ticker", placeholder="e.g. SPY").upper().strip()
                action = st.selectbox("Action", ["BUY", "SHORT"])
                entry_date = st.date_input("Entry Date", value=date.today())
            with col2:
                entry_price = st.number_input("Entry Price ($)", min_value=0.01, step=0.01)
                shares = st.number_input("Shares", min_value=0.01, step=1.0)
                position_size = entry_price * shares
                st.metric("Position Size", f"${position_size:,.0f}")
            with col3:
                regime = st.selectbox("Market Regime at Entry",
                                       ["Risk-On", "Risk-Off", "Inflationary", "Recessionary", "Neutral"])
                strategy = st.selectbox("Strategy Used",
                                         ["Momentum", "Mean Reversion", "Pairs Trading",
                                          "Macro/Regime", "Technical", "Options", "Other"])
                tools_used = st.multiselect("Apollo Quant Tools Used",
                                             ["Market Regime", "Dual Momentum", "Cross-Sectional Momentum",
                                              "Mean Reversion Scanner", "MACD", "Pairs Trading",
                                              "Relative Strength", "Yield Curve", "Fear & Greed",
                                              "Macro Research", "Implied Move", "Black-Scholes"])

            reason = st.text_area("Trade Thesis (1-3 sentences)",
                                   placeholder="e.g. Market regime is Risk-Off. Dual momentum switched to bonds. TLT showing RSI oversold at 28 with yield curve normalizing. Long TLT as rates expected to fall.",
                                   height=100)

            submitted = st.form_submit_button("✅ Log Trade")

            if submitted:
                if not ticker:
                    st.error("Please enter a ticker.")
                elif entry_price <= 0:
                    st.error("Please enter a valid entry price.")
                elif shares <= 0:
                    st.error("Please enter a valid number of shares.")
                elif not reason:
                    st.error("Please enter your trade thesis. This is the most important field.")
                else:
                    trade = {
                        "ticker": ticker,
                        "action": action,
                        "entry_date": str(entry_date),
                        "entry_price": entry_price,
                        "shares": shares,
                        "reason": reason,
                        "regime": regime,
                        "strategy": strategy,
                        "tools_used": tools_used,
                        "status": "Open",
                        "exit_price": None,
                        "exit_date": None,
                        "logged_at": datetime.now(timezone.utc).isoformat(),
                    }
                    st.session_state.trades.append(trade)
                    _save_trades(st.session_state.trades)
                    st.success(f"✅ Trade logged: {action} {shares:.0f} shares of {ticker} at ${entry_price:.2f}")
                    st.balloons()

        # ---- Close a Trade ----
        st.markdown("### 🔒 Close an Open Trade")
        open_trades = [(i, t) for i, t in enumerate(st.session_state.trades) if t.get("status") == "Open"]

        if open_trades:
            trade_labels = [f"{t['action']} {t['ticker']} @ ${t['entry_price']:.2f} ({t['entry_date']})"
                            for _, t in open_trades]
            selected_label = st.selectbox("Select trade to close", trade_labels, key="close_select")
            selected_idx = open_trades[trade_labels.index(selected_label)][0]

            col_a, col_b = st.columns(2)
            with col_a:
                exit_price = st.number_input("Exit Price ($)", min_value=0.01, step=0.01, key="exit_price")
            with col_b:
                exit_date = st.date_input("Exit Date", value=date.today(), key="exit_date")

            if st.button("🔒 Close Trade", key="close_btn"):
                st.session_state.trades[selected_idx]["status"] = "Closed"
                st.session_state.trades[selected_idx]["exit_price"] = exit_price
                st.session_state.trades[selected_idx]["exit_date"] = str(exit_date)
                _save_trades(st.session_state.trades)
                t = st.session_state.trades[selected_idx]
                if t["action"] == "BUY":
                    pnl = (exit_price - t["entry_price"]) * t["shares"]
                    ret = (exit_price - t["entry_price"]) / t["entry_price"] * 100
                else:
                    pnl = (t["entry_price"] - exit_price) * t["shares"]
                    ret = (t["entry_price"] - exit_price) / t["entry_price"] * 100
                if pnl >= 0:
                    st.success(f"✅ Trade closed: {ret:+.1f}% | P&L: ${pnl:+,.0f}")
                else:
                    st.error(f"Trade closed: {ret:+.1f}% | P&L: ${pnl:+,.0f}")
        else:
            st.info("No open trades to close.")

        # ---- Delete a Trade ----
        st.markdown("### 🗑️ Delete a Trade")
        if st.session_state.trades:
            del_labels = [f"{t['action']} {t['ticker']} @ ${t['entry_price']:.2f} ({t['entry_date']})"
                          for t in st.session_state.trades]
            del_label = st.selectbox("Select trade to delete", del_labels, key="del_select")
            del_idx = del_labels.index(del_label)
            if st.button("🗑️ Delete Trade", key="del_btn"):
                deleted = st.session_state.trades.pop(del_idx)
                _save_trades(st.session_state.trades)
                st.warning(f"Deleted: {deleted['action']} {deleted['ticker']}")
                st.rerun()

    # =========================
    # TAB 3 — TRADE JOURNAL
    # =========================
    with tab3:
        st.markdown("### 📋 Full Trade Journal")
        st.caption("Your complete trade history with reasoning — your interview talking points.")

        if not st.session_state.trades:
            st.info("No trades logged yet.")
            return

        df = _compute_analytics(st.session_state.trades)
        if df is None:
            return

        # ---- Filters ----
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.multiselect("Status", ["Open", "Closed"], default=["Open", "Closed"], key="j_status")
        with col2:
            action_filter = st.multiselect("Action", ["BUY", "SHORT"], default=["BUY", "SHORT"], key="j_action")
        with col3:
            sector_filter = st.multiselect("Sector", df["sector"].unique().tolist(),
                                            default=df["sector"].unique().tolist(), key="j_sector")

        filtered = df[
            (df["status"].isin(status_filter)) &
            (df["action"].isin(action_filter)) &
            (df["sector"].isin(sector_filter))
        ].sort_values("entry_date", ascending=False)

        st.markdown(f"**{len(filtered)} trades** | Total P&L: **${filtered['pnl_dollar'].sum():+,.0f}**")

        for _, row in filtered.iterrows():
            pnl_color = "🟢" if row["pnl_dollar"] >= 0 else "🔴"
            status_tag = "🔵 Open" if row["status"] == "Open" else "⚫ Closed"

            with st.expander(
                f"{pnl_color} {row['action']} {row['ticker']} | Entry: ${row['entry_price']:.2f} | "
                f"P&L: {row['pnl_pct']:+.1f}% (${row['pnl_dollar']:+,.0f}) | {status_tag}"
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Entry Date", str(row["entry_date"])[:10])
                c2.metric("Entry Price", f"${row['entry_price']:.2f}")
                c3.metric("Current/Exit", f"${row['current_price']:.2f}")
                c4.metric("P&L", f"{row['pnl_pct']:+.1f}%")

                st.markdown(f"**📝 Trade Thesis:**")
                st.info(row["reason"])

                trade_data = st.session_state.trades[
                    next(i for i, t in enumerate(st.session_state.trades)
                         if t["ticker"] == row["ticker"] and t["entry_date"] == str(row["entry_date"])[:10])
                ]
                if trade_data.get("regime"):
                    st.markdown(f"**Regime at Entry:** {trade_data['regime']} | **Strategy:** {trade_data.get('strategy', 'N/A')}")
                if trade_data.get("tools_used"):
                    st.markdown(f"**Tools Used:** {', '.join(trade_data['tools_used'])}")

        # ---- Export ----
        st.markdown("### 📥 Export Trade Journal")
        if not filtered.empty:
            csv = filtered.to_csv(index=False)
            st.download_button(
                "⬇️ Download as CSV",
                csv,
                file_name=f"apollo_quant_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )