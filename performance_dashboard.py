# performance_dashboard.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime, timezone, date

TRADES_FILE = "trades.json"

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

@st.cache_data(ttl=300, show_spinner=False)
def _get_current_price(ticker):
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
def _get_spy_return(start_date):
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

SECTOR_MAP = {
    "SPY": "Broad Market", "QQQ": "Technology", "IWM": "Small Cap",
    "TLT": "Fixed Income", "IEF": "Fixed Income", "GLD": "Commodities",
    "USO": "Commodities", "HYG": "Fixed Income", "XLK": "Technology",
    "XLF": "Financials", "XLE": "Energy", "XLV": "Healthcare",
    "XLY": "Consumer Disc", "XLP": "Consumer Staples", "XLI": "Industrials",
    "XLU": "Utilities", "XLC": "Comm Services", "XLRE": "Real Estate",
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "META": "Technology", "AMD": "Technology",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "XOM": "Energy", "CVX": "Energy", "TSLA": "Consumer Disc",
    "AMZN": "Consumer Disc", "V": "Financials", "MA": "Financials",
    "UNH": "Healthcare", "JNJ": "Healthcare",
}

def _get_sector(ticker):
    return SECTOR_MAP.get(ticker.upper(), "Other")

def _compute_analytics(trades):
    if not trades:
        return None
    rows = []
    for t in trades:
        current_price = _get_current_price(t["ticker"])
        if current_price is None:
            current_price = t["entry_price"]

        if t.get("status") == "Closed" and t.get("exit_price"):
            exit_p = t["exit_price"]
            if t["action"] == "BUY":
                pnl_pct = (exit_p - t["entry_price"]) / t["entry_price"] * 100
                pnl_dollar = (exit_p - t["entry_price"]) * t["shares"]
            else:
                pnl_pct = (t["entry_price"] - exit_p) / t["entry_price"] * 100
                pnl_dollar = (t["entry_price"] - exit_p) * t["shares"]
            current_price = exit_p
        else:
            if t["action"] == "BUY":
                pnl_pct = (current_price - t["entry_price"]) / t["entry_price"] * 100
                pnl_dollar = (current_price - t["entry_price"]) * t["shares"]
            else:
                pnl_pct = (t["entry_price"] - current_price) / t["entry_price"] * 100
                pnl_dollar = (t["entry_price"] - current_price) * t["shares"]

        rows.append({
            "ticker": t["ticker"],
            "action": t["action"],
            "entry_date": t["entry_date"],
            "entry_price": t["entry_price"],
            "shares": t["shares"],
            "current_price": current_price,
            "status": t.get("status", "Open"),
            "reason": t.get("reason", ""),
            "regime": t.get("regime", ""),
            "strategy": t.get("strategy", ""),
            "tools_used": t.get("tools_used", []),
            "sector": _get_sector(t["ticker"]),
            "pnl_pct": pnl_pct,
            "pnl_dollar": pnl_dollar,
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

def run_performance_dashboard():
    _init_trades()

    st.subheader("📊 Trading Performance Dashboard")
    st.markdown(
        "Track your **Investopedia paper trades** vs S&P 500. "
        "Log every trade with your reasoning. Build a quantitative track record for interviews."
    )

    tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "➕ Log Trade", "📋 Trade Journal"])

    # =========================
    # TAB 1 — DASHBOARD
    # =========================
    with tab1:
        if not st.session_state.trades:
            st.info("No trades logged yet. Go to the **Log Trade** tab to add your first Investopedia trade.")
        else:
            df = _compute_analytics(st.session_state.trades)
            if df is not None and not df.empty:
                total_invested = df["position_value"].sum()
                total_pnl = df["pnl_dollar"].sum()
                total_return = total_pnl / total_invested * 100 if total_invested > 0 else 0
                win_rate = (df["pnl_dollar"] > 0).mean() * 100
                avg_win = df[df["pnl_dollar"] > 0]["pnl_pct"].mean() if len(df[df["pnl_dollar"] > 0]) > 0 else 0
                avg_loss = df[df["pnl_dollar"] < 0]["pnl_pct"].mean() if len(df[df["pnl_dollar"] < 0]) > 0 else 0

                first_date = df["entry_date"].min().strftime("%Y-%m-%d")
                spy_return = _get_spy_return(first_date)

                daily_ret = df["pnl_pct"] / 100
                sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if len(df) >= 3 and daily_ret.std() > 0 else np.nan

                st.markdown("### 🎯 Portfolio Overview")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total P&L", f"${total_pnl:+,.0f}", delta=f"{total_return:+.1f}%")
                c2.metric("vs S&P 500",
                          f"{total_return - spy_return:+.1f}%" if spy_return else "N/A",
                          delta="Outperforming ✅" if spy_return and total_return > spy_return else "Underperforming")
                c3.metric("Win Rate", f"{win_rate:.0f}%")
                c4.metric("Sharpe Ratio", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Total Trades", len(df))
                c6.metric("Open", len(df[df["status"] == "Open"]))
                c7.metric("Avg Win", f"{avg_win:+.1f}%")
                c8.metric("Avg Loss", f"{avg_loss:+.1f}%")

                if spy_return:
                    if total_return > spy_return:
                        st.success(f"✅ Outperforming S&P 500 by {total_return - spy_return:.1f}% since first trade")
                    else:
                        st.warning(f"📉 Underperforming S&P 500 by {spy_return - total_return:.1f}% since first trade")

                st.markdown("### 💼 Current Positions")
                disp = df[["ticker","action","entry_date","entry_price","shares",
                            "current_price","pnl_pct","pnl_dollar","status","sector"]].copy()
                disp["entry_date"] = disp["entry_date"].dt.strftime("%Y-%m-%d")
                disp["pnl_pct"] = disp["pnl_pct"].round(2)
                disp["pnl_dollar"] = disp["pnl_dollar"].round(0)
                disp.columns = ["Ticker","Action","Entry Date","Entry $","Shares","Current $","P&L %","P&L $","Status","Sector"]
                st.dataframe(disp, use_container_width=True)

                st.markdown("### 📊 Trade P&L")
                fig, ax = plt.subplots(figsize=(12, 4))
                fig.patch.set_facecolor("#0e1117"); _dark_ax(ax)
                df_s = df.sort_values("entry_date")
                colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df_s["pnl_dollar"]]
                ax.bar(range(len(df_s)), df_s["pnl_dollar"], color=colors, alpha=0.85)
                ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)
                ax.set_xticks(range(len(df_s)))
                ax.set_xticklabels(df_s["ticker"].tolist(), rotation=45, ha="right", color="white", fontsize=8)
                ax.set_title("P&L by Trade ($)", color="white", fontsize=11)
                ax.set_ylabel("P&L ($)", color="white", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("### 🥧 Sector Attribution")
                sector_pnl = df.groupby("sector")["pnl_dollar"].sum().sort_values(ascending=False)
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                fig2.patch.set_facecolor("#0e1117"); _dark_ax(ax2)
                colors2 = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sector_pnl.values]
                ax2.barh(sector_pnl.index, sector_pnl.values, color=colors2, alpha=0.85)
                ax2.axvline(0, color="white", linewidth=0.5, alpha=0.3)
                ax2.set_title("P&L by Sector ($)", color="white", fontsize=11)
                plt.tight_layout()
                st.pyplot(fig2)

    # =========================
    # TAB 2 — LOG TRADE
    # =========================
    with tab2:
        st.markdown("### ➕ Log a New Trade")
        st.caption("Fill in the details from your Investopedia trade and hit Log Trade.")

        col1, col2, col3 = st.columns(3)
        with col1:
            ticker = st.text_input("Ticker", placeholder="e.g. TLT", key="lt_ticker").upper().strip()
            action = st.selectbox("Action", ["BUY", "SHORT"], key="lt_action")
            entry_date = st.date_input("Entry Date", value=date.today(), key="lt_date")
        with col2:
            entry_price = st.number_input("Entry Price ($)", min_value=0.01, value=100.0, step=0.01, key="lt_price")
            shares = st.number_input("Shares", min_value=1.0, value=10.0, step=1.0, key="lt_shares")
            st.metric("Position Size", f"${entry_price * shares:,.0f}")
        with col3:
            regime = st.selectbox("Market Regime at Entry",
                                   ["Risk-On", "Risk-Off", "Inflationary", "Recessionary", "Neutral"],
                                   key="lt_regime")
            strategy = st.selectbox("Strategy",
                                     ["Momentum", "Mean Reversion", "Pairs Trading",
                                      "Macro/Regime", "Technical", "Options", "Other"],
                                     key="lt_strategy")

        tools_used = st.multiselect(
            "Apollo Quant Tools Used",
            ["Market Regime", "Dual Momentum", "Cross-Sectional Momentum",
             "Mean Reversion Scanner", "MACD", "Pairs Trading",
             "Relative Strength", "Yield Curve", "Fear & Greed",
             "Macro Research", "Implied Move", "Black-Scholes"],
            key="lt_tools"
        )

        reason = st.text_area(
            "Trade Thesis (required)",
            placeholder="e.g. Regime classifier shows Risk-Off. Dual momentum in bond mode. TLT RSI oversold at 28. Yield curve normalizing. Long TLT expecting rates to fall.",
            height=120,
            key="lt_reason"
        )

        if st.button("✅ Log Trade", key="lt_submit"):
            if not ticker:
                st.error("Please enter a ticker.")
            elif entry_price <= 0:
                st.error("Please enter a valid entry price.")
            elif not reason:
                st.error("Please enter your trade thesis — this is the most important field.")
            else:
                trade = {
                    "ticker": ticker,
                    "action": action,
                    "entry_date": str(entry_date),
                    "entry_price": float(entry_price),
                    "shares": float(shares),
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
                st.success(f"✅ Logged: {action} {int(shares)} shares of {ticker} at ${entry_price:.2f}")
                st.balloons()

        st.markdown("---")
        st.markdown("### 🔒 Close an Open Trade")
        open_trades = [(i, t) for i, t in enumerate(st.session_state.trades) if t.get("status") == "Open"]
        if open_trades:
            labels = [f"{t['action']} {t['ticker']} @ ${t['entry_price']:.2f} ({t['entry_date']})" for _, t in open_trades]
            sel = st.selectbox("Select trade to close", labels, key="ct_select")
            sel_idx = open_trades[labels.index(sel)][0]
            col_a, col_b = st.columns(2)
            with col_a:
                exit_price = st.number_input("Exit Price ($)", min_value=0.01, value=100.0, step=0.01, key="ct_price")
            with col_b:
                exit_date = st.date_input("Exit Date", value=date.today(), key="ct_date")
            if st.button("🔒 Close Trade", key="ct_submit"):
                st.session_state.trades[sel_idx]["status"] = "Closed"
                st.session_state.trades[sel_idx]["exit_price"] = float(exit_price)
                st.session_state.trades[sel_idx]["exit_date"] = str(exit_date)
                _save_trades(st.session_state.trades)
                t = st.session_state.trades[sel_idx]
                if t["action"] == "BUY":
                    pnl = (exit_price - t["entry_price"]) * t["shares"]
                    ret = (exit_price - t["entry_price"]) / t["entry_price"] * 100
                else:
                    pnl = (t["entry_price"] - exit_price) * t["shares"]
                    ret = (t["entry_price"] - exit_price) / t["entry_price"] * 100
                if pnl >= 0:
                    st.success(f"✅ Closed: {ret:+.1f}% | P&L: ${pnl:+,.0f}")
                else:
                    st.error(f"Closed: {ret:+.1f}% | P&L: ${pnl:+,.0f}")
        else:
            st.info("No open trades to close.")

        st.markdown("---")
        st.markdown("### 🗑️ Delete a Trade")
        if st.session_state.trades:
            del_labels = [f"{t['action']} {t['ticker']} @ ${t['entry_price']:.2f} ({t['entry_date']})"
                          for t in st.session_state.trades]
            del_sel = st.selectbox("Select trade to delete", del_labels, key="dt_select")
            del_idx = del_labels.index(del_sel)
            if st.button("🗑️ Delete", key="dt_submit"):
                deleted = st.session_state.trades.pop(del_idx)
                _save_trades(st.session_state.trades)
                st.warning(f"Deleted: {deleted['action']} {deleted['ticker']}")
                st.rerun()

    # =========================
    # TAB 3 — TRADE JOURNAL
    # =========================
    with tab3:
        st.markdown("### 📋 Trade Journal")
        if not st.session_state.trades:
            st.info("No trades logged yet.")
        else:
            df = _compute_analytics(st.session_state.trades)
            if df is not None and not df.empty:
                df_show = df.sort_values("entry_date", ascending=False)
                for i, (_, row) in enumerate(df_show.iterrows()):
                    pnl_icon = "🟢" if row["pnl_dollar"] >= 0 else "🔴"
                    status_tag = "🔵 Open" if row["status"] == "Open" else "⚫ Closed"
                    with st.expander(
                        f"{pnl_icon} {row['action']} {row['ticker']} | "
                        f"Entry: ${row['entry_price']:.2f} | "
                        f"P&L: {row['pnl_pct']:+.1f}% (${row['pnl_dollar']:+,.0f}) | {status_tag}"
                    ):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Entry Date", str(row["entry_date"])[:10])
                        c2.metric("Entry Price", f"${row['entry_price']:.2f}")
                        c3.metric("Current Price", f"${row['current_price']:.2f}")
                        c4.metric("P&L", f"{row['pnl_pct']:+.1f}%")
                        st.markdown("**📝 Trade Thesis:**")
                        st.info(row["reason"] if row["reason"] else "No thesis logged.")
                        if row["regime"]:
                            st.markdown(f"**Regime:** {row['regime']} | **Strategy:** {row['strategy']}")
                        if row["tools_used"]:
                            st.markdown(f"**Tools Used:** {', '.join(row['tools_used'])}")

                csv = df_show.to_csv(index=False)
                st.download_button(
                    "⬇️ Export to CSV",
                    csv,
                    file_name=f"apollo_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
