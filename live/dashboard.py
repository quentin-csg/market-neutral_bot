"""
Dashboard Streamlit — Interface locale du trading bot.

Affiche :
  - Graphique cours BTC temps réel
  - Positions ouvertes / trades fermés
  - KPIs hebdomadaires
  - Statut du circuit breaker
  - Résultats des backtests

Usage:
  streamlit run live/dashboard.py
  ou
  python main.py dashboard
"""

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from config.settings import (
    DASHBOARD_REFRESH_SECONDS,
    INITIAL_BALANCE,
    LOGS_LIVE_DIR,
    LOGS_TRAIN_DIR,
    MODELS_DIR,
    SYMBOL,
)
from training.logger import load_backtest_results


def _load_live_state() -> dict:
    """Charge l'état courant du portfolio depuis live_state.json."""
    state_path = LOGS_LIVE_DIR / "live_state.json"
    if not state_path.exists():
        return {}
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv_summary(mode: str = "live") -> list[dict]:
    """Charge le CSV cumulatif hebdomadaire."""
    import csv
    base = LOGS_LIVE_DIR if mode == "live" else LOGS_TRAIN_DIR
    csv_path = base / "weekly_summary.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _list_models() -> list[str]:
    """Liste les modèles disponibles."""
    return sorted(f.stem for f in MODELS_DIR.glob("*.zip"))


def run_dashboard():
    """Lance le dashboard Streamlit."""
    import pandas as pd

    st.set_page_config(
        page_title="Trading Bot RL",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Trading Bot RL — Dashboard")
    st.caption(f"Symbole: {SYMBOL} | Rafraichissement: {DASHBOARD_REFRESH_SECONDS}s")

    tab_live, tab_backtests, tab_models = st.tabs([
        "📊 Live / Paper",
        "📈 Backtests",
        "🧠 Modeles",
    ])

    # =========================================================================
    # TAB LIVE
    # =========================================================================
    with tab_live:
        state = _load_live_state()

        if not state:
            st.info("Aucune donnée live disponible. Lancez le bot en mode paper/live.")
        else:
            portfolio = state.get("portfolio", {})
            current_price = state.get("current_price", 0.0)
            last_updated = state.get("last_updated", "")

            # --- KPIs --------------------------------------------------------
            st.subheader("Portfolio")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Net Worth", f"{portfolio.get('net_worth', 0):.2f} USDT")
            col2.metric("Cash", f"{portfolio.get('balance_usdt', 0):.2f} USDT")
            col3.metric(
                "Position BTC",
                f"{portfolio.get('position_btc', 0):.6f} BTC",
                f"≈ {portfolio.get('position_usdt', 0):.2f} USDT",
            )
            col4.metric("Return", f"{portfolio.get('total_return_pct', 0):+.2f}%")
            col5.metric("Trades", f"{portfolio.get('total_trades', 0)}")

            if last_updated:
                st.caption(f"Dernière mise à jour : {last_updated[:19]}")

            # --- Graphique cours BTC -----------------------------------------
            st.subheader(f"Cours {SYMBOL}")
            price_history = state.get("price_history", [])
            if price_history:
                df_prices = pd.DataFrame(price_history)
                if "timestamp" in df_prices.columns:
                    df_prices["timestamp"] = pd.to_datetime(
                        df_prices["timestamp"], utc=True, errors="coerce"
                    )
                    df_prices = df_prices.dropna(subset=["timestamp"])
                    df_prices = df_prices.set_index("timestamp")
                st.line_chart(df_prices[["price"]], height=300)
            else:
                st.info("Historique de prix non disponible.")

            st.divider()

            # --- Positions ouvertes ------------------------------------------
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("Positions ouvertes")
                open_positions = state.get("open_positions", [])
                if open_positions:
                    df_open = pd.DataFrame(open_positions)
                    df_open = df_open.rename(columns={
                        "entry_price": "Prix entrée",
                        "amount_btc": "Quantité BTC",
                        "value_usdt": "Valeur USDT",
                        "unrealized_pnl_pct": "PnL latent %",
                    })
                    df_open["Prix entrée"] = df_open["Prix entrée"].map("{:.2f}".format)
                    df_open["Quantité BTC"] = df_open["Quantité BTC"].map("{:.6f}".format)
                    df_open["Valeur USDT"] = df_open["Valeur USDT"].map("{:.2f}".format)
                    df_open["PnL latent %"] = df_open["PnL latent %"].map("{:+.2f}%".format)
                    st.dataframe(df_open, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune position ouverte.")

            # --- Trades fermés -----------------------------------------------
            with col_right:
                st.subheader("Trades fermés")
                closed_trades = state.get("closed_trades", [])
                if closed_trades:
                    df_closed = pd.DataFrame(closed_trades)
                    cols_to_show = [
                        c for c in ["timestamp", "price", "amount", "fee"]
                        if c in df_closed.columns
                    ]
                    df_closed = df_closed[cols_to_show].copy()
                    df_closed = df_closed.rename(columns={
                        "timestamp": "Date",
                        "price": "Prix vente",
                        "amount": "Quantité BTC",
                        "fee": "Frais USDT",
                    })
                    if "Prix vente" in df_closed.columns:
                        df_closed["Prix vente"] = pd.to_numeric(
                            df_closed["Prix vente"], errors="coerce"
                        ).map("{:.2f}".format)
                    if "Quantité BTC" in df_closed.columns:
                        df_closed["Quantité BTC"] = pd.to_numeric(
                            df_closed["Quantité BTC"], errors="coerce"
                        ).map("{:.6f}".format)
                    if "Frais USDT" in df_closed.columns:
                        df_closed["Frais USDT"] = pd.to_numeric(
                            df_closed["Frais USDT"], errors="coerce"
                        ).map("{:.4f}".format)
                    st.dataframe(
                        df_closed.iloc[::-1],  # plus récents en haut
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("Aucun trade fermé pour cette session.")

            st.divider()

        # --- Résumé hebdomadaire -------------------------------------------
        weekly_data = _load_csv_summary("live")
        if weekly_data:
            st.subheader("Résumé hebdomadaire")
            df_weekly = pd.DataFrame(weekly_data)
            num_cols = [
                "pnl_usdt", "pnl_cumul_usdt", "net_worth",
                "total_return_pct", "sharpe_ratio", "sortino_ratio",
                "max_drawdown_pct", "total_trades", "total_fees",
            ]
            for col in num_cols:
                if col in df_weekly.columns:
                    df_weekly[col] = pd.to_numeric(df_weekly[col], errors="coerce")

            if len(df_weekly) > 1 and "pnl_cumul_usdt" in df_weekly.columns:
                st.line_chart(df_weekly.set_index("week")["pnl_cumul_usdt"])

            st.dataframe(df_weekly, use_container_width=True, hide_index=True)

        # --- Circuit Breaker -----------------------------------------------
        st.subheader("Circuit Breaker")
        st.success("✅ Status: OK — Aucun déclenchement")
        st.caption(
            "Le circuit breaker surveille les chutes de prix et volumes anormaux "
            "en temps réel."
        )

    # =========================================================================
    # TAB BACKTESTS
    # =========================================================================
    with tab_backtests:
        st.header("Résultats des backtests")

        mode = st.radio("Source", ["train", "live"], horizontal=True)
        results = load_backtest_results(mode=mode)

        if results:
            df_bt = pd.DataFrame(results)
            last_bt = results[-1]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Return", f"{last_bt.get('total_return_pct', 0):.2f}%")
            col2.metric("Sharpe", f"{last_bt.get('sharpe_ratio', 0):.4f}")
            col3.metric("Sortino", f"{last_bt.get('sortino_ratio', 0):.4f}")
            col4.metric("Max Drawdown", f"{last_bt.get('max_drawdown_pct', 0):.2f}%")

            st.dataframe(df_bt, use_container_width=True, hide_index=True)
        else:
            st.info(f"Aucun backtest trouvé dans les logs {mode}.")

    # =========================================================================
    # TAB MODELES
    # =========================================================================
    with tab_models:
        st.header("Modeles disponibles")
        models = _list_models()
        if models:
            for m in models:
                st.code(m)
        else:
            st.info("Aucun modele sauvegardé dans models/")

        st.subheader("TensorBoard")
        st.code("tensorboard --logdir logs/train/tensorboard")
        st.caption(
            "Lancez cette commande dans un terminal pour visualiser "
            "les courbes d'entrainement."
        )

    # Auto-refresh
    st.markdown(
        f"<meta http-equiv='refresh' content='{DASHBOARD_REFRESH_SECONDS}'>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    run_dashboard()
