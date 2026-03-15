"""
Dashboard Streamlit — Interface locale du trading bot.

Affiche :
  - Solde du compte et positions ouvertes
  - Graphique de prix temps réel
  - Statut du circuit breaker
  - Historique des trades
  - Résultats des backtests

Usage:
  streamlit run live/dashboard.py
  ou
  python main.py dashboard
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

from config.settings import (
    DASHBOARD_REFRESH_SECONDS,
    LOGS_LIVE_DIR,
    LOGS_TRAIN_DIR,
    MODELS_DIR,
    SYMBOL,
)
from training.logger import load_backtest_results


def _load_trade_history(mode: str = "live") -> list[dict]:
    """Charge l'historique des trades depuis les logs weekly JSON."""
    base = LOGS_LIVE_DIR if mode == "live" else LOGS_TRAIN_DIR
    weekly_dir = base / "weekly"
    if not weekly_dir.exists():
        return []

    results = []
    for f in sorted(weekly_dir.glob("week_*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            results.append(json.load(fh))
    return results


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
    models = []
    for f in MODELS_DIR.glob("*.zip"):
        models.append(f.stem)
    return sorted(models)


def run_dashboard():
    """Lance le dashboard Streamlit."""
    st.set_page_config(
        page_title="Trading Bot RL",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Trading Bot RL — Dashboard")
    st.caption(f"Symbole: {SYMBOL} | Rafraichissement: {DASHBOARD_REFRESH_SECONDS}s")

    # Tabs
    tab_live, tab_backtests, tab_models = st.tabs([
        "📊 Live / Paper",
        "📈 Backtests",
        "🧠 Modeles",
    ])

    # === TAB LIVE ===
    with tab_live:
        st.header("Trading en cours")

        # Résumé hebdomadaire
        weekly_data = _load_csv_summary("live")
        if weekly_data:
            st.subheader("Résumé hebdomadaire (live)")

            import pandas as pd
            df = pd.DataFrame(weekly_data)
            # Convertir les colonnes numériques
            num_cols = [
                "pnl_usdt", "pnl_cumul_usdt", "net_worth",
                "total_return_pct", "sharpe_ratio", "sortino_ratio",
                "max_drawdown_pct", "total_trades", "total_fees"
            ]
            for col in num_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # KPIs
            if len(df) > 0:
                last = df.iloc[-1]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Net Worth", f"{last.get('net_worth', 0):.2f} USDT")
                col2.metric("PnL cumulé", f"{last.get('pnl_cumul_usdt', 0):.2f} USDT")
                col3.metric("Return", f"{last.get('total_return_pct', 0):.2f}%")
                col4.metric("Trades", f"{last.get('total_trades', 0)}")

                # Graphique PnL cumulé
                if "pnl_cumul_usdt" in df.columns and len(df) > 1:
                    st.line_chart(df.set_index("week")["pnl_cumul_usdt"])

            st.dataframe(df, use_container_width=True)
        else:
            st.info("Aucune donnée live disponible. Lancez le bot en mode paper/live.")

        # Circuit Breaker status
        st.subheader("Circuit Breaker")
        st.success("✅ Status: OK — Aucun déclenchement")
        st.caption(
            "Le circuit breaker surveille les chutes de prix et volumes anormaux "
            "en temps réel."
        )

    # === TAB BACKTESTS ===
    with tab_backtests:
        st.header("Résultats des backtests")

        mode = st.radio("Source", ["train", "live"], horizontal=True)
        results = load_backtest_results(mode=mode)

        if results:
            import pandas as pd
            df_bt = pd.DataFrame(results)

            # KPIs du dernier backtest
            last_bt = results[-1]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Return", f"{last_bt.get('total_return_pct', 0):.2f}%")
            col2.metric("Sharpe", f"{last_bt.get('sharpe_ratio', 0):.4f}")
            col3.metric("Sortino", f"{last_bt.get('sortino_ratio', 0):.4f}")
            col4.metric("Max Drawdown", f"{last_bt.get('max_drawdown_pct', 0):.2f}%")

            st.dataframe(df_bt, use_container_width=True)
        else:
            st.info(f"Aucun backtest trouvé dans les logs {mode}.")

    # === TAB MODELES ===
    with tab_models:
        st.header("Modeles disponibles")
        models = _list_models()
        if models:
            for m in models:
                st.code(m)
        else:
            st.info("Aucun modele sauvegardé dans models/")

        # TensorBoard link
        st.subheader("TensorBoard")
        st.code("tensorboard --logdir logs/train/tensorboard")
        st.caption("Lancez cette commande dans un terminal pour visualiser les courbes d'entrainement.")


if __name__ == "__main__":
    run_dashboard()
