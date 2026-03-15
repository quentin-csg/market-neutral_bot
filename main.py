"""
Trading Bot RL — Point d'entrée principal.
Usage:
    python main.py train        Lancer l'entraînement
    python main.py backtest     Lancer le backtest sur données de test
    python main.py live         Lancer le trading live/paper
    python main.py dashboard    Lancer le dashboard Streamlit
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Trading Bot RL — Bot de trading crypto basé sur PPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python main.py train          Entraîner le modèle PPO
  python main.py backtest       Backtest sur données 2024+
  python main.py live           Lancer en mode paper/live trading
  python main.py dashboard      Ouvrir le dashboard Streamlit
        """,
    )

    parser.add_argument(
        "command",
        choices=["train", "backtest", "live", "dashboard"],
        help="Commande à exécuter",
    )

    parser.add_argument(
        "--live-mode",
        action="store_true",
        default=False,
        help="Activer le trading réel (par défaut: paper trading)",
    )

    args = parser.parse_args()

    if args.command == "train":
        print("🚀 Lancement de l'entraînement...")
        # from training.train import run_training
        # run_training()
        print("⚠️  Module training pas encore implémenté (Phase 6)")

    elif args.command == "backtest":
        print("📊 Lancement du backtest...")
        # from training.backtest import run_backtest
        # run_backtest()
        print("⚠️  Module backtest pas encore implémenté (Phase 6)")

    elif args.command == "live":
        mode = "LIVE" if args.live_mode else "PAPER"
        print(f"🤖 Lancement du trading en mode {mode}...")
        # from live.executor import run_live
        # run_live(live_mode=args.live_mode)
        print("⚠️  Module live pas encore implémenté (Phase 7)")

    elif args.command == "dashboard":
        print("📈 Lancement du dashboard Streamlit...")
        # from live.dashboard import run_dashboard
        # run_dashboard()
        print("⚠️  Module dashboard pas encore implémenté (Phase 7)")


if __name__ == "__main__":
    main()
