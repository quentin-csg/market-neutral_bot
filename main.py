"""
Trading Bot RL — Point d'entrée principal.
Usage:
    python main.py train          Lancer l'entraînement
    python main.py backtest       Lancer le backtest sur données de test
    python main.py walk-forward   Lancer la walk-forward validation
    python main.py live           Lancer le trading live/paper
    python main.py dashboard      Lancer le dashboard Streamlit
"""

import argparse
import logging
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Trading Bot RL — Bot de trading crypto basé sur PPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python main.py train                    Entraîner le modèle PPO
  python main.py train --timesteps 500000 Entraîner avec 500k steps
  python main.py backtest                 Backtest sur données 2024+
  python main.py backtest --model v1      Backtest avec un modèle spécifique
  python main.py live                     Lancer en mode paper trading
  python main.py live --live-mode         Lancer en mode live réel
  python main.py dashboard               Ouvrir le dashboard Streamlit
        """,
    )

    parser.add_argument(
        "command",
        choices=["train", "backtest", "walk-forward", "live", "dashboard"],
        help="Commande à exécuter",
    )

    parser.add_argument(
        "--model",
        default="ppo_trading",
        help="Nom du modèle (défaut: ppo_trading)",
    )

    parser.add_argument(
        "--live-mode",
        action="store_true",
        default=False,
        help="Activer le trading réel (par défaut: paper trading)",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Nombre de timesteps d'entraînement (défaut: config)",
    )

    parser.add_argument(
        "--nlp",
        action="store_true",
        default=False,
        help="Inclure l'analyse NLP FinBERT",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "train":
        from training.train import train

        kwargs = {
            "model_name": args.model,
            "include_nlp": args.nlp,
        }
        if args.timesteps:
            kwargs["total_timesteps"] = args.timesteps

        print("Lancement de l'entraînement...")
        train(**kwargs)

    elif args.command == "backtest":
        from training.backtest import backtest

        print("Lancement du backtest...")
        backtest(
            model_name=args.model,
            include_nlp=args.nlp,
        )

    elif args.command == "walk-forward":
        from training.walk_forward import walk_forward_validate

        print("Lancement de la walk-forward validation...")
        kwargs = {"include_nlp": args.nlp}
        if args.timesteps:
            kwargs["total_timesteps"] = args.timesteps
        walk_forward_validate(**kwargs)

    elif args.command == "live":
        from live.executor import run_live

        mode = "LIVE" if args.live_mode else "PAPER"
        print(f"Lancement du trading en mode {mode}...")
        run_live(
            model_name=args.model,
            live_mode=args.live_mode,
            include_nlp=args.nlp,
        )

    elif args.command == "dashboard":
        print("Lancement du dashboard Streamlit...")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run",
             "live/dashboard.py", "--server.headless", "true"],
            check=True,
        )


if __name__ == "__main__":
    main()
