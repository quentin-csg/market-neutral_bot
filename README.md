# Trading Bot RL

Bot de trading crypto basé sur **PPO** (Proximal Policy Optimization) avec CNN 1D, walk-forward validation, et dashboard temps réel. 100% local et gratuit.

**Stack :** Python 3.10+ · Stable-Baselines3 · Gymnasium · FinBERT · ccxt · Streamlit

---

## Installation

```bash
pip install -r requirements.txt
```

> Le modèle FinBERT (`ProsusAI/finbert`, ~500 Mo) est téléchargé automatiquement au premier lancement avec `--nlp`.

---

## Utilisation

```bash
# 1. Entraînement (1M steps, BTC/USDT 1h, 2020-2023)
python main.py train --model mon_modele

# 2. Backtest rapide (données 2024+)
python main.py backtest --model mon_modele

# 3. Walk-forward validation (expanding window, plusieurs heures)
python main.py walk-forward

# 4. Paper trading (tick toutes les heures, Ctrl+C pour arrêter)
python main.py live --model mon_modele

# 5. Live trading (argent réel — tester en paper d'abord !)
python main.py live --model mon_modele --live-mode

# Dashboard Streamlit (port 8501)
python main.py dashboard

# Visualiser l'entraînement
tensorboard --logdir logs/train/tensorboard
```

Pour le live trading, définir les clés API en variables d'environnement :

```bash
export EXCHANGE_API_KEY=...
export EXCHANGE_API_SECRET=...
```

---

## Métriques clés

| Métrique | Bon | Mauvais |
| --- | --- | --- |
| Total return | > 0% | < -10% |
| Sharpe ratio | > 1.0 | < 0 |
| Max drawdown | < 15% | > 25% |
| Nombre de trades | 50–300 | 0 ou > 500 |

Le **walk-forward** est la validation la plus fiable : il ré-entraîne et backteste sur plusieurs fenêtres temporelles successives. Un Sharpe moyen > 0.5 stable sur tous les folds est un bon signal.

---

## Structure

```
├── config/settings.py       # Hyperparamètres et configuration
├── data/                    # Fetchers (ccxt, yfinance, Fear&Greed, RSS)
├── features/                # Indicateurs techniques, FinBERT, scaler
├── env/trading_env.py       # Environnement Gymnasium
├── agent/                   # PPO + CNN 1D + reward
├── training/                # Train, backtest, walk-forward, logger
├── live/                    # Executor, circuit breaker, dashboard
├── models/                  # Modèles sauvegardés (.zip)
├── logs/                    # TensorBoard, backtests, walk-forward
└── main.py                  # CLI principal
```

---

## Licence

Projet personnel — usage libre.
