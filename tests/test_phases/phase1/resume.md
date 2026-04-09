# Phase 1 — Résumé

## Objectif
Mise en place des fondations du projet : structure, configuration centralisée, dépendances et point d'entrée CLI.

## Ce qui a été implémenté

### 📁 Structure du projet
Architecture modulaire avec séparation des responsabilités :
| Dossier | Rôle |
|---------|------|
| `config/` | Configuration centralisée (settings.py) |
| `data/` | Ingestion des données (crypto, macro, sentiment, news) |
| `features/` | Feature engineering (indicateurs techniques, NLP, scaling) |
| `env/` | Environnement Gymnasium pour la simulation de trading |
| `agent/` | Agent RL (PPO, fonctions de reward) |
| `training/` | Entraînement, backtest, logging |
| `live/` | Exécution live/paper, circuit breaker, dashboard |
| `tests/` | Tests unitaires |
| `models/` | Modèles sauvegardés (.zip) — gitignored |
| `logs/` | TensorBoard + logs hebdo — gitignored |
| `notebooks/` | Exploration / debug |
| `test_phases/` | Checklists de validation par phase |

### ⚙️ Configuration centralisée (`config/settings.py`)
Tous les paramètres du bot en un seul fichier, zéro magic number :
- **Crypto** : exchange (Binance), symbol (BTC/USDT), timeframe (1h, 4h)
- **Clés API** : via variables d'environnement (`EXCHANGE_API_KEY`, `EXCHANGE_API_SECRET`)
- **Macro** : symboles yfinance (QQQ, SPY)
- **Sentiment** : URL API Alternative.me (Fear & Greed)
- **News** : flux RSS (CoinDesk, CoinTelegraph, Yahoo Finance) + mots-clés de filtrage
- **NLP** : modèle FinBERT (`ProsusAI/finbert`), batch size, max articles
- **Indicateurs techniques** : périodes SMA, RSI, ATR, Bollinger
- **Environnement Gym** : capital initial (10 000 USDT), frais (0.1%), slippage (0-0.05%)
- **PPO** : hyperparamètres complets (lr, batch_size, n_epochs, gamma, etc.)
- **Reward** : fenêtre Sharpe (24), seuil drawdown (15%), pénalité position sizing
- **Entraînement** : split train/test (2020-2023 / 2024+), timesteps, checkpoints
- **Circuit breaker** : seuils chute prix (3%), spike volume (5x), fenêtre (5 min)
- **Dashboard** : port Streamlit (8501), refresh (10s)

### 📦 Dépendances (`requirements.txt`)
16 librairies organisées par catégorie :
| Catégorie | Librairies |
|-----------|-----------|
| Data Pipeline | `ccxt`, `yfinance`, `feedparser`, `requests` |
| Data Processing | `pandas`, `numpy` |
| Indicateurs Techniques | `pandas-ta` |
| NLP | `transformers`, `torch` |
| Scaling | `scikit-learn` |
| Environnement RL | `gymnasium` |
| Agent RL | `stable-baselines3` |
| Monitoring | `tensorboard` |
| Live Trading | `websockets` |
| Dashboard | `streamlit`, `plotly` |
| Tests | `pytest` |

### 🚀 Point d'entrée CLI (`main.py`)
Interface en ligne de commande avec 4 modes :
```
python main.py train        → Entraîner le modèle PPO
python main.py backtest     → Backtester sur données 2024+
python main.py live         → Trading paper/live (--live-mode)
python main.py dashboard    → Dashboard Streamlit
```

### 🛡️ .gitignore
Configuré pour exclure : `__pycache__`, `.env`, `models/`, `logs/`, fichiers temporaires.
Le dossier `env/` (module Gymnasium) n'est PAS ignoré.

## Technologies utilisées dans cette phase
- **Python 3** — langage principal
- **argparse** — CLI
- **pathlib** — gestion des chemins
- **os.getenv** — variables d'environnement pour les secrets

## Statut
✅ **Phase 1 validée** — Toutes les vérifications de `test_phase1.md` passées.
