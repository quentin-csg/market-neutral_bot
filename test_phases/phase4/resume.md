# Phase 4 — Résumé

## Fichiers créés

### `agent/reward.py` — Fonctions de récompense
- **log_return_reward** : récompense basée sur les log-returns du portefeuille (pénalise plus les grosses pertes)
- **sharpe_reward** : ratio de Sharpe sur fenêtre glissante (force la régularité)
- **sortino_reward** : comme Sharpe mais ne pénalise que la volatilité baissière
- **drawdown_penalty** : pénalité exponentielle quand le drawdown dépasse le seuil configuré (15%)
- **position_size_penalty** : pénalité quadratique pour les positions trop grosses par rapport au capital
- **transaction_cost_penalty** : pénalité proportionnelle aux frais pour empêcher l'hyper-trading
- **compute_reward** : combine toutes les composantes avec pondérations ajustables

### `env/trading_env.py` — Environnement Gymnasium
- **Classe `TradingEnv(gymnasium.Env)`** compatible Stable-Baselines3
- **Observation space** : features marché normalisées + 3 features portfolio (solde, position ratio, PnL non réalisé)
- **Action space** : continu `[-1, +1]` → vente 100% à achat 100%
- **Frais** : 0.1% par trade (configurable via `config/settings.py`)
- **Slippage** : aléatoire entre 0% et 0.05% (configurable)
- **Zone morte** : actions < 5% ignorées (anti-bruit)
- **Terminaison** : fin des données OU portefeuille ruiné (< 10% du capital initial)
- **Stats portfolio** : return total, max drawdown, Sharpe, Sortino, nombre de trades, frais totaux
- **Render** : modes "human" (console) et "log" (logging Python)

### `tests/test_env.py` — 30 tests
- 13 tests pour les fonctions de reward
- 17 tests pour l'environnement TradingEnv

## Technologies utilisées
- **Gymnasium** (ex-OpenAI Gym) : interface standard pour environnements RL
- **NumPy** : calculs vectorisés (log-returns, Sharpe, drawdown)
- **config/settings.py** : tous les paramètres (frais, slippage, seuils) centralisés

## Architecture
```
agent/
└── reward.py          # Fonctions de récompense modulaires
env/
└── trading_env.py     # Environnement Gymnasium
tests/
└── test_env.py        # 30 tests unitaires
```

## Paramètres configurables (dans `config/settings.py`)
| Paramètre | Valeur | Description |
|---|---|---|
| INITIAL_BALANCE | 10 000 USDT | Capital initial |
| TRADING_FEE | 0.1% | Frais par trade |
| SLIPPAGE_MIN | 0% | Slippage minimum |
| SLIPPAGE_MAX | 0.05% | Slippage maximum |
| SHARPE_WINDOW | 24 bougies | Fenêtre pour Sharpe/Sortino |
| MAX_DRAWDOWN_PENALTY | 15% | Seuil de drawdown exponentiel |
| POSITION_SIZE_PENALTY_FACTOR | 0.1 | Facteur de pénalité position |

## Remarque importante (entraînement progressif)
L'architecture supporte toutes les features, mais l'entraînement doit se faire par étapes :
- **V1** : OHLCV + RSI + SMA uniquement
- **V2** : + données macro (QQQ, SPY, Fear & Greed)
- **V3** : + sentiment NLP (FinBERT)

Cela permet à l'agent de converger progressivement et évite le bruit excessif.

## Tests
- **68/68 tests passent** (16 data + 22 features + 30 env)
- Aucune régression sur les phases précédentes
