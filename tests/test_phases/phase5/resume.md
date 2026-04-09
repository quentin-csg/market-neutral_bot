# Phase 5 — Resume

## Fichiers crees

### `agent/model.py` — Configuration PPO + environnements vectorises
- **`make_env()`** : factory function pour creer un TradingEnv (compatible SubprocVecEnv)
- **`make_vec_env()`** : cree un environnement vectorise avec VecFrameStack
  - Support DummyVecEnv (mono-process) et SubprocVecEnv (multi-process)
  - VecFrameStack empile N observations pour donner de la memoire a l'agent
- **`create_agent()`** : instancie un PPO avec hyperparametres de config/settings.py
  - Policy : MlpPolicy (reseau MLP 64x64)
  - Integration TensorBoard pour le monitoring
- **`save_agent()` / `load_agent()`** : persistance du modele au format .zip
- **`get_feature_set()`** : retourne les features pour l'entrainement progressif

### `tests/test_agent.py` — 10 tests
- Import, feature sets, creation env, creation agent, predict, save/load, frame stacking, micro-entrainement

## Entrainement progressif (feature sets)

| Version | Features | Description |
|---|---|---|
| V1 | close, open, high, low, volume, rsi, rsi_normalized, sma_50, sma_200, sma_trend | OHLCV + techniques de base |
| V2 | V1 + qqq_close, spy_close, fear_greed, funding_rate, atr, bb, zscore, volume_ratio, log_returns, is_weekend | + macro + indicateurs avances |
| V3 | V2 + sentiment_score, n_articles, price_to_sma_long | + NLP FinBERT |

## Technologies utilisees
- **Stable-Baselines3** : framework RL (PPO)
- **SubprocVecEnv / DummyVecEnv** : parallelisation des environnements
- **VecFrameStack** : empilage des observations (24 frames = 1 jour de memoire H1)

## Architecture
```
agent/
├── model.py       # PPO config + vec env + save/load
└── reward.py      # Fonctions de recompense (Phase 4)
```

## Parametres (dans `config/settings.py`)
| Parametre | Valeur | Description |
|---|---|---|
| PPO learning_rate | 3e-4 | Taux d'apprentissage |
| PPO n_steps | 2048 | Steps par update |
| PPO batch_size | 64 | Taille du batch |
| PPO n_epochs | 10 | Epochs par update |
| PPO gamma | 0.99 | Facteur de discount |
| FRAME_STACK_SIZE | 24 | Bougies empilees |
| N_ENVS | 4 | Environnements paralleles |

## Tests
- **78/78 tests passent** (16 data + 22 features + 30 env + 10 agent)
- Aucune regression sur les phases precedentes
