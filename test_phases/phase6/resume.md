# Resume Phase 6 - Entrainement & Backtest

## Fichiers crees

| Fichier | Description |
|---------|-------------|
| `training/train.py` | Script d'entrainement PPO complet |
| `training/backtest.py` | Script de validation sur donnees de test |
| `training/logger.py` | Systeme de logs (weekly + backtest + TensorBoard) |
| `tests/test_training.py` | 12 tests unitaires et d'integration |

## Fonctionnalites implementees

### training/train.py
- Chargement automatique des donnees via `build_full_pipeline()`
- Creation d'environnements vectorises (SubprocVecEnv ou DummyVecEnv)
- Callbacks: CheckpointCallback (sauvegarde tous les N steps) + TensorBoard
- Sauvegarde du scaler pour reutilisation en backtest/live
- Sauvegarde du modele final dans `models/`
- Parametres configurables: timesteps, n_envs, frame_stack, seed, feature_columns
- Support NLP optionnel (include_nlp=True/False)
- Barre de progression (progress_bar=True)

### training/backtest.py
- Chargement d'un modele sauvegarde
- Evaluation en mode deterministe (pas d'exploration)
- Metriques calculees: total_return, net_worth, sharpe, sortino, drawdown, trades, fees
- Sauvegarde automatique des resultats en JSON
- Unwrap correct de VecFrameStack -> DummyVecEnv -> TradingEnv

### training/logger.py
- Logs hebdomadaires en JSON dans `logs/weekly/`
- Resultats de backtest en JSON dans `logs/backtests/`
- Chargement et filtrage des resultats pour comparaison
- Serialisation automatique des types numpy
- Affichage formate des statistiques

## Technologies utilisees
- stable-baselines3 (PPO, callbacks, vec envs)
- TensorBoard (via tb_log_name pour nommer les runs)
- JSON (format de stockage des logs)
- pytest + unittest.mock (tests avec mocks pour eviter les appels reseau)

## Architecture des logs
```
logs/
  tensorboard/        # Logs TensorBoard (metriques en temps reel)
    ppo_trading_YYYYMMDD_HHMMSS_0/
  weekly/             # Resumes hebdomadaires
    week_2024_W01.json
  backtests/          # Resultats de backtest
    backtest_ppo_trading_run001.json
models/
  ppo_trading_YYYYMMDD_HHMMSS.zip   # Modele final
  checkpoints/                       # Checkpoints intermediaires
    ppo_trading_50000_steps.zip
```

## Utilisation

### Lancer un entrainement
```python
from training.train import train

model_path = train(
    total_timesteps=1_000_000,
    n_envs=4,
    model_name="mon_modele",
    use_subproc=True,
)
```

### Lancer un backtest
```python
from training.backtest import backtest

stats = backtest(model_name="mon_modele")
```

### Visualiser dans TensorBoard
```
tensorboard --logdir logs/tensorboard
```

## Tests
- 12 nouveaux tests (6 logger + 2 training + 2 backtest + 2 integration)
- Total projet: 90 tests, tous passent
