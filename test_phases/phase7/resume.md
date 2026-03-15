# Resume Phase 7 - Live / Paper Trading

## Fichiers crees

| Fichier | Description |
|---------|-------------|
| `live/executor.py` | Boucle principale de trading (paper/live) |
| `live/circuit_breaker.py` | Surveillance temps reel des conditions de marche |
| `live/dashboard.py` | Dashboard Streamlit local |
| `tests/test_live.py` | 20 tests unitaires |
| `main.py` | CLI complet (train/backtest/live/dashboard) |

## Fonctionnalites implementees

### live/executor.py
- **PaperPortfolio** : simulation de portefeuille (achat/vente/hold)
  - Zone morte 5% (action trop faible = hold)
  - Frais de 0.1% par trade
  - Historique des trades
  - Statistiques (net worth, return, trades, fees)
- **LiveExecutor** : boucle de trading
  - Mode paper (PaperPortfolio) et mode live (ordres reels via ccxt)
  - Scheduler toutes les heures
  - Fetch donnees recentes -> features -> prediction PPO -> ordre
  - Logs automatiques dans logs/live/
  - Ctrl+C pour arreter proprement

### live/circuit_breaker.py
- **CircuitBreaker** : surveillance continue
  - Polling bougies 1 minute via ccxt
  - Detection chute de prix > 3% en 5 minutes
  - Detection volume anormal > 5x la moyenne
  - Fermeture instantanee des positions si declenche
  - Etat trigger/reset
  - Mode paper (simulation) et live (ordres reels)

### live/dashboard.py
- **Dashboard Streamlit** avec 3 onglets :
  - Live/Paper : KPIs (net worth, PnL, return, trades) + graphique PnL cumule
  - Backtests : resultats des backtests (train ou live)
  - Modeles : liste des modeles sauvegardes + lien TensorBoard
- Charge les donnees depuis les CSV et JSON de logs

### main.py (mis a jour)
- 4 commandes fonctionnelles : `train`, `backtest`, `live`, `dashboard`
- Arguments : `--model`, `--live-mode`, `--timesteps`, `--nlp`
- Logging configure automatiquement

## Technologies utilisees
- ccxt (ordres live/paper, fetch OHLCV)
- Streamlit (dashboard local)
- stable-baselines3 (prediction PPO)
- subprocess (lancement Streamlit depuis CLI)

## Architecture live
```
live/
  executor.py         # Boucle principale (1 tick/heure)
  circuit_breaker.py  # Surveillance temps reel (1 check/minute)
  dashboard.py        # Interface Streamlit (port 8501)
```

## Utilisation

### Lancer le paper trading
```bash
python main.py live --model ppo_trading
```

### Lancer le live trading (ATTENTION: argent reel)
```bash
export EXCHANGE_API_KEY="..."
export EXCHANGE_API_SECRET="..."
python main.py live --model ppo_trading --live-mode
```

### Lancer le dashboard
```bash
python main.py dashboard
```

### Lancer le circuit breaker seul
```python
from live.circuit_breaker import run_circuit_breaker
run_circuit_breaker(live_mode=False)
```

## Tests
- 20 nouveaux tests (7 portfolio + 8 circuit breaker + 3 executor + 2 CLI)
- Total projet: 114 tests, tous passent
