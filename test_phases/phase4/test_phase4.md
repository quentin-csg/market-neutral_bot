# Phase 4 — Checklist de validation

## Environnement Gymnasium (`env/trading_env.py`)

### Tests automatiques (pytest)
```
python -m pytest tests/test_env.py -v
```
**30 tests a verifier :**

### Reward (`agent/reward.py`)
- [ ] Import de toutes les fonctions de reward
- [ ] `log_return_reward` positif quand net_worth augmente
- [ ] `log_return_reward` negatif quand net_worth baisse
- [ ] `log_return_reward` retourne -1.0 si net_worth = 0
- [ ] `sharpe_reward` retourne 0 avec trop peu de donnees
- [ ] `sharpe_reward` calcul correct avec donnees suffisantes
- [ ] `sortino_reward` positif avec des returns positifs
- [ ] `drawdown_penalty` = 0 quand pas de drawdown
- [ ] `drawdown_penalty` negatif avec drawdown
- [ ] `drawdown_penalty` exponentielle au-dela du seuil
- [ ] `position_size_penalty` plus severe pour grosses positions
- [ ] `transaction_cost_penalty` proportionnelle au cout
- [ ] `compute_reward` retourne un tuple (float, dict)

### TradingEnv
- [ ] Import reussi
- [ ] Init avec observation_space et action_space valides
- [ ] `reset()` retourne (obs, info) corrects
- [ ] Hold (action ~0) = pas de trade
- [ ] Buy (action > 0) = position augmente, balance diminue
- [ ] Buy then Sell = position revient a 0
- [ ] Frais de 0.1% appliques (net_worth < initial apres trade)
- [ ] Slippage applique (entry_price > prix marche)
- [ ] Episode complet se termine correctement
- [ ] Statistiques de portfolio coherentes
- [ ] Terminaison si portefeuille ruine (< 10% du capital)
- [ ] Observation dans observation_space
- [ ] Actions dans action_space + observations valides apres step
- [ ] Render mode "human" ne crashe pas
- [ ] Zone morte (action < 5% = pas de trade)
- [ ] Feature columns personnalisables
- [ ] Conformite API Gymnasium (obs, reward, terminated, truncated, info)

## Tests manuels (REPL Python)

Lancer `python` puis copier-coller chaque bloc (SANS les `>>>`).

### Test 1 — Episode complet avec stats
```
import numpy as np, pandas as pd
from env.trading_env import TradingEnv
np.random.seed(42)
n = 500
prices = 42000 + np.cumsum(np.random.randn(n) * 100)
df = pd.DataFrame({'close': prices, 'open': prices + np.random.randn(n) * 50, 'high': prices + abs(np.random.randn(n) * 100), 'low': prices - abs(np.random.randn(n) * 100), 'volume': np.abs(np.random.randn(n) * 500 + 1000), 'rsi': np.random.rand(n) * 2 - 1})
env = TradingEnv(df=df, initial_balance=10000.0)
obs, info = env.reset(seed=42)
print(f"Obs shape: {obs.shape}, dtype: {obs.dtype}")
print(f"Balance: {info['balance']}, Position: {info['position']}")
```
```
terminated = False
while not terminated:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

stats = env.get_portfolio_stats()
for k, v in stats.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

```
- [X] Episode complet sans crash
- [X] Stats de portfolio coherentes (return, drawdown, Sharpe)

### Test 2 — Verification des frais
```
import numpy as np, pandas as pd
from env.trading_env import TradingEnv
df = pd.DataFrame({'close': [10000.0] * 50, 'rsi': [0.0] * 50})
env = TradingEnv(df=df, initial_balance=10000.0, slippage_min=0.0, slippage_max=0.0)
env.reset(seed=42)
env.step(np.array([1.0]))
print(f"Fees paid: {env.total_fees_paid:.2f}")
print(f"Position value: {env.position * 10000:.2f}")
print(f"Net worth: {env.balance + env.position * 10000:.2f}")
```
- [X] Frais = ~10 USDT (0.1% de 10000)
- [X] Net worth < 10000 apres achat (a cause des frais)

### Test 3 — Reward composite
```
from agent.reward import compute_reward
total, components = compute_reward(net_worth=10500, prev_net_worth=10000, peak_net_worth=10500, position_ratio=0.5, trade_cost=10.0, returns_history=[0.01, 0.005, -0.002, 0.003, 0.008, 0.002, -0.001])
print(f"Reward total: {total:.6f}")
for k, v in components.items():
    print(f"  {k}: {v:+.6f}")

```
- [X] log_return positif (~0.0488)
- [X] drawdown = 0 (on est au pic)
- [X] position_size negatif
- [X] transaction negatif
- [X] reward total coherent

## Tous les tests (regression)
```
python -m pytest tests/ -v
```
- [ ] 68/68 tests passent (16 data + 22 features + 30 env)
