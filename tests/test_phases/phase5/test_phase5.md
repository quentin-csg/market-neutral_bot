# Phase 5 — Checklist de validation

## Agent PPO (`agent/model.py`)

### Tests automatiques (pytest)
```
python -m pytest tests/test_agent.py -v
```
**10 tests a verifier :**

- [X] Import de toutes les fonctions (create_agent, make_vec_env, save_agent, load_agent, get_feature_set)
- [X] Feature sets V1/V2/V3 coherents (V1 < V2 < V3 en nombre de features)
- [X] Feature set invalide leve une erreur
- [X] make_env cree un TradingEnv fonctionnel
- [X] make_vec_env cree un environnement vectorise avec la bonne shape
- [X] create_agent instancie un PPO sans crash
- [X] predict retourne une action dans [-1, +1]
- [X] save/load fonctionne (predictions identiques apres reload)
- [X] VecFrameStack modifie la shape (x frame_stack)
- [X] Micro-entrainement de 128 steps sans crash

## Tests manuels (REPL Python)

### Test 1 — Creation agent + predict
```
import numpy as np, pandas as pd
from agent.model import create_agent, make_vec_env, get_feature_set
np.random.seed(42)
n = 300
prices = 42000 + np.cumsum(np.random.randn(n) * 100)
df = pd.DataFrame({'close': prices, 'open': prices, 'high': prices + 50, 'low': prices - 50, 'volume': np.abs(np.random.randn(n) * 500 + 1000), 'rsi': np.random.rand(n) * 2 - 1, 'rsi_normalized': np.random.rand(n) * 2 - 1, 'sma_50': prices, 'sma_200': prices, 'sma_trend': np.random.choice([-1.0, 1.0], n)})
print(f"Features V1: {get_feature_set('v1')}")
vec_env = make_vec_env(df, n_envs=1, feature_columns=get_feature_set('v1'), use_subproc=False, frame_stack=4)
obs = vec_env.reset()
print(f"Obs shape: {obs.shape}")
agent = create_agent(vec_env, hyperparams={'verbose': 0, 'n_steps': 64}, tensorboard_log=None, seed=42)
action, _ = agent.predict(obs, deterministic=True)
print(f"Action: {action}")
vec_env.close()
```
- [X] Obs shape = (1, 52) soit 13 features * 4 frames
- [X] Action dans [-1, +1]

### Test 2 — Micro-entrainement
```
import numpy as np, pandas as pd
from agent.model import create_agent, make_vec_env
np.random.seed(42)
n = 500
prices = 42000 + np.cumsum(np.random.randn(n) * 100)
df = pd.DataFrame({'close': prices, 'rsi': np.random.rand(n) * 2 - 1})
vec_env = make_vec_env(df, n_envs=2, feature_columns=['close', 'rsi'], use_subproc=False, frame_stack=8)
agent = create_agent(vec_env, hyperparams={'verbose': 0, 'n_steps': 64, 'batch_size': 32}, tensorboard_log=None, seed=42)
agent.learn(total_timesteps=256)
obs = vec_env.reset()
action, _ = agent.predict(obs, deterministic=True)
print(f"Apres entrainement - Action: {action}, shape: {action.shape}")
vec_env.close()
print('OK')
```
- [X] Entrainement sans crash
- [X] Prediction fonctionnelle apres entrainement

## Tous les tests (regression)
```
python -m pytest tests/ -v
```
- [X] 78/78 tests passent (16 data + 22 features + 30 env + 10 agent)
