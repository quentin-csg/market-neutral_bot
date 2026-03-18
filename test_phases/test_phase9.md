# Phase 9 — Architecture RL complète

## Objectifs

Corriger les bugs critiques de reward, réduire le bruit dans les features, améliorer l'architecture réseau, et ajouter une validation statistique robuste.

---

## Changements appliqués

### Step 1 — Correction des rewards (`agent/reward.py`)

**Problèmes corrigés :**
- `sharpe_reward()` : annualisé avec `sqrt(8760)` pour returns horaires
- `sortino_reward()` : edge case sans downside returns → `min(mean / 1e-8, 3.0)` au lieu de `mean * 10`
- `drawdown_penalty()` : cappé à `-DRAWDOWN_PENALTY_CAP` (2.0) pour éviter les spikes explosifs

**Nouveau paramètre config :** `DRAWDOWN_PENALTY_CAP = 2.0`

---

### Step 2 — Feature sets progressifs (`agent/model.py`)

**3 sets progressifs sans doublons :**

| Set | Features | Contenu |
|-----|----------|---------|
| V1  | 7        | RSI, SMA trend, price_to_sma_long, ATR, Bollinger, log_return |
| V2  | 14       | V1 + volume, momentum multi-horizon, Fear&Greed, funding_rate, is_weekend |
| V3  | 18       | V2 + MACD hist, ADX, RSI 4h, SMA trend 4h |

**Features live-only** (pas incluses dans V1/V2/V3 mais disponibles en live) :
- `orderbook_imbalance`, `oi_change_pct`

---

### Step 3 — VecNormalize (`agent/model.py`, `training/train.py`, `training/backtest.py`)

Pipeline : `DummyVecEnv → VecNormalize(norm_obs=False, norm_reward=True) → VecFrameStack`

- Normalisation des rewards seulement (les obs sont déjà normalisées par RobustScaler)
- Sauvegarde automatique après training : `models/vec_normalize.pkl`
- Chargement en mode eval pour backtest/live

**Nouvelles fonctions :** `save_vec_normalize()`, `load_vec_normalize()`, `_get_vec_normalize()`

---

### Step 4 — Executor refactorisé (`live/executor.py`)

**Avant :** recréait un vec_env complet + rechargait le modèle à chaque tick (~30s)

**Après :** modèle chargé une seule fois au démarrage, tick < 5s

- `_load_model()` : PPO + FeatureScaler chargés au `__init__()`
- `_build_observation(dataset)` : construit l'obs stacked (frame_stack × n_features) directement sans environnement Gymnasium
- `tick()` : fetch → build obs → predict → execute

---

### Step 5 — CNN 1D feature extractor (`agent/model.py`)

Remplace le MLP flat sur les observations empilées par un CNN 1D qui détecte les patterns temporels.

**Architecture :**
```
Input: (batch, n_stack × n_features)
  → Reshape: (batch, n_features, n_stack)
  → Conv1d(n_features→32, kernel=3) → ReLU
  → Conv1d(32→64, kernel=3) → ReLU
  → AdaptiveAvgPool1d(1) → Flatten
  → Linear(64→128) → ReLU
Output: (batch, 128)
```

**Config :** `USE_CNN = True`, `CNN_FEATURES_DIM = 128`

---

### Step 6 — Early stopping (`training/train.py`)

`EarlyStoppingCallback` : stop si pas d'amélioration du mean reward après `EARLY_STOPPING_PATIENCE = 5` checks.

---

### Step 7 — Walk-Forward Validation (`training/walk_forward.py`)

Expanding window : train_start fixe, train_end avance par `step_months`, test couvre `test_months` suivants.

**Commande :** `python main.py walk-forward`

**Config :** `WF_TRAIN_MONTHS=24, WF_TEST_MONTHS=3, WF_STEP_MONTHS=3`

**Sorties :** métriques par fold + agrégat (mean/std/min/max) → JSON dans `logs/walk_forward/`

---

### Step 8 — MACD + ADX (`features/technical.py`)

- `add_macd()` : MACD(12,26,9). Normalisation : `macd_hist_normalized = histogram / close`
- `add_adx()` : ADX(14). Normalisation : `adx_normalized = adx / 100`

**Config :** `MACD_FAST=12, MACD_SLOW=26, MACD_SIGNAL=9, ADX_PERIOD=14`

---

### Step 9 — Orderbook imbalance + Open Interest (`data/pipeline.py`)

- Fetch snapshot OB et OI dans `build_dataset()`
- Fallback à 0.0 si indisponible (pas de données historiques)
- Déclarés comme `FEATURES_LIVE_ONLY` (poids ~0 pendant training, signal utile en live)

---

### Step 10 — Multi-timeframe 4h (`features/technical.py`, `data/pipeline.py`)

- Fetch OHLCV 4h dans `build_full_pipeline()`
- `add_multi_timeframe_features(df_1h, df_4h)` : RSI 4h + SMA trend 4h
- Alignement via `merge_asof(direction="backward")`
- Ajoutés au set V3 : `rsi_4h_normalized`, `sma_trend_4h`

---

## Tests ajoutés / mis à jour

| Fichier | Classe | Nouvelles méthodes |
|---------|--------|--------------------|
| `tests/test_env.py` | `TestRewardFixes` | `test_drawdown_penalty_capped`, `test_sharpe_annualized`, `test_sortino_no_downside_capped`, `test_sortino_no_downside_zero_mean` |
| `tests/test_agent.py` | `TestModel` | `test_import_vecnormalize_functions`, `test_vecnormalize_in_wrapper_chain`, `test_no_vecnormalize_when_disabled`, `test_cnn_extractor_predict`, `test_cnn_disabled_uses_mlp` |
| `tests/test_features.py` | `TestTechnicalIndicators` | `test_add_macd`, `test_add_adx`, `test_add_macd_in_all_indicators`, `test_add_multi_timeframe_features`, `test_multi_timeframe_empty_4h`, `test_multi_timeframe_rsi_range` |
| `tests/test_walk_forward.py` | `TestGenerateFolds`, `TestAggregateResults`, `TestWalkForwardIntegration` | Tous nouveaux |

---

## Vérification

```bash
# Tests unitaires
python -m pytest tests/ -v

# Training court
python main.py train --timesteps 50000

# Backtest
python main.py backtest

# Walk-forward
python main.py walk-forward

# Live paper
python main.py live --live-mode paper
```

---

## Corrections post-review (Phase 9 — suite)

Corrections appliquées après review approfondie du code :

### Fix 1 — Fee de vente symétrique (`env/trading_env.py`)

**Avant :** `fee = revenue * trading_fee` (fee calculé sur le prix post-slippage)
**Après :** `fee = btc_to_sell * current_price * trading_fee` (fee calculé sur la valeur nominale, symétrique avec l'achat)

L'achat calcule le fee sur `cash_to_spend` (valeur nominale). La vente doit faire de même.

### Fix 2 — Sortino cohérent entre reward et stats (`env/trading_env.py`)

`get_portfolio_stats()` retournait 0.0 quand pas de returns négatifs.
`sortino_reward()` retournait `min(mean / 1e-8, 3.0)` pour le même cas.

**Après :** `get_portfolio_stats()` utilise la même logique que `sortino_reward()` + annualisation `sqrt(8760)` sur Sharpe et Sortino (les stats reportées correspondent maintenant à ce que l'agent a réellement appris).

### Fix 4 — Fenêtre Sharpe/Sortino élargie (`config/settings.py`)

`SHARPE_WINDOW = 24` → `SHARPE_WINDOW = 72`

24 bougies = 1 jour. Trop court pour un signal statistiquement stable. 72 bougies = 3 jours, réduction significative du bruit de reward.

### Fix 5 — Poids de reward normalisés (`agent/reward.py`)

| Composante | Avant | Après |
| ---------- | ----- | ----- |
| log_return | 1.0 | 1.0 |
| sharpe | 0.5 | 0.3 |
| drawdown | 1.0 | 0.5 |
| position_size | 0.3 | 0.1 |
| transaction | 1.0 | 0.1 |
| **Total** | **3.8** | **2.0** |

Somme = 3.8 → 2.0. La magnitude élevée déstabilisait VecNormalize et ralentissait la convergence PPO.

### Fix 8 — batch_size PPO (`config/settings.py` + `tests/test_agent.py`)

`batch_size = 64` → `batch_size = 256`

Avec `n_steps=2048` et `n_envs=4` : 8192 transitions par update. Un batch de 64 = 128 mini-batches/epoch = gradient trop bruité. 256 = 32 mini-batches, convergence plus stable.

Tests mis à jour : les tests avec `n_steps=64` passent maintenant aussi `batch_size=32` (contrainte SB3 : `batch_size ≤ n_steps × n_envs`).

---

## Note

Après ces changements, un **retrain complet est obligatoire** — les modèles pré-existants sont incompatibles (obs space différent, VecNormalize requis, poids de reward modifiés).
