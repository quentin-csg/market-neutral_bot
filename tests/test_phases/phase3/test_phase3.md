# Phase 3 — Checklist de validation

## Toutes les commandes sont à copier-coller dans le REPL Python.

## Lancer `python` depuis le dossier Trading-bot.

## Seul le pytest se lance depuis PowerShell directement.

## Imports

- [X] **Tous les modules features s'importent** :
  ```python
  from features.technical import add_all_indicators, add_sma, add_rsi, add_atr, add_bollinger, add_zscore
  from features.nlp import analyze_sentiment, compute_hourly_sentiment, add_sentiment_to_dataframe
  from features.scaler import FeatureScaler, normalize_features
  from data.pipeline import build_full_pipeline
  print('OK: Tous les modules importés')
  ```

## Indicateurs Techniques (pandas-ta)

- [X] **Tous les indicateurs sont calculés sur des données synthétiques** :
  ```python
  import numpy as np, pandas as pd
  from features.technical import add_all_indicators
  np.random.seed(42)
  n = 300
  df = pd.DataFrame({
      'timestamp': pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC'),
      'open': 42000 + np.cumsum(np.random.randn(n) * 100),
      'high': 42100 + np.cumsum(np.random.randn(n) * 100),
      'low': 41900 + np.cumsum(np.random.randn(n) * 100),
      'close': 42000 + np.cumsum(np.random.randn(n) * 100),
      'volume': np.abs(np.random.randn(n) * 500 + 1000),
  })
  result = add_all_indicators(df)
  print(f"Colonnes: {len(result.columns)}")
  print(result.columns.tolist())
  ```
  → Doit contenir : sma_50, sma_200, sma_trend, rsi, rsi_normalized, atr, atr_pct, bb_upper, bb_middle, bb_lower, bb_position, zscore, volume_ratio, volume_direction, log_return, log_return_5h, log_return_24h

- [X] **RSI normalisé entre -1 et +1** :
  ```python
  rsi_norm = result['rsi_normalized'].dropna()
  print(f"RSI normalisé: min={rsi_norm.min():.2f}, max={rsi_norm.max():.2f}")
  assert rsi_norm.between(-1, 1).all()
  print('OK')
  ```

- [X] **Bollinger Bands : upper > lower** :
  ```python
  valid = result.dropna(subset=['bb_upper', 'bb_lower'])
  assert (valid['bb_upper'] >= valid['bb_lower']).all()
  print(f"BB position: min={valid['bb_position'].min():.2f}, max={valid['bb_position'].max():.2f}")
  print('OK')
  ```

- [X] **Log-returns calculés** :
  ```python
  print(f"Log-return 1h: {result['log_return'].dropna().mean():.6f}")
  print(f"Log-return 24h: {result['log_return_24h'].dropna().mean():.6f}")
  print('OK')
  ```

## NLP — FinBERT

- [X] **Analyse de sentiment sur des titres** :
  ```python
  from features.nlp import analyze_sentiment
  texts = [
      "Bitcoin surges to new all-time high amid ETF inflows",
      "Crypto market crashes as FED raises rates",
      "SEC approves new Bitcoin ETF applications",
  ]
  results = analyze_sentiment(texts)
  for r in results:
      print(f"  [{r['label']:>8}] score={r['score']:+.3f} | {r['text'][:60]}")
  ```
  → Le premier titre devrait être positif, le second négatif

- [X] **Sentiment horaire moyen** :
  ```python
  from features.nlp import compute_hourly_sentiment
  result = compute_hourly_sentiment(texts)
  print(f"Score moyen: {result['sentiment_score']:.3f}")
  print(f"Articles: {result['n_articles']}, +{result['n_positive']} -{result['n_negative']} ={result['n_neutral']}")
  ```
  → Score entre -1 et +1

## Scaler (RobustScaler)

- [X] **Normalisation entre -1 et +1** :
  ```python
  from features.scaler import FeatureScaler
  import numpy as np, pandas as pd
  df = pd.DataFrame({
      'close': np.random.randn(200) * 1000 + 42000,
      'volume': np.random.randn(200) * 500 + 1000,
      'rsi': np.random.randn(200) * 20 + 50,
      'is_weekend': [0, 1] * 100,
  })
  scaler = FeatureScaler()
  result = scaler.fit_transform(df)
  print(f"close: [{result['close'].min():.2f}, {result['close'].max():.2f}]")
  print(f"volume: [{result['volume'].min():.2f}, {result['volume'].max():.2f}]")
  print(f"is_weekend non modifié: {sorted(result['is_weekend'].unique())}")
  ```
  → close et volume entre -1 et +1, is_weekend inchangé [0, 1]

- [X] **Sauvegarde et chargement du scaler** :
  ```python
  scaler.save()
  scaler2 = FeatureScaler()
  scaler2.load()
  print(f"Colonnes chargées: {scaler2.feature_columns}")
  print(f"Fitted: {scaler2.is_fitted}")
  ```

## Pipeline complet (Phase 2 → Phase 3)

- [X] **build_full_pipeline fonctionne de bout en bout** :
  ```python
  from data.pipeline import build_full_pipeline
  dataset, scaler = build_full_pipeline(start='2024-06-01', end='2024-06-07', include_nlp=False)
  print(f"Dataset: {len(dataset)} lignes, {len(dataset.columns)} colonnes")
  print(f"Colonnes: {dataset.columns.tolist()}")
  print(f"Scaler fitted: {scaler.is_fitted}")
  ```
  → Dataset avec données crypto + macro + sentiment + indicateurs techniques, tout normalisé

## Tests unitaires (depuis PowerShell)

- [X] **Tous les tests passent** :
  ```
  python -m pytest tests/test_features.py -v
  ```
  → 22 tests doivent passer (22 passed)

- [X] **Tests data + features ensemble** :
  ```
  python -m pytest tests/test_data.py tests/test_features.py -v
  ```
  → 38 tests doivent passer (38 passed)

---

✅ **Phase 3 validée** quand toutes les cases sont cochées.
Passe ensuite à la **Phase 4 — Environnement Gymnasium**.
