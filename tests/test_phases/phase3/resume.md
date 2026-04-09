# Phase 3 — Résumé

## Objectif
Implémenter le Feature Engineering complet : indicateurs techniques, analyse NLP via FinBERT, et normalisation RobustScaler (BLOC 2 du instruction.md).

## Ce qui a été implémenté

### 📈 `features/technical.py` — Indicateurs Techniques (pandas-ta)

| Fonction | Indicateur | Description |
|----------|-----------|-------------|
| `add_sma()` | SMA 50/200 | Moyennes mobiles + signal de croisement (`sma_trend` = ±1) + distance prix/SMA |
| `add_rsi()` | RSI 14 | RSI brut (0-100) + RSI normalisé (-1/+1) |
| `add_atr()` | ATR 14 | Average True Range (volatilité) + ATR en % du prix |
| `add_bollinger()` | Bandes de Bollinger | Upper/Middle/Lower bands + position relative du prix (-1/+1) + bandwidth |
| `add_zscore()` | Z-Score | Nombre d'écarts-types vs moyenne (détection outliers) |
| `add_volume_features()` | Volume | Volume ratio (vs MA20) + volume directionnel (volume × signe du prix) |
| `add_returns()` | Log-returns | Log-returns 1h, 5h, 24h (pénalisent les grosses pertes, cf instruction.md) |
| `add_all_indicators()` | Tous | Applique tous les indicateurs en un seul appel |

**Note** : Order Book imbalance, Funding Rate et Open Interest sont récupérés par le data pipeline (Phase 2) et passés directement comme features.

### 🧠 `features/nlp.py` — Analyse NLP FinBERT

| Fonction | Description |
|----------|-------------|
| `analyze_sentiment()` | Analyse une liste de textes avec FinBERT → score par article (-1/+1) |
| `compute_hourly_sentiment()` | Calcule le score moyen pour une heure, avec pondération des mots-clés |
| `add_sentiment_to_dataframe()` | Ajoute `sentiment_score` et `n_articles` au DataFrame principal |

- Modèle : `ProsusAI/finbert` (local, CPU/GPU)
- Lazy loading avec cache global (le modèle n'est chargé qu'une fois)
- Traitement par batch (configurable : `NLP_BATCH_SIZE = 8`)
- Pondération ×1.5 pour les articles contenant des mots-clés (BTC, FED, ETF...)
- Score clampé entre -1 (panique) et +1 (euphorie)
- Heures sans news → sentiment = 0 (stagnant, comme spécifié pour le weekend)

### 📏 `features/scaler.py` — Normalisation RobustScaler

| Fonction | Description |
|----------|-------------|
| `FeatureScaler.fit()` | Ajuste le scaler sur les données d'entraînement |
| `FeatureScaler.transform()` | Normalise les features, clampé entre -1 et +1 |
| `FeatureScaler.fit_transform()` | Ajuste + normalise en un seul appel |
| `FeatureScaler.save()` / `load()` | Sauvegarde/chargement du scaler (pour mode live) |
| `normalize_features()` | Fonction utilitaire simple |

- **RobustScaler** : résistant aux outliers (utilise médiane + IQR au lieu de mean + std)
- Résultat clampé entre **-1 et +1**
- Gestion des `inf` et `NaN` (remplacés par 0 après normalisation)
- Colonnes exclues de la normalisation : `timestamp`, `is_weekend`, `n_articles`, compteurs
- Sauvegarde en `.pkl` pour réutilisation en mode live

### 🔗 Pipeline intégré (Phase 2 → Phase 3)

Ajout de `build_full_pipeline()` dans `data/pipeline.py` :
```
Données brutes (ccxt, yfinance, Alternative.me, RSS)
    → Merge aligné sur grille 1h
        → Indicateurs techniques (pandas-ta)
            → Sentiment NLP (FinBERT)
                → Normalisation (RobustScaler)
                    → DataFrame prêt pour l'environnement Gym (Phase 4)
```

### 🧪 `tests/test_features.py` — 22 Tests unitaires
- 10 tests TechnicalIndicators (SMA, RSI, ATR, Bollinger, Z-Score, volume, returns, all, empty)
- 6 tests NLP (import, empty, no titles, basic analysis, hourly sentiment, no news)
- 6 tests Scaler (import, fit_transform, transform sans fit, save/load, utility, inf values)

**Total avec Phase 2 : 38/38 tests passent** ✅

## Technologies utilisées dans cette phase
| Technologie | Usage |
|------------|-------|
| **pandas-ta** | Calcul des indicateurs techniques (SMA, RSI, ATR, Bollinger) |
| **transformers** | Pipeline FinBERT pour l'analyse de sentiment NLP |
| **torch** | Backend PyTorch (CPU/GPU) pour FinBERT |
| **scikit-learn** | RobustScaler pour la normalisation des features |
| **pickle** | Sérialisation du scaler pour réutilisation en live |

## Statut
✅ **Phase 3 validée** — 22/22 tests passent. Indicateurs techniques, NLP et normalisation opérationnels.
Pipeline complet data → features intégré.
