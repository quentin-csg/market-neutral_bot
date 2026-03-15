"""
Analyse NLP locale via FinBERT.
Les titres des flux RSS sont analysés pour obtenir un score de sentiment
moyen par heure (-1 = panique, +1 = euphorie).
Le modèle tourne 100% en local (CPU/GPU).
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

# Supprimer les warnings HuggingFace Hub (requêtes non authentifiées)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline  # noqa: E402

from config.settings import FINBERT_MODEL, NEWS_KEYWORDS, NLP_BATCH_SIZE, NLP_MAX_ARTICLES

logger = logging.getLogger(__name__)

# Cache global du pipeline pour éviter de recharger le modèle à chaque appel
_sentiment_pipeline = None


def _get_pipeline():
    """Charge le pipeline FinBERT (lazy loading + cache)."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        logger.info(f"Chargement du modèle FinBERT ({FINBERT_MODEL})...")
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device == 0 else "CPU"
        logger.info(f"  Utilisation du {device_name}")

        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            device=device,
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT chargé avec succès")
    return _sentiment_pipeline


def analyze_sentiment(texts: list[str]) -> list[dict]:
    """
    Analyse le sentiment d'une liste de textes avec FinBERT.

    Args:
        texts: Liste de titres/textes à analyser

    Returns:
        Liste de dicts avec: text, label (positive/negative/neutral), score (-1 à +1)
    """
    if not texts:
        return []

    # Limiter le nombre d'articles
    texts = texts[:NLP_MAX_ARTICLES]

    pipe = _get_pipeline()

    results = []
    # Traiter par batch
    for i in range(0, len(texts), NLP_BATCH_SIZE):
        batch = texts[i:i + NLP_BATCH_SIZE]
        try:
            predictions = pipe(batch)
            for text, pred in zip(batch, predictions):
                # FinBERT retourne: positive, negative, neutral
                label = pred["label"].lower()
                raw_score = pred["score"]

                # Convertir en score continu -1 à +1
                if label == "positive":
                    score = raw_score
                elif label == "negative":
                    score = -raw_score
                else:  # neutral
                    score = 0.0

                results.append({
                    "text": text,
                    "label": label,
                    "raw_score": raw_score,
                    "score": score,
                })
        except Exception as e:
            logger.error(f"Erreur analyse batch FinBERT: {e}")
            for text in batch:
                results.append({
                    "text": text,
                    "label": "error",
                    "raw_score": 0.0,
                    "score": 0.0,
                })

    return results


def compute_hourly_sentiment(
    titles: list[str],
    keywords: list[str] = NEWS_KEYWORDS,
) -> dict:
    """
    Calcule le score de sentiment moyen pour une heure donnée.

    Args:
        titles: Liste de titres d'articles pour cette heure
        keywords: Mots-clés pour pondérer les articles pertinents

    Returns:
        Dict avec: mean_score (-1/+1), n_articles, n_positive, n_negative, n_neutral
    """
    if not titles:
        return {
            "sentiment_score": 0.0,
            "n_articles": 0,
            "n_positive": 0,
            "n_negative": 0,
            "n_neutral": 0,
        }

    results = analyze_sentiment(titles)

    scores = []
    n_positive = 0
    n_negative = 0
    n_neutral = 0

    for r in results:
        # Pondérer les articles avec des mots-clés pertinents
        text_lower = r["text"].lower()
        weight = 1.5 if any(kw.lower() in text_lower for kw in keywords) else 1.0

        scores.append(r["score"] * weight)

        if r["label"] == "positive":
            n_positive += 1
        elif r["label"] == "negative":
            n_negative += 1
        else:
            n_neutral += 1

    mean_score = float(np.mean(scores)) if scores else 0.0
    # Clamp entre -1 et +1
    mean_score = max(-1.0, min(1.0, mean_score))

    return {
        "sentiment_score": mean_score,
        "n_articles": len(results),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_neutral": n_neutral,
    }


def add_sentiment_to_dataframe(
    df: pd.DataFrame,
    news_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Ajoute une colonne sentiment_score au DataFrame principal.
    Analyse les titres des news par heure et merge sur le timestamp.

    Args:
        df: DataFrame principal avec colonne 'timestamp' (grille 1h)
        news_df: DataFrame de news avec colonnes 'timestamp', 'title'
                 (résultat de data.news_fetcher.fetch_news)

    Returns:
        DataFrame enrichi avec sentiment_score, n_articles
    """
    df = df.copy()

    if news_df is None or news_df.empty:
        df["sentiment_score"] = 0.0
        df["n_articles"] = 0
        logger.warning("Pas de news disponibles, sentiment = 0")
        return df

    # Arrondir les timestamps des news à l'heure
    news_df = news_df.copy()
    news_df["hour"] = news_df["timestamp"].dt.floor("h")

    # Grouper les titres par heure
    hourly_titles = news_df.groupby("hour")["title"].apply(list).to_dict()

    # Analyser le sentiment pour chaque heure
    sentiment_data = []
    for hour, titles in hourly_titles.items():
        result = compute_hourly_sentiment(titles)
        result["timestamp"] = hour
        sentiment_data.append(result)

    if sentiment_data:
        sentiment_df = pd.DataFrame(sentiment_data)
        df = pd.merge(df, sentiment_df, on="timestamp", how="left")
    else:
        df["sentiment_score"] = 0.0
        df["n_articles"] = 0

    # Fill NaN (heures sans news) avec 0
    df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
    df["n_articles"] = df["n_articles"].fillna(0).astype(int)

    # Colonnes optionnelles
    for col in ["n_positive", "n_negative", "n_neutral"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    logger.info(
        f"Sentiment NLP ajouté: {len(sentiment_data)} heures analysées, "
        f"score moyen = {df['sentiment_score'].mean():.3f}"
    )
    return df
