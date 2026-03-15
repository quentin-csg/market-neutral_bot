"""
Système de logs pour le trading bot.
  - Résumé hebdomadaire (PnL, nb trades, Sharpe) dans logs/
  - Intégration TensorBoard
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import LOGS_DIR

logger = logging.getLogger(__name__)

# Sous-dossiers de logs
WEEKLY_LOG_DIR = LOGS_DIR / "weekly"
BACKTEST_LOG_DIR = LOGS_DIR / "backtests"


def _ensure_dirs():
    """Crée les dossiers de logs si nécessaire."""
    WEEKLY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_weekly_summary(
    stats: dict,
    week_label: Optional[str] = None,
) -> Path:
    """
    Sauvegarde un résumé hebdomadaire au format JSON.

    Args:
        stats: dict de métriques (PnL, Sharpe, trades, etc.)
        week_label: label de la semaine (auto-généré si None)

    Returns:
        Chemin du fichier sauvegardé
    """
    _ensure_dirs()

    if week_label is None:
        now = datetime.now()
        week_label = f"{now.year}_W{now.isocalendar()[1]:02d}"

    filepath = WEEKLY_LOG_DIR / f"week_{week_label}.json"

    entry = {
        "week": week_label,
        "timestamp": datetime.now().isoformat(),
        **{k: _serialize(v) for k, v in stats.items()},
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)

    logger.info(f"Résumé hebdomadaire sauvegardé: {filepath}")
    return filepath


def log_backtest_result(
    stats: dict,
    model_name: str = "ppo_trading",
    run_name: Optional[str] = None,
) -> Path:
    """
    Sauvegarde les résultats d'un backtest au format JSON.

    Args:
        stats: dict de métriques du backtest
        model_name: nom du modèle utilisé
        run_name: nom du run (auto-généré si None)

    Returns:
        Chemin du fichier sauvegardé
    """
    _ensure_dirs()

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    filepath = BACKTEST_LOG_DIR / f"backtest_{model_name}_{run_name}.json"

    entry = {
        "run_name": run_name,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        **{k: _serialize(v) for k, v in stats.items()},
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)

    logger.info(f"Résultat backtest sauvegardé: {filepath}")
    return filepath


def load_backtest_results(model_name: Optional[str] = None) -> list[dict]:
    """
    Charge tous les résultats de backtest pour comparaison.

    Args:
        model_name: filtrer par nom de modèle (None = tous)

    Returns:
        Liste de dicts de résultats, triés par date
    """
    _ensure_dirs()
    results = []

    for filepath in sorted(BACKTEST_LOG_DIR.glob("backtest_*.json")):
        if model_name and model_name not in filepath.name:
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            results.append(json.load(f))

    return results


def print_stats(stats: dict, title: str = "Résultats") -> None:
    """Affiche les statistiques de manière formatée."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:.<35s} {value:>10.4f}")
        else:
            print(f"  {key:.<35s} {str(value):>10s}")
    print(f"{'='*50}\n")


def _serialize(value):
    """Convertit les types numpy pour la sérialisation JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
