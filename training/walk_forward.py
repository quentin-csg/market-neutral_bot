"""
Walk-Forward Validation.

Valide la robustesse du modèle PPO avec des fenêtres glissantes :
  - Expanding window : le train grandit, le test avance
  - Agrège les métriques (Sharpe, return, drawdown) sur tous les folds
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from dateutil.relativedelta import relativedelta

from config.settings import (
    TOTAL_TIMESTEPS,
    TRAIN_START,
    WF_PURGE_HOURS,
    WF_STEP_MONTHS,
    WF_TEST_MONTHS,
    WF_TRAIN_MONTHS,
)
from training.backtest import backtest
from training.logger import log_walk_forward_result
from training.train import train

logger = logging.getLogger(__name__)


def generate_folds(
    data_start: str = TRAIN_START,
    data_end: Optional[str] = None,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    purge_hours: int = WF_PURGE_HOURS,
) -> list[dict]:
    """
    Génère les folds pour walk-forward validation (expanding window).

    Le train_start reste fixe, train_end avance de step_months à chaque fold.
    Un gap de purge (purge_hours) est inséré entre train et test pour éliminer
    le data leakage dû aux features à fenêtre glissante (SMA, RSI, frame stack...).

    Args:
        data_start: date de début des données
        data_end: date de fin (None = aujourd'hui)
        train_months: durée initiale du train en mois
        test_months: durée du test en mois
        step_months: pas d'avancement en mois
        purge_hours: nombre d'heures de gap entre train et test

    Returns:
        Liste de dicts avec fold_id, train_start, train_end, test_start, test_end
    """
    start = datetime.strptime(data_start, "%Y-%m-%d")
    end = datetime.strptime(data_end, "%Y-%m-%d") if data_end else datetime.now()

    folds = []
    fold_id = 1
    train_end = start + relativedelta(months=train_months)

    while True:
        # Gap de purge : le test commence purge_hours après la fin du train
        test_start = train_end + timedelta(hours=purge_hours)
        test_end = test_start + relativedelta(months=test_months)

        if test_end > end:
            break

        folds.append({
            "fold_id": fold_id,
            "train_start": start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d %H:%M"),
            "test_end": test_end.strftime("%Y-%m-%d"),
        })

        train_end += relativedelta(months=step_months)
        fold_id += 1

    return folds


def aggregate_results(fold_results: list[dict]) -> dict:
    """
    Agrège les métriques sur tous les folds.

    Returns:
        Dict avec mean/std/min/max de chaque métrique clé
    """
    if not fold_results:
        return {}

    metrics = ["total_return_pct", "sharpe_ratio", "sortino_ratio",
               "max_drawdown_pct", "total_trades"]

    agg = {}
    for metric in metrics:
        values = [f.get(metric, 0.0) for f in fold_results if metric in f]
        if values:
            agg[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    agg["n_folds"] = len(fold_results)
    return agg


def walk_forward_validate(
    feature_columns: Optional[list[str]] = None,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    purge_hours: int = WF_PURGE_HOURS,
    total_timesteps: int = TOTAL_TIMESTEPS,
    include_nlp: bool = False,
    data_start: str = TRAIN_START,
    data_end: Optional[str] = None,
) -> dict:
    """
    Exécute la walk-forward validation complète.

    Pour chaque fold :
      1. Entraîne un modèle sur la fenêtre train
      2. Backteste sur la fenêtre test
      3. Collecte les métriques

    Returns:
        Dict avec fold_results et aggregate
    """
    folds = generate_folds(
        data_start=data_start,
        data_end=data_end,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        purge_hours=purge_hours,
    )

    if not folds:
        print("Aucun fold généré. Vérifiez les dates et paramètres.")
        return {"fold_results": [], "aggregate": {}}

    print(f"=== Walk-Forward Validation: {len(folds)} folds (purge={purge_hours}h) ===")
    for f in folds:
        print(
            f"  Fold {f['fold_id']}: Train {f['train_start']} → {f['train_end']} | "
            f"Purge {purge_hours}h | Test {f['test_start']} → {f['test_end']}"
        )
    print()

    fold_results = []
    prev_model_name = None  # Pour le warm start inter-folds

    for fold in folds:
        fold_id = fold["fold_id"]
        model_name = f"wf_fold_{fold_id}"

        print(f"\n--- Fold {fold_id}/{len(folds)} ---")

        # 1. Entraîner (warm start depuis le fold précédent si disponible)
        print(f"[Train] {fold['train_start']} → {fold['train_end']}")
        if prev_model_name:
            print(f"  → Warm start depuis fold {fold_id - 1}: {prev_model_name}")
        try:
            train(
                train_start=fold["train_start"],
                train_end=fold["train_end"],
                total_timesteps=total_timesteps,
                feature_columns=feature_columns,
                model_name=model_name,
                include_nlp=include_nlp,
                use_subproc=False,
                warm_start_model=prev_model_name,
            )
        except Exception as e:
            logger.error(f"Fold {fold_id} train échoué: {e}")
            print(f"  ⚠ Train échoué: {e}")
            continue

        prev_model_name = model_name  # Fold suivant part d'ici

        # 2. Backtester
        print(f"[Test]  {fold['test_start']} → {fold['test_end']}")
        try:
            stats = backtest(
                model_name=model_name,
                test_start=fold["test_start"],
                test_end=fold["test_end"],
                feature_columns=feature_columns,
                include_nlp=include_nlp,
                save_results=False,
            )
            stats["fold_id"] = fold_id
            stats["fold"] = fold
            fold_results.append(stats)
        except Exception as e:
            logger.error(f"Fold {fold_id} backtest échoué: {e}")
            print(f"  ⚠ Backtest échoué: {e}")
            continue

    # 3. Agréger
    agg = aggregate_results(fold_results)

    print(f"\n=== Résultats agrégés ({len(fold_results)}/{len(folds)} folds) ===")
    for metric, vals in agg.items():
        if isinstance(vals, dict):
            print(f"  {metric}: mean={vals['mean']:.4f} ± {vals['std']:.4f} "
                  f"[{vals['min']:.4f}, {vals['max']:.4f}]")

    # 4. Sauvegarder
    result = {"fold_results": fold_results, "aggregate": agg}
    log_walk_forward_result(fold_results, agg)

    return result
