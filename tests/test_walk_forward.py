"""
Tests unitaires pour training/walk_forward.py (Phase 9 — Walk-Forward Validation).
"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch


class TestGenerateFolds(unittest.TestCase):
    """Tests pour generate_folds()."""

    def test_basic_folds(self):
        from training.walk_forward import generate_folds
        folds = generate_folds(
            data_start="2022-01-01",
            data_end="2024-01-01",
            train_months=12,
            test_months=3,
            step_months=3,
            purge_hours=0,  # sans purge pour compter les folds comme avant
        )
        # 2022-01 + 12m train = 2023-01 test_end
        # Folds: 2023-01→04, 2023-04→07, 2023-07→10, 2023-10→2024-01
        self.assertEqual(len(folds), 4)

    def test_fold_structure(self):
        from training.walk_forward import generate_folds
        folds = generate_folds(
            data_start="2022-01-01",
            data_end="2023-12-01",
            train_months=12,
            test_months=3,
            step_months=3,
        )
        self.assertGreater(len(folds), 0)
        fold = folds[0]
        self.assertIn("fold_id", fold)
        self.assertIn("train_start", fold)
        self.assertIn("train_end", fold)
        self.assertIn("test_start", fold)
        self.assertIn("test_end", fold)

    def test_fold_ids_sequential(self):
        from training.walk_forward import generate_folds
        folds = generate_folds(
            data_start="2021-01-01",
            data_end="2024-01-01",
            train_months=12,
            test_months=3,
            step_months=3,
        )
        ids = [f["fold_id"] for f in folds]
        self.assertEqual(ids, list(range(1, len(folds) + 1)))

    def test_train_start_fixed(self):
        """Dans expanding window, le train_start est toujours le même."""
        from training.walk_forward import generate_folds
        folds = generate_folds(
            data_start="2022-01-01",
            data_end="2024-01-01",
            train_months=12,
            test_months=3,
            step_months=3,
        )
        train_starts = [f["train_start"] for f in folds]
        self.assertTrue(all(s == "2022-01-01" for s in train_starts))

    def test_test_start_after_train_end_with_purge(self):
        """test_start = train_end + purge_hours pour chaque fold."""
        from datetime import datetime, timedelta
        from training.walk_forward import generate_folds
        # Avec purge
        folds = generate_folds(
            data_start="2022-01-01",
            data_end="2024-06-01",
            train_months=12,
            test_months=3,
            step_months=3,
            purge_hours=200,
        )
        for fold in folds:
            train_end = datetime.strptime(fold["train_end"], "%Y-%m-%d")
            test_start = datetime.strptime(fold["test_start"], "%Y-%m-%d %H:%M")
            expected = train_end + timedelta(hours=200)
            self.assertEqual(test_start, expected)

        # Sans purge, test_start == train_end
        folds_no_purge = generate_folds(
            data_start="2022-01-01",
            data_end="2024-06-01",
            train_months=12,
            test_months=3,
            step_months=3,
            purge_hours=0,
        )
        for fold in folds_no_purge:
            self.assertEqual(fold["test_start"], fold["train_end"] + " 00:00")

    def test_no_folds_when_data_too_short(self):
        """Retourne 0 folds si les données ne couvrent pas au moins 1 fold."""
        from training.walk_forward import generate_folds
        folds = generate_folds(
            data_start="2022-01-01",
            data_end="2022-06-01",
            train_months=12,
            test_months=3,
            step_months=3,
        )
        self.assertEqual(len(folds), 0)

    def test_dates_are_strings(self):
        """Les dates dans les folds sont des strings parseables."""
        from training.walk_forward import generate_folds
        folds = generate_folds(
            data_start="2022-01-01",
            data_end="2024-01-01",
            train_months=12,
            test_months=3,
            step_months=3,
            purge_hours=0,
        )
        for fold in folds:
            for key in ("train_start", "train_end", "test_start", "test_end"):
                self.assertIsInstance(fold[key], str)
            # train_start, train_end, test_end = %Y-%m-%d
            datetime.strptime(fold["train_start"], "%Y-%m-%d")
            datetime.strptime(fold["train_end"], "%Y-%m-%d")
            datetime.strptime(fold["test_end"], "%Y-%m-%d")
            # test_start = %Y-%m-%d %H:%M (inclut le purge offset)
            datetime.strptime(fold["test_start"], "%Y-%m-%d %H:%M")

    def test_expanding_window(self):
        """train_end avance de step_months à chaque fold."""
        from training.walk_forward import generate_folds
        folds = generate_folds(
            data_start="2022-01-01",
            data_end="2024-06-01",
            train_months=12,
            test_months=3,
            step_months=3,
        )
        if len(folds) >= 2:
            end1 = datetime.strptime(folds[0]["train_end"], "%Y-%m-%d")
            end2 = datetime.strptime(folds[1]["train_end"], "%Y-%m-%d")
            # diff = 3 mois
            self.assertGreater(end2, end1)


class TestAggregateResults(unittest.TestCase):
    """Tests pour aggregate_results()."""

    def test_empty_input(self):
        from training.walk_forward import aggregate_results
        result = aggregate_results([])
        self.assertEqual(result, {})

    def test_single_fold(self):
        from training.walk_forward import aggregate_results
        fold_results = [{
            "total_return_pct": 5.0,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.5,
            "max_drawdown_pct": 3.0,
            "total_trades": 20,
        }]
        agg = aggregate_results(fold_results)
        self.assertIn("total_return_pct", agg)
        self.assertIn("n_folds", agg)
        self.assertEqual(agg["n_folds"], 1)
        # Avec 1 fold, mean == valeur originale
        self.assertAlmostEqual(agg["total_return_pct"]["mean"], 5.0)
        self.assertAlmostEqual(agg["total_return_pct"]["min"], 5.0)
        self.assertAlmostEqual(agg["total_return_pct"]["max"], 5.0)

    def test_multiple_folds(self):
        from training.walk_forward import aggregate_results
        fold_results = [
            {"total_return_pct": 2.0, "sharpe_ratio": 0.8, "max_drawdown_pct": 5.0,
             "sortino_ratio": 1.0, "total_trades": 10},
            {"total_return_pct": 4.0, "sharpe_ratio": 1.2, "max_drawdown_pct": 3.0,
             "sortino_ratio": 1.5, "total_trades": 15},
            {"total_return_pct": 6.0, "sharpe_ratio": 1.6, "max_drawdown_pct": 2.0,
             "sortino_ratio": 2.0, "total_trades": 20},
        ]
        agg = aggregate_results(fold_results)
        self.assertAlmostEqual(agg["total_return_pct"]["mean"], 4.0, places=5)
        self.assertAlmostEqual(agg["sharpe_ratio"]["min"], 0.8, places=5)
        self.assertAlmostEqual(agg["sharpe_ratio"]["max"], 1.6, places=5)
        self.assertEqual(agg["n_folds"], 3)

    def test_agg_keys(self):
        from training.walk_forward import aggregate_results
        fold_results = [{"total_return_pct": 1.0, "sharpe_ratio": 0.5,
                         "sortino_ratio": 0.6, "max_drawdown_pct": 4.0,
                         "total_trades": 5}]
        agg = aggregate_results(fold_results)
        for metric in ["total_return_pct", "sharpe_ratio", "sortino_ratio",
                       "max_drawdown_pct", "total_trades"]:
            self.assertIn(metric, agg)
            self.assertIn("mean", agg[metric])
            self.assertIn("std", agg[metric])
            self.assertIn("min", agg[metric])
            self.assertIn("max", agg[metric])

    def test_missing_metric_ignored(self):
        """Un fold sans une métrique n'empêche pas l'agrégation."""
        from training.walk_forward import aggregate_results
        fold_results = [
            {"total_return_pct": 3.0, "sharpe_ratio": 1.0,
             "sortino_ratio": 1.2, "max_drawdown_pct": 4.0, "total_trades": 10},
            {"total_return_pct": 5.0},  # fold incomplet
        ]
        agg = aggregate_results(fold_results)
        # Sharpe ne doit avoir qu'1 valeur
        self.assertEqual(len([f for f in fold_results if "sharpe_ratio" in f]), 1)
        self.assertAlmostEqual(agg["sharpe_ratio"]["mean"], 1.0)

    def test_values_are_floats(self):
        """Toutes les valeurs agrégées sont des floats Python."""
        from training.walk_forward import aggregate_results
        fold_results = [{"total_return_pct": 2.0, "sharpe_ratio": 1.0,
                         "sortino_ratio": 1.0, "max_drawdown_pct": 2.0,
                         "total_trades": 10}]
        agg = aggregate_results(fold_results)
        self.assertIsInstance(agg["total_return_pct"]["mean"], float)
        self.assertIsInstance(agg["total_return_pct"]["std"], float)


class TestWalkForwardIntegration(unittest.TestCase):
    """Tests d'intégration légers pour walk_forward_validate()."""

    def test_no_folds_returns_empty(self):
        """Retourne une structure vide si aucun fold n'est généré."""
        from training.walk_forward import walk_forward_validate
        result = walk_forward_validate(
            data_start="2022-01-01",
            data_end="2022-06-01",  # trop court
            train_months=12,
            test_months=3,
            step_months=3,
        )
        self.assertEqual(result["fold_results"], [])
        self.assertEqual(result["aggregate"], {})

    def test_walk_forward_calls_train_and_backtest(self):
        """Vérifie que walk_forward appelle train() et backtest() pour chaque fold."""
        from training.walk_forward import walk_forward_validate

        with patch("training.walk_forward.train") as mock_train, \
             patch("training.walk_forward.backtest") as mock_backtest, \
             patch("training.walk_forward.log_walk_forward_result"):

            mock_train.return_value = None
            mock_backtest.return_value = {
                "total_return_pct": 3.0,
                "sharpe_ratio": 1.0,
                "sortino_ratio": 1.2,
                "max_drawdown_pct": 2.5,
                "total_trades": 10,
            }

            result = walk_forward_validate(
                data_start="2022-01-01",
                data_end="2023-06-01",
                train_months=12,
                test_months=3,
                step_months=3,
            )

        # 1 fold: train 2022-01 → 2023-01, test 2023-01 → 2023-04
        self.assertEqual(mock_train.call_count, 1)
        self.assertEqual(mock_backtest.call_count, 1)
        self.assertEqual(len(result["fold_results"]), 1)
        self.assertIn("aggregate", result)

    def test_warm_start_passed_between_folds(self):
        """Le fold N+1 reçoit le modèle du fold N comme warm_start_model."""
        from training.walk_forward import walk_forward_validate

        train_calls = []

        def capture_train(**kwargs):
            train_calls.append(kwargs.get("warm_start_model"))

        with patch("training.walk_forward.train", side_effect=capture_train) as mock_train, \
             patch("training.walk_forward.backtest") as mock_backtest, \
             patch("training.walk_forward.log_walk_forward_result"):

            mock_backtest.return_value = {
                "total_return_pct": 2.0, "sharpe_ratio": 1.0,
                "sortino_ratio": 1.2, "max_drawdown_pct": 3.0, "total_trades": 5,
            }

            walk_forward_validate(
                data_start="2020-01-01",
                data_end="2023-06-01",
                train_months=24,
                test_months=3,
                step_months=3,
            )

        self.assertGreaterEqual(mock_train.call_count, 2)
        # Fold 1 : pas de warm start
        self.assertIsNone(train_calls[0])
        # Fold 2+ : warm start depuis le fold précédent
        self.assertEqual(train_calls[1], "wf_fold_1")

    def test_first_fold_no_warm_start(self):
        """Le premier fold démarre toujours from scratch (pas de warm start)."""
        from training.walk_forward import walk_forward_validate

        train_kwargs_list = []

        def capture_train(**kwargs):
            train_kwargs_list.append(kwargs)

        with patch("training.walk_forward.train", side_effect=capture_train), \
             patch("training.walk_forward.backtest") as mock_backtest, \
             patch("training.walk_forward.log_walk_forward_result"):

            mock_backtest.return_value = {
                "total_return_pct": 2.0, "sharpe_ratio": 1.0,
                "sortino_ratio": 1.0, "max_drawdown_pct": 3.0, "total_trades": 5,
            }

            walk_forward_validate(
                data_start="2022-01-01",
                data_end="2023-06-01",
                train_months=12,
                test_months=3,
                step_months=3,
            )

        self.assertGreater(len(train_kwargs_list), 0)
        self.assertIsNone(train_kwargs_list[0].get("warm_start_model"))


if __name__ == "__main__":
    unittest.main()
