"""
Tests unitaires pour l'environnement Gymnasium et les fonctions de reward.
"""

import unittest

import numpy as np
import pandas as pd


class TestReward(unittest.TestCase):
    """Tests des fonctions de récompense."""

    def test_import(self):
        from agent.reward import (
            compute_reward,
            drawdown_penalty,
            log_return_reward,
            position_size_penalty,
            sharpe_reward,
            sortino_reward,
            transaction_cost_penalty,
        )
        self.assertTrue(callable(compute_reward))

    def test_log_return_positive(self):
        from agent.reward import log_return_reward
        reward = log_return_reward(10500, 10000)
        self.assertGreater(reward, 0)
        self.assertAlmostEqual(reward, np.log(10500 / 10000), places=6)

    def test_log_return_negative(self):
        from agent.reward import log_return_reward
        reward = log_return_reward(9500, 10000)
        self.assertLess(reward, 0)

    def test_log_return_zero_worth(self):
        from agent.reward import log_return_reward
        reward = log_return_reward(0, 10000)
        self.assertEqual(reward, -1.0)

    def test_sharpe_not_enough_data(self):
        from agent.reward import sharpe_reward
        result = sharpe_reward([0.01], window=24)
        self.assertEqual(result, 0.0)

    def test_sharpe_with_data(self):
        from agent.reward import sharpe_reward
        returns = list(np.random.randn(30) * 0.01 + 0.001)
        result = sharpe_reward(returns, window=24)
        self.assertIsInstance(result, float)

    def test_sortino_positive_returns(self):
        from agent.reward import sortino_reward
        returns = [0.01, 0.02, 0.005, 0.01, 0.015, 0.008, 0.012]
        result = sortino_reward(returns, window=24)
        self.assertGreater(result, 0)

    def test_drawdown_no_loss(self):
        from agent.reward import drawdown_penalty
        penalty = drawdown_penalty(10000, 10000)
        self.assertEqual(penalty, 0.0)

    def test_drawdown_small_loss(self):
        from agent.reward import drawdown_penalty
        penalty = drawdown_penalty(9500, 10000, threshold=0.15)
        self.assertLess(penalty, 0)

    def test_drawdown_exponential(self):
        from agent.reward import drawdown_penalty
        # Au-delà du seuil, la pénalité doit être plus sévère
        mild = drawdown_penalty(9000, 10000, threshold=0.15)
        severe = drawdown_penalty(7000, 10000, threshold=0.15)
        self.assertLess(severe, mild)

    def test_position_size_penalty(self):
        from agent.reward import position_size_penalty
        # Petite position → petite pénalité
        small = position_size_penalty(0.1)
        # Grosse position → grosse pénalité
        large = position_size_penalty(0.9)
        self.assertLess(large, small)
        self.assertLess(small, 0)

    def test_transaction_cost_penalty(self):
        from agent.reward import transaction_cost_penalty
        penalty = transaction_cost_penalty(10, 10000)
        self.assertAlmostEqual(penalty, -0.001, places=4)

    def test_compute_reward_returns_tuple(self):
        from agent.reward import compute_reward
        total, components = compute_reward(
            net_worth=10100,
            prev_net_worth=10000,
            peak_net_worth=10100,
            position_ratio=0.5,
            trade_cost=5.0,
            returns_history=[0.01, 0.005, -0.002, 0.003],
        )
        self.assertIsInstance(total, float)
        self.assertIsInstance(components, dict)
        self.assertIn("log_return", components)
        self.assertIn("sharpe", components)
        self.assertIn("drawdown", components)
        self.assertIn("position_size", components)
        self.assertIn("transaction", components)


class TestTradingEnv(unittest.TestCase):
    """Tests de l'environnement Gymnasium TradingEnv."""

    @classmethod
    def setUpClass(cls):
        """Crée un DataFrame de test avec des données synthétiques."""
        np.random.seed(42)
        n = 200
        prices = 42000 + np.cumsum(np.random.randn(n) * 100)
        cls.test_df = pd.DataFrame({
            "close": prices,
            "open": prices + np.random.randn(n) * 50,
            "high": prices + abs(np.random.randn(n) * 100),
            "low": prices - abs(np.random.randn(n) * 100),
            "volume": np.abs(np.random.randn(n) * 500 + 1000),
            "rsi": np.random.rand(n) * 2 - 1,  # déjà normalisé
            "sma_trend": np.random.choice([-1, 1], n).astype(float),
        })

    def _make_env(self, **kwargs):
        from env.trading_env import TradingEnv
        return TradingEnv(df=self.test_df.copy(), **kwargs)

    def test_import(self):
        from env.trading_env import TradingEnv
        self.assertTrue(callable(TradingEnv))

    def test_init(self):
        env = self._make_env()
        self.assertIsNotNone(env.observation_space)
        self.assertIsNotNone(env.action_space)

    def test_reset(self):
        env = self._make_env()
        obs, info = env.reset(seed=42)
        self.assertEqual(obs.shape, (env.n_obs,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(info["balance"], 10000.0)
        self.assertEqual(info["position"], 0.0)

    def test_step_hold(self):
        """Action ~0 = hold, pas de trade."""
        env = self._make_env()
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
        self.assertEqual(info["total_trades"], 0)
        self.assertAlmostEqual(info["balance"], 10000.0)

    def test_step_buy(self):
        """Achat avec action positive."""
        env = self._make_env()
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(np.array([1.0]))
        self.assertEqual(info["total_trades"], 1)
        self.assertGreater(info["position"], 0)
        self.assertLess(info["balance"], 10000.0)

    def test_step_buy_then_sell(self):
        """Achat puis vente complète."""
        env = self._make_env()
        env.reset(seed=42)

        # Acheter 100%
        env.step(np.array([1.0]))
        self.assertGreater(env.position, 0)

        # Vendre 100%
        env.step(np.array([-1.0]))
        self.assertAlmostEqual(env.position, 0.0, places=8)
        self.assertEqual(env.total_trades, 2)

    def test_fees_applied(self):
        """Vérifie que les frais sont bien déduits."""
        env = self._make_env(trading_fee=0.001, slippage_min=0.0, slippage_max=0.0)
        env.reset(seed=42)

        # Achat 100% → le balance doit être ~0, mais le net_worth < initial
        # à cause des frais
        env.step(np.array([1.0]))
        price = env.prices[env.current_step]
        position_value = env.position * price
        net_worth = env.balance + position_value

        # Le net worth doit être inférieur au capital initial à cause des frais
        self.assertLess(net_worth, 10000.0)
        self.assertGreater(env.total_fees_paid, 0)

    def test_slippage_applied(self):
        """Vérifie que le slippage affecte le prix d'exécution."""
        env = self._make_env(slippage_min=0.01, slippage_max=0.01)
        env.reset(seed=42)
        env.step(np.array([1.0]))
        # Avec 1% de slippage, l'entry price doit être > prix marché
        self.assertGreater(env.entry_price, env.prices[0])

    def test_episode_complete(self):
        """Vérifie qu'un épisode se termine correctement."""
        env = self._make_env()
        env.reset(seed=42)

        terminated = False
        steps = 0
        while not terminated:
            action = np.array([env.np_random.uniform(-0.3, 0.3)])
            _, _, terminated, truncated, info = env.step(action)
            steps += 1

        self.assertGreater(steps, 0)
        self.assertEqual(steps, len(self.test_df) - 1)

    def test_portfolio_stats(self):
        """Vérifie le calcul des statistiques de portfolio."""
        env = self._make_env()
        env.reset(seed=42)

        for _ in range(50):
            action = np.array([env.np_random.uniform(-0.5, 0.5)])
            env.step(action)

        stats = env.get_portfolio_stats()
        self.assertIn("total_return_pct", stats)
        self.assertIn("max_drawdown_pct", stats)
        self.assertIn("sharpe_ratio", stats)
        self.assertIn("total_trades", stats)

    def test_ruin_terminates(self):
        """Vérifie que l'épisode se termine si le portfolio est ruiné."""
        # Créer un env avec prix qui chute massivement (90%+ de perte)
        n = 100
        prices = [10000 * (0.95 ** i) for i in range(n)]  # -5% par step
        df = pd.DataFrame({
            "close": prices,
            "rsi": [0.0] * n,
        })
        from env.trading_env import TradingEnv
        env = TradingEnv(df=df, initial_balance=10000.0)
        env.reset(seed=42)

        # Acheter 100% et laisser le prix chuter
        env.step(np.array([1.0]))

        terminated = False
        for _ in range(90):
            _, _, terminated, _, info = env.step(np.array([0.0]))
            if terminated:
                break

        self.assertTrue(terminated)
        self.assertLess(info["net_worth"], 10000.0 * 0.1)

    def test_observation_space_check(self):
        """Vérifie que l'observation est dans l'espace défini."""
        env = self._make_env()
        obs, _ = env.reset(seed=42)
        self.assertTrue(env.observation_space.contains(obs))

    def test_action_space_check(self):
        """Vérifie que les actions sont dans l'espace défini."""
        env = self._make_env()
        env.reset(seed=42)
        for _ in range(10):
            action = env.action_space.sample()
            self.assertTrue(env.action_space.contains(action))
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break
            self.assertTrue(env.observation_space.contains(obs))

    def test_render_human(self):
        """Vérifie que le render en mode human ne crashe pas."""
        env = self._make_env(render_mode="human")
        env.reset(seed=42)
        env.step(np.array([0.5]))  # Doit afficher dans la console

    def test_dead_zone(self):
        """Action < 5% = pas de trade."""
        env = self._make_env()
        env.reset(seed=42)
        env.step(np.array([0.03]))
        self.assertEqual(env.total_trades, 0)

    def test_custom_feature_columns(self):
        """Vérifie qu'on peut spécifier les colonnes manuellement."""
        env = self._make_env(feature_columns=["close", "rsi"])
        obs, _ = env.reset(seed=42)
        # 2 features marché + 3 portfolio = 5
        self.assertEqual(obs.shape, (5,))

    def test_gymnasium_api_check(self):
        """Vérifie la conformité basique avec l'API Gymnasium."""
        env = self._make_env()
        obs, info = env.reset(seed=42)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(info, dict)

        result = env.step(env.action_space.sample())
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)


if __name__ == "__main__":
    unittest.main()
