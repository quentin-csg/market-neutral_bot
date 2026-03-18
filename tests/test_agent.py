"""
Tests unitaires pour l'agent PPO (Phase 5).
"""

import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def _make_test_df(n=200):
    """Crée un DataFrame de test synthétique avec toutes les features V2."""
    np.random.seed(42)
    prices = 42000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame({
        # V1 features
        "rsi_normalized": np.random.rand(n) * 2 - 1,
        "sma_trend": np.random.choice([-1.0, 1.0], n),
        "price_to_sma_long": np.random.randn(n) * 0.05,
        "atr_pct": np.abs(np.random.randn(n) * 0.01),
        "bb_position": np.random.rand(n) * 2 - 1,
        "bb_bandwidth": np.abs(np.random.randn(n) * 0.02),
        "log_return": np.random.randn(n) * 0.01,
        # V2 features
        "volume_ratio": np.abs(np.random.randn(n) + 1),
        "volume_direction": np.random.randn(n),
        "log_return_5h": np.random.randn(n) * 0.02,
        "log_return_24h": np.random.randn(n) * 0.05,
        "fear_greed_normalized": np.random.rand(n) * 2 - 1,
        "funding_rate": np.random.randn(n) * 0.001,
        "is_weekend": np.random.choice([0, 1], n),
        # Colonnes OHLCV (nécessaires pour l'env)
        "close": prices,
        "open": prices + np.random.randn(n) * 50,
        "high": prices + np.abs(np.random.randn(n) * 100),
        "low": prices - np.abs(np.random.randn(n) * 100),
        "volume": np.abs(np.random.randn(n) * 500 + 1000),
    })


class TestModel(unittest.TestCase):
    """Tests pour agent/model.py"""

    def test_import(self):
        from agent.model import (
            create_agent,
            get_feature_set,
            load_agent,
            make_env,
            make_vec_env,
            save_agent,
        )
        self.assertTrue(callable(create_agent))
        self.assertTrue(callable(make_vec_env))

    def test_feature_sets(self):
        from agent.model import get_feature_set
        v1 = get_feature_set("v1")
        v2 = get_feature_set("v2")
        v3 = get_feature_set("v3")
        self.assertGreater(len(v1), 0)
        self.assertGreater(len(v2), len(v1))
        self.assertGreaterEqual(len(v3), len(v2))
        # V1 : features tech de base
        self.assertIn("rsi_normalized", v1)
        self.assertIn("sma_trend", v1)
        self.assertIn("log_return", v1)
        # V2 : + macro
        self.assertIn("fear_greed_normalized", v2)
        self.assertIn("funding_rate", v2)
        # Vérifier pas de doublons (rsi brut, close brut supprimés)
        self.assertNotIn("close", v1)
        self.assertNotIn("rsi", v1)
        self.assertNotIn("zscore", v2)

    def test_feature_set_invalid(self):
        from agent.model import get_feature_set
        with self.assertRaises(ValueError):
            get_feature_set("v99")

    def test_make_env(self):
        from agent.model import make_env
        df = _make_test_df()
        env_fn = make_env(df, feature_columns=["rsi_normalized", "sma_trend"])
        env = env_fn()
        obs, info = env.reset()
        self.assertEqual(obs.shape[0], 5)  # 2 features + 3 portfolio
        env.close()

    def test_make_vec_env(self):
        from agent.model import make_vec_env
        df = _make_test_df()
        vec_env = make_vec_env(
            df, n_envs=2, feature_columns=["rsi_normalized", "sma_trend"],
            use_subproc=False, frame_stack=4,
        )
        obs = vec_env.reset()
        # shape = (n_envs, n_features * frame_stack)
        self.assertEqual(obs.shape[0], 2)  # n_envs
        self.assertEqual(obs.shape[1], 5 * 4)  # (2 features + 3 portfolio) * 4 frames
        vec_env.close()

    def test_create_agent(self):
        from agent.model import create_agent, make_vec_env
        df = _make_test_df()
        vec_env = make_vec_env(
            df, n_envs=1, feature_columns=["rsi_normalized", "sma_trend"],
            use_subproc=False, frame_stack=4,
        )
        agent = create_agent(
            vec_env,
            hyperparams={"verbose": 0, "n_steps": 64, "batch_size": 32},
            tensorboard_log=None,
            seed=42,
        )
        self.assertIsNotNone(agent)
        self.assertIsNotNone(agent.policy)
        vec_env.close()

    def test_predict(self):
        """L'agent peut faire un forward pass (predict)."""
        from agent.model import create_agent, make_vec_env
        df = _make_test_df()
        vec_env = make_vec_env(
            df, n_envs=1, feature_columns=["rsi_normalized", "sma_trend"],
            use_subproc=False, frame_stack=4,
        )
        agent = create_agent(
            vec_env,
            hyperparams={"verbose": 0, "n_steps": 64, "batch_size": 32},
            tensorboard_log=None,
            seed=42,
            use_cnn=False,
        )
        obs = vec_env.reset()
        action, _states = agent.predict(obs, deterministic=True)
        self.assertEqual(action.shape, (1, 1))
        self.assertTrue(-1.0 <= action[0][0] <= 1.0)
        vec_env.close()

    def test_save_load(self):
        """Save et load fonctionnent correctement."""
        from agent.model import create_agent, load_agent, make_vec_env, save_agent
        tmp_dir = Path("models/_test_tmp")
        try:
            df = _make_test_df()
            vec_env = make_vec_env(
                df, n_envs=1, feature_columns=["rsi_normalized", "sma_trend"],
                use_subproc=False, frame_stack=4,
            )
            agent = create_agent(
                vec_env,
                hyperparams={"verbose": 0, "n_steps": 64, "batch_size": 32},
                tensorboard_log=None,
                seed=42,
                use_cnn=False,
            )

            # Sauvegarder
            save_agent(agent, name="test_model", path=tmp_dir)
            self.assertTrue((tmp_dir / "test_model.zip").exists())

            # Charger
            agent2 = load_agent(vec_env, name="test_model", path=tmp_dir)
            self.assertIsNotNone(agent2)

            # Vérifier que les prédictions sont identiques
            obs = vec_env.reset()
            a1, _ = agent.predict(obs, deterministic=True)
            a2, _ = agent2.predict(obs, deterministic=True)
            np.testing.assert_array_almost_equal(a1, a2)

            vec_env.close()
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

    def test_frame_stack_changes_obs_shape(self):
        """VecFrameStack modifie la shape de l'observation."""
        from agent.model import make_vec_env
        df = _make_test_df()

        # Sans frame stack (stack=1)
        env1 = make_vec_env(
            df, n_envs=1, feature_columns=["rsi_normalized", "sma_trend"],
            use_subproc=False, frame_stack=1,
        )
        obs1 = env1.reset()

        # Avec frame stack (stack=8)
        env2 = make_vec_env(
            df, n_envs=1, feature_columns=["rsi_normalized", "sma_trend"],
            use_subproc=False, frame_stack=8,
        )
        obs2 = env2.reset()

        self.assertEqual(obs1.shape[1], 5)      # 2 features + 3 portfolio
        self.assertEqual(obs2.shape[1], 5 * 8)  # x8 frames

        env1.close()
        env2.close()

    def test_short_training(self):
        """L'agent peut s'entraîner quelques steps sans crash."""
        from agent.model import create_agent, make_vec_env
        df = _make_test_df(n=300)
        vec_env = make_vec_env(
            df, n_envs=1, feature_columns=["rsi_normalized", "sma_trend"],
            use_subproc=False, frame_stack=4,
        )
        agent = create_agent(
            vec_env,
            hyperparams={"verbose": 0, "n_steps": 64, "batch_size": 32},
            tensorboard_log=None,
            seed=42,
            use_cnn=False,
        )
        # Entraîner 128 steps (2 updates)
        agent.learn(total_timesteps=128)

        # Vérifier que l'agent fonctionne après entraînement
        obs = vec_env.reset()
        action, _ = agent.predict(obs, deterministic=True)
        self.assertEqual(action.shape, (1, 1))

        vec_env.close()


    def test_import_vecnormalize_functions(self):
        """save_vec_normalize et load_vec_normalize sont importables."""
        from agent.model import load_vec_normalize, save_vec_normalize
        self.assertTrue(callable(save_vec_normalize))
        self.assertTrue(callable(load_vec_normalize))

    def test_vecnormalize_in_wrapper_chain(self):
        """VecNormalize est présent dans la wrapper chain quand normalize=True."""
        from stable_baselines3.common.vec_env import VecNormalize
        from agent.model import make_vec_env, _get_vec_normalize
        df = _make_test_df()
        vec_env = make_vec_env(
            df, n_envs=1, feature_columns=["rsi_normalized", "sma_trend"],
            use_subproc=False, frame_stack=4, normalize=True,
        )
        vn = _get_vec_normalize(vec_env)
        self.assertIsInstance(vn, VecNormalize)
        self.assertFalse(vn.norm_obs)
        self.assertTrue(vn.norm_reward)
        vec_env.close()

    def test_no_vecnormalize_when_disabled(self):
        """Pas de VecNormalize quand normalize=False."""
        from agent.model import make_vec_env, _get_vec_normalize
        df = _make_test_df()
        vec_env = make_vec_env(
            df, n_envs=1, feature_columns=["rsi_normalized", "sma_trend"],
            use_subproc=False, frame_stack=4, normalize=False,
        )
        vn = _get_vec_normalize(vec_env)
        self.assertIsNone(vn)
        vec_env.close()

    def test_cnn_extractor_predict(self):
        """Le CNN 1D permet au modèle de faire un forward pass."""
        from agent.model import create_agent, make_vec_env
        df = _make_test_df()
        n_features = 5  # 2 features + 3 portfolio
        frame_stack = 4
        vec_env = make_vec_env(
            df, n_envs=1, feature_columns=["rsi_normalized", "sma_trend"],
            use_subproc=False, frame_stack=frame_stack, normalize=False,
        )
        agent = create_agent(
            vec_env,
            hyperparams={"verbose": 0, "n_steps": 64, "batch_size": 32},
            tensorboard_log=None,
            seed=42,
            use_cnn=True,
            n_features=n_features,
            frame_stack=frame_stack,
        )
        obs = vec_env.reset()
        action, _ = agent.predict(obs, deterministic=True)
        self.assertEqual(action.shape[0], 1)
        vec_env.close()

    def test_cnn_disabled_uses_mlp(self):
        """Avec use_cnn=False, le modèle utilise MlpPolicy sans extractor custom."""
        from agent.model import create_agent, make_vec_env
        df = _make_test_df()
        vec_env = make_vec_env(
            df, n_envs=1, feature_columns=["rsi_normalized", "sma_trend"],
            use_subproc=False, frame_stack=4, normalize=False,
        )
        agent = create_agent(
            vec_env,
            hyperparams={"verbose": 0, "n_steps": 64, "batch_size": 32},
            tensorboard_log=None,
            seed=42,
            use_cnn=False,
        )
        obs = vec_env.reset()
        action, _ = agent.predict(obs, deterministic=True)
        self.assertIsNotNone(action)
        vec_env.close()


if __name__ == "__main__":
    unittest.main()
