"""
Tests unitaires pour le module features (Phase 3 — Feature Engineering).
"""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv_df(n: int = 300) -> pd.DataFrame:
    """Crée un DataFrame OHLCV synthétique pour les tests."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 42000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 30
    volume = np.abs(np.random.randn(n) * 500 + 1000)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestTechnicalIndicators:
    """Tests pour features/technical.py"""

    def test_import(self):
        from features.technical import (
            add_all_indicators,
            add_sma,
            add_rsi,
            add_atr,
            add_bollinger,
            add_zscore,
            add_volume_features,
            add_returns,
            add_macd,
            add_adx,
            add_multi_timeframe_features,
            add_candle_features,
            add_price_position_features,
        )
        assert callable(add_all_indicators)

    def test_add_sma(self):
        from features.technical import add_sma
        df = _make_ohlcv_df()
        result = add_sma(df)
        assert "sma_50" in result.columns
        assert "sma_200" in result.columns
        assert "sma_trend" in result.columns
        assert "price_to_sma_long" in result.columns
        # SMA trend doit être -1 ou +1
        valid_values = result["sma_trend"].dropna().unique()
        assert set(valid_values).issubset({-1.0, 1.0})

    def test_add_rsi(self):
        from features.technical import add_rsi
        df = _make_ohlcv_df()
        result = add_rsi(df)
        assert "rsi" in result.columns
        assert "rsi_normalized" in result.columns
        # RSI brut entre 0 et 100
        rsi_valid = result["rsi"].dropna()
        assert rsi_valid.between(0, 100).all()
        # RSI normalisé entre -1 et +1
        rsi_norm = result["rsi_normalized"].dropna()
        assert rsi_norm.between(-1, 1).all()

    def test_add_atr(self):
        from features.technical import add_atr
        df = _make_ohlcv_df()
        result = add_atr(df)
        assert "atr" in result.columns
        assert "atr_pct" in result.columns
        # ATR doit être positif
        atr_valid = result["atr"].dropna()
        assert (atr_valid >= 0).all()

    def test_add_bollinger(self):
        from features.technical import add_bollinger
        df = _make_ohlcv_df()
        result = add_bollinger(df)
        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_position" in result.columns
        # bb_upper > bb_lower
        valid = result.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_add_zscore(self):
        from features.technical import add_zscore
        df = _make_ohlcv_df()
        result = add_zscore(df)
        assert "zscore" in result.columns

    def test_add_volume_features(self):
        from features.technical import add_volume_features
        df = _make_ohlcv_df()
        result = add_volume_features(df)
        assert "volume_ratio" in result.columns
        assert "volume_direction" in result.columns

    def test_add_returns(self):
        from features.technical import add_returns
        df = _make_ohlcv_df()
        result = add_returns(df)
        assert "log_return" in result.columns
        assert "log_return_5h" in result.columns
        assert "log_return_24h" in result.columns

    def test_add_all_indicators(self):
        from features.technical import add_all_indicators
        df = _make_ohlcv_df()
        result = add_all_indicators(df)
        # Doit avoir plus de colonnes qu'à l'entrée
        assert len(result.columns) > len(df.columns)
        # Colonnes originales préservées
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            assert col in result.columns
        assert "candle_body" in result.columns
        assert "upper_wick_ratio" in result.columns
        assert "lower_wick_ratio" in result.columns
        assert "price_to_high_20" in result.columns
        assert "price_to_low_20" in result.columns

    def test_add_macd(self):
        from features.technical import add_macd
        df = _make_ohlcv_df()
        result = add_macd(df)
        assert "macd" in result.columns
        assert "macd_histogram" in result.columns
        assert "macd_signal_line" in result.columns
        assert "macd_hist_normalized" in result.columns
        # macd_hist_normalized doit être scale-indépendant (~0 autour de 0)
        valid = result["macd_hist_normalized"].dropna()
        assert len(valid) > 0
        # Les valeurs normalisées doivent être petites par rapport à 1
        assert valid.abs().max() < 1.0

    def test_add_adx(self):
        from features.technical import add_adx
        df = _make_ohlcv_df()
        result = add_adx(df)
        assert "adx" in result.columns
        assert "adx_normalized" in result.columns
        # ADX normalisé entre 0 et 1
        valid = result["adx_normalized"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_add_macd_in_all_indicators(self):
        from features.technical import add_all_indicators
        df = _make_ohlcv_df()
        result = add_all_indicators(df)
        assert "macd_hist_normalized" in result.columns
        assert "adx_normalized" in result.columns

    def test_add_multi_timeframe_features(self):
        from features.technical import add_multi_timeframe_features
        df_1h = _make_ohlcv_df(n=300)
        # Simuler un DataFrame 4h (1 point toutes les 4 heures)
        df_4h = df_1h.iloc[::4].copy().reset_index(drop=True)
        result = add_multi_timeframe_features(df_1h, df_4h)
        assert "rsi_4h_normalized" in result.columns
        assert "sma_trend_4h" in result.columns
        # Même nombre de lignes que df_1h
        assert len(result) == len(df_1h)

    def test_multi_timeframe_empty_4h(self):
        """Retourne NaN si le DataFrame 4h est vide."""
        from features.technical import add_multi_timeframe_features
        df_1h = _make_ohlcv_df(n=100)
        df_4h_empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        result = add_multi_timeframe_features(df_1h, df_4h_empty)
        assert "rsi_4h_normalized" in result.columns
        assert result["rsi_4h_normalized"].isna().all()

    def test_multi_timeframe_rsi_range(self):
        """RSI 4h normalisé entre -1 et +1."""
        from features.technical import add_multi_timeframe_features
        df_1h = _make_ohlcv_df(n=400)
        df_4h = df_1h.iloc[::4].copy().reset_index(drop=True)
        result = add_multi_timeframe_features(df_1h, df_4h)
        valid = result["rsi_4h_normalized"].dropna()
        if len(valid) > 0:
            assert valid.between(-1, 1).all()

    def test_empty_dataframe(self):
        from features.technical import add_all_indicators
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        result = add_all_indicators(df)
        assert isinstance(result, pd.DataFrame)

    def test_add_candle_features(self):
        from features.technical import add_candle_features
        df = _make_ohlcv_df()
        result = add_candle_features(df)
        assert "candle_body" in result.columns
        assert "upper_wick_ratio" in result.columns
        assert "lower_wick_ratio" in result.columns
        # candle_body entre -1 et +1
        valid_body = result["candle_body"].dropna()
        assert len(valid_body) > 0
        assert (valid_body >= -1).all() and (valid_body <= 1).all()
        # wick ratios entre 0 et 1
        for col in ["upper_wick_ratio", "lower_wick_ratio"]:
            valid = result[col].dropna()
            assert len(valid) > 0
            assert (valid >= 0).all() and (valid <= 1).all()

    def test_add_price_position_features(self):
        from features.technical import add_price_position_features
        df = _make_ohlcv_df()
        result = add_price_position_features(df, window=20)
        assert "price_to_high_20" in result.columns
        assert "price_to_low_20" in result.columns
        # price_to_high_20 doit être ≤ 0 (prix ≤ high récent)
        valid_high = result["price_to_high_20"].dropna()
        assert len(valid_high) > 0
        assert (valid_high <= 1e-10).all()  # petit epsilon pour float precision
        # price_to_low_20 doit être ≥ 0 (prix ≥ low récent)
        valid_low = result["price_to_low_20"].dropna()
        assert len(valid_low) > 0
        assert (valid_low >= -1e-10).all()

    def test_candle_features_in_all_indicators(self):
        from features.technical import add_all_indicators
        df = _make_ohlcv_df()
        result = add_all_indicators(df)
        for col in ["candle_body", "upper_wick_ratio", "lower_wick_ratio",
                    "price_to_high_20", "price_to_low_20"]:
            assert col in result.columns, f"{col} manquant dans add_all_indicators"

    def test_candle_features_missing_open(self):
        """Sans colonne open, retourne NaN."""
        from features.technical import add_candle_features
        df = _make_ohlcv_df().drop(columns=["open"])
        result = add_candle_features(df)
        assert "candle_body" in result.columns
        assert result["candle_body"].isna().all()


class TestNLP:
    """Tests pour features/nlp.py"""

    def test_import(self):
        from features.nlp import (
            analyze_sentiment,
            compute_hourly_sentiment,
            add_sentiment_to_dataframe,
        )
        assert callable(analyze_sentiment)
        assert callable(compute_hourly_sentiment)
        assert callable(add_sentiment_to_dataframe)

    def test_analyze_empty_list(self):
        from features.nlp import analyze_sentiment
        result = analyze_sentiment([])
        assert result == []

    def test_compute_hourly_sentiment_no_titles(self):
        from features.nlp import compute_hourly_sentiment
        result = compute_hourly_sentiment([])
        assert result["sentiment_score"] == 0.0
        assert result["n_articles"] == 0

    def test_analyze_sentiment_basic(self):
        """Test avec des titres réels (nécessite le modèle FinBERT)."""
        from features.nlp import analyze_sentiment
        texts = [
            "Bitcoin surges to new all-time high",
            "Crypto market crashes, investors panic",
            "Fed announces interest rate decision",
        ]
        results = analyze_sentiment(texts)
        assert len(results) == 3
        for r in results:
            assert "text" in r
            assert "label" in r
            assert "score" in r
            assert -1 <= r["score"] <= 1

    def test_compute_hourly_sentiment_with_titles(self):
        from features.nlp import compute_hourly_sentiment
        titles = [
            "Bitcoin price reaches record high as ETF inflows surge",
            "Major crypto exchange hacked, millions lost",
        ]
        result = compute_hourly_sentiment(titles)
        assert -1 <= result["sentiment_score"] <= 1
        assert result["n_articles"] == 2

    def test_add_sentiment_to_dataframe_no_news(self):
        from features.nlp import add_sentiment_to_dataframe
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"),
            "close": range(24),
        })
        result = add_sentiment_to_dataframe(df, news_df=None)
        assert "sentiment_score" in result.columns
        assert (result["sentiment_score"] == 0.0).all()


class TestScaler:
    """Tests pour features/scaler.py"""

    def test_import(self):
        from features.scaler import FeatureScaler, normalize_features
        assert callable(normalize_features)

    def test_fit_transform(self):
        from features.scaler import FeatureScaler
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"),
            "close": np.random.randn(100) * 1000 + 42000,
            "volume": np.random.randn(100) * 500 + 1000,
            "rsi": np.random.randn(100) * 20 + 50,
            "is_weekend": [0, 1] * 50,
        })

        scaler = FeatureScaler()
        result = scaler.fit_transform(df)

        # Les colonnes normalisées doivent être entre -1 et +1
        for col in ["close", "volume", "rsi"]:
            assert result[col].between(-1, 1).all(), f"{col} hors limites"

        # is_weekend ne doit PAS être normalisé
        assert "is_weekend" in result.columns

        # timestamp ne doit PAS être normalisé
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])

    def test_transform_without_fit(self):
        from features.scaler import FeatureScaler
        scaler = FeatureScaler()
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(RuntimeError):
            scaler.transform(df)

    def test_save_load(self, tmp_path):
        from features.scaler import FeatureScaler
        df = pd.DataFrame({
            "close": np.random.randn(100) * 1000 + 42000,
            "volume": np.random.randn(100) * 500 + 1000,
        })

        # Fit et save
        scaler = FeatureScaler()
        scaler.fit(df)
        save_path = tmp_path / "test_scaler.pkl"
        scaler.save(save_path)

        # Load et vérifier
        scaler2 = FeatureScaler()
        scaler2.load(save_path)
        assert scaler2.is_fitted
        assert scaler2.feature_columns == scaler.feature_columns

        # Les résultats doivent être identiques
        result1 = scaler.transform(df)
        result2 = scaler2.transform(df)
        pd.testing.assert_frame_equal(result1, result2)

    def test_normalize_features_utility(self):
        from features.scaler import normalize_features
        df = pd.DataFrame({
            "close": np.random.randn(100) * 1000 + 42000,
            "rsi": np.random.randn(100) * 20 + 50,
        })

        df_scaled, scaler = normalize_features(df, fit=True)
        assert scaler.is_fitted
        assert df_scaled["close"].between(-1, 1).all()

    def test_handles_inf_values(self):
        from features.scaler import FeatureScaler
        df = pd.DataFrame({
            "close": [1.0, 2.0, np.inf, 4.0, -np.inf],
            "volume": [100, 200, 300, 400, 500],
        })
        scaler = FeatureScaler()
        result = scaler.fit_transform(df)
        # Pas de NaN ni d'inf dans le résultat
        assert not result["close"].isna().any()
        assert not np.isinf(result["close"]).any()
