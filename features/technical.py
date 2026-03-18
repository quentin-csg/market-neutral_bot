"""
Calcul des indicateurs techniques via pandas-ta.
Indicateurs : SMA 50/200, RSI, ATR, Bandes de Bollinger, Z-Score.
Les données Order Book, Funding Rate et Open Interest sont récupérées
par le data pipeline (Phase 2) et simplement passées ici.
"""

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta

from config.settings import (
    ADX_PERIOD,
    ATR_PERIOD,
    BOLLINGER_PERIOD,
    BOLLINGER_STD,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    PRICE_POSITION_WINDOW,
    RSI_PERIOD,
    SMA_LONG,
    SMA_SHORT,
)

logger = logging.getLogger(__name__)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute tous les indicateurs techniques au DataFrame OHLCV.

    Args:
        df: DataFrame avec colonnes: timestamp, open, high, low, close, volume

    Returns:
        DataFrame enrichi avec tous les indicateurs techniques.
    """
    df = df.copy()

    if df.empty or "close" not in df.columns:
        logger.warning("DataFrame vide ou sans colonne 'close'")
        return df

    df = add_sma(df)
    df = add_rsi(df)
    df = add_atr(df)
    df = add_bollinger(df)
    df = add_zscore(df)
    df = add_volume_features(df)
    df = add_returns(df)
    df = add_macd(df)
    df = add_adx(df)
    df = add_candle_features(df)
    df = add_price_position_features(df)

    logger.info(f"Indicateurs techniques ajoutés: {len(df.columns)} colonnes")
    return df


def add_sma(
    df: pd.DataFrame,
    short: int = SMA_SHORT,
    long: int = SMA_LONG,
) -> pd.DataFrame:
    """Ajoute les SMA courte et longue + signal de croisement."""
    df = df.copy()

    sma_short = ta.sma(df["close"], length=short)
    sma_long = ta.sma(df["close"], length=long)

    # pandas-ta retourne None si pas assez de données → créer une Series de NaN
    df[f"sma_{short}"] = sma_short if sma_short is not None else np.nan
    df[f"sma_{long}"] = sma_long if sma_long is not None else np.nan

    # Signal de tendance : 1 si SMA courte > SMA longue (bullish), -1 sinon
    # NaN quand les SMAs n'ont pas assez de données (warmup)
    mask = df[f"sma_{short}"].notna() & df[f"sma_{long}"].notna()
    df["sma_trend"] = np.nan
    df.loc[mask, "sma_trend"] = np.where(
        df.loc[mask, f"sma_{short}"] > df.loc[mask, f"sma_{long}"], 1.0, -1.0
    )

    # Distance relative entre le prix et la SMA longue
    df["price_to_sma_long"] = np.where(
        df[f"sma_{long}"].notna() & (df[f"sma_{long}"] != 0),
        (df["close"] - df[f"sma_{long}"]) / df[f"sma_{long}"],
        np.nan,
    )

    return df


def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Ajoute le RSI normalisé entre -1 et +1 (au lieu de 0-100)."""
    df = df.copy()
    rsi_raw = ta.rsi(df["close"], length=period)
    if rsi_raw is None:
        df["rsi"] = np.nan
        df["rsi_normalized"] = np.nan
    else:
        df["rsi"] = rsi_raw
        df["rsi_normalized"] = (rsi_raw - 50) / 50.0
    return df


def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """Ajoute l'ATR (Average True Range) comme mesure de volatilité."""
    df = df.copy()
    atr_raw = ta.atr(df["high"], df["low"], df["close"], length=period)
    if atr_raw is None:
        df["atr"] = np.nan
        df["atr_pct"] = np.nan
    else:
        df["atr"] = atr_raw
        df["atr_pct"] = atr_raw / df["close"]
    return df


def add_bollinger(
    df: pd.DataFrame,
    period: int = BOLLINGER_PERIOD,
    std: int = BOLLINGER_STD,
) -> pd.DataFrame:
    """Ajoute les Bandes de Bollinger + position relative du prix."""
    df = df.copy()
    bbands = ta.bbands(df["close"], length=period, std=std)

    if bbands is not None and not bbands.empty:
        df["bb_lower"] = bbands.iloc[:, 0]   # BBL — Bande inférieure
        df["bb_middle"] = bbands.iloc[:, 1]  # BBM — Bande médiane (SMA)
        df["bb_upper"] = bbands.iloc[:, 2]   # BBU — Bande supérieure
        df["bb_bandwidth"] = bbands.iloc[:, 3] if bbands.shape[1] > 3 else None
        df["bb_percent"] = bbands.iloc[:, 4] if bbands.shape[1] > 4 else None

        # Position relative du prix dans les bandes (-1 = bande basse, +1 = bande haute)
        band_range = df["bb_upper"] - df["bb_lower"]
        df["bb_position"] = np.where(
            band_range > 0,
            2 * (df["close"] - df["bb_lower"]) / band_range - 1,
            0,
        )
    else:
        logger.warning("Bollinger Bands non calculées")

    return df


def add_zscore(
    df: pd.DataFrame,
    period: int = BOLLINGER_PERIOD,
) -> pd.DataFrame:
    """
    Ajoute le Z-Score du prix (nombre d'écarts-types par rapport à la moyenne).
    Utile pour détecter les valeurs extrêmes.
    """
    df = df.copy()
    rolling_mean = df["close"].rolling(window=period).mean()
    rolling_std = df["close"].rolling(window=period).std()
    df["zscore"] = np.where(
        rolling_std > 0,
        (df["close"] - rolling_mean) / rolling_std,
        0,
    )
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des features liées au volume."""
    df = df.copy()
    # Volume relatif (ratio vs moyenne mobile 20 périodes)
    vol_ma = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = np.where(vol_ma > 0, df["volume"] / vol_ma, 1.0)

    # Volume direction (volume × direction du prix)
    price_change = df["close"].diff()
    df["volume_direction"] = np.sign(price_change) * df["volume_ratio"]

    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les log-returns (comme recommandé dans instruction.md).
    Les log-returns pénalisent plus les grosses pertes.
    """
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["log_return_5h"] = np.log(df["close"] / df["close"].shift(5))
    df["log_return_24h"] = np.log(df["close"] / df["close"].shift(24))
    return df


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les features de structure des bougies japonaises.

    Colonnes ajoutées :
    - candle_body        : corps de la bougie normalisé (-1 à +1)
    - upper_wick_ratio   : mèche haute en % de la range (0 à 1)
    - lower_wick_ratio   : mèche basse en % de la range (0 à 1)
    """
    df = df.copy()
    if "open" not in df.columns or "high" not in df.columns or "low" not in df.columns:
        logger.warning("Colonnes open/high/low manquantes pour candle features")
        for col in ["candle_body", "upper_wick_ratio", "lower_wick_ratio"]:
            df[col] = np.nan
        return df

    wick_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["candle_body"] = ((df["close"] - df["open"]) / wick_range).clip(-1, 1)
    df["upper_wick_ratio"] = (
        (df["high"] - df[["open", "close"]].max(axis=1)) / wick_range
    ).clip(0, 1)
    df["lower_wick_ratio"] = (
        (df[["open", "close"]].min(axis=1) - df["low"]) / wick_range
    ).clip(0, 1)
    return df


def add_price_position_features(
    df: pd.DataFrame,
    window: int = PRICE_POSITION_WINDOW,
) -> pd.DataFrame:
    """
    Ajoute la position du prix par rapport aux hauts/bas récents.

    Colonnes ajoutées :
    - price_to_high_20 : distance au plus haut sur N périodes (≤ 0)
    - price_to_low_20  : distance au plus bas sur N périodes (≥ 0)

    Ces features indiquent si le prix est proche d'une résistance ou d'un support.
    """
    df = df.copy()
    if "high" not in df.columns or "low" not in df.columns:
        logger.warning("Colonnes high/low manquantes pour price position features")
        df["price_to_high_20"] = np.nan
        df["price_to_low_20"] = np.nan
        return df

    high_n = df["high"].rolling(window=window).max()
    low_n = df["low"].rolling(window=window).min()

    df["price_to_high_20"] = np.where(
        high_n.notna() & (high_n > 0),
        (df["close"] - high_n) / high_n,
        np.nan,
    )
    df["price_to_low_20"] = np.where(
        low_n.notna() & (low_n > 0),
        (df["close"] - low_n) / low_n,
        np.nan,
    )
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """
    Ajoute le MACD (Moving Average Convergence Divergence).

    Colonnes ajoutées :
    - macd_hist_normalized : histogramme MACD normalisé par le prix
      (signal de momentum + retournement, indépendant de l'échelle)
    """
    df = df.copy()
    macd_result = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    if macd_result is not None and not macd_result.empty:
        # pandas-ta retourne [MACD, Histogram, Signal] (ordre varie selon version)
        # On cherche la colonne histogram par nom
        hist_cols = [c for c in macd_result.columns if "h" in c.lower()]
        macd_cols = [c for c in macd_result.columns if "macd" in c.lower() and "h" not in c.lower() and "s" not in c.lower()]
        sig_cols = [c for c in macd_result.columns if "s" in c.lower()]

        df["macd"] = macd_result[macd_cols[0]] if macd_cols else np.nan
        df["macd_histogram"] = macd_result[hist_cols[0]] if hist_cols else np.nan
        df["macd_signal_line"] = macd_result[sig_cols[0]] if sig_cols else np.nan

        # Normalisation par le prix : scale-independent
        close_nonzero = df["close"].replace(0, np.nan)
        df["macd_hist_normalized"] = df["macd_histogram"] / close_nonzero
    else:
        logger.warning("MACD non calculé")
        for col in ["macd", "macd_histogram", "macd_signal_line", "macd_hist_normalized"]:
            df[col] = np.nan
    return df


def add_multi_timeframe_features(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    rsi_period: int = RSI_PERIOD,
    sma_short: int = SMA_SHORT,
    sma_long: int = SMA_LONG,
) -> pd.DataFrame:
    """
    Calcule des indicateurs sur le timeframe 4h et les merge sur la grille 1h.

    Le contexte macro 4h apporte une vue de tendance plus stable que le 1h.
    Utilise merge_asof (backward) pour aligner les valeurs 4h sur chaque bougie 1h.

    Colonnes ajoutées :
    - rsi_4h_normalized : RSI 4h normalisé (-1 à +1)
    - sma_trend_4h     : Tendance SMA50/SMA200 sur 4h (+1/-1)

    Args:
        df_1h: DataFrame 1h (avec colonne 'timestamp')
        df_4h: DataFrame 4h brut (OHLCV avec colonne 'timestamp')

    Returns:
        df_1h enrichi avec les features 4h
    """
    if df_4h.empty or "close" not in df_4h.columns:
        logger.warning("Données 4h vides — features multi-timeframe ignorées")
        df_1h = df_1h.copy()
        df_1h["rsi_4h_normalized"] = np.nan
        df_1h["sma_trend_4h"] = np.nan
        return df_1h

    df_4h = df_4h.copy()

    # RSI 4h
    rsi_4h = ta.rsi(df_4h["close"], length=rsi_period)
    df_4h["rsi_4h_normalized"] = (rsi_4h - 50) / 50.0 if rsi_4h is not None else np.nan

    # SMA trend 4h
    sma_s = ta.sma(df_4h["close"], length=sma_short)
    sma_l = ta.sma(df_4h["close"], length=sma_long)
    if sma_s is not None and sma_l is not None:
        mask = sma_s.notna() & sma_l.notna()
        df_4h["sma_trend_4h"] = np.nan
        df_4h.loc[mask, "sma_trend_4h"] = np.where(
            sma_s[mask] > sma_l[mask], 1.0, -1.0
        )
    else:
        df_4h["sma_trend_4h"] = np.nan

    # Aligner les timestamps
    df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"]).dt.floor("4h")
    df_4h = df_4h.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    df_1h = df_1h.copy().sort_values("timestamp")
    df_4h_slim = df_4h[["timestamp", "rsi_4h_normalized", "sma_trend_4h"]]

    result = pd.merge_asof(
        df_1h,
        df_4h_slim,
        on="timestamp",
        direction="backward",
    )

    logger.info("Features multi-timeframe (4h) ajoutées: rsi_4h_normalized, sma_trend_4h")
    return result


def add_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """
    Ajoute l'ADX (Average Directional Index).

    Mesure la force de la tendance (0-100), indépendamment de la direction.
    ADX > 25 = tendance forte. ADX < 20 = marché en range.

    Colonnes ajoutées :
    - adx_normalized : ADX normalisé entre 0 et 1
    """
    df = df.copy()

    if "high" not in df.columns or "low" not in df.columns:
        logger.warning("Colonnes high/low manquantes pour ADX")
        df["adx_normalized"] = np.nan
        return df

    adx_result = ta.adx(df["high"], df["low"], df["close"], length=period)
    if adx_result is not None and not adx_result.empty:
        adx_cols = [c for c in adx_result.columns if "adx" in c.lower() and "d" not in c.lower()]
        if adx_cols:
            df["adx"] = adx_result[adx_cols[0]]
            df["adx_normalized"] = df["adx"] / 100.0
        else:
            df["adx"] = adx_result.iloc[:, 0]
            df["adx_normalized"] = df["adx"] / 100.0
    else:
        logger.warning("ADX non calculé")
        df["adx"] = np.nan
        df["adx_normalized"] = np.nan
    return df
