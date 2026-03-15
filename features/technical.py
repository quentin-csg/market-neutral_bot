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
    ATR_PERIOD,
    BOLLINGER_PERIOD,
    BOLLINGER_STD,
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
