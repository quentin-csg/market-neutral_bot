"""
Fonctions de récompense pour l'agent RL.
Composantes :
  - Variation de Net Worth (log-returns)
  - Ratio de Sharpe/Sortino sur fenêtre glissante
  - Pénalité de drawdown exponentielle
  - Pénalité de position sizing
  - Pénalité de coûts de transaction
"""

import logging

import numpy as np

from config.settings import (
    MAX_DRAWDOWN_PENALTY,
    POSITION_SIZE_PENALTY_FACTOR,
    SHARPE_WINDOW,
)

logger = logging.getLogger(__name__)


def log_return_reward(net_worth: float, prev_net_worth: float) -> float:
    """
    Récompense basée sur le log-return de la valeur du portefeuille.
    Les log-returns pénalisent plus les grosses pertes (instruction.md).
    """
    if prev_net_worth <= 0 or net_worth <= 0:
        return -1.0
    return float(np.log(net_worth / prev_net_worth))


def sharpe_reward(returns_history: list[float], window: int = SHARPE_WINDOW) -> float:
    """
    Ratio de Sharpe sur fenêtre glissante.
    Force l'agent à chercher la régularité plutôt que les gros coups.

    Args:
        returns_history: historique des log-returns récents
        window: taille de la fenêtre glissante

    Returns:
        Ratio de Sharpe (mean / std) ou 0 si pas assez de données
    """
    if len(returns_history) < max(2, window // 4):
        return 0.0

    recent = np.array(returns_history[-window:])
    std = np.std(recent)
    if std < 1e-10:
        return 0.0
    return float(np.mean(recent) / std)


def sortino_reward(returns_history: list[float], window: int = SHARPE_WINDOW) -> float:
    """
    Ratio de Sortino sur fenêtre glissante.
    Comme Sharpe mais ne pénalise que la volatilité baissière.
    """
    if len(returns_history) < max(2, window // 4):
        return 0.0

    recent = np.array(returns_history[-window:])
    downside = recent[recent < 0]
    if len(downside) < 1:
        return float(np.mean(recent)) * 10 if np.mean(recent) > 0 else 0.0

    downside_std = np.std(downside)
    if downside_std < 1e-10:
        return 0.0
    return float(np.mean(recent) / downside_std)


def drawdown_penalty(
    net_worth: float,
    peak_net_worth: float,
    threshold: float = MAX_DRAWDOWN_PENALTY,
) -> float:
    """
    Pénalité exponentielle de drawdown (instruction.md).
    Plus le drawdown est profond, plus la pénalité croît exponentiellement.

    Args:
        net_worth: valeur actuelle du portefeuille
        peak_net_worth: pic historique du portefeuille
        threshold: seuil de drawdown au-delà duquel la pénalité explose

    Returns:
        Pénalité négative (0 si pas de drawdown)
    """
    if peak_net_worth <= 0:
        return 0.0

    drawdown = (peak_net_worth - net_worth) / peak_net_worth
    if drawdown <= 0:
        return 0.0

    # Pénalité exponentielle au-delà du seuil
    if drawdown > threshold:
        return -float(np.exp(drawdown / threshold) - 1)

    # Pénalité linéaire sous le seuil
    return -drawdown


def position_size_penalty(
    position_ratio: float,
    factor: float = POSITION_SIZE_PENALTY_FACTOR,
) -> float:
    """
    Pénalité quadratique pour les positions trop grosses (instruction.md).
    Plus la position est grande par rapport au capital, plus le risque augmente.

    Args:
        position_ratio: ratio |position_value| / net_worth (entre 0 et 1)
        factor: facteur multiplicatif de la pénalité

    Returns:
        Pénalité négative
    """
    return -factor * position_ratio ** 2


def transaction_cost_penalty(trade_cost: float, net_worth: float) -> float:
    """
    Pénalité proportionnelle aux coûts de transaction.
    Empêche l'hyper-trading (instruction.md).

    Args:
        trade_cost: coût total du trade (frais + slippage) en USDT
        net_worth: valeur totale du portefeuille

    Returns:
        Pénalité négative proportionnelle au coût relatif
    """
    if net_worth <= 0:
        return 0.0
    return -trade_cost / net_worth


def compute_reward(
    net_worth: float,
    prev_net_worth: float,
    peak_net_worth: float,
    position_ratio: float,
    trade_cost: float,
    returns_history: list[float],
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Calcule la récompense totale pondérée.

    Args:
        net_worth: valeur actuelle du portefeuille
        prev_net_worth: valeur au pas précédent
        peak_net_worth: pic historique
        position_ratio: ratio |position_value| / net_worth
        trade_cost: coût du trade effectué ce step
        returns_history: historique des log-returns
        weights: pondération de chaque composante (optionnel)

    Returns:
        Tuple (reward_total, dict des composantes pour logging)
    """
    if weights is None:
        weights = {
            "log_return": 1.0,
            "sharpe": 0.5,
            "drawdown": 1.0,
            "position_size": 0.3,
            "transaction": 1.0,
        }

    components = {
        "log_return": log_return_reward(net_worth, prev_net_worth),
        "sharpe": sharpe_reward(returns_history),
        "drawdown": drawdown_penalty(net_worth, peak_net_worth),
        "position_size": position_size_penalty(position_ratio),
        "transaction": transaction_cost_penalty(trade_cost, net_worth),
    }

    total = sum(weights.get(k, 1.0) * v for k, v in components.items())

    return float(total), components
