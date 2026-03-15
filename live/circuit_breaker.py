"""
Circuit Breaker — Surveillance temps réel.

Script léger qui tourne en parallèle du bot principal.
Surveille les bougies 1 minute via polling ccxt.
Si le volume explose ou le prix chute brutalement, coupe les positions.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import ccxt
import numpy as np

from config.settings import (
    API_KEY,
    API_SECRET,
    CB_CHECK_INTERVAL,
    CB_LOOKBACK_MINUTES,
    CB_PRICE_DROP_THRESHOLD,
    CB_VOLUME_SPIKE_FACTOR,
    EXCHANGE,
    SYMBOL,
)

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Surveillance temps réel des conditions de marché anormales.

    Détecte :
    - Chute de prix > seuil en N minutes
    - Volume anormal (> X fois la moyenne)

    Si déclenché → coupe les positions instantanément.
    """

    def __init__(
        self,
        symbol: str = SYMBOL,
        price_drop_threshold: float = CB_PRICE_DROP_THRESHOLD,
        volume_spike_factor: float = CB_VOLUME_SPIKE_FACTOR,
        lookback_minutes: int = CB_LOOKBACK_MINUTES,
        live_mode: bool = False,
    ):
        self.symbol = symbol
        self.price_drop_threshold = price_drop_threshold
        self.volume_spike_factor = volume_spike_factor
        self.lookback_minutes = lookback_minutes
        self.live_mode = live_mode

        self.triggered = False
        self.trigger_reason = ""
        self.trigger_time: Optional[datetime] = None
        self.running = False

        # Historique des prix/volumes pour la détection
        self.price_history: list[float] = []
        self.volume_history: list[float] = []

        # Exchange
        self.exchange = self._init_exchange()

        logger.info(
            f"CircuitBreaker initialisé: drop>{price_drop_threshold*100}%, "
            f"volume>{volume_spike_factor}x, lookback={lookback_minutes}min"
        )

    def _init_exchange(self) -> ccxt.Exchange:
        """Initialise la connexion à l'exchange."""
        exchange_class = getattr(ccxt, EXCHANGE)
        config = {"enableRateLimit": True}

        if self.live_mode and API_KEY and API_SECRET:
            config["apiKey"] = API_KEY
            config["secret"] = API_SECRET

        return exchange_class(config)

    def check_conditions(self) -> dict:
        """
        Vérifie les conditions de marché.

        Returns:
            Dict avec les résultats de la vérification
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "price_drop_detected": False,
            "volume_spike_detected": False,
            "current_price": 0.0,
            "price_change_pct": 0.0,
            "volume_ratio": 0.0,
        }

        try:
            # Fetch les dernières bougies 1min
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe=CB_CHECK_INTERVAL,
                limit=self.lookback_minutes + 20,
            )

            if not ohlcv or len(ohlcv) < self.lookback_minutes:
                logger.warning("Pas assez de données pour le circuit breaker")
                return result

            # Extraire prix et volumes
            prices = [c[4] for c in ohlcv]  # close prices
            volumes = [c[5] for c in ohlcv]  # volumes

            current_price = prices[-1]
            result["current_price"] = current_price

            # Vérifier la chute de prix
            recent_prices = prices[-self.lookback_minutes:]
            if recent_prices:
                max_recent = max(recent_prices)
                price_change = (current_price - max_recent) / max_recent
                result["price_change_pct"] = price_change

                if price_change < -self.price_drop_threshold:
                    result["price_drop_detected"] = True
                    logger.warning(
                        f"CIRCUIT BREAKER: Chute de prix détectée! "
                        f"{price_change*100:.2f}% en {self.lookback_minutes}min"
                    )

            # Vérifier le volume anormal
            if len(volumes) > self.lookback_minutes:
                avg_volume = np.mean(volumes[:-self.lookback_minutes])
                recent_volume = np.mean(volumes[-self.lookback_minutes:])

                if avg_volume > 0:
                    volume_ratio = recent_volume / avg_volume
                    result["volume_ratio"] = volume_ratio

                    if volume_ratio > self.volume_spike_factor:
                        result["volume_spike_detected"] = True
                        logger.warning(
                            f"CIRCUIT BREAKER: Volume anormal! "
                            f"{volume_ratio:.1f}x la moyenne"
                        )

        except Exception as e:
            logger.error(f"Erreur circuit breaker check: {e}")

        return result

    def close_all_positions(self) -> dict:
        """
        Ferme toutes les positions ouvertes.

        Returns:
            Dict avec les détails de la fermeture
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "action": "close_all",
            "success": False,
        }

        if not self.live_mode:
            logger.info("CIRCUIT BREAKER [PAPER]: Positions coupées (simulé)")
            result["success"] = True
            result["mode"] = "paper"
            return result

        try:
            # Mode live : fermer les positions réelles
            balance = self.exchange.fetch_balance()
            btc_balance = balance.get("BTC", {}).get("free", 0)

            if btc_balance > 0.00001:
                order = self.exchange.create_market_sell_order(
                    self.symbol, btc_balance
                )
                result["success"] = True
                result["order"] = order
                logger.info(
                    f"CIRCUIT BREAKER [LIVE]: Vendu {btc_balance:.6f} BTC"
                )
            else:
                result["success"] = True
                result["message"] = "Pas de position ouverte"

        except Exception as e:
            logger.error(f"Erreur fermeture positions: {e}")
            result["error"] = str(e)

        return result

    def trigger(self, reason: str) -> dict:
        """Déclenche le circuit breaker."""
        self.triggered = True
        self.trigger_reason = reason
        self.trigger_time = datetime.now()

        logger.critical(
            f"CIRCUIT BREAKER DÉCLENCHÉ: {reason} "
            f"à {self.trigger_time.strftime('%H:%M:%S')}"
        )

        return self.close_all_positions()

    def reset(self):
        """Réinitialise le circuit breaker après déclenchement."""
        self.triggered = False
        self.trigger_reason = ""
        self.trigger_time = None
        logger.info("Circuit breaker réinitialisé")

    @property
    def status(self) -> dict:
        """Retourne le statut actuel du circuit breaker."""
        return {
            "triggered": self.triggered,
            "reason": self.trigger_reason,
            "trigger_time": (
                self.trigger_time.isoformat() if self.trigger_time else None
            ),
            "config": {
                "price_drop_threshold": self.price_drop_threshold,
                "volume_spike_factor": self.volume_spike_factor,
                "lookback_minutes": self.lookback_minutes,
            },
        }

    def run(self, check_interval_seconds: int = 60):
        """
        Boucle de surveillance continue.

        Args:
            check_interval_seconds: intervalle entre les vérifications (défaut: 60s)
        """
        print(f"=== Circuit Breaker démarré ===")
        print(f"Symbole: {self.symbol}")
        print(f"Seuil chute: {self.price_drop_threshold*100}%")
        print(f"Seuil volume: {self.volume_spike_factor}x")
        print(f"Fenêtre: {self.lookback_minutes} minutes")
        print(f"Vérification toutes les {check_interval_seconds}s")
        print(f"Appuyez sur Ctrl+C pour arrêter.\n")

        self.running = True

        try:
            while self.running:
                conditions = self.check_conditions()

                if conditions["price_drop_detected"]:
                    self.trigger(
                        f"Chute prix: {conditions['price_change_pct']*100:.2f}%"
                    )
                elif conditions["volume_spike_detected"]:
                    self.trigger(
                        f"Volume anormal: {conditions['volume_ratio']:.1f}x"
                    )
                else:
                    price = conditions["current_price"]
                    change = conditions["price_change_pct"] * 100
                    vol = conditions["volume_ratio"]
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"Prix: {price:.2f} | "
                        f"Variation: {change:+.2f}% | "
                        f"Volume ratio: {vol:.1f}x | "
                        f"Status: OK"
                    )

                if self.triggered:
                    print("⚠ CIRCUIT BREAKER DÉCLENCHÉ — Positions coupées")
                    print(f"  Raison: {self.trigger_reason}")
                    # Rester en mode triggered, ne plus trader
                    # L'utilisateur doit reset manuellement

                time.sleep(check_interval_seconds)

        except KeyboardInterrupt:
            print("\n=== Circuit Breaker arrêté ===")
        finally:
            self.running = False


def run_circuit_breaker(live_mode: bool = False):
    """Point d'entrée pour lancer le circuit breaker."""
    cb = CircuitBreaker(live_mode=live_mode)
    cb.run()
