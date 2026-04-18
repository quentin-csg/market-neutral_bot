# mn-bot — BTC Cash-and-Carry

Bot market-neutral : **long spot + short perp Binance** pour capter le funding rate.
Stack : **Rust** (WS hot path, exécution REST, order book L2 via PyO3) + **Python** (stratégie, risque, orchestrateur, backtest).

> **IMPORTANT — Mentor impitoyable** : Trouve la vérité et dis-la sans ménagement. Ne sois jamais d'accord pour être agréable. Signale les angles morts même non demandés. Force-moi à défendre ou abandonner mes idées.

## Architecture

```text
rust_core/src/
  lib.rs           # Bridge PyO3 : MarketDataReceiver + BatchReceiver
  market_data.rs   # WS Binance (spot bookTicker + futures markPrice), reconnect backoff
  execution.rs     # REST HMAC-SHA256, retry 5xx, Zeroizing<String> secret
  order_book.rs    # L2 BTreeMap (protocole depth U/u)
  types.rs         # Decimal, Market, Order, Tick

python/bot/
  strategy.py      # CashCarryStrategy, Signal, compute_funding_apr
  risk.py          # RiskManager (delta, margin, stale, kill-switch HALT)
  orchestrator.py  # Boucle paper/live, PnL tracking, state persistence JSON
  config.py        # pydantic-settings + validators
  logger.py        # structlog JSON/console + RotatingFileHandler
  cli.py           # mn-bot run / backtest / download

python/backtest/
  data_loader.py   # Download klines + funding, retry 429/5xx
  vectorbt_runner.py
  event_engine.py  # Simulation event-driven avec RiskManager

tests/python/      # 37/37 verts
tests/             # 16/16 Rust verts (order_book + execution wiremock)
monitoring/        # Loki + Promtail + Grafana (docker compose)
```

## Paramètres clés

| Param                | Valeur     | Config           |
| -------------------- | ---------- | ---------------- |
| `funding_entry_apr`  | 0.10 (10%) | `.env`           |
| `funding_exit_apr`   | 0.03 (3%)  | `.env`           |
| `kelly_fraction`     | 0.5        | `.env`           |
| `max_notional_usdt`  | variable   | `.env`           |
| `max_spread_pct`     | 0.001      | `StrategyConfig` |
| `max_delta_pct`      | 0.02       | `RiskManager`    |
| `margin_buffer_mult` | 3.0        | `RiskManager`    |
| `stale_tick_seconds` | 5          | `RiskManager`    |

## Commandes

```bash
pytest tests/python -v          # 37/37
cargo test -p rust_core         # 16/16
cargo build --release
maturin develop --release       # build + installe le module Python

mn-bot run --mode paper
mn-bot backtest --engine {vectorbt,event}
mn-bot download --start YYYY-MM-DD --end YYYY-MM-DD

touch HALT   # kill-switch (stoppe à la prochaine itération)
rm HALT

cd monitoring && docker compose up -d   # Grafana → http://localhost:3000
```

---

## ✅ Fait

### Infrastructure

- Rust toolchain + `cargo build --release` OK
- `maturin develop --release` — module Python buildé
- **37/37 tests Python verts**, **16/16 tests Rust verts**
- CI GitHub Actions : `pytest` + `cargo test` + `ruff` + `cargo clippy` sur push/PR

### P1 (avant paper trading)

- HALT path absolu (`Path(__file__).resolve().parents[2] / "HALT"`), frais paper trading, position state sync
- Logger : validation explicite du log level (raise `ValueError` si invalide)
- Backtest event-driven lit les paramètres depuis `.env`
- Log spam `reverse_funding_trigger` supprimé

### P2 (sans clé API)

- Retry HTTP avec backoff exponentiel (500ms→1s→2s, max 3 tentatives, retry sur 5xx + timeout)
- `Zeroizing<String>` pour `api_secret` — zéroïsé en mémoire à la destruction
- Nouvelle méthode `futures_account_info` (prête pour la synchro equity)
- Tests wiremock : 200, retry 500, fail fast 400, cancel, futures account

### P3 — Tests Python

- `test_orchestrator.py` — PnL round-trip avec FakeReceiver (ENTER/EXIT/force_exit/kill-switch)
- `test_event_engine.py` — fixtures Parquet synthétiques, funding capturé, empty data raises
- `test_data_loader.py` — retry 429/5xx via pytest-httpx, fail-fast 400, max retries

### P3 partiel (antérieur)

- Tests `order_book.rs` : `stale_update`, `vwap_bid/ask`, `vwap_empty`, `vwap_depth`
- Tests `execution.rs` via wiremock (5 tests)

### P4 — Améliorations mineures

- `recvWindow` exposé via `Settings.recv_window_ms` + `ExecutionConfig.recv_window_ms`
- Slippage backtest paramétrable via `Settings.backtest_slippage_pct` (défaut 0.05%)
- Delta recalculé sur mark-to-market (`check_delta` accepte qty + prix mark courants)
- GIL batching : `BatchReceiver` dans `rust_core/src/lib.rs`, orchestrator consomme via `receiver.batches(32)` — une acquisition GIL pour jusqu'à 32 ticks

### P5 — Observabilité

- Log sink fichier JSON rotatif (`Settings.log_file`, `RotatingFileHandler`)
- Events WS reconnect structurés dans `market_data.rs` (champ `market` pour filtrage Loki)
- Stack monitoring `monitoring/` : Loki + Promtail + Grafana (docker compose)
- Règles d'alerte Loki : `RiskError` rate > 0 sur 5min, WS reconnects > 3/h

### Divers

- Backtest validé : +10.59% sur 2021, +2.42% sur 2024 avec seuil 10% APR
- README.md créé

---

## ❌ Reste à faire

### P1 — Paper trading (en cours)

- [ ] Laisser `mn-bot run --mode paper` tourner 24-48h sans interruption

---

### P2 — Nécessite clé API Binance

#### P2.1 — Synchro equity live via `futures_account_info`

La méthode Rust existe déjà (`execution.rs:82`). Ce qui manque :

**Étape A — Exposer `ExecutionClient` à Python via PyO3 (`rust_core/src/lib.rs`)**

Ajouter un `#[pyclass]` wrapper minimal :

```rust
#[pyclass]
pub struct PyExecutionClient(ExecutionClient);

#[pymethods]
impl PyExecutionClient {
    /// Returns (equity_usdt, maintenance_margin_usdt)
    fn get_equity<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // appel futures_account_info + parse totalWalletBalance + totalMaintMargin
    }
}

#[pyfunction]
fn create_execution_client(
    api_key: &str, api_secret: &str, testnet: bool, recv_window_ms: u64
) -> PyResult<PyExecutionClient> { ... }
```

Le JSON `/fapi/v2/account` retourne (entre autres) :

```json
{ "totalWalletBalance": "1234.56", "totalMaintMargin": "12.34", ... }
```

Extraire `data["totalWalletBalance"]` et `data["totalMaintMargin"]` → convertir en `Decimal`.

**Étape B — Modifier `Orchestrator.__init__` (`python/bot/orchestrator.py`)**

```python
from mn_bot._rust import create_execution_client  # ajout

class Orchestrator:
    def __init__(self, settings):
        ...
        self._exec_client = None
        if settings.bot_mode == BotMode.live:
            self._exec_client = create_execution_client(
                settings.binance_api_key,
                settings.binance_api_secret,
                settings.binance_testnet,
                settings.recv_window_ms,
            )
```

**Étape C — Ajouter `_refresh_equity` et la tâche périodique (`orchestrator.py`)**

```python
async def _refresh_equity(self, interval_s: int = 30) -> None:
    while True:
        await asyncio.sleep(interval_s)
        if self._exec_client is None:
            return
        try:
            equity, maint = await self._exec_client.get_equity()
            self.portfolio.equity = equity
            self.portfolio.maintenance_margin = maint
            log.debug("equity_refreshed", equity=str(equity), maint=str(maint))
        except Exception as e:
            log.warning("equity_refresh_failed", error=str(e))

async def run(self) -> None:
    ...
    refresh_task = asyncio.create_task(self._refresh_equity(interval_s=30))
    try:
        async for batch in receiver.batches(32):
            ...
    finally:
        refresh_task.cancel()
```

**Tests à ajouter** : mock `get_equity` dans `test_orchestrator.py` (retourne `(Decimal("1050"), Decimal("5"))`) et vérifier que `portfolio.equity` est mis à jour après la synchro.

#### P2.2 — Compte Binance Testnet

- Créer un compte sur [testnet.binancefuture.com](https://testnet.binancefuture.com)
- Générer API key + secret, les mettre dans `.env` :

  ```env
  BINANCE_API_KEY=xxx
  BINANCE_API_SECRET=yyy
  BINANCE_TESTNET=true
  ```

#### P2.3 — Limiter l'exposition initiale

```env
MAX_NOTIONAL_USDT=100
```

---

### Post-paper (nécessite données live)

#### Alertes latence > 100ms

Ajouter dans `orchestrator.py` (`_on_both_ticks`) :

```python
import time
latency_ms = int(time.time() * 1000) - int(perp["ts_ms"])
if latency_ms > 100:
    log.warning("high_latency", latency_ms=latency_ms)
```

Puis règle Loki dans `monitoring/loki/rules.yml` :

```yaml
- alert: HighLatency
  expr: count_over_time({app="mn-bot"} |= "high_latency" [5m]) > 3
```

#### Réévaluer Kelly fraction

Après 1 mois live : si Sharpe > 1.5 et drawdown < 5%, monter dans `.env` :

```env
KELLY_FRACTION=0.7
```

#### Routing alertes Loki

Remplacer dans `monitoring/loki-config.yml` :

```yaml
alertmanager_url: http://localhost:9093
```

par l'URL Alertmanager réel + configurer un receiver Slack/PagerDuty.
