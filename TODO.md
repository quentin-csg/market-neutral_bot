# TODO — mn-bot

## ✅ Fait

### Infrastructure

- Rust toolchain + `cargo build --release` OK
- `maturin develop --release` — module Python buildé
- 24/24 tests Python verts, **16/16 tests Rust verts**

### P1 (avant paper trading)

- HALT path absolu, frais paper trading, position state sync, logger validation
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

### Divers

- Backtest validé : +10.59% sur 2021, +2.42% sur 2024 avec seuil 10% APR
- README.md créé

---

## ❌ Reste à faire

### P1 — Paper trading (en cours)

- [ ] Laisser `mn-bot run --mode paper` tourner 24-48h sans interruption

### P2 — Nécessite clé API Binance

- [ ] Intégration orchestrator : appel périodique `futures_account_info` pour refresh `PortfolioState.equity`
- [ ] Créer compte Binance Testnet, valider 24h (testnet.binancefuture.com)
- [ ] Baisser `MAX_NOTIONAL_USDT` à 50-100 pour les premières semaines live

### P4.5 — GIL batching (optionnel)

- [ ] Batcher N ticks avant `Python::with_gil` dans `rust_core/src/lib.rs` (~20% latence gagnée)

### P5 — Production

- [ ] Monitoring : logs JSON dans Loki/Grafana
- [ ] Alertes sur : `RiskError`, reconnexions WS > 3/h, latence > 100ms
- [ ] Réévaluer Kelly fraction (0.5 → 0.7 après 1 mois de données live)
