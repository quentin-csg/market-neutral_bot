# Phase 7 — Checklist de validation

## PaperPortfolio (`live/executor.py`)

### Tests automatiques (pytest)
```
python -m pytest tests/test_live.py::TestPaperPortfolio -v
```
**7 tests a verifier :**

- [X] Portefeuille initialise avec le bon solde
- [X] Achat : balance diminue, position augmente, trade compté
- [X] Vente : position diminue, balance augmente
- [X] Zone morte (action < 5%) = hold, pas de trade
- [X] Frais de 0.1% correctement appliques
- [X] get_stats retourne toutes les metriques
- [X] Historique des trades conserve

## CircuitBreaker (`live/circuit_breaker.py`)

### Tests automatiques (pytest)
```
python -m pytest tests/test_live.py::TestCircuitBreaker -v
```
**8 tests a verifier :**

- [X] Demarre en etat non declenche
- [X] Status retourne la config correcte
- [X] Declenchement met le bon etat + raison + timestamp
- [X] Reset remet en etat normal
- [X] Detection chute de prix (>3% en 5min)
- [X] Detection volume anormal (>5x la moyenne)
- [X] Pas de faux positif en conditions normales
- [X] Fermeture positions en mode paper

## LiveExecutor (`live/executor.py`)

### Tests automatiques (pytest)
```
python -m pytest tests/test_live.py::TestLiveExecutor -v
```
**3 tests a verifier :**

- [X] Initialisation en mode paper (pas d'exchange)
- [X] Mode live sans API keys leve une erreur
- [X] Paper portfolio accessible dans l'executeur

## CLI (`main.py`)

### Tests automatiques (pytest)
```
python -m pytest tests/test_live.py::TestCLI -v
```
**2 tests a verifier :**

- [X] Help affiche les 4 commandes
- [X] Arguments parses sans erreur

## Tests manuels (REPL Python)

Lancer `python` puis copier-coller chaque bloc.

### Test 1 — Paper Portfolio achat/vente
```python
from live.executor import PaperPortfolio

p = PaperPortfolio(initial_balance=10000.0)
print(f"Initial: {p.balance} USDT, {p.position} BTC")

order1 = p.execute_order(0.5, current_price=50000.0)
print(f"Apres achat 50%: {p.balance:.2f} USDT, {p.position:.6f} BTC")
print(f"  Type: {order1['type']}, Fee: {order1['fee']:.2f}")

order2 = p.execute_order(-1.0, current_price=51000.0)
print(f"Apres vente 100%: {p.balance:.2f} USDT, {p.position:.6f} BTC")
print(f"  Type: {order2['type']}, Fee: {order2['fee']:.2f}")

stats = p.get_stats(current_price=51000.0)
print(f"Net worth: {stats['net_worth']:.2f}, Return: {stats['total_return_pct']:.2f}%")
print(f"Trades: {stats['total_trades']}, Fees: {stats['total_fees']:.2f}")
```
- [X] Achat reduit la balance, augmente la position
- [X] Vente vide la position, augmente la balance
- [X] Fees > 0 apres chaque trade
- [X] Net worth coherent

### Test 2 — Zone morte
```python
from live.executor import PaperPortfolio

p = PaperPortfolio(initial_balance=10000.0)
order = p.execute_order(0.03, current_price=50000.0)
print(f"Action 0.03 -> Type: {order['type']}")
print(f"Balance inchangee: {p.balance == 10000.0}")
print(f"Trades: {p.total_trades}")
```
- [X] Type = hold
- [X] Balance inchangee
- [X] 0 trades

### Test 3 — Circuit Breaker etat
```python
from unittest.mock import patch, MagicMock
from live.circuit_breaker import CircuitBreaker

with patch.object(CircuitBreaker, "_init_exchange", return_value=MagicMock()):
    cb = CircuitBreaker(live_mode=False)
    print(f"Triggered: {cb.triggered}")
    print(f"Status: {cb.status}")
    cb.trigger("Test crash -5%")
    print(f"Apres trigger: {cb.triggered}")
    print(f"Raison: {cb.trigger_reason}")
    cb.reset()
    print(f"Apres reset: {cb.triggered}")
```
- [X] Triggered = False au debut
- [X] Triggered = True apres trigger
- [X] Triggered = False apres reset

### Test 4 — CLI help
```python
import subprocess, sys
result = subprocess.run([sys.executable, "main.py", "--help"], capture_output=True, text=True)
print(result.stdout)
```
- [X] Affiche les 4 commandes (train, backtest, live, dashboard)
- [X] Affiche les options (--model, --live-mode, --timesteps, --nlp)

### Test 5 — LiveExecutor init
```python
from live.executor import LiveExecutor

executor = LiveExecutor(model_name="test", live_mode=False)
print(f"Mode: {'LIVE' if executor.live_mode else 'PAPER'}")
print(f"Model: {executor.model_name}")
print(f"Interval: {executor.interval}s")
print(f"Portfolio balance: {executor.paper_portfolio.balance}")
print(f"Exchange: {executor.exchange}")
```
- [X] Mode PAPER
- [X] Portfolio balance = 10000.0
- [X] Exchange = None

## Tous les tests (regression)
```
python -m pytest tests/ -v
```
- [X] 114/114 tests passent
