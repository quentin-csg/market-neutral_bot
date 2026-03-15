# Phase 1 — Checklist de validation

## Exécute chaque commande ci-dessous dans ton terminal (depuis le dossier Trading-bot)

## et coche la case quand c'est OK

## Structure du projet

- [X] **Les dossiers existent** :
  ```
  dir
  ```
  Vérifier que ces dossiers sont présents :
  `config/`, `data/`, `features/`, `env/`, `agent/`, `training/`, `live/`, `tests/`, `models/`, `logs/`, `notebooks/`, `test_phases/`

- [X] **Les packages Python sont initialisés** (`__init__.py` présents) :
  ```
  dir config\__init__.py
  dir data\__init__.py
  dir features\__init__.py
  dir env\__init__.py
  dir agent\__init__.py
  dir training\__init__.py
  dir live\__init__.py
  dir tests\__init__.py
  ```
  → Chaque commande doit trouver le fichier.

## Configuration centralisée

- [X] **`config/settings.py` est importable** :
  ```
  python -c "from config import settings; print('OK: settings importé')"
  ```

- [X] **Les chemins de base sont corrects** :
  ```
  python -c "from config.settings import BASE_DIR, MODELS_DIR, LOGS_DIR; print(f'BASE: {BASE_DIR}'); print(f'MODELS: {MODELS_DIR}'); print(f'LOGS: {LOGS_DIR}')"
  ```

- [X] **Les paramètres crypto sont accessibles** :
  ```
  python -c "from config.settings import EXCHANGE, SYMBOL, TIMEFRAME; print(f'{EXCHANGE} | {SYMBOL} | {TIMEFRAME}')"
  ```
  → Doit afficher : `binance | BTC/USDT | 1h`

- [X] **Les hyperparamètres PPO sont un dict valide** :
  ```
  python -c "from config.settings import PPO_HYPERPARAMS; print(f'PPO params: {len(PPO_HYPERPARAMS)} clés'); assert 'learning_rate' in PPO_HYPERPARAMS"
  ```

- [X] **Les seuils du circuit breaker sont définis** :
  ```
  python -c "from config.settings import CB_PRICE_DROP_THRESHOLD, CB_VOLUME_SPIKE_FACTOR; print(f'Drop: {CB_PRICE_DROP_THRESHOLD}, Spike: {CB_VOLUME_SPIKE_FACTOR}')"
  ```
  → Doit afficher : `Drop: 0.03, Spike: 5.0`

- [X] **Les clés API sont vides par défaut (sécurité)** :
  ```
  python -c "from config.settings import API_KEY, API_SECRET; assert API_KEY == '', 'API_KEY devrait être vide!'; assert API_SECRET == '', 'API_SECRET devrait être vide!'; print('OK: clés API vides par défaut')"
  ```

## requirements.txt

- [X] **Le fichier requirements.txt existe et contient les bonnes dépendances** :
  ```
  python -c "lines = open('requirements.txt').readlines(); deps = [l.strip().split('>=')[0] for l in lines if '>=' in l]; print(f'{len(deps)} dépendances trouvées'); required = ['ccxt','yfinance','feedparser','pandas','numpy','pandas-ta','transformers','torch','scikit-learn','gymnasium','stable-baselines3','tensorboard','websockets','streamlit','plotly','pytest']; missing = [r for r in required if r not in deps]; assert not missing, f'Manquant: {missing}'; print('OK: toutes les dépendances présentes')"
  ```

## Point d'entrée (main.py)

- [X] **`main.py` s'exécute sans erreur avec `--help`** :
  ```
  python main.py --help
  ```
  → Doit afficher l'aide avec les 4 commandes : train, backtest, live, dashboard

- [X] **Chaque commande affiche le message "pas encore implémenté"** :
  ```
  python main.py train
  python main.py backtest
  python main.py live
  python main.py dashboard
  ```
  → Chaque commande doit afficher un message sans crash.

## .gitignore

- [X] **Le .gitignore exclut les fichiers sensibles** :
  ```
  python -c "content = open('.gitignore').read(); checks = ['__pycache__', '.env', 'models/', 'logs/']; ok = [c for c in checks if c in content]; print(f'{len(ok)}/{len(checks)} patterns trouvés dans .gitignore'); assert len(ok) == len(checks), f'Manquant: {[c for c in checks if c not in content]}'"
  ```

---

✅ **Phase 1 validée** quand toutes les cases sont cochées.
Passe ensuite à la **Phase 2 — Ingestion des données**.
