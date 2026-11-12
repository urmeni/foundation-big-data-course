# Projet 5A — Foundation of Big Data : Approche Small Data vs Approche Big Data (Hadoop / Spark)

Ce dépôt présente une étude comparative complète entre deux paradigmes de traitement et d'apprentissage automatique :
1. **Partie 1 — Approche Small Data (`Etude Small Data/`)** : Analyse exploratoire, modélisation supervisée (Classification & Régression) avec `pandas` / `scikit-learn` / `XGBoost` et déploiement d'une application interactive **Streamlit** sur le jeu de données **Air Quality & PM2.5** (`srisyra02/air-quality-dataset`, **500 observations**).
2. **Partie 2 — Approche Big Data (`Etude Big Data/`)** : Architecture distribuée **Hadoop / Apache Parquet / Apache Spark (PySpark SQL & MLlib)** appliquée à un jeu de données massif de **37 345 126 lignes** (~2,55 Go en CSV vs ~289 Mo en Parquet) retraçant l'historique *tick-by-tick* des mouvements de cotes de football entre 2018 et 2026 (`archangel0x01/football-odds-movements-2018-2026`).

---

## Architecture du Dépôt

```text
Etude/
├── README.md                                      # Documentation générale du projet (Small Data + Big Data)
├── .gitignore                                     # Exclusion des caches, logs Spark et fichiers CSV lourds
│
├── Etude Small Data/                              # PARTIE 1 : Approche Small Data (In-Memory / Scikit-Learn)
│   ├── part1_small_data_air_quality.ipynb         # Notebook complet exécuté (EDA, Prétraitement, ML, Évaluation)
│   ├── app.py                                     # Application web interactive Streamlit (Prédiction & Simulation)
│   └── models/                                    # Artefacts sérialisés (joblib)
│       ├── model.joblib                           # Meilleur classifieur (Random Forest Classifier)
│       ├── regressor.joblib                       # Meilleur régresseur (Random Forest Regressor)
│       ├── scaler.joblib                          # StandardScaler entraîné
│       ├── encoder.joblib                         # LabelEncoder des classes de qualité de l'air
│       └── metadata.joblib                        # Métriques, hyperparamètres et statistiques descriptives
│
└── Etude Big Data/                                # PARTIE 2 : Approche Big Data (Hadoop / Parquet / PySpark MLlib)
    ├── part2_big_data_spark.ipynb                 # Notebook PySpark complet exécuté (7 sections, 37.3M lignes)
    ├── README.md                                  # Documentation dédiée à l'architecture Big Data
    ├── .gitignore                                 # Règles Git spécifiques à Spark/Hadoop
    ├── models_spark/
    │   └── football_rf_model/                     # PipelineModel Spark ML distribué sauvegardé
    └── data_spark/
        └── football_features.parquet/             # Table de features consolidée au format Apache Parquet
```

---

## Synthèse Comparative : Small Data vs Big Data

| Dimension | Partie 1 — Approche Small Data | Partie 2 — Approche Big Data (Hadoop / Spark) |
| :--- | :--- | :--- |
| **Dataset Kaggle** | `srisyra02/air-quality-dataset` | `archangel0x01/football-odds-movements-2018-2026` |
| **Volumétrie** | **500 lignes** × 10 colonnes (~25 Ko) | **37 345 126 lignes** réparties sur 5 tables |
| **Format de stockage** | Fichier tabulaire CSV en mémoire vive | Stockage colonnaire **Apache Parquet** (**289,45 Mo** vs **2 550,47 Mo** en CSV, compression **×8,81**) |
| **Moteur de calcul** | Processus Python unique (`pandas`, `numpy`) | Exécution distribuée multi-partitions **Apache Spark 4.3.0** (`PySpark SQL`, Catalyst Optimizer, DAG) |
| **Ingénierie des variables** | Transformations vectorisées locales | Agrégations distribuées *MapReduce/Spark* sur 11,6M de ticks (`odds_1x2`) et 251k cotes (`match_closing`) + jointures distribuées |
| **Algorithmes ML** | `scikit-learn` & `XGBoost` (Classification & Régression) | `pyspark.ml` (`Pipeline`, `StringIndexer`, `VectorAssembler`, `StandardScaler`, `LogisticRegression`, `RandomForestClassifier`) |
| **Persistance & Déploiement** | `joblib` (`models/*.joblib`) + Dashboard **Streamlit** (`app.py`) | Format natif distribué `PipelineModel` (`models_spark/football_rf_model`) + Data Lake **Parquet** (`data_spark/football_features.parquet`) |

---

## Détail de la Partie 1 — Approche Small Data (Qualité de l'Air & PM2.5)

### 1. Problématique & Données
- **Source** : `srisyra02/air-quality-dataset` (500 relevés de capteurs environnementaux).
- **Variables explicatives (9 capteurs)** : `PM10`, `NOx`, `NO2`, `SO2`, `VOCs`, `CO`, `CO2`, `CH4`, `Temperature`.
- **Double formulation supervisée** :
  1. **Classification multiclasse (`Air_Quality_Class`)** : `Bon (Low)` ($\le 35\ \mu\text{g/m}^3$), `Modéré (Moderate)` ($35\text{–}60\ \mu\text{g/m}^3$), `Élevé (High)` ($> 60\ \mu\text{g/m}^3$).
  2. **Régression continue (`PM2.5`)** : Prédiction directe de la concentration en particules fines $\text{PM}_{2.5}$ ($\mu\text{g/m}^3$).

### 2. Résultats des Modèles (Small Data)

#### Classification (`Air_Quality_Class`)
| Modèle | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** *(Retenu)* | **73,00 %** | **71,33 %** | **73,00 %** | **70,71 %** |
| **XGBoost Classifier** | 72,00 % | 70,24 % | 72,00 % | 70,47 % |
| **Logistic Regression** | 72,00 % | 69,18 % | 72,00 % | 67,30 % |

#### Régression (`PM2.5`)
| Modèle | RMSE ($\mu\text{g/m}^3$) | MAE ($\mu\text{g/m}^3$) | Score $R^2$ |
| :--- | :---: | :---: | :---: |
| **Random Forest Regressor** *(Retenu)* | **12,25** | **9,70** | **0,7158** |
| **Linear Regression** | 13,08 | 10,10 | 0,6758 |
| **XGBoost Regressor** | 13,32 | 10,81 | 0,6641 |

---

## Détail de la Partie 2 — Approche Big Data (Hadoop / Spark — 37,3M de lignes)

### 1. Architecture & Volumétrie des 5 Tables Parquet
Le dataset `archangel0x01/football-odds-movements-2018-2026` comprend **37 345 126 lignes** :
- `matches.parquet` : **63 597 lignes** (métadonnées des rencontres, équipes, scores temps réglementaire et mi-temps, ligues, saisons).
- `match_closing.parquet` : **251 326 lignes** (cotes d'ouverture et de clôture 1X2, Asian Handicap et Over/Under par bookmaker).
- `odds_1x2.parquet` : **11 607 578 lignes** (historique *tick-by-tick* horodaté des variations de cotes 1X2 pré-match).
- `odds_ah.parquet` : **15 614 752 lignes** (variations *tick-by-tick* Asian Handicap).
- `odds_ou.parquet` : **9 807 873 lignes** (variations *tick-by-tick* Over/Under).

Grâce au format colonnaire **Apache Parquet**, l'empreinte disque passe de **2 550,47 Mo (~2,50 Go en CSV)** à **289,45 Mo en Parquet** (soit un facteur de compression **×8,81** et une réduction d'I/O de **88,7 %**), tout en activant le **Column Pruning** et le **Predicate Pushdown** dans le moteur **Catalyst** de Spark.

### 2. Feature Engineering Distribué & Garantie *Zero Data Leakage*
- **Agrégation distribuée de `odds_1x2` (11,6M de lignes $\rightarrow$ 1 ligne par `match_id`)** : Extraction du nombre de mouvements (`n_movements_1x2`), des écarts-types (`std_home_odds`, `std_draw_odds`, `std_away_odds`) et des amplitudes min-max (`range_home_odds`, `range_away_odds`).
- **Agrégation distribuée de `match_closing` (251k lignes $\rightarrow$ 1 ligne par `match_id`)** : Consensus inter-bookmakers sur les cotes d'ouverture et de clôture 1X2, Asian Handicap et Over/Under.
- **Variables financières créées en PySpark** : Probabilités implicites normalisées (`prob_home_open`, `prob_home_close`, etc.), marge des bookmakers (`overround_open`, `overround_close`), dérive des cotes (`drift_home`, `drift_draw`, `drift_away`), déplacement de probabilité (*Smart Money* `prob_shift_home`, `prob_shift_away`), dérive du handicap asiatique (`ah_line_drift`) et ratio de force (`odds_ratio_home_away`).
- **Aucune fuite de données (*Zero Data Leakage*)** : Seules des variables connues strictement **avant le coup d'envoi** sont injectées dans le `VectorAssembler` (37 features).

### 3. Résultats des Modèles Distribués (Spark MLlib — Test = 12 670 matchs)

| Modèle Distribué (Spark MLlib) | Temps d'entraînement | Accuracy | F1-Score (Weighted) | Precision (Weighted) | Recall (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier (100 arbres, maxDepth=8)** | 15,46 s | **52,38 %** | **44,61 %** | 46,05 % | **52,38 %** |
| **Logistic Regression Multinomiale (`regParam=0.01`)** | **14,48 s** | 52,23 % | 44,35 % | **50,10 %** | 52,23 % |

*(En prédiction de football 1X2 pré-match sans fuite de données, l'aléatoire est à 33,3 % et la classe majoritaire à 44,2 % ; nos modèles distribués atteignent ~52,4 % d'Accuracy en ~15 secondes sur l'ensemble du catalogue historique mondial).*

---

## Instructions d'Exécution

### Prérequis
- **Python** `3.12+` (Anaconda recommandé)
- **Java OpenJDK** `17+` (requis pour la JVM Apache Spark)
- Packages principaux : `pyspark>=4.0`, `pyarrow`, `kagglehub`, `scikit-learn`, `xgboost`, `pandas`, `matplotlib`, `seaborn`, `streamlit`, `joblib`.

### 1. Exécuter la Partie 1 (Small Data & Application Streamlit)
```bash
cd "Etude Small Data"
# Lancer le tableau de bord interactif Streamlit :
streamlit run app.py
```

### 2. Exécuter la Partie 2 (Big Data PySpark Notebook)
```bash
cd "Etude Big Data"
# Exécuter le notebook PySpark de bout en bout en ligne de commande :
jupyter nbconvert --to notebook --execute --inplace part2_big_data_spark.ipynb
```
