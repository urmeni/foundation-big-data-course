# Projet 5A — Foundation of Big Data : Approche Small Data vs Approche Big Data (Hadoop / Spark)

Ce dossier (`Etude Big Data/`) contient la **Partie 2 (Approche Big Data Hadoop / Apache Spark)** de l'étude comparative, articulée avec la **Partie 1 (Approche Small Data — Air Quality & PM2.5)** située dans `../Etude Small Data/`.

---

## Vue d'Ensemble du Projet (Partie 1 & Partie 2)

| Dimension | Partie 1 — Approche Small Data (`../Etude Small Data/`) | Partie 2 — Approche Big Data (`./`) |
| :--- | :--- | :--- |
| **Dataset Kaggle** | `srisyra02/air-quality-dataset` | `archangel0x01/football-odds-movements-2018-2026` |
| **Volumétrie** | **500 lignes** × 10 colonnes | **37 345 126 lignes** réparties sur 5 tables |
| **Stockage** | CSV en mémoire locale (~25 Ko) | **Apache Parquet** (**289,45 Mo** vs **2 550,47 Mo** en CSV, compression **×8,81**) |
| **Stack Technique** | `pandas`, `scikit-learn`, `XGBoost`, `Streamlit` | **Apache Spark 4.3.0** (`PySpark SQL`, Catalyst Optimizer, `Spark MLlib`) |
| **Meilleur Modèle** | Random Forest Classifier (**73,0 %** Accuracy) & Regressor ($R^2 = 0,716$) | Random Forest Spark MLlib (**52,38 %** Accuracy sur 3 issues 1X2, 12 670 matchs de test) |
| **Artefacts produits** | `models/*.joblib` + `app.py` (Streamlit) | `models_spark/football_rf_model` + `data_spark/football_features.parquet` |

---

## Contenu du Dossier `Etude Big Data/`

- `part2_big_data_spark.ipynb` : Notebook Jupyter PySpark intégralement exécuté structuré en **7 sections** :
  1. **Architecture Big Data & Data Ingestion (Hadoop/Spark)** : Téléchargement via `kagglehub`, benchmark de compression CSV (2,55 Go) vs Parquet (289 Mo), initialisation de la `SparkSession` (`4g` driver memory, 8 partitions de shuffle), lecture distribuée des **37 345 126 lignes** en 2,96 secondes et inspection du plan physique Catalyst (`explain()` : *Column Pruning* & *Predicate Pushdown*).
  2. **EDA Distribuée (Spark SQL & DataFrames)** : Statistiques descriptives distribuées, requêtes analytiques Spark SQL sur les 63 597 matchs et les 11,6M de mouvements `odds_1x2`, et génération de 3 figures analytiques (répartition 1X2 & buts, top 12 des ligues & volume de ticks par bookmaker, courbe de calibration empirique des probabilités implicites de clôture).
  3. **Prétraitement Distribué (PySpark)** : Déduplication, traitement des valeurs manquantes (`dropna`, `fillna`), filtrage des cotes valides ($> 1.0$) et construction de la cible multiclasse `target_label` (`Home Win`, `Draw`, `Away Win`).
  4. **Transformation & Feature Engineering (Spark ML Pipeline)** : Agrégation distribuée des 11,6M de lignes `odds_1x2` (volatilité *tick-by-tick*) et des 251k lignes `match_closing` au niveau `match_id`, création de 37 variables explicatives strictement pré-match (*Zero Data Leakage*) et assemblage via `StringIndexer`, `VectorAssembler` et `StandardScaler`.
  5. **ML Distribué (Spark MLlib)** : Split distribué 80/20 avec mise en cache RAM (`.cache()`) et entraînement parallèle de `LogisticRegression` (14,48 s) et `RandomForestClassifier` (15,46 s).
  6. **EVAL (Évaluation Spark MLlib)** : Évaluation multiclasse distribuée (`Accuracy`, `F1-Score`, `Weighted Precision`, `Weighted Recall`), matrice de confusion et graphique du Top 15 des `featureImportances` du Random Forest.
  7. **Déploiement Big Data & Préparation GitHub** : Sauvegarde du `PipelineModel` complet dans `models_spark/football_rf_model`, export Parquet dans `data_spark/football_features.parquet` et vérification de rechargement à froid.

---

## Reproduire l'Exécution

```bash
/home/urmenus/anaconda3/bin/jupyter nbconvert --to notebook --execute --inplace part2_big_data_spark.ipynb
```
