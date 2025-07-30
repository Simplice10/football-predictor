# ⚽ Football Predictor App

Cette application de prédiction de football permet de prédire :
- Le **score exact**
- Le **résultat à la mi-temps et à la fin du match (HT/FT)**
- Le **nombre de corners**
- Le **nombre de fautes**
- Les **tirs** et **tirs cadrés**
- Les **cartons jaunes**

Elle est basée sur un modèle d'apprentissage automatique utilisant `RandomForest` et les données historiques des matchs.

---

## 🚀 Démo en ligne

👉 *À venir via Streamlit Cloud ou autre*.

---

## 🛠️ Installation locale

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/Simplice10/football-predictor.git
   cd football-predictor
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Lancer l'application Streamlit**
   ```bash
   streamlit run football_predictor_app.py
   ```

---

## 📊 Données utilisées

Le fichier `all_matches_combined.csv` contient les données historiques avec les colonnes :
- Buts mi-temps (`HTHG`, `HTAG`)
- Score final (`FTHG`, `FTAG`)
- Corners (`HC`, `AC`)
- Fautes (`HF`, `AF`)
- Tirs (`HS`, `AS`) et tirs cadrés (`HST`, `AST`)
- Cartons jaunes (`HY`, `AY`)
- Résultats (`HTR`, `FTR`)

---

## 🧠 Modèles utilisés

L'application utilise plusieurs modèles `RandomForest` pour chaque type de prédiction, entraînés sur les données des matchs passés.

---

## ✍️ Auteur

- **Simplice10** - [GitHub](https://github.com/Simplice10)
