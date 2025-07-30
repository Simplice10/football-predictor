
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from difflib import get_close_matches
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="⚽ Prédiction Football", layout="centered")

# Chargement et nettoyage
@st.cache_data
def load_data():
    df = pd.read_csv("all_matches_combined.csv")
    return df.dropna(subset=['HomeTeam', 'AwayTeam', 'HTHG', 'HTAG', 'FTHG', 'FTAG'])

# Encodage des équipes
@st.cache_resource
def prepare_models(df):
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    df['HomeTeam_enc'] = le_home.fit_transform(df['HomeTeam'])
    df['AwayTeam_enc'] = le_away.fit_transform(df['AwayTeam'])
    X_base = df[['HomeTeam_enc', 'AwayTeam_enc', 'HTHG', 'HTAG']]

    models = {}

    # Score exact
    y_home = df['FTHG']
    y_away = df['FTAG']
    model_home = RandomForestRegressor().fit(X_base, y_home)
    model_away = RandomForestRegressor().fit(X_base, y_away)
    models['score_home'] = model_home
    models['score_away'] = model_away

    # HT/FT
    if 'HTR' in df.columns and 'FTR' in df.columns:
        df['HTFT'] = df['HTR'].astype(str) + '/' + df['FTR'].astype(str)
        y_htft = df['HTFT']
        model_htft = RandomForestClassifier().fit(X_base, y_htft)
        models['htft'] = model_htft

    # Corners
    if 'HC' in df.columns and 'AC' in df.columns:
        models['home_corners'] = RandomForestRegressor().fit(X_base, df['HC'])
        models['away_corners'] = RandomForestRegressor().fit(X_base, df['AC'])

    # Fautes
    if 'HF' in df.columns and 'AF' in df.columns:
        models['home_fouls'] = RandomForestRegressor().fit(X_base, df['HF'])
        models['away_fouls'] = RandomForestRegressor().fit(X_base, df['AF'])

    # Tirs
    if 'HS' in df.columns and 'AS' in df.columns:
        models['home_shots'] = RandomForestRegressor().fit(X_base, df['HS'])
        models['away_shots'] = RandomForestRegressor().fit(X_base, df['AS'])

    # Tirs cadrés
    if 'HST' in df.columns and 'AST' in df.columns:
        models['home_ontarget'] = RandomForestRegressor().fit(X_base, df['HST'])
        models['away_ontarget'] = RandomForestRegressor().fit(X_base, df['AST'])

    # Jaunes
    if 'HY' in df.columns and 'AY' in df.columns:
        models['home_yellow'] = RandomForestRegressor().fit(X_base, df['HY'])
        models['away_yellow'] = RandomForestRegressor().fit(X_base, df['AY'])

    return models, le_home, le_away, df['HomeTeam'].unique().tolist(), df['AwayTeam'].unique().tolist(), df

# Interface principale

def main():
    st.title("⚽ Application de Prédiction Football")

    df = load_data()
    models, le_home, le_away, home_teams, away_teams, full_df = prepare_models(df)

    st.markdown("**✍️ Remplis les informations suivantes pour prédire le match :**")

    col1, col2 = st.columns(2)
    with col1:
        home_team_input = st.text_input("🏠 Équipe à domicile")
    with col2:
        away_team_input = st.text_input("🚗 Équipe à l'extérieur")

    hthg = st.number_input("⏱️ Buts à la mi-temps (domicile)", 0, 10, 0)
    htag = st.number_input("⏱️ Buts à la mi-temps (extérieur)", 0, 10, 0)

    if st.button("🔍 Prédire"):
        try:
            home_team_match = get_close_matches(home_team_input, home_teams, n=1, cutoff=0.6)
            away_team_match = get_close_matches(away_team_input, away_teams, n=1, cutoff=0.6)

            if not home_team_match or not away_team_match:
                raise ValueError("Aucune correspondance trouvée pour les noms d'équipes saisis.")

            home_team = home_team_match[0]
            away_team = away_team_match[0]

            x_input = [[
                le_home.transform([home_team])[0],
                le_away.transform([away_team])[0],
                hthg,
                htag
            ]]

            st.subheader("🔮 Résultats de la prédiction")
            pred_home = round(models['score_home'].predict(x_input)[0])
            pred_away = round(models['score_away'].predict(x_input)[0])
            st.success(f"Score final prédit : {home_team} {pred_home} - {pred_away} {away_team}")

            if 'htft' in models:
                pred_htft = models['htft'].predict(x_input)[0]
                st.info(f"Prédiction HT/FT : {pred_htft}")

            if 'home_corners' in models:
                hc = round(models['home_corners'].predict(x_input)[0])
                ac = round(models['away_corners'].predict(x_input)[0])
                st.write(f"🏁 Corners : {home_team} {hc} - {ac} {away_team}")

            if 'home_fouls' in models:
                hf = round(models['home_fouls'].predict(x_input)[0])
                af = round(models['away_fouls'].predict(x_input)[0])
                st.write(f"❌ Fautes : {home_team} {hf} - {af} {away_team}")

            if 'home_shots' in models:
                hs = round(models['home_shots'].predict(x_input)[0])
                as_ = round(models['away_shots'].predict(x_input)[0])
                st.write(f"🎯 Tirs : {home_team} {hs} - {as_} {away_team}")

            if 'home_ontarget' in models:
                hst = round(models['home_ontarget'].predict(x_input)[0])
                ast = round(models['away_ontarget'].predict(x_input)[0])
                st.write(f"🎯 Tirs cadrés : {home_team} {hst} - {ast} {away_team}")

            if 'home_yellow' in models:
                hy = round(models['home_yellow'].predict(x_input)[0])
                ay = round(models['away_yellow'].predict(x_input)[0])
                st.write(f"🟨 Cartons jaunes : {home_team} {hy} - {ay} {away_team}")

            # Historique des confrontations
            head_to_head = full_df[(full_df['HomeTeam'] == home_team) & (full_df['AwayTeam'] == away_team)]
            if not head_to_head.empty:
                st.subheader("📊 Derniers affrontements")
                st.dataframe(head_to_head[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].sort_values(by='Date', ascending=False).head(5))

                # Moyennes statistiques
                st.subheader("📈 Statistiques moyennes sur les 5 derniers matchs")
                recent_matches = head_to_head.sort_values(by='Date', ascending=False).head(5)
                stats = {}
                for col in ['FTHG', 'FTAG', 'HC', 'AC', 'HF', 'AF', 'HS', 'AS', 'HST', 'AST', 'HY', 'AY']:
                    if col in recent_matches.columns:
                        stats[col] = recent_matches[col].mean()

                for stat, val in stats.items():
                    st.write(f"{stat} : {val:.2f}")

        except Exception as e:
            st.error(f"❗ Erreur : {e}")

if __name__ == '__main__':
    main()
