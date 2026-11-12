import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION DE LA PAGE
# ==============================================================================
st.set_page_config(
    page_title="Qualité de l'Air & PM2.5 — Approche Small Data",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CHARGEMENT DES ARTEFACTS (AVEC CACHE)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


@st.cache_resource
def load_artifacts():
    """Charge les modèles, le scaler, l'encodeur et les métadonnées depuis models/."""
    clf = joblib.load(os.path.join(MODELS_DIR, "model.joblib"))
    reg = joblib.load(os.path.join(MODELS_DIR, "regressor.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.joblib"))
    metadata = joblib.load(os.path.join(MODELS_DIR, "metadata.joblib"))
    return clf, reg, scaler, encoder, metadata


try:
    clf_model, reg_model, scaler, label_encoder, metadata = load_artifacts()
except Exception as e:
    st.error(f"Erreur lors du chargement des artefacts dans `{MODELS_DIR}` : {e}")
    st.stop()

FEATURE_NAMES = metadata["feature_names"]
FEATURE_STATS = metadata["feature_stats"]
CLASS_THRESHOLDS = metadata.get(
    "class_thresholds",
    {
        "Bon (Low)": "<= 35 µg/m³",
        "Modéré (Moderate)": "35 - 60 µg/m³",
        "Élevé (High)": "> 60 µg/m³",
    },
)

FEATURE_UNITS = {
    "PM10": "µg/m³",
    "NOx": "ppb",
    "NO2": "ppb",
    "SO2": "ppb",
    "VOCs": "ppb",
    "CO": "ppm",
    "CO2": "ppm",
    "CH4": "ppm",
    "Temperature": "°C",
}

FEATURE_DESCRIPTIONS = {
    "PM10": "Particules en suspension < 10 µm",
    "NOx": "Oxydes d'azote totaux",
    "NO2": "Dioxyde d'azote",
    "SO2": "Dioxyde de soufre",
    "VOCs": "Composés organiques volatils",
    "CO": "Monoxyde de carbone",
    "CO2": "Dioxyde de carbone",
    "CH4": "Méthane",
    "Temperature": "Température ambiante",
}

# ==============================================================================
# PROFILS PRÉ-CONFIGURÉS
# ==============================================================================
PRESET_PROFILES = {
    "Valeurs médianes du dataset": {
        f: float(FEATURE_STATS[f]["median"]) for f in FEATURE_NAMES
    },
    "Air pur (Pollution faible)": {
        "PM10": 48.25,
        "NOx": 117.53,
        "NO2": 41.82,
        "SO2": 5.86,
        "VOCs": 107.30,
        "CO": 2.43,
        "CO2": 435.19,
        "CH4": 1.82,
        "Temperature": 33.22,
    },
    "Pollution urbaine modérée": {
        "PM10": 69.67,
        "NOx": 37.02,
        "NO2": 14.69,
        "SO2": 9.31,
        "VOCs": 100.15,
        "CO": 1.32,
        "CO2": 400.59,
        "CH4": 2.04,
        "Temperature": 29.96,
    },
    "Pic de pollution sévère": {
        "PM10": 114.61,
        "NOx": 126.11,
        "NO2": 72.52,
        "SO2": 27.91,
        "VOCs": 140.71,
        "CO": 2.98,
        "CO2": 529.13,
        "CH4": 2.41,
        "Temperature": 34.02,
    },
    "Saisie personnalisée": None,
}

# ==============================================================================
# BARRE LATÉRALE : SÉLECTION DU PROFIL & SLIDERS DES 9 CAPTEURS
# ==============================================================================
st.sidebar.title("🎛️ Paramètres des Capteurs")
st.sidebar.markdown(
    "Sélectionnez un profil type ou ajustez manuellement les 9 capteurs "
    "atmosphériques pour simuler une mesure en temps réel."
)

selected_profile = st.sidebar.selectbox(
    "Profil pré-configuré :",
    options=list(PRESET_PROFILES.keys()),
    index=0,
    help="Charge automatiquement un jeu de valeurs réalistes issu du dataset.",
)

# Initialisation / mise à jour du session_state lors d'un changement de profil
if "last_profile" not in st.session_state:
    st.session_state["last_profile"] = selected_profile
    for f in FEATURE_NAMES:
        st.session_state[f"slider_{f}"] = float(
            PRESET_PROFILES["Valeurs médianes du dataset"][f]
        )

if selected_profile != st.session_state["last_profile"]:
    st.session_state["last_profile"] = selected_profile
    if PRESET_PROFILES[selected_profile] is not None:
        for f in FEATURE_NAMES:
            val = float(PRESET_PROFILES[selected_profile][f])
            f_min = float(FEATURE_STATS[f]["min"])
            f_max = float(FEATURE_STATS[f]["max"])
            st.session_state[f"slider_{f}"] = min(max(val, f_min), f_max)

st.sidebar.markdown("---")
st.sidebar.subheader("📏 Valeurs des 9 capteurs")

user_inputs = {}
for f in FEATURE_NAMES:
    f_min = float(FEATURE_STATS[f]["min"])
    f_max = float(FEATURE_STATS[f]["max"])
    f_med = float(FEATURE_STATS[f]["median"])
    unit = FEATURE_UNITS.get(f, "")
    desc = FEATURE_DESCRIPTIONS.get(f, "")
    step = round((f_max - f_min) / 200.0, 4)
    if step <= 0:
        step = 0.01

    default_val = st.session_state.get(f"slider_{f}", f_med)
    default_val = min(max(float(default_val), f_min), f_max)

    user_inputs[f] = st.sidebar.slider(
        label=f"{f} ({unit})",
        min_value=f_min,
        max_value=f_max,
        value=default_val,
        step=step,
        key=f"slider_{f}",
        help=f"{desc} | Min: {f_min:.2f}, Médiane: {f_med:.2f}, Max: {f_max:.2f} {unit}",
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    f"📊 **Dataset** : `{metadata.get('dataset_name', 'srisyra02/air-quality-dataset')}` "
    f"({metadata.get('n_samples', 500)} échantillons, {len(FEATURE_NAMES)} capteurs)"
)

# ==============================================================================
# CALCUL DES PRÉDICTIONS EN DIRECT
# ==============================================================================
input_df = pd.DataFrame([user_inputs])[FEATURE_NAMES]
scaled_array = scaler.transform(input_df)
scaled_df = pd.DataFrame(scaled_array, columns=FEATURE_NAMES)

# Classification (Air_Quality_Class)
pred_class_idx = clf_model.predict(scaled_df)[0]
pred_class_label = label_encoder.inverse_transform([pred_class_idx])[0]
pred_probas = clf_model.predict_proba(scaled_df)[0]
class_labels = list(label_encoder.classes_)
proba_dict = {cls: float(prob) for cls, prob in zip(class_labels, pred_probas)}

# Régression (PM2.5)
pred_pm25 = float(reg_model.predict(scaled_df)[0])

# Configuration visuelle selon la classe prédite
STYLE_MAP = {
    "Bon (Low)": {
        "color": "#1e8449",
        "bg": "#d4efdf",
        "border": "#27ae60",
        "icon": "🟢",
        "rec": "Qualité de l'air satisfaisante. Aucune restriction d'activité extérieure.",
    },
    "Modéré (Moderate)": {
        "color": "#b9770e",
        "bg": "#fcf3cf",
        "border": "#f39c12",
        "icon": "🟠",
        "rec": "Qualité de l'air modérée. Les personnes sensibles devraient limiter les efforts prolongés.",
    },
    "Élevé (High)": {
        "color": "#922b21",
        "bg": "#fadbd8",
        "border": "#e74c3c",
        "icon": "🔴",
        "rec": "Pollution élevée ! Évitez les activités physiques intenses en extérieur.",
    },
}
current_style = STYLE_MAP.get(
    pred_class_label,
    {
        "color": "#2c3e50",
        "bg": "#ebedef",
        "border": "#7f8c8d",
        "icon": "⚪",
        "rec": "",
    },
)

# ==============================================================================
# PANNEAU PRINCIPAL : EN-TÊTE & PRÉDICTIONS EN DIRECT
# ==============================================================================
st.title("🌿 Surveillance & Prédiction de la Qualité de l'Air (Approche Small Data)")
st.markdown(
    "Application interactive d'inférence en temps réel basée sur les modèles **Random Forest** "
    "(Classification de la qualité de l'air & Régression de la concentration en **PM2.5**) "
    "entraînés sur le jeu de données `srisyra02/air-quality-dataset`."
)

st.markdown("### 🎯 Résultats de la Prédiction en Direct")

col_cls, col_reg, col_prob = st.columns([1.25, 1.0, 1.25])

with col_cls:
    threshold_info = CLASS_THRESHOLDS.get(pred_class_label, "")
    st.markdown(
        f"""
        <div style="
            background-color: {current_style['bg']};
            border-left: 8px solid {current_style['border']};
            padding: 18px 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        ">
            <div style="font-size: 0.9rem; font-weight: 600; color: #555; text-transform: uppercase;">
                Classe de Qualité de l'Air Prédite
            </div>
            <div style="font-size: 1.75rem; font-weight: 800; color: {current_style['color']}; margin: 8px 0;">
                {current_style['icon']} {pred_class_label}
            </div>
            <div style="font-size: 0.88rem; color: #333; margin-bottom: 6px;">
                <strong>Seuil de référence PM2.5 :</strong> {threshold_info}
            </div>
            <div style="font-size: 0.85rem; color: #444;">
                {current_style['rec']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_reg:
    # Détermination de la catégorie théorique du PM2.5 estimé
    if pred_pm25 <= 35.0:
        pm25_cat = "Zone Basse (≤ 35 µg/m³)"
        pm25_color = "#27ae60"
    elif pred_pm25 <= 60.0:
        pm25_cat = "Zone Modérée (35 - 60 µg/m³)"
        pm25_color = "#f39c12"
    else:
        pm25_cat = "Zone Élevée (> 60 µg/m³)"
        pm25_color = "#e74c3c"

    st.markdown(
        f"""
        <div style="
            background-color: #f8f9f9;
            border-left: 8px solid {pm25_color};
            padding: 18px 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        ">
            <div style="font-size: 0.9rem; font-weight: 600; color: #555; text-transform: uppercase;">
                Concentration PM2.5 Estimée
            </div>
            <div style="font-size: 1.95rem; font-weight: 800; color: {pm25_color}; margin: 8px 0;">
                {pred_pm25:.2f} µg/m³
            </div>
            <div style="font-size: 0.88rem; color: #333; margin-bottom: 6px;">
                <strong>Plage :</strong> {pm25_cat}
            </div>
            <div style="font-size: 0.82rem; color: #666;">
                Modèle : <code>{metadata.get('best_regressor_name', 'Random Forest Regressor')}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_prob:
    st.markdown("**📊 Probabilités par classe (`predict_proba`)**")
    bar_colors = {
        "Bon (Low)": "#2ecc71",
        "Modéré (Moderate)": "#f39c12",
        "Élevé (High)": "#e74c3c",
    }
    for cls_name in ["Bon (Low)", "Modéré (Moderate)", "Élevé (High)"]:
        prob_pct = proba_dict.get(cls_name, 0.0) * 100.0
        c_hex = bar_colors.get(cls_name, "#3498db")
        st.markdown(
            f"""
            <div style="margin-bottom: 8px;">
                <div style="display:flex; justify-content:space-between; font-size:0.86rem; font-weight:600;">
                    <span>{cls_name}</span>
                    <span>{prob_pct:.1f} %</span>
                </div>
                <div style="width:100%; background-color:#e5e8e8; border-radius:6px; height:12px; margin-top:3px;">
                    <div style="width:{prob_pct:.1f}%; background-color:{c_hex}; height:12px; border-radius:6px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# ==============================================================================
# ONGLETS D'ANALYSE : CAPTEURS & STANDARDISATION | PERFORMANCES | IMPORTANCE
# ==============================================================================
tab_sensors, tab_perf, tab_fi = st.tabs(
    [
        "📋 Valeurs Saisies & Standardisation (StandardScaler)",
        "🏆 Performances Comparatives des Modèles",
        "🔍 Importance des Variables (Feature Importance)",
    ]
)

# ------------------------------------------------------------------------------
# ONGLET 1 : RÉSUMÉ DES CAPTEURS ET VALEURS STANDARDISÉES
# ------------------------------------------------------------------------------
with tab_sensors:
    st.subheader("Résumé des Capteurs et Transformation `StandardScaler`")
    st.markdown(
        "Les modèles utilisent les variables centrées-réduites via le `StandardScaler` "
        "ajusté sur l'ensemble d'entraînement ($Z = \\frac{X - \\mu}{\\sigma}$). "
        "Un score $Z > 0$ indique une valeur supérieure à la moyenne du dataset."
    )

    summary_rows = []
    for f in FEATURE_NAMES:
        raw_val = float(user_inputs[f])
        z_val = float(scaled_df[f].iloc[0])
        stats_f = FEATURE_STATS[f]
        summary_rows.append(
            {
                "Capteur": f,
                "Description": FEATURE_DESCRIPTIONS.get(f, ""),
                "Unité": FEATURE_UNITS.get(f, ""),
                "Valeur Saisie": round(raw_val, 3),
                "Valeur Standardisée (Z-score)": round(z_val, 3),
                "Min Dataset": round(float(stats_f["min"]), 2),
                "Médiane Dataset": round(float(stats_f["median"]), 2),
                "Moyenne Dataset": round(float(stats_f["mean"]), 2),
                "Max Dataset": round(float(stats_f["max"]), 2),
            }
        )
    summary_table = pd.DataFrame(summary_rows)

    col_tbl, col_zchart = st.columns([1.35, 1.0])
    with col_tbl:
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

    with col_zchart:
        fig_z, ax_z = plt.subplots(figsize=(6, 4.2))
        z_vals = summary_table["Valeur Standardisée (Z-score)"].values
        sensors = summary_table["Capteur"].values
        colors_z = ["#e74c3c" if z > 1.0 else ("#2ecc71" if z < -0.5 else "#3498db") for z in z_vals]
        y_pos = np.arange(len(sensors))
        ax_z.barh(y_pos, z_vals, color=colors_z, edgecolor="black", alpha=0.85)
        ax_z.axvline(0, color="black", linewidth=1, linestyle="--")
        ax_z.set_yticks(y_pos)
        ax_z.set_yticklabels(sensors)
        ax_z.invert_yaxis()
        ax_z.set_xlabel("Score standardisé (Z-score)")
        ax_z.set_title("Écart par rapport à la moyenne du dataset (en σ)")
        ax_z.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_z)
        plt.close(fig_z)

# ------------------------------------------------------------------------------
# ONGLET 2 : PERFORMANCES COMPARATIVES DES MODÈLES ENTRAÎNÉS
# ------------------------------------------------------------------------------
with tab_perf:
    st.subheader("Comparaison des Modèles Entraînés (Validation Test 20%)")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("#### 🎯 Classification (`Air_Quality_Class`)")
        clf_metrics_df = pd.DataFrame(metadata["classification_metrics"])
        st.dataframe(
            clf_metrics_df.style.format(
                {
                    "Accuracy": "{:.2%}",
                    "Precision (weighted)": "{:.2%}",
                    "Recall (weighted)": "{:.2%}",
                    "F1-Score (weighted)": "{:.2%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.success(
            f"🏆 **Meilleur classifieur retenu** : `{metadata['best_classifier_name']}` "
            f"(Accuracy = {clf_metrics_df.iloc[0]['Accuracy']:.1%}, "
            f"F1-pondéré = {clf_metrics_df.iloc[0]['F1-Score (weighted)']:.1%})"
        )

        fig_c, ax_c = plt.subplots(figsize=(5.5, 3.4))
        x_idx = np.arange(len(clf_metrics_df))
        width = 0.35
        ax_c.bar(
            x_idx - width / 2,
            clf_metrics_df["Accuracy"],
            width,
            label="Accuracy",
            color="#2980b9",
        )
        ax_c.bar(
            x_idx + width / 2,
            clf_metrics_df["F1-Score (weighted)"],
            width,
            label="F1-Score (pondéré)",
            color="#27ae60",
        )
        ax_c.set_xticks(x_idx)
        ax_c.set_xticklabels(clf_metrics_df["Modèle"])
        ax_c.set_ylim(0.5, 0.85)
        ax_c.set_ylabel("Score")
        ax_c.set_title("Performances en Classification")
        ax_c.legend()
        ax_c.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_c)
        plt.close(fig_c)

    with col_p2:
        st.markdown("#### 📈 Régression (`PM2.5` en µg/m³)")
        reg_metrics_df = pd.DataFrame(metadata["regression_metrics"])
        st.dataframe(
            reg_metrics_df.style.format(
                {
                    "MAE (µg/m³)": "{:.2f}",
                    "RMSE (µg/m³)": "{:.2f}",
                    "R² Score": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.success(
            f"🏆 **Meilleur régresseur retenu** : `{metadata['best_regressor_name']}` "
            f"(R² = {reg_metrics_df.iloc[0]['R² Score']:.3f}, "
            f"MAE = {reg_metrics_df.iloc[0]['MAE (µg/m³)']:.2f} µg/m³)"
        )

        fig_r, ax_r = plt.subplots(figsize=(5.5, 3.4))
        x_idx_r = np.arange(len(reg_metrics_df))
        ax_r.bar(
            x_idx_r - width / 2,
            reg_metrics_df["MAE (µg/m³)"],
            width,
            label="MAE (µg/m³)",
            color="#e67e22",
        )
        ax_r.bar(
            x_idx_r + width / 2,
            reg_metrics_df["RMSE (µg/m³)"],
            width,
            label="RMSE (µg/m³)",
            color="#c0392b",
        )
        ax_r.set_xticks(x_idx_r)
        ax_r.set_xticklabels(reg_metrics_df["Modèle"], rotation=10)
        ax_r.set_ylabel("Erreur (µg/m³)")
        ax_r.set_title("Erreurs de Régression PM2.5 (plus bas = meilleur)")
        ax_r.legend()
        ax_r.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_r)
        plt.close(fig_r)

# ------------------------------------------------------------------------------
# ONGLET 3 : IMPORTANCE DES VARIABLES (FEATURE IMPORTANCE)
# ------------------------------------------------------------------------------
with tab_fi:
    st.subheader("Importance des Variables (Feature Importance)")
    st.markdown(
        "Contribution relative de chacun des 9 capteurs dans les décisions du "
        "**RandomForestClassifier** (classe de qualité de l'air) et du "
        "**RandomForestRegressor** (estimation continue de `PM2.5`)."
    )

    fi_clf = pd.Series(clf_model.feature_importances_, index=FEATURE_NAMES).sort_values(
        ascending=True
    )
    fi_reg = pd.Series(reg_model.feature_importances_, index=FEATURE_NAMES).sort_values(
        ascending=True
    )

    col_fi1, col_fi2 = st.columns(2)

    with col_fi1:
        st.markdown("#### 🌲 Classification (`RandomForestClassifier`)")
        fig_fi1, ax_fi1 = plt.subplots(figsize=(5.5, 4.0))
        ax_fi1.barh(
            fi_clf.index,
            fi_clf.values * 100.0,
            color="#2ecc71",
            edgecolor="black",
            alpha=0.85,
        )
        for i, v in enumerate(fi_clf.values * 100.0):
            ax_fi1.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
        ax_fi1.set_xlim(0, max(fi_clf.values * 100.0) * 1.18)
        ax_fi1.set_xlabel("Importance relative (%)")
        ax_fi1.set_title("Importance des capteurs — Classification")
        ax_fi1.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_fi1)
        plt.close(fig_fi1)

    with col_fi2:
        st.markdown("#### 🌲 Régression PM2.5 (`RandomForestRegressor`)")
        fig_fi2, ax_fi2 = plt.subplots(figsize=(5.5, 4.0))
        ax_fi2.barh(
            fi_reg.index,
            fi_reg.values * 100.0,
            color="#3498db",
            edgecolor="black",
            alpha=0.85,
        )
        for i, v in enumerate(fi_reg.values * 100.0):
            ax_fi2.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=9)
        ax_fi2.set_xlim(0, max(fi_reg.values * 100.0) * 1.18)
        ax_fi2.set_xlabel("Importance relative (%)")
        ax_fi2.set_title("Importance des capteurs — Régression PM2.5")
        ax_fi2.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_fi2)
        plt.close(fig_fi2)

    fi_table = pd.DataFrame(
        {
            "Capteur": FEATURE_NAMES,
            "Description": [FEATURE_DESCRIPTIONS[f] for f in FEATURE_NAMES],
            "Importance Classification (%)": [
                round(float(c) * 100, 2) for c in clf_model.feature_importances_
            ],
            "Importance Régression PM2.5 (%)": [
                round(float(r) * 100, 2) for r in reg_model.feature_importances_
            ],
        }
    ).sort_values(by="Importance Classification (%)", ascending=False)
    st.dataframe(fi_table, use_container_width=True, hide_index=True)
    st.info(
        "💡 **Analyse** : Le capteur **CO2** est le prédicteur dominant à la fois pour la "
        "régression du PM2.5 (~71.9 %) et la classification (~20.2 %), suivi par **SO2** (15.8 %), "
        "**PM10** (12.5 %) et **NO2** (12.0 %) pour la discrimination des classes de pollution."
    )
