import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import plotly.express as px
from xgboost import XGBClassifier

# ────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────
DATA_PATH = 'cleaned_data.csv'          
st.set_page_config(
    page_title="Appendicitis Prediction Tool",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Appendicitis Prediction Tool")
st.markdown("""
🩺 Appendicitis Prediction Assistant  
Enter patient details to get predictions for:  
• **Diagnosis** (e.g. appendicitis, no appendicitis etc.)  
• **Severity** (uncomplicated, complicated)  
• **Management** (conservative, primary surgical, secondary surgical etc.)  

The predictions are powered by three XGBoost models trained on real clinical data with SMOTE oversampling for better performance on rare classes.
""")

# ────────────────────────────────────────────────────────────────
# TRAIN / LOAD ALL THREE MODELS (cached)
# ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading or training models…")
def get_all_models():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"File not found: {DATA_PATH}\nPlease place cleaned_data.csv in the app folder.")
        st.stop()

    # Common feature preparation
    X_raw = df.drop(columns=["Management", "Diagnosis", "Severity"])
    X = pd.get_dummies(X_raw, drop_first=False)
    feature_names = X.columns.tolist()

    models = {}
    scalers = {}
    encoders = {}

    for target_col in ["Diagnosis", "Severity", "Management"]:
        y_raw = df[target_col]
        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        # Remove rare classes (<2 samples)
        counts = pd.Series(y).value_counts()
        rare = counts[counts < 2].index
        if len(rare) > 0:
            mask = ~pd.Series(y).isin(rare)
            X_target = X[mask].reset_index(drop=True)
            y = y[mask]
        else:
            X_target = X.copy()

        # Split
        X_train, _, y_train, _ = train_test_split(
            X_target, y, test_size=0.2, random_state=42, stratify=y
        )

        # SMOTE
        min_count = pd.Series(y_train).value_counts().min()
        k = min(5, max(1, min_count - 1)) if min_count > 1 else 1
        smote = SMOTE(k_neighbors=k, random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_res)

        # Model
        model = XGBClassifier(
            eval_metric="mlogloss",
            random_state=42,
            n_estimators=250,
            max_depth=6,
            learning_rate=0.1,
            colsample_bytree=0.8
        )
        model.fit(X_train_scaled, y_res)

        models[target_col] = model
        scalers[target_col] = scaler
        encoders[target_col] = le

    return models, scalers, encoders, feature_names


# Load everything once
models, scalers, le_dict, feature_names = get_all_models()


# ────────────────────────────────────────────────────────────────
# SIDEBAR – PATIENT INPUT
# ────────────────────────────────────────────────────────────────
def user_input_features():
    st.sidebar.header("Patient Clinical Features")

    # --- Demographic ---
    with st.sidebar.subheader("Demographics"):
        Age = st.sidebar.slider("Age", 1, 100, 12)
        BMI = st.sidebar.number_input("BMI", min_value=7.00, max_value=50.0, value=18.0, step=0.1)
        Sex = st.sidebar.selectbox("Sex", options=["male", "female"])
        Height = st.sidebar.number_input("Height", min_value=50, max_value=200, value=50)
        Weight = st.sidebar.number_input("Weight", min_value=2, max_value=200, value=18)


    # --- Symptoms ---
    with st.sidebar.subheader("Symptoms"):
        Migratory_Pain = st.sidebar.selectbox("Migratory_Pain", ["yes", "no","unknown"])
        Lower_Right_And_Pain = st.sidebar.selectbox("Lower Right Abdominal Pain", ["yes", "no","unknown"])
        Loss_of_Appetite = st.sidebar.selectbox("Loss of Appetite", ["yes", "no","unknown"])
        Nausea = st.sidebar.selectbox("Nausea", ["yes", "no","unknown"])
        Coughing_Pain = st.sidebar.selectbox("Coughing", ["yes", "no","unknown"])
        Contralateral_Rebound_Tenderness = st.sidebar.selectbox("C ontralateralRebound Tenderness", ["yes", "no","unknown"])
        Psoas_Sign = st.sidebar.selectbox("Psoas Sign", ["yes", "no","unknown"])
        Ipsilateral_Rebound_Tenderness = st.sidebar.selectbox("Ipsilateral Rebound Tenderness", ["yes", "no","unknown"])
        Diagnosis_Presumptive = st.sidebar.selectbox("Diagnosis_Presumptive", ["Appendicitis", "Appendicitis With Mesenteric Lymph Node Inflammation","No Appendicitis","Gastroenteritis","Chronic Appendicitis",
                                                                           "Ovarian Torsion","Prolonged Gastroenteritis","Diabetic Ketoacidosis With Myocarditis","Chronic Abdominal Pain","Sepsis With Accompanying Appendicitis",
                                                                           "Adnexal Torsion","Abdominal Adhesions With Partial Bowel Obstruction","Adhesions Of The Ascending Colon","Adhesive Ileus","Perforated Appendicitis"])

    # --- Clinical Signs ---
    with st.sidebar.subheader("Clinical & Lab Findings"):
        Body_Temperature = st.sidebar.slider("Body Temperature", 20.0, 41.0, 37.0, step=0.1)
        RBC_Count = st.sidebar.number_input("RBC Count (×10⁹/L)", 2.0, 40.0, 12.0, step=0.1)
        Neutrophil_Percentage = st.sidebar.slider("Neutrophil %", 20.0, 100.0, 50.0, step=0.5)
        Neutrophilia = st.sidebar.selectbox("Neutrophilia", ["yes", "no","unknown"])
        CRP = st.sidebar.number_input("CRP (mg/L)", 0, 500, 10, step=10)
        Alvarado_Score = st.sidebar.number_input("Alvarado_Score", min_value=0, max_value=10, value=1)
        Paedriatic_Appendicitis_Score = st.sidebar.number_input("Paedriatic_Appendicitis_Score", min_value=0, max_value=10, value=1)
        Peritonitis = st.sidebar.selectbox("Peritonitis", ["generalized","local", "no","unknown"])


    # --- Ultrasound Findings ---
    with st.sidebar.subheader("Ultrasound Findings"):
        US_Performed = st.sidebar.selectbox("Was Ultrasound Performed?", ["yes", "no","unknown"])
        Appendix_on_US = st.sidebar.selectbox("Appendix Visualized on US", ["yes", "no","unknown"])
        Appendix_Diameter = st.sidebar.number_input("Appendix Diameter (mm)", 0.0, 20.0, 7.5, step=0.1)
        Free_Fluids = st.sidebar.selectbox("Free Fluid in Abdomen", ["yes", "no","unknown"])
        Appendicolith = st.sidebar.selectbox("Appendicolith (Fecalith)", ["yes", "no","unknown"])
        Target_Sign = st.sidebar.selectbox("Target Sign on US", ["yes", "no"])
        US_Number = st.sidebar.number_input("US_Number", 1, 1000, 20)
        Appendix_Wall_Layers = st.sidebar.selectbox("Appendix_Wall_Layers", ["intact", "raised","unknown","partially raised","upset"])
        Perfusion = st.sidebar.selectbox("Perfusion", ["hyperperfused", "hypoperfused", "unknown","no","present"])
        Perforation = st.sidebar.selectbox("Perforation Signs", ["yes", "no", "not excluded", "unknown","suspected"])
        Surrounding_Tissue_Reaction = st.sidebar.selectbox("Surrounding Tissue Reaction", ["yes", "no", "unknown"])
        Appendicular_Abscess = st.sidebar.selectbox("Appendicular Abscess", ["yes", "no", "unknown","suscepted"])
        Abscess_Location = st.sidebar.selectbox("Abscess Location",["Unknown","Pelvic Cavity","Behind The Bladder","Right Lower Abdomen",
                                                                 "Around The Cecum","Right Psoas Muscle Region","Right Mid-Abdomen"])
        Pathological_Lymph_Nodes = st.sidebar.selectbox("Pathological Lymph Nodes", ["yes", "no", "unknown"])
        Lymph_Nodes_Location = st.sidebar.selectbox("Lymph Nodes Location", ["Right lower abdomen","unknown","Ileocecal","Lower abdomen","Periumbilical","Mesenteric and right lower abdomen",
                                                                          "Mesenteric","Right lower abdomen and periumbilical","re UB","Right lower abdomen and ileocecal","Right lower and middle abdomen",
                                                                          "Middle abdomen","Right middle abdomen","Inguinal","Periappendiceal","Lymphadenopathy","Mesenteric and left inguinal","Multiple locations","Around the appendix","Ovarian cysts"])
        Bowel_Wall_Thickening = st.sidebar.selectbox("Bowel Wall Thickening", ["yes", "no", "unknown"])
        Conglomerate_of_Bowel_Loops = st.sidebar.selectbox("Conglomerate of Bowel Loops", ["yes", "no", "unknown"])
        Ileus = st.sidebar.selectbox("Ileus", ["yes", "no", "unknown"])
        Coprostasis = st.sidebar.selectbox("Coproostasis", ["yes", "no", "unknown"])
        Meteorism = st.sidebar.selectbox("Meteorism", ["yes", "no", "unknown"])
        Enteritis = st.sidebar.selectbox("Enteritis", ["yes", "no", "unknown"])
        Gynecological_Findings = st.sidebar.selectbox("Gynecological Findings", ["unknown","Ovarian cyst","Uterine cyst","Bilateral ovarian cysts with abnormal right ovary perfusion",
                                                                              "Normal ovaries","Right ovarian cyst","No gynecological cause","Suspected ovarian torsion","Yes",
                                                                            "Ovarian cysts","Normal finding"])

    # --- Other ---
    with st.sidebar.subheader("Other"):
        Lenght_of_Stay = st.sidebar.number_input("Lenght_of_Stay", min_value=1, max_value=30, value=18, step=10)
        Dysuria = st.sidebar.selectbox("Dysuria", ["yes", "no","unknown"])
        Stool = st.sidebar.selectbox("Stool Pattern", ["normal", "diarrhea", "constipation","constipationand diarrhea","unknown"])
        Hemoglobin = st.sidebar.number_input("Hemoglobin", 1.0, 40.0, 7.5, step=0.1)
        RDW  = st.sidebar.number_input("RDW", 10.0, 100.0, 20.0, step=0.1)
        Thrombocyte_Count =  st.sidebar.number_input("Thrombocyte_Count", 90, 800, 100)
        Ketones_in_Urine = st.sidebar.selectbox("Ketones_in_Urine", ["+", "++", "+++","no","unknown"])
        RBC_in_Urine = st.sidebar.selectbox("RBC_in_Urine", ["+", "++", "+++","no","unknown"])
        WBC_in_Urine = st.sidebar.selectbox("WBC_in_Urine", ["+", "++", "+++","no","unknown"])
   

    # Create dictionary
    data = {
        "Age": Age,
        "BMI": BMI,
        "Sex": Sex,
        "Height": Height,
        "Weight": Weight,
        "Migratory_Pain": Migratory_Pain,
        "Lower_Right_And_Pain": Lower_Right_And_Pain,
        "Contralateral_Rebound_Tenderness": Contralateral_Rebound_Tenderness,
        "Coughing_Pain": Coughing_Pain,
        "Nausea": Nausea,
        "Loss_of_Appetite": Loss_of_Appetite,
        "Body_Temperature": Body_Temperature,
        "Neutrophil_Percentage": Neutrophil_Percentage,
        "Neutrophilia": Neutrophilia,
        "RBC_Count": RBC_Count,
        "Hemoglobin": Hemoglobin,
        "RDW": RDW,
        "Thrombocyte_Count": Thrombocyte_Count,
        "Ketones_in_Urine": Ketones_in_Urine,
        "RBC_in_Urine": RBC_in_Urine,
        "WBC_in_Urine": WBC_in_Urine,
        "CRP": CRP,
        "Dysuria": Dysuria,
        "Stool": Stool,
        "Peritonitis": Peritonitis,
        "Psoas_Sign": Psoas_Sign,
        "Ipsilateral_Rebound_Tenderness": Ipsilateral_Rebound_Tenderness,
        "US_Performed": US_Performed,
        "US_Number": US_Number,
        "Appendix_on_US": Appendix_on_US,
        "Appendix_Diameter": Appendix_Diameter,
        "Free_Fluids": Free_Fluids,
        "Appendix_Wall_Layers": Appendix_Wall_Layers,
        "Target_Sign": Target_Sign,
        "Appendicolith": Appendicolith,
        "Perfusion": Perfusion,
        "Perforation": Perforation,
        "Surrounding_Tissue_Reaction": Surrounding_Tissue_Reaction,
        "Appendicular_Abscess": Appendicular_Abscess,
        "Abscess_Location": Abscess_Location,
        "Pathological_Lymph_Nodes": Pathological_Lymph_Nodes,
        "Lymph_Nodes_Location": Lymph_Nodes_Location,
        "Bowel_Wall_Thickening": Bowel_Wall_Thickening,
        "Conglomerate_of_Bowel_Loops": Conglomerate_of_Bowel_Loops,
        "Ileus": Ileus,
        "Coprostasis": Coprostasis,
        "Meteorism": Meteorism,
        "Enteritis": Enteritis,
        "Gynecological_Findings": Gynecological_Findings}
    return pd.DataFrame([data])

input_df = user_input_features()
input_encoded = pd.get_dummies(input_df, drop_first=False)
input_aligned = input_encoded.reindex(columns=feature_names, fill_value=0)


# ────────────────────────────────────────────────────────────────
# PREDICTION
# ────────────────────────────────────────────────────────────────
if st.button("Generate Predictions", type="primary", use_container_width=True):
    with st.spinner("Analyzing…"):
        results = {}

        for target in ["Diagnosis", "Severity", "Management"]:
            X_scaled = scalers[target].transform(input_aligned)
            model = models[target]
            le = le_dict[target]

            pred_idx = model.predict(X_scaled)[0]
            pred_label = le.inverse_transform([pred_idx])[0]
            probs = model.predict_proba(X_scaled)[0]
            confidence = probs.max()

            results[target] = {
                "label": pred_label,
                "confidence": confidence,
                "top_classes": sorted(zip(le.classes_, probs),key=lambda x: x[1], reverse=True)[:5],
                "prob_dict": dict(zip(le.classes_, probs))
                                      
            }

    # ── RESULTS LAYOUT ──────────────────────────────────────────────
    st.success("Predictions ready", icon="✅")
    st.divider()

    col_diag, col_sev, col_mgmt = st.columns(3)

    with col_diag:
        r = results["Diagnosis"]
        st.subheader("Diagnosis")
        st.metric(r["label"], f"{r['confidence']:.1%}")
       

    with col_sev:
        r = results["Severity"]
        st.subheader("Severity")
        st.metric(r["label"], f"{r['confidence']:.1%}")
        
            

    with col_mgmt:
        r = results["Management"]
        st.subheader("Management")
        st.metric(r["label"], f"{r['confidence']:.1%}")

    # ── DETAILED PROBABILITIES ──────────────────────────────────────
   



st.divider()
with st.expander("Detailed probabilities – all classes", expanded=False):
        tabs = st.tabs(["Diagnosis", "Severity", "Management"])

        for i, target in enumerate(["Diagnosis", "Severity", "Management"]):
            with tabs[i]:
                r = results[target]   

                df_prob = pd.DataFrame({
                    "Class": list(r["prob_dict"].keys()),
                    "Probability": [p * 100 for p in r["prob_dict"].values()]
                }).sort_values("Probability", ascending=False).head(10)

                df_prob["Probability"] = df_prob["Probability"].round(1)

                fig = px.bar(
                    df_prob,
                    x="Probability",
                    y="Class",
                    orientation="h",
                    title=f"{target} – Top Probabilities",
                    text="Probability",
                    color_discrete_sequence=["#4CAF50"],
                    template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
                )

                fig.update_traces(
                    texttemplate="%{text}%",
                    textposition="auto",
                    marker_line_color="rgba(0,0,0,0)",
                    hovertemplate="%{y}<br>Probability: %{x:.1f}%"
                )

                fig.update_layout(
                    xaxis_title="Probability (%)",
                    yaxis_title=None,
                    height=350 + 30 * len(df_prob),
                    margin=dict(l=20, r=20, t=60, b=40),
                    xaxis_range=[0, max(60, df_prob["Probability"].max() + 10)]
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
         st.info("Fill in the sidebar values and click **Generate Predictions**")

# ────────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("XGBoost • SMOTE • StandardScaler • Multi-target prediction • Clinical use requires validation")

st.caption ("Developed by Vicrise Healthtech Consult• © 2025 All rights reserved.")
