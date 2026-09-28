import os
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# ==================================================
# CONFIGURATION & THEME
# ==================================================
PAGE_TITLE = "PCOS Predictive AI & Explainer"
PAGE_ICON = "🔬"

CUSTOM_CSS = """
<style>
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6, #1D4ED8);
        color: white !important;
        border-radius: 8px;
        font-weight: 600;
        border: none;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    .confidence-high { color: #DC2626; font-weight: bold; font-size: 1.2rem; }
    .confidence-low { color: #16A34A; font-weight: bold; font-size: 1.2rem; }
</style>
"""

# ==================================================
# MOCK MACHINE LEARNING MODEL (XGBoost)
# ==================================================
@st.cache_resource
def train_mock_model():
    """
    Trains a mock XGBoost classifier on synthetic data to simulate 
    a real predictive model without needing an external .csv file.
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Generate Synthetic Data based on clinical patterns
    df = pd.DataFrame({
        'Testosterone': np.random.normal(40, 20, n_samples),
        'SHBG': np.random.normal(60, 25, n_samples),
        'LH': np.random.normal(8, 4, n_samples),
        'FSH': np.random.normal(6, 2, n_samples),
        'AMH': np.random.normal(3, 2, n_samples),
        'Fasting_Insulin': np.random.normal(8, 4, n_samples),
        'Fasting_Glucose': np.random.normal(90, 10, n_samples),
        'DHEA_S': np.random.normal(150, 50, n_samples),
        'TSH': np.random.normal(2.0, 1.0, n_samples),
        'Prolactin': np.random.normal(12, 5, n_samples)
    })
    
    # Create target (PCOS = 1) based on combinations of markers
    lh_fsh_ratio = df['LH'] / df['FSH']
    pcos_probability = (
        (df['Testosterone'] > 60).astype(int) * 0.3 +
        (lh_fsh_ratio > 2.0).astype(int) * 0.3 +
        (df['AMH'] > 4.5).astype(int) * 0.2 +
        (df['Fasting_Insulin'] > 12).astype(int) * 0.2
    )
    
    df['PCOS_Target'] = (pcos_probability + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    
    X = df.drop(columns=['PCOS_Target'])
    y = df['PCOS_Target']
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
    return model, X.columns

# ==================================================
# MAIN APP
# ==================================================
def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not groq_api_key:
        st.error("⚠️ Missing GROQ_API_KEY in .env file.")
        st.stop()
    model, feature_names = train_mock_model()

    st.title("🔬 Predictive PCOS Analyzer with SHAP Explainability")
    st.caption("Powered by XGBoost, SHAP, and Generative AI")
    st.markdown("---")

    col1, col2 = st.columns([1, 1.5])

    # Left Column: User Inputs (Expanded Biomarkers)
    with col1:
        st.subheader("📝 Input Blood Test Metrics")
        
        testosterone = st.number_input("Testosterone (total) ng/dL", min_value=0.0, value=75.0)
        shbg = st.number_input("SHBG nmol/L", min_value=0.0, value=30.0)
        lh = st.number_input("LH (Luteinizing Hormone) mIU/mL", min_value=0.0, value=15.5)
        fsh = st.number_input("FSH (Follicle-Stimulating Hormone) mIU/mL", min_value=0.0, value=5.2)
        amh = st.number_input("AMH (Anti-Müllerian Hormone) ng/mL", min_value=0.0, value=6.2)
        insulin = st.number_input("Fasting Insulin (mU/L)", min_value=0.0, value=16.0)
        glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=0.0, value=95.0)
        dhea_s = st.number_input("DHEA-S (ug/dL)", min_value=0.0, value=200.0)
        tsh = st.number_input("TSH (mIU/L)", min_value=0.0, value=2.1)
        prolactin = st.number_input("Prolactin (ng/mL)", min_value=0.0, value=14.0)
        
        st.markdown("---")
        ai_model_choice = st.selectbox("🧠 Select Summarization Engine", ["Groq (Llama-3)", "Google (Gemini 1.5 Flash)"])
        analyze_btn = st.button("Generate Predictive Analysis", use_container_width=True)

    # Right Column: ML Prediction & SHAP Analysis
    with col2:
        st.subheader("📊 Predictive Results & Explainability")
        
        if analyze_btn:
            with st.spinner("Running ML classifier and calculating SHAP values..."):
                
                # 1. Prepare Input Data
                input_data = pd.DataFrame([[
                    testosterone, shbg, lh, fsh, amh, insulin, glucose, dhea_s, tsh, prolactin
                ]], columns=feature_names)
                
                # 2. Get Confidence Score
                probabilities = model.predict_proba(input_data)[0]
                pcos_prob = probabilities[1] * 100
                
                # 3. Get SHAP Values for Explainability
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_data)
                
                # Extract top 3 driving features
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP_Value': shap_values[0]
                }).sort_values(by='SHAP_Value', key=abs, ascending=False)
                
                top_features = feature_importance.head(3)
                drivers_text = ", ".join([f"{row['Feature']} (Impact: {row['SHAP_Value']:.2f})" for _, row in top_features.iterrows()])

                # Display Confidence Score
                st.markdown("### Model Prediction")
                if pcos_prob > 50:
                    st.markdown(f"**Likelihood of PCOS:** <span class='confidence-high'>{pcos_prob:.1f}% (High Confidence)</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Likelihood of PCOS:** <span class='confidence-low'>{pcos_prob:.1f}% (Low Likelihood)</span>", unsafe_allow_html=True)
                
                st.markdown("### SHAP Explainability")
                st.info(f"**Primary drivers of this prediction:** {drivers_text}")

                # 4. LLM Patient-Friendly Translation
                prompt_template = PromptTemplate.from_template("""
                You are a clinical AI assistant. You have just run an XGBoost model to predict the likelihood of PCOS.
                
                MODEL RESULTS:
                - Confidence Score for PCOS: {pcos_prob:.1f}%
                - SHAP Explainability (Top Drivers): {drivers_text}
                
                PATIENT RAW DATA:
                - Testosterone: {testosterone}, SHBG: {shbg}, LH: {lh}, FSH: {fsh}, AMH: {amh}, Fasting Insulin: {insulin}
                
                TASK:
                Write a concise, plain-language explanation of these results. Explain *why* the model made this prediction based on the SHAP top drivers. 
                Keep it clinical but easy to understand. Do not provide a definitive medical diagnosis. Ensure cases are flagged for a doctor if necessary.
                """)
                
                formatted_prompt = prompt_template.format(
                    pcos_prob=pcos_prob, drivers_text=drivers_text, testosterone=testosterone, 
                    shbg=shbg, lh=lh, fsh=fsh, amh=amh, insulin=insulin
                )

                try:
                    if "Groq" in ai_model_choice:
                        llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key, temperature=0.2)
                    else:
                        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0.2)
                        
                    response = llm.invoke(formatted_prompt)
                    
                    st.markdown("### Plain-Language Summary")
                    with st.container(border=True):
                        st.markdown(response.content)
                        
                except Exception as e:
                    st.error(f"Error connecting to LLM: {e}")
        else:
            st.info("Enter biomarkers on the left and click 'Generate Predictive Analysis' to run the ML model and view SHAP drivers.")

if __name__ == "__main__":
    main()