import os
import io
import re
import json
import base64

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# ==================================================
# CONFIG
# ==================================================
PAGE_TITLE = "PCOS Blood Report Analyzer"
PAGE_ICON = "🔬"

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Vision model used to read report images. Groq retires models often, so if
# this errors, pick a current vision model from console.groq.com/docs/vision
# and set GROQ_VISION_MODEL in .env.
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")

MIN_MARKERS_REQUIRED = 4  # refuse to predict if fewer values than this are available

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

# --------------------------------------------------
# Biomarkers: the model's feature name, display label, the unit the model
# expects, and multipliers to convert common report units into that unit.
# --------------------------------------------------
BIOMARKERS = {
    "Testosterone": {
        "label": "Testosterone (total)", "unit": "ng/dL",
        "hint": "total testosterone",
        "conversions": {"ng/dl": 1, "nmol/l": 28.84, "ng/ml": 100, "ug/l": 100},
    },
    "SHBG": {
        "label": "SHBG", "unit": "nmol/L",
        "hint": "sex hormone binding globulin",
        "conversions": {"nmol/l": 1},
    },
    "LH": {
        "label": "LH", "unit": "mIU/mL",
        "hint": "luteinizing hormone",
        "conversions": {"miu/ml": 1, "iu/l": 1, "u/l": 1, "miu/l": 0.001},
    },
    "FSH": {
        "label": "FSH", "unit": "mIU/mL",
        "hint": "follicle stimulating hormone",
        "conversions": {"miu/ml": 1, "iu/l": 1, "u/l": 1, "miu/l": 0.001},
    },
    "AMH": {
        "label": "AMH", "unit": "ng/mL",
        "hint": "anti-mullerian hormone",
        "conversions": {"ng/ml": 1, "ug/l": 1, "pmol/l": 1 / 7.143, "ng/dl": 0.01},
    },
    "Fasting_Insulin": {
        "label": "Fasting Insulin", "unit": "mU/L",
        "hint": "fasting insulin",
        "conversions": {"mu/l": 1, "miu/l": 1, "uiu/ml": 1, "pmol/l": 1 / 6.945},
    },
    "Fasting_Glucose": {
        "label": "Fasting Glucose", "unit": "mg/dL",
        "hint": "fasting blood glucose / fasting plasma glucose",
        "conversions": {"mg/dl": 1, "mmol/l": 18.016},
    },
    "DHEA_S": {
        "label": "DHEA-S", "unit": "ug/dL",
        "hint": "DHEA sulfate / DHEAS",
        "conversions": {"ug/dl": 1, "mcg/dl": 1, "umol/l": 36.85, "ug/ml": 100, "nmol/l": 0.03685},
    },
    "TSH": {
        "label": "TSH", "unit": "mIU/L",
        "hint": "thyroid stimulating hormone",
        "conversions": {"miu/l": 1, "uiu/ml": 1, "mu/l": 1},
    },
    "Prolactin": {
        "label": "Prolactin", "unit": "ng/mL",
        "hint": "prolactin",
        "conversions": {"ng/ml": 1, "ug/l": 1, "miu/l": 1 / 21.2, "uiu/ml": 1 / 21.2},
    },
}

EXTRACTION_PROMPT = """You are reading a medical blood test report. Find the patient's RESULT (not the reference range) for each marker below, if present.

Markers (use these exact keys):
{marker_list}

Return ONLY a JSON object. Each key maps to an object: {{"value": <number or null>, "unit": <unit string exactly as printed, or null>}}.
Use null when a marker is not in the report. Never guess or estimate. No commentary, no markdown."""


# ==================================================
# MOCK MODEL (synthetic data, demo only)
# ==================================================
@st.cache_resource
def train_mock_model():
    """Trains an XGBoost classifier on synthetic data (no real patients)."""
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "Testosterone": np.random.normal(40, 20, n),
        "SHBG": np.random.normal(60, 25, n),
        "LH": np.random.normal(8, 4, n),
        "FSH": np.random.normal(6, 2, n),
        "AMH": np.random.normal(3, 2, n),
        "Fasting_Insulin": np.random.normal(8, 4, n),
        "Fasting_Glucose": np.random.normal(90, 10, n),
        "DHEA_S": np.random.normal(150, 50, n),
        "TSH": np.random.normal(2.0, 1.0, n),
        "Prolactin": np.random.normal(12, 5, n),
    })

    lh_fsh_ratio = df["LH"] / df["FSH"]
    score = (
        (df["Testosterone"] > 60).astype(int) * 0.3
        + (lh_fsh_ratio > 2.0).astype(int) * 0.3
        + (df["AMH"] > 4.5).astype(int) * 0.2
        + (df["Fasting_Insulin"] > 12).astype(int) * 0.2
    )
    y = (score + np.random.normal(0, 0.1, n) > 0.5).astype(int)

    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(df, y)

    return model, list(df.columns), df.median()


# ==================================================
# REPORT READING (PDF / image -> biomarker values)
# ==================================================
def content_to_text(content):
    """LLM message content can be a string or a list of blocks."""
    if not isinstance(content, str):
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    # Reasoning models can include their thinking in <think> tags; drop it.
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def parse_json_response(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("The model did not return readable JSON.")
    return json.loads(match.group(0))


def build_extraction_prompt():
    marker_list = "\n".join(f"- {key}: {cfg['hint']}" for key, cfg in BIOMARKERS.items())
    return EXTRACTION_PROMPT.format(marker_list=marker_list)


def extract_from_image(image_bytes, mime_type, groq_api_key):
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    llm = ChatGroq(model=GROQ_VISION_MODEL, groq_api_key=groq_api_key, temperature=0)
    message = HumanMessage(content=[
        {"type": "text", "text": build_extraction_prompt()},
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
    ])
    return parse_json_response(content_to_text(llm.invoke([message]).content))


def extract_from_pdf(pdf_bytes, groq_api_key):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()

    if len(text) < 50:
        raise ValueError("No readable text in this PDF (likely a scan). Upload it as an image instead.")

    llm = ChatGroq(model=GROQ_MODEL, groq_api_key=groq_api_key, temperature=0)
    prompt = f"{build_extraction_prompt()}\n\nREPORT TEXT:\n{text[:12000]}"
    return parse_json_response(content_to_text(llm.invoke(prompt).content))


def normalize_unit(unit):
    return unit.lower().replace(" ", "").replace("µ", "u").replace("μ", "u")


def convert_to_model_units(raw):
    """Turns the LLM's raw {value, unit} output into model-ready numbers, plus notes."""
    values, notes = {}, []

    for key, cfg in BIOMARKERS.items():
        item = raw.get(key) or {}
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            continue

        unit = item.get("unit")
        if unit:
            factor = cfg["conversions"].get(normalize_unit(str(unit)))
            if factor is None:
                notes.append(f"{cfg['label']}: unrecognised unit '{unit}', please enter it manually.")
                continue
            if factor != 1:
                notes.append(f"{cfg['label']}: converted {value:g} {unit} to {cfg['unit']}.")
            value *= factor
        else:
            notes.append(f"{cfg['label']}: no unit found, assumed {cfg['unit']}.")

        values[key] = round(value, 2)

    return values, notes


# ==================================================
# PREDICTION & SUMMARY
# ==================================================
def run_prediction(model, feature_names, medians, entered):
    """Imputes missing values with training medians, then returns probability + SHAP values."""
    imputed = [k for k in feature_names if entered.get(k) is None]
    row = {k: (medians[k] if entered.get(k) is None else entered[k]) for k in feature_names}
    input_df = pd.DataFrame([row], columns=feature_names)

    prob = model.predict_proba(input_df)[0][1] * 100
    shap_values = shap.TreeExplainer(model).shap_values(input_df)
    shap_series = pd.Series(shap_values[0], index=feature_names)
    return prob, shap_series, imputed


def build_summary_prompt(prob, drivers_text, entered, imputed):
    values_text = ", ".join(
        f"{BIOMARKERS[k]['label']}: {v:g} {BIOMARKERS[k]['unit']}"
        for k, v in entered.items() if v is not None
    )
    imputed_text = ", ".join(BIOMARKERS[k]["label"] for k in imputed) or "none"

    return f"""You are a clinical AI assistant. A demo XGBoost model (trained on synthetic data) estimated the likelihood of PCOS from a patient's blood test report.

MODEL RESULTS:
- Estimated PCOS likelihood: {prob:.1f}%
- Top SHAP drivers: {drivers_text}

VALUES READ FROM THE REPORT:
{values_text}

MARKERS NOT FOUND (filled with population averages): {imputed_text}

TASK:
Write a concise, plain-language explanation of these results, explaining why the model leaned this way based on the SHAP drivers.
State clearly that this is a screening estimate from a demo model and NOT a diagnosis, that PCOS diagnosis also needs clinical signs and often an ultrasound, and recommend seeing a doctor, especially if the likelihood is elevated or key markers were missing."""


# ==================================================
# UI SECTIONS
# ==================================================
def render_upload_section(groq_api_key):
    st.subheader("1. Upload your blood test report")
    uploaded = st.file_uploader("PDF or image of the report", type=["pdf", "png", "jpg", "jpeg"])

    if st.button("Read report", use_container_width=True, disabled=uploaded is None):
        try:
            with st.spinner("Reading your report..."):
                ext = os.path.splitext(uploaded.name)[1].lower()
                if ext == ".pdf":
                    raw = extract_from_pdf(uploaded.getvalue(), groq_api_key)
                else:
                    mime = "image/png" if ext == ".png" else "image/jpeg"
                    raw = extract_from_image(uploaded.getvalue(), mime, groq_api_key)

                values, notes = convert_to_model_units(raw)

            # Overwrite every field so values from a previous report don't linger.
            for key in BIOMARKERS:
                st.session_state[f"in_{key}"] = values.get(key)
            st.session_state["extraction_notes"] = notes
            st.session_state["extracted_count"] = len(values)

        except Exception as e:
            st.error(f"Couldn't read the report: {e}")

    if "extracted_count" in st.session_state:
        st.success(f"Found {st.session_state['extracted_count']} of {len(BIOMARKERS)} markers. Please check them below.")
        for note in st.session_state.get("extraction_notes", []):
            st.caption(f"• {note}")


def render_values_section():
    st.subheader("2. Check the values")
    st.caption("Reading from images can make mistakes, so fix anything that looks wrong. Leave blank if it isn't on your report.")

    entered = {}
    left, right = st.columns(2)
    for i, (key, cfg) in enumerate(BIOMARKERS.items()):
        with (left if i % 2 == 0 else right):
            entered[key] = st.number_input(
                f"{cfg['label']} ({cfg['unit']})",
                min_value=0.0, value=None, key=f"in_{key}",
            )
    return entered


def render_results(prob, shap_series, imputed, entered, groq_api_key):
    st.markdown("### Model Estimate")
    if prob > 50:
        st.markdown(f"**Estimated likelihood of PCOS:** <span class='confidence-high'>{prob:.1f}% (Elevated)</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"**Estimated likelihood of PCOS:** <span class='confidence-low'>{prob:.1f}% (Lower)</span>", unsafe_allow_html=True)

    if imputed:
        names = ", ".join(BIOMARKERS[k]["label"] for k in imputed)
        st.warning(f"Not found on the report, filled with population averages: {names}. Treat this estimate with extra caution.")

    top = shap_series.reindex(shap_series.abs().sort_values(ascending=False).index).head(3)
    drivers_text = ", ".join(f"{BIOMARKERS[k]['label']} (impact {v:+.2f})" for k, v in top.items())
    st.markdown("### What drove this estimate")
    st.info(f"**Top drivers:** {drivers_text}")
    st.bar_chart(shap_series.rename(index=lambda k: BIOMARKERS[k]["label"]))
    st.caption("Positive bars push toward PCOS, negative bars push away.")

    try:
        llm = ChatGroq(model=GROQ_MODEL, groq_api_key=groq_api_key, temperature=0.2)
        with st.spinner("Writing a plain-language summary..."):
            response = llm.invoke(build_summary_prompt(prob, drivers_text, entered, imputed))
        st.markdown("### Plain-Language Summary")
        with st.container(border=True):
            st.markdown(content_to_text(response.content))
    except Exception as e:
        st.error(f"Error connecting to LLM: {e}")


# ==================================================
# MAIN
# ==================================================
def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        st.error("⚠️ Missing GROQ_API_KEY in .env file.")
        st.stop()

    model, feature_names, medians = train_mock_model()

    st.title("🔬 PCOS Blood Report Analyzer")
    st.caption("Upload a report, check the extracted values, and get an explainable risk estimate.")
    st.warning(
        "Demo tool, not a diagnosis. The prediction model is trained on synthetic data, "
        "and PCOS is diagnosed from clinical signs and imaging as well as blood work. "
        "Please see a doctor for any health decision."
    )
    st.markdown("---")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        render_upload_section(groq_api_key)
        st.markdown("---")
        entered = render_values_section()
        st.markdown("---")
        analyze_btn = st.button("Analyze", use_container_width=True)

    with col2:
        st.subheader("3. Results")

        if not analyze_btn:
            st.info("Upload your report, check the values on the left, then click Analyze.")
            return

        provided = sum(v is not None for v in entered.values())
        if provided < MIN_MARKERS_REQUIRED:
            st.error(f"Only {provided} markers available. At least {MIN_MARKERS_REQUIRED} are needed for a meaningful estimate.")
            return

        with st.spinner("Running the model and calculating SHAP values..."):
            prob, shap_series, imputed = run_prediction(model, feature_names, medians, entered)

        render_results(prob, shap_series, imputed, entered, groq_api_key)


if __name__ == "__main__":
    main()