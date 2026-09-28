# 🩸 Predictive PCOS Analyzer & Clinical Explainer

An AI-powered diagnostic support tool that analyzes blood test biomarkers to predict the likelihood of Polycystic Ovary Syndrome (PCOS). 

Instead of relying on basic rule-based comparisons, this application utilizes a trained **XGBoost Machine Learning Classifier** to detect complex patterns across multiple hormonal markers. It integrates **SHAP (SHapley Additive exPlanations)** to provide transparent reasoning for its predictions and uses **Generative AI (Groq / Google Gemini)** to generate patient-friendly clinical summaries.

## 🚀 Key Features

* **Machine Learning Prediction:** Uses an XGBoost classifier trained on multi-variable blood markers (Testosterone, LH:FSH ratio, AMH, Insulin, etc.) to generate a specific confidence score.
* **SHAP Explainability:** Opens the ML "black box" by calculating the exact impact of each biomarker on the final prediction, highlighting the top driving factors.
* **LLM-Powered Clinical Summaries:** Connects to Groq (Llama-3) and Google Gemini (Flash 1.5) to translate raw SHAP values and probability scores into plain-language, actionable insights.
* **Local Processing:** Fast, interactive UI built entirely in Python using Streamlit.

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Native Python UI)
* **Machine Learning:** XGBoost, Scikit-Learn, Pandas, NumPy
* **Explainable AI:** SHAP (TreeExplainer)
* **Generative AI / LLMs:** LangChain, Groq API (Llama 3), Google AI Studio (Gemini 1.5 Flash)

## 💻 Installation and Setup

Follow these steps to run the project locally on your machine.

**1. Clone the repository**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name