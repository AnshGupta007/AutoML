"""
AutoMLPro Streamlit Dashboard — main application entry point.
Run with: streamlit run src/dashboard/app.py
"""
import streamlit as st
from pathlib import Path

# Must be the very first Streamlit call
st.set_page_config(
    page_title="AutoMLPro Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom global CSS
st.markdown("""
<style>
  /* Dark background */
  [data-testid="stAppViewContainer"] { background: #0f172a; }
  [data-testid="stSidebar"] { background: #1e293b; }
  [data-testid="stHeader"] { background: transparent; }

  /* Typography */
  h1, h2, h3 { color: #e2e8f0; }
  p, label, .stMarkdown { color: #94a3b8; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #0284c7, #6366f1);
    color: white; border: none; border-radius: 8px;
    padding: 8px 20px; font-weight: 600;
    transition: opacity .2s;
  }
  .stButton > button:hover { opacity: 0.85; }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: #1e293b; border-radius: 12px;
    padding: 16px; border: 1px solid #334155;
  }
  [data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 1.5rem !important; }
  [data-testid="stMetricLabel"] { color: #94a3b8 !important; }

  /* Tab styling */
  .stTabs [data-baseweb="tab"] { color: #94a3b8; }
  .stTabs [aria-selected="true"] { color: #38bdf8; border-bottom-color: #38bdf8; }

  /* Upload area */
  [data-testid="stFileUploader"] { border: 1px dashed #334155; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
MODELS_DIR = "models"

st.sidebar.markdown("""
<div style='text-align:center;padding:16px 0 8px 0'>
  <span style='font-size:2.5rem'>🤖</span><br/>
  <span style='font-size:1.1rem;font-weight:700;color:#38bdf8'>AutoMLPro</span><br/>
  <span style='font-size:0.75rem;color:#64748b'>Intelligent ML Pipeline</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

PAGES = {
    "📊 Data Overview": "01_data_overview",
    "🏆 Training Results": "02_training_results",
    "🔍 Model Explain": "03_model_explain",
    "🎯 Predictions": "04_predictions",
    "📡 Monitoring": "05_monitoring",
}

page_label = st.sidebar.radio("Navigate to", list(PAGES.keys()), key="nav")
st.sidebar.divider()

# Models directory override
models_dir_input = st.sidebar.text_input("Models directory", value=MODELS_DIR, key="models_dir_input")

# Quick actions
st.sidebar.markdown("**⚡ Quick Actions**")
if st.sidebar.button("🔄 Reload Page"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("AutoMLPro v1.0.0 · Built with Streamlit")

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
page_module = PAGES[page_label]

if page_module == "01_data_overview":
    from src.dashboard.pages import page_01_data_overview as pg
    pg.render()
elif page_module == "02_training_results":
    from src.dashboard.pages import page_02_training_results as pg
    pg.render(models_dir=models_dir_input)
elif page_module == "03_model_explain":
    from src.dashboard.pages import page_03_model_explain as pg
    pg.render(models_dir=models_dir_input)
elif page_module == "04_predictions":
    from src.dashboard.pages import page_04_predictions as pg
    pg.render(models_dir=models_dir_input)
elif page_module == "05_monitoring":
    from src.dashboard.pages import page_05_monitoring as pg
    pg.render(models_dir=models_dir_input)
