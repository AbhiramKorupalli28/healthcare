"""Healthcare Analytics for Disease Pattern Detection: Streamlit entry point.

Run with:  streamlit run app.py
"""
import streamlit as st

from src.app_state import active_dataset_path, set_active_dataset
from src.data_loader import list_local_datasets
from src.ui import inject_css

st.set_page_config(page_title="Healthcare Analytics", page_icon="🩺", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()

pages = [
    st.Page("pages/overview.py", title="Overview", icon="📊", default=True),
    st.Page("pages/dataset_explorer.py", title="Dataset Explorer", icon="🗂️"),
    st.Page("pages/disease_analytics.py", title="Disease Analytics", icon="🧬"),
    st.Page("pages/risk_analysis.py", title="Risk Analysis", icon="⚠️"),
    st.Page("pages/bigdata_processing.py", title="Big Data Processing", icon="⚡"),
    st.Page("pages/reports.py", title="Reports", icon="📑"),
]

with st.sidebar:
    st.markdown("### 🩺 Healthcare Analytics")
    available = list_local_datasets()
    if available:
        names = [d.name for d in available]
        current = active_dataset_path().name
        idx = names.index(current) if current in names else 0
        chosen = st.selectbox("📂 Active Dataset", names, index=idx)
        chosen_path = available[names.index(chosen)]
        if str(chosen_path) != str(active_dataset_path()):
            set_active_dataset(chosen_path)
            st.rerun()

    active = active_dataset_path()
    if active.exists():
        st.caption(f"{active.stat().st_size / 1e6:.2f} MB · loaded from disk")
    st.divider()

navigation = st.navigation(pages)
navigation.run()
