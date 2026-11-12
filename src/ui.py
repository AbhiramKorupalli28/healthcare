"""Streamlit presentation helpers: modern clean light SaaS theme with Forest Green accents."""
import streamlit as st
from config import DATASET_INFO, THEME

CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }}

  /* App Background */
  .stApp {{
    background-color: #F8FAFC;
    color: #111827;
  }}

  /* Sidebar Styling */
  [data-testid="stSidebar"] {{
    background-color: #FFFFFF;
    border-right: 1px solid #E5E7EB;
    padding-top: 1.5rem;
  }}

  /* Active Navigation Links */
  [data-testid="stSidebarNavLink"][aria-current="page"] {{
    background-color: #EBF5EE !important;
    color: #0E4D2A !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
  }}
  [data-testid="stSidebarNavLink"][aria-current="page"] span,
  [data-testid="stSidebarNavLink"][aria-current="page"] svg {{
    color: #0E4D2A !important;
    fill: #0E4D2A !important;
  }}
  [data-testid="stSidebarNavLink"] {{
    border-radius: 12px;
    margin-bottom: 4px;
    color: #4B5563;
    transition: all 0.15s ease;
  }}
  [data-testid="stSidebarNavLink"]:hover {{
    background-color: #F3F4F6;
    color: #111827;
  }}

  /* Top Metric Cards */
  .stat-card {{
    border-radius: 16px;
    padding: 20px 22px;
    min-height: 148px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
  }}
  .stat-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
  }}
  .hero-card {{
    background-color: #0E4D2A;
    color: #FFFFFF;
    border: 1px solid #0E4D2A;
  }}
  .white-card {{
    background-color: #FFFFFF;
    color: #111827;
    border: 1px solid #E5E7EB;
  }}
  .stat-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .stat-label {{
    font-size: 0.82rem;
    font-weight: 500;
    color: #6B7280;
  }}
  .stat-label-light {{
    font-size: 0.82rem;
    font-weight: 500;
    color: #D1FAE5;
  }}
  .stat-icon-circle {{
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background-color: #F3F4F6;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    color: #9CA3AF;
  }}
  .stat-icon-circle-light {{
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background-color: rgba(255, 255, 255, 0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    color: #FFFFFF;
  }}
  .stat-value {{
    font-size: 2.1rem;
    font-weight: 800;
    color: #111827;
    line-height: 1.1;
    margin: 6px 0 10px;
  }}
  .stat-value-light {{
    font-size: 2.1rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.1;
    margin: 6px 0 10px;
  }}
  .stat-badge-green {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    background-color: #EBF5EE;
    color: #166534;
    width: fit-content;
  }}
  .stat-badge-dark {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    background-color: rgba(0, 0, 0, 0.22);
    color: #E6F4EA;
    width: fit-content;
  }}

  /* Content Cards */
  .content-card {{
    background-color: #FFFFFF;
    border-radius: 16px;
    border: 1px solid #E5E7EB;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
  }}
  .card-title {{
    font-size: 1.22rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 4px;
  }}
  .card-subtitle {{
    font-size: 0.85rem;
    color: #6B7280;
    margin-bottom: 18px;
  }}

  /* Model Comparison Table */
  .model-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin-bottom: 18px;
  }}
  .model-table th {{
    text-align: left;
    padding: 12px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #4B5563;
    border-bottom: 1px solid #E5E7EB;
    background-color: #FAFBFB;
  }}
  .model-table td {{
    padding: 13px 14px;
    font-size: 0.88rem;
    color: #374151;
    border-bottom: 1px solid #F3F4F6;
  }}
  .model-table tr:last-child td {{
    border-bottom: none;
  }}
  .model-table td.highlight {{
    font-weight: 700;
    color: #0E4D2A;
  }}

  /* Green Summary Callout Box */
  .summary-box {{
    background-color: #EBF5EE;
    border: 1px solid #C6E7D0;
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 0.84rem;
    color: #166534;
    line-height: 1.55;
  }}

  /* Buttons */
  button[kind="primary"], .action-btn {{
    background-color: #0E4D2A !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 8px 18px !important;
  }}
  button[kind="primary"]:hover {{
    background-color: #0A381E !important;
  }}

  /* Sidebar Links Section */
  .sidebar-section-label {{
    font-size: 0.7rem;
    font-weight: 600;
    color: #9CA3AF;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 16px 0 8px 4px;
  }}
  .sidebar-link {{
    display: block;
    color: #4B5563;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 6px 8px;
    text-decoration: none;
    border-radius: 8px;
  }}
  .sidebar-link:hover {{
    color: #0E4D2A;
    background-color: #F3F4F6;
  }}
</style>
"""

def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

def render_top_header(title: str = "Dashboard", subtitle: str = "Smart Healthcare Big Data Analytics (1.6M+ Records)") -> None:
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 22px;">
      <div>
        <h1 style="font-size: 2rem; font-weight: 800; color: #111827; margin: 0; line-height: 1.15;">{title}</h1>
        <div style="font-size: 0.88rem; color: #6B7280; margin-top: 4px;">{subtitle}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def render_metric_cards(c1_val: str, c2_val: str, c3_val: str, c4_val: str,
                        c1_badge: str = "1.6M Validated Records",
                        c2_badge: str = "Kaggle 20 Tables Master",
                        c3_badge: str = "Clinical & Vitals",
                        c4_badge: str = "Random Forest"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-card hero-card">
          <div class="stat-header">
            <span class="stat-label-light">Total Observations</span>
            <span class="stat-icon-circle-light">✓</span>
          </div>
          <div class="stat-value-light">{c1_val}</div>
          <div class="stat-badge-dark">{c1_badge}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card white-card">
          <div class="stat-header">
            <span class="stat-label">Original Columns</span>
            <span class="stat-icon-circle">✎</span>
          </div>
          <div class="stat-value">{c2_val}</div>
          <div class="stat-badge-green">{c2_badge}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card white-card">
          <div class="stat-header">
            <span class="stat-label">Selected Features</span>
            <span class="stat-icon-circle">✎</span>
          </div>
          <div class="stat-value">{c3_val}</div>
          <div class="stat-badge-green">{c3_badge}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-card white-card">
          <div class="stat-header">
            <span class="stat-label">Best ML Model</span>
            <span class="stat-icon-circle">✎</span>
          </div>
          <div class="stat-value">{c4_val}</div>
          <div class="stat-badge-green">{c4_badge}</div>
        </div>
        """, unsafe_allow_html=True)

def render_model_comparison_card():
    st.markdown("""
    <div class="content-card">
      <div class="card-title">Model Comparison</div>
      <div class="card-subtitle">Empirical classification quality metrics and execution benchmarks.</div>
      <table class="model-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Random Forest</th>
            <th>Decision Tree</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><b>Heart Disease Accuracy</b></td>
            <td class="highlight">96.92%</td>
            <td>89.98%</td>
            <td>Cardiovascular condition detection benchmark</td>
          </tr>
          <tr>
            <td><b>Heart Disease Macro F1</b></td>
            <td class="highlight">0.8444</td>
            <td>0.7202</td>
            <td>Balanced metric accounting for rare positive cases</td>
          </tr>
          <tr>
            <td><b>Diabetes Accuracy</b></td>
            <td class="highlight">92.28%</td>
            <td>49.53%</td>
            <td>Chronic endocrine disorder classification</td>
          </tr>
          <tr>
            <td><b>Hypertension Accuracy</b></td>
            <td class="highlight">89.72%</td>
            <td>77.45%</td>
            <td>AHA blood pressure & chronic hypertension status</td>
          </tr>
          <tr>
            <td><b>Execution Time</b></td>
            <td>1.60 s</td>
            <td>0.30 s</td>
            <td>Evaluation over 50,000 stratified Spark partitions</td>
          </tr>
          <tr>
            <td><b>Complexity / Estimators</b></td>
            <td>200 Trees</td>
            <td>Max Depth 6</td>
            <td>Balanced class weights to mitigate clinical imbalance</td>
          </tr>
        </tbody>
      </table>
      <div class="summary-box">
        <b>Summary:</b> Random Forest achieved superior classification quality metrics (96.92% Heart Disease accuracy, 92.28% Diabetes accuracy, 0.8444 Macro F1) with balanced class weights, executing distributed evaluation over Spark-sampled partitions.
      </div>
    </div>
    """, unsafe_allow_html=True)

def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"<h2 style='font-weight: 700; color: #111827; margin-bottom: 2px;'>{title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div style='color: #6B7280; font-size: 0.88rem; margin-bottom: 18px;'>{subtitle}</div>", unsafe_allow_html=True)

def metric_card(label: str, value: str, sub: str = "") -> None:
    with st.container(border=True):
        st.caption(label.upper())
        st.markdown(f"#### {value}")
        if sub:
            st.caption(sub)

def storage_badge(mode: str) -> str:
    cls = "stat-badge-green" if mode == "HDFS" else "stat-badge-dark"
    return f"<span class='{cls}'>{mode}</span>"

def source_note() -> None:
    st.markdown("<div class='summary-box'><b>Source:</b> " + DATASET_INFO["name"] + " (" + DATASET_INFO["author"] +
                "). Population health records processed with Apache Spark & PySpark.</div>",
                unsafe_allow_html=True)

def note(text: str) -> None:
    st.markdown(f"<div class='summary-box'>{text}</div>", unsafe_allow_html=True)

def pretty(name) -> str:
    return str(name).replace("_", " ").strip().title()
