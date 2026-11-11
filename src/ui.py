"""Streamlit presentation helpers: dark navy theme, metric cards, badges and notices."""
import streamlit as st

from config import DATASET_INFO, THEME

CSS = f"""
<style>
  .stApp {{ background: {THEME['bg']}; color: {THEME['text']}; }}
  [data-testid="stSidebar"] {{ background: {THEME['panel']}; border-right: 1px solid rgba(124,58,237,.25); }}
  h1, h2, h3 {{ color: {THEME['text']}; letter-spacing: .2px; }}
  .hc-card {{ background: linear-gradient(135deg, {THEME['panel2']}, {THEME['panel']});
             border: 1px solid rgba(124,58,237,.35); border-radius: 14px; padding: 16px 18px; min-height: 112px; }}
  .hc-lbl {{ color: {THEME['muted']}; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }}
  .hc-val {{ color: {THEME['text']}; font-size: 1.7rem; font-weight: 700; margin-top: 4px; }}
  .hc-sub {{ color: {THEME['lilac']}; font-size: .82rem; margin-top: 4px; }}
  .hc-badge {{ display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: .75rem; font-weight: 600; }}
  .hc-hdfs {{ background: rgba(52,211,153,.15); color: {THEME['green']}; border: 1px solid {THEME['green']}; }}
  .hc-local {{ background: rgba(251,191,36,.12); color: {THEME['amber']}; border: 1px solid {THEME['amber']}; }}
  .hc-note {{ background: rgba(59,130,246,.10); border-left: 4px solid {THEME['blue']};
             padding: 10px 14px; border-radius: 8px; color: {THEME['text']}; font-size: .9rem; margin: 8px 0 14px; }}
  .hc-warn {{ background: rgba(248,113,113,.10); border-left: 4px solid {THEME['red']}; }}
  .hc-sub-title {{ color: {THEME['muted']}; margin-top: -10px; margin-bottom: 18px; }}
  div[data-testid="stMetricValue"] {{ color: {THEME['text']}; }}
  [data-testid="stSidebarNav"] {{ display: none !important; }}   /* automatic page list from pages/; st.navigation is the menu */
  div[data-testid="stVerticalBlockBorderWrapper"] {{ background: {THEME['panel']}; border-color: rgba(124,58,237,.45) !important; border-radius: 14px; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"<div class='hc-sub-title'>{subtitle}</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = "") -> None:
    """A bordered card: label, large value, and a line of context."""
    with st.container(border=True):
        st.caption(label.upper())
        st.markdown(f"#### {value}")
        if sub:
            st.caption(sub)


def storage_badge(mode: str) -> str:
    """HDFS in green, anything else in amber with its label (LOCAL FALLBACK (not HDFS))."""
    cls = "hc-hdfs" if mode == "HDFS" else "hc-local"
    return f"<span class='hc-badge {cls}'>{mode}</span>"


def source_note() -> None:
    """Where the records come from and what the answers are, shown on every analysis page."""
    st.markdown("<div class='hc-note'><b>Source:</b> " + DATASET_INFO["name"] + " (" + DATASET_INFO["author"] +
                "). Survey answers as recorded by CDC; counts are unweighted respondents, not population rates.</div>",
                unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<div class='hc-note'>{text}</div>", unsafe_allow_html=True)


def pretty(name) -> str:
    """'heart_disease' -> 'Heart Disease' for chart labels."""
    return str(name).replace("_", " ").strip().title()
