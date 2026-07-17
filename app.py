"""Home page for the OMR exam application."""

from pathlib import Path

import streamlit as st


APP_ROOT = Path(__file__).resolve().parent
GENERATED_DIR = APP_ROOT / "generated_sheets"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="OMR Exam Manager",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📝 OMR Exam Manager")
st.caption(
    "Create printable answer sheets, scan responses, "
    "review detections, and grade exams."
)

st.info(
    "Start by creating an exam. The app will generate an A4 PDF "
    "answer sheet and a JSON file containing the marker and bubble "
    "coordinates."
)

left, middle, right = st.columns(3)

with left:
    st.subheader("1. Create")
    st.write(
        "Define the exam, choices, question count, "
        "and optional answer key."
    )

with middle:
    st.subheader("2. Scan")
    st.write(
        "Upload a completed sheet and align it using "
        "the four corner markers."
    )

with right:
    st.subheader("3. Grade")
    st.write(
        "Review detected answers, resolve uncertain marks, "
        "and calculate results."
    )

st.divider()

st.page_link(
    "pages/1_Create_Exam.py",
    label="Create your first exam",
    icon="➕",
    use_container_width=True,
)

with st.sidebar:
    st.header("Project status")
    st.success("Sheet generator ready")
    st.warning("Scanning and grading pages are not connected yet")
    st.caption(f"Generated sheets: `{GENERATED_DIR.name}/`")