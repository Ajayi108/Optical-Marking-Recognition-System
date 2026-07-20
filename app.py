# Displays the OMR application home page.

from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
PAGES_DIR = APP_ROOT / "pages"
GENERATED_DIR = APP_ROOT / "generated_sheets"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="OMR Exam Manager", page_icon="📝", layout="wide", initial_sidebar_state="expanded")

# Displays a page link only when the page exists.
def show_page_link(filename: str, label: str, icon: str) -> None:
    page_path = PAGES_DIR / filename
    if page_path.exists():
        st.page_link(f"pages/{filename}", label=label, icon=icon, use_container_width=True)
    else:
        st.info(f"{label} is not available yet.")

st.title("📝 OMR Exam Manager")
st.caption("Create printable answer sheets, scan responses, review detections, and grade exams.")
st.info("Start by creating an exam. The app will generate an A4 PDF answer sheet and JSON metadata containing ArUco marker and bubble coordinates.")

left, middle, right = st.columns(3)

with left:
    st.subheader("1. Create")
    st.write("Define the exam title, exam ID, question count, choices, and answer key.")
    show_page_link("1_Create_Exam.py", "Create exam", "➕")

with middle:
    st.subheader("2. Scan")
    st.write("Upload a completed sheet and detect the four unique ArUco markers.")
    show_page_link("2_Scan_Sheet.py", "Scan sheet", "📷")

with right:
    st.subheader("3. Review and grade")
    st.write("Review detected answers and calculate the final score.")
    show_page_link("3_Review_Answers.py", "Review answers", "🔍")
    show_page_link("4_Results.py", "View results", "📊")

st.divider()

with st.sidebar:
    st.header("Project status")
    st.success("Sheet generator ready")
    st.success("ArUco marker detection ready")
    st.warning("Review and results pages are not created yet")
    st.caption(f"Generated sheets folder: `{GENERATED_DIR.name}/`")