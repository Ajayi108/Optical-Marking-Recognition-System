# Displays saved OMR results and exports.

from __future__ import annotations

import csv
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import cv2
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
SCANNED_DIR = APP_ROOT / "scanned_sheets"
SCANNED_DIR.mkdir(parents=True, exist_ok=True)

# Streamlit page files need the project root on the import path.
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from database.database import exam_summary, list_results

st.set_page_config(page_title="Results | OMR", layout="wide")


def to_rgb(image: Any) -> Any:
    # Converts an OpenCV BGR image to RGB for Streamlit.
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_result_files() -> list[dict[str, Any]]:
    # Loads saved result JSON files from the scanned sheets directory.
    results: list[dict[str, Any]] = []

    # JSON result files carry the full grading audit trail for each submission.
    for path in sorted(SCANNED_DIR.glob("*_result.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        result["_path"] = str(path)
        # Keep the source path with the row so downloads can use stable names.
        results.append(result)

    return results


def summary_row(result: dict[str, Any]) -> dict[str, Any]:
    # Builds one row for the results summary table.
    metadata = result.get("metadata", {})
    exam = metadata.get("exam", {})
    grading = result.get("grading", {})
    student = result.get("student_name") or result.get("student_id") or "Unknown"
    score = grading.get("score")
    total = grading.get("total")
    percentage = grading.get("percentage")

    return {
        # Empty strings render more cleanly than None in Streamlit dataframes.
        "Result": result.get("result_id", Path(result.get("_path", "")).stem),
        "Exam ID": exam.get("exam_id", ""),
        "Title": exam.get("title", ""),
        "Student": student,
        "Student ID": result.get("student_id", ""),
        "Score": "" if score is None else score,
        "Total": "" if not total else total,
        "Percent": "" if percentage is None else percentage,
        "Reviewed at": result.get("reviewed_at", ""),
    }


def csv_text(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    # Serializes rows to CSV text.
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def result_detail_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    # Returns question-level grading rows for display and export.
    rows = []

    for row in result.get("grading", {}).get("rows", []):
        rows.append(
            {
                "Question": row.get("question", ""),
                "Answer": row.get("answer", ""),
                "Correct answer": row.get("correct_answer", ""),
                "Result": row.get("result", ""),
            }
        )

    return rows


st.page_link("app.py", label="Back to home")
st.title("Results")
st.write("View saved submissions, inspect grading, and export class results.")

file_results = load_result_files()
database_results = list_results()

if not file_results and not database_results:
    # Results cannot appear until the review page saves at least one submission.
    st.info("No reviewed results have been saved yet.")
    st.page_link("pages/3_Review_Answers.py", label="Go to Review Answers")
    st.stop()

summary_rows = [summary_row(result) for result in file_results]

if summary_rows:
    # Exam filtering keeps a large class run manageable.
    exams = sorted({str(row["Exam ID"]) for row in summary_rows if row["Exam ID"]})
    selected_exam = st.selectbox("Exam filter", options=["All exams"] + exams)

    if selected_exam != "All exams":
        # Keep the visible result objects in the same filtered set as the table rows.
        visible_rows = [row for row in summary_rows if row["Exam ID"] == selected_exam]
        visible_results = [result for result in file_results if summary_row(result)["Exam ID"] == selected_exam]
    else:
        visible_rows = summary_rows
        visible_results = file_results

    metric_a, metric_b, metric_c = st.columns(3)
    # Ungraded submissions have no percentage and are skipped in aggregates.
    graded_percentages = [float(row["Percent"]) for row in visible_rows if row["Percent"] != ""]
    metric_a.metric("Submissions", len(visible_rows))
    metric_b.metric("Average", f"{sum(graded_percentages) / len(graded_percentages):.2f}%" if graded_percentages else "N/A")
    metric_c.metric("Best", f"{max(graded_percentages):.2f}%" if graded_percentages else "N/A")

    st.subheader("Saved submissions")
    st.dataframe(visible_rows, hide_index=True, use_container_width=True)
    st.download_button(
        # Summary CSV is intended for class-level export.
        "Download summary CSV",
        data=csv_text(visible_rows, list(visible_rows[0].keys()) if visible_rows else []),
        file_name="omr_results_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

    selected_result_id = st.selectbox("Open result", options=[row["Result"] for row in visible_rows])
    # Detail view reuses the exact result JSON, not a recomputed grade.
    selected_result = next(result for result in visible_results if summary_row(result)["Result"] == selected_result_id)
    selected_summary = summary_row(selected_result)
    grading = selected_result.get("grading", {})
    status_counts = grading.get("status_counts", {})

    st.divider()
    st.subheader("Submission detail")
    detail_a, detail_b, detail_c, detail_d = st.columns(4)
    detail_a.metric("Student", selected_summary["Student"])
    detail_b.metric("Score", "N/A" if grading.get("score") is None else f"{grading.get('score')} / {grading.get('total')}")
    detail_c.metric("Percent", "N/A" if grading.get("percentage") is None else f"{grading.get('percentage')}%")
    detail_d.metric("Exam", selected_summary["Exam ID"])

    left, right = st.columns([2, 1])

    with left:
        # Question-level rows make it easy to audit a student's final score.
        rows = result_detail_rows(selected_result)
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.download_button(
            "Download detail CSV",
            data=csv_text(rows, ["Question", "Answer", "Correct answer", "Result"]),
            file_name=f"{selected_result_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download result JSON",
            data=json.dumps(selected_result, indent=2),
            file_name=f"{selected_result_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    with right:
        st.write("Status counts")
        status_rows = [{"status": key, "count": value} for key, value in status_counts.items()]

        # The chart highlights blanks, multiples, and other grading categories at a glance.
        if status_rows:
            st.bar_chart(status_rows, x="status", y="count")

        overlay_path = Path(str(selected_result.get("review_overlay_path", "")))

        # The saved overlay lets the reviewer see exactly what the detector saw.
        if overlay_path.exists():
            overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)

            if overlay is not None:
                st.image(to_rgb(overlay), use_container_width=True)

db_exam_summary = exam_summary()

if db_exam_summary:
    # SQLite aggregates are separate from JSON file display and useful for quick checks.
    st.divider()
    st.subheader("Database exam summary")
    st.dataframe(db_exam_summary, hide_index=True, use_container_width=True)
