# Reviews detected answers, allows corrections, and saves graded results.

from __future__ import annotations

import csv
import json
import re
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import cv2
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
SCANNED_DIR = APP_ROOT / "scanned_sheets"
SCANNED_DIR.mkdir(parents=True, exist_ok=True)

# Streamlit page files run from their own folder, so add the app root for imports.
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from database.database import save_result, save_scan, utc_now
from omr.bubble_detection import (
    DetectionSettings,
    answers_to_mapping,
    detect_answers,
    draw_detection_overlay,
    settings_from_metadata,
    statuses_to_mapping,
    summarize_detections,
)
from omr.grading import grade_answers

st.set_page_config(page_title="Review Answers | OMR", layout="wide")


def safe_name(value: str) -> str:
    # Converts text into a safe filename part.
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return cleaned.strip("._-") or "review"


def to_rgb(image: Any) -> Any:
    # Converts an OpenCV BGR image to RGB for Streamlit.
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_scan_result(uploaded_scan: Any, selected_scan: str) -> tuple[dict[str, Any], str]:
    # Loads a scan JSON from disk or from an upload.
    # Uploads are useful when a scan was produced outside the normal scanned_sheets folder.
    if uploaded_scan is not None:
        return json.loads(uploaded_scan.getvalue().decode("utf-8")), uploaded_scan.name

    if not selected_scan:
        raise ValueError("Select or upload a scan JSON file.")

    scan_path = SCANNED_DIR / selected_scan

    if not scan_path.exists():
        raise ValueError("The selected scan file does not exist.")

    return json.loads(scan_path.read_text(encoding="utf-8")), scan_path.name


def build_review_rows(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Builds data-editor rows from bubble detections.
    rows: list[dict[str, Any]] = []

    for detection in detections:
        # Fill ratios help the operator understand why a row was marked or flagged.
        ratios = ", ".join(f"{bubble['choice']}:{bubble['fill_ratio']:.2f}" for bubble in detection["bubbles"])
        rows.append(
            {
                "Question": int(detection["question"]),
                "Detected": detection.get("selected_choice") or "",
                "Reviewed answer": detection.get("selected_choice") or "",
                "Status": str(detection.get("status", "")),
                "Confidence": float(detection.get("confidence", 0.0)),
                "Fill ratios": ratios,
            }
        )

    return rows


def edited_rows_to_answers(
    edited_rows: list[dict[str, Any]],
    detections: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    # Converts edited rows to reviewed answers and grading statuses.
    original_statuses = statuses_to_mapping(detections)
    answers: dict[str, str] = {}
    statuses: dict[str, str] = {}

    for row in edited_rows:
        question = str(int(row["Question"]))
        reviewed = str(row.get("Reviewed answer", "")).strip().upper()
        original_status = original_statuses.get(question, "blank")

        # A manual answer overrides blank or multiple detection during grading.
        if reviewed:
            answers[question] = reviewed
            statuses[question] = "marked" if original_status != "multiple" else "manual"
        else:
            statuses[question] = original_status

    return answers, statuses


def csv_text(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    # Serializes rows to CSV text.
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def result_path_for(scan_name: str, student_id: str, student_name: str) -> Path:
    # Builds a stable result path for a reviewed scan.
    stem = Path(scan_name).stem.replace("_scan", "")
    student_part = safe_name(student_id or student_name or "student")
    return SCANNED_DIR / f"{stem}_{student_part}_result.json"


st.page_link("app.py", label="Back to home")
st.title("Review Answers")
st.write("Check the detected marks, correct any uncertain answers, and save the graded result.")

scan_files = sorted(SCANNED_DIR.glob("*_scan.json"), key=lambda path: path.stat().st_mtime, reverse=True)
scan_names = [path.name for path in scan_files]

# Tuning controls live in the sidebar so the review table stays focused.
with st.sidebar:
    st.header("Detection tuning")
    st.caption("Adjust only when the preview shows blank or overfilled detections.")

selected_scan = st.selectbox(
    "Saved scan",
    options=[""] + scan_names,
    format_func=lambda value: "Choose a saved scan JSON" if not value else value,
)
uploaded_scan = st.file_uploader("Or upload a scan JSON", type=["json"])

if not selected_scan and uploaded_scan is None:
    # Review depends on the aligned image produced by the Scan Sheet page.
    st.info("Scan a sheet first, then return here to review detected answers.")
    st.page_link("pages/2_Scan_Sheet.py", label="Go to Scan Sheet")
    st.stop()

try:
    # Load the scan record and the metadata that describes the generated sheet layout.
    scan_result, scan_name = load_scan_result(uploaded_scan, selected_scan)
    metadata = scan_result.get("metadata")

    if not isinstance(metadata, dict):
        raise ValueError("The scan JSON does not include the original sheet metadata.")

    aligned_path = Path(str(scan_result.get("aligned_path", "")))

    if not aligned_path.exists():
        raise ValueError("The aligned scan image is missing. Re-run the Scan Sheet step.")

    aligned_image = cv2.imread(str(aligned_path), cv2.IMREAD_COLOR)

    if aligned_image is None:
        raise ValueError("The aligned scan image could not be opened.")

    save_scan(scan_result, SCANNED_DIR / scan_name)
    defaults = settings_from_metadata(metadata)

    # Sliders let the user recover from faint pencil marks or very dark scans.
    with st.sidebar:
        minimum_fill_ratio = st.slider(
            "Minimum fill ratio",
            min_value=0.05,
            max_value=0.75,
            value=float(defaults.minimum_fill_ratio),
            step=0.01,
        )
        multiple_mark_margin = st.slider(
            "Multiple mark margin",
            min_value=0.01,
            max_value=0.30,
            value=float(defaults.multiple_mark_margin),
            step=0.01,
        )
        inner_radius_scale = st.slider(
            "Inner bubble area",
            min_value=0.45,
            max_value=0.95,
            value=float(defaults.inner_radius_scale),
            step=0.01,
        )

    settings = DetectionSettings(
        minimum_fill_ratio=minimum_fill_ratio,
        multiple_mark_margin=multiple_mark_margin,
        inner_radius_scale=inner_radius_scale,
    )
    detections = detect_answers(aligned_image, metadata, settings)
    detection_summary = summarize_detections(detections)
    overlay = draw_detection_overlay(aligned_image, metadata, detections, settings)
    # Exam choices come from metadata so custom labels work in the review dropdown.
    exam = metadata.get("exam", {})
    choices = [str(choice) for choice in exam.get("choices", [])]
    answer_key = metadata.get("answer_key")

    if not answer_key:
        st.warning("This sheet metadata has no answer key, so saved reviews will be stored without a score.")

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Questions", int(exam.get("num_questions", len(detections))))
    metric_b.metric("Marked", detection_summary.get("marked", 0))
    metric_c.metric("Blank", detection_summary.get("blank", 0))
    metric_d.metric("Multiple", detection_summary.get("multiple", 0))

    preview_column, detail_column = st.columns([2, 1])

    # The overlay gives a fast visual audit before the user touches the table.
    with preview_column:
        st.subheader("Detection preview")
        st.image(to_rgb(overlay), use_container_width=True)

    with detail_column:
        st.subheader("Scan details")
        st.write(f"Exam: `{exam.get('exam_id', '')}`")
        st.write(f"Title: {exam.get('title', '')}")
        st.write(f"Sheet ID: `{scan_result.get('sheet_id', metadata.get('sheet_id', ''))}`")
        st.write(f"Metadata: `{scan_result.get('metadata_name', '')}`")
        st.download_button(
            "Download detected answers JSON",
            data=json.dumps(
                {
                    "answers": answers_to_mapping(detections),
                    "statuses": statuses_to_mapping(detections),
                    "detections": detections,
                },
                indent=2,
            ),
            file_name=f"{Path(scan_name).stem}_detections.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()
    st.subheader("Review table")
    review_rows = build_review_rows(detections)

    # The form batches table edits so partial changes do not save accidentally.
    with st.form("review_form"):
        student_left, student_right = st.columns(2)

        with student_left:
            student_name = st.text_input("Student name", value="")

        with student_right:
            student_id = st.text_input("Student ID", value="")

        edited_rows = st.data_editor(
            review_rows,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            disabled=["Question", "Detected", "Status", "Confidence", "Fill ratios"],
            column_config={
                "Reviewed answer": st.column_config.SelectboxColumn(
                    "Reviewed answer",
                    options=[""] + choices,
                    required=False,
                ),
                "Confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
            },
        )
        submitted = st.form_submit_button("Save reviewed result", type="primary", use_container_width=True)

    if submitted:
        # Convert the edited table to clean answer/status mappings before grading.
        reviewed_answers, reviewed_statuses = edited_rows_to_answers(edited_rows, detections)
        grading = grade_answers(
            reviewed_answers,
            answer_key,
            int(exam.get("num_questions", len(detections))),
            reviewed_statuses,
        )
        result_path = result_path_for(scan_name, student_id, student_name)
        overlay_path = result_path.with_name(result_path.stem.replace("_result", "_review_overlay") + ".png")
        cv2.imwrite(str(overlay_path), overlay)
        # The JSON result keeps the full audit trail for this reviewed submission.
        result_data = {
            "result_id": result_path.stem,
            "scan_id": Path(scan_name).stem,
            "scan_json_path": str(SCANNED_DIR / scan_name),
            "result_json_path": str(result_path),
            "review_overlay_path": str(overlay_path),
            "reviewed_at": utc_now(),
            "student_name": student_name.strip(),
            "student_id": student_id.strip(),
            "sheet_id": str(scan_result.get("sheet_id", metadata.get("sheet_id", ""))),
            "metadata_name": str(scan_result.get("metadata_name", "")),
            "metadata": metadata,
            "detection_settings": settings.__dict__,
            "detected_answers": answers_to_mapping(detections),
            "detection_statuses": statuses_to_mapping(detections),
            "reviewed_answers": reviewed_answers,
            "reviewed_statuses": reviewed_statuses,
            "grading": grading,
            "review_rows": edited_rows,
        }
        result_path.write_text(json.dumps(result_data, indent=2) + "\n", encoding="utf-8")
        save_result(result_data, result_path)

        if grading["has_answer_key"]:
            st.success(f"Saved result: {grading['score']} / {grading['total']} ({grading['percentage']}%).")
        else:
            st.success("Saved reviewed answers. No answer key was available for grading.")

        result_rows = [
            {
                "Question": row["question"],
                "Answer": row["answer"],
                "Correct answer": row["correct_answer"],
                "Result": row["result"],
            }
            for row in grading["rows"]
        ]
        st.download_button(
            "Download graded CSV",
            data=csv_text(result_rows, ["Question", "Answer", "Correct answer", "Result"]),
            file_name=f"{result_path.stem}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.page_link("pages/4_Results.py", label="View saved results")

except json.JSONDecodeError:
    st.error("The selected scan file is not valid JSON.")

except ValueError as error:
    st.error(str(error))

except Exception as error:
    st.exception(error)
