# Uploads, detects, aligns, previews, and saves completed OMR sheets.

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = APP_ROOT / "generated_sheets"
SCANNED_DIR = APP_ROOT / "scanned_sheets"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
SCANNED_DIR.mkdir(parents=True, exist_ok=True)

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from omr.marker_detection import MarkerDetectionError, detect_sheet_markers, draw_marker_preview, markers_to_dict

st.set_page_config(page_title="Scan Sheet | OMR", page_icon="📷", layout="wide")


# Converts text into a safe filename.
def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return cleaned.strip("._-") or "scan"


# Converts an OpenCV BGR image to RGB for Streamlit.
def to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Loads metadata from an uploaded JSON file or generated JSON path.
def load_metadata(uploaded_metadata: Any, selected_metadata: str) -> tuple[dict[str, Any], str]:
    if uploaded_metadata is not None:
        metadata = json.loads(uploaded_metadata.getvalue().decode("utf-8"))
        return metadata, uploaded_metadata.name
    if not selected_metadata:
        raise ValueError("Select or upload a metadata JSON file.")
    metadata_path = GENERATED_DIR / selected_metadata
    if not metadata_path.exists():
        raise ValueError("The selected metadata file does not exist.")
    return json.loads(metadata_path.read_text(encoding="utf-8")), metadata_path.name


# Validates that the metadata contains alignment information.
def validate_metadata(metadata: dict[str, Any]) -> None:
    if "alignment" not in metadata:
        raise ValueError("The JSON file does not contain alignment metadata.")
    if "destination_marker_centers_px" not in metadata["alignment"]:
        raise ValueError("The JSON file does not contain destination marker centers.")
    if "warp_target_size_px" not in metadata["alignment"]:
        raise ValueError("The JSON file does not contain a warp target size.")
    if len(metadata["alignment"]["destination_marker_centers_px"]) != 4:
        raise ValueError("The JSON file must contain four destination marker centers.")


# Aligns a tilted sheet using the four detected ArUco marker centers.
def align_sheet(image: np.ndarray, source_points: np.ndarray, destination_points: list[list[int]], output_size: list[int]) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_points, dtype=np.float32)
    destination = np.asarray(destination_points, dtype=np.float32)
    width, height = int(output_size[0]), int(output_size[1])
    matrix = cv2.getPerspectiveTransform(source, destination)
    aligned = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return aligned, matrix


# Saves an OpenCV image and raises an error when saving fails.
def save_image(path: Path, image: np.ndarray) -> None:
    saved = cv2.imwrite(str(path), image)
    if not saved:
        raise OSError(f"Could not save image: {path}")


# Saves scan information for the Review Answers page.
def save_scan_result(sheet_id: str, original: np.ndarray, preview: np.ndarray, aligned: np.ndarray, metadata: dict[str, Any], metadata_name: str, detection: dict[str, Any], matrix: np.ndarray) -> dict[str, str]:
    stem = safe_name(sheet_id)
    original_path = SCANNED_DIR / f"{stem}_original.png"
    preview_path = SCANNED_DIR / f"{stem}_markers.png"
    aligned_path = SCANNED_DIR / f"{stem}_aligned.png"
    result_path = SCANNED_DIR / f"{stem}_scan.json"
    save_image(original_path, original)
    save_image(preview_path, preview)
    save_image(aligned_path, aligned)
    result_data = {"sheet_id": sheet_id, "metadata_name": metadata_name, "original_path": str(original_path), "preview_path": str(preview_path), "aligned_path": str(aligned_path), "detected_ids": detection["detected_ids"], "markers": markers_to_dict(detection["markers"]), "transform_matrix": matrix.round(8).tolist(), "metadata": metadata}
    result_path.write_text(json.dumps(result_data, indent=2) + "\n", encoding="utf-8")
    return {"sheet_id": sheet_id, "original_path": str(original_path), "preview_path": str(preview_path), "aligned_path": str(aligned_path), "result_path": str(result_path), "metadata_name": metadata_name}


st.page_link("app.py", label="Back to home", icon="🏠")
st.title("📷 Scan OMR Sheet")
st.write("Upload a completed answer sheet and select the JSON metadata generated with that sheet.")

with st.sidebar:
    st.header("Scanning requirements")
    st.write("Keep all four ArUco markers visible.")
    st.write("Avoid covering the markers with fingers.")
    st.write("Use good lighting and avoid strong shadows.")
    st.write("The photograph may be rotated or tilted.")

metadata_files = sorted(GENERATED_DIR.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
metadata_names = [path.name for path in metadata_files]

left, right = st.columns(2)

with left:
    uploaded_image = st.file_uploader("Upload completed OMR sheet", type=["jpg", "jpeg", "png", "webp"])

with right:
    selected_metadata = st.selectbox("Select generated metadata", options=[""] + metadata_names, format_func=lambda value: "Choose a generated JSON file" if not value else value)
    uploaded_metadata = st.file_uploader("Or upload metadata JSON", type=["json"])

scan_clicked = st.button("Detect and align sheet", type="primary", use_container_width=True)

if scan_clicked:
    try:
        if uploaded_image is None:
            raise ValueError("Upload a completed OMR sheet image.")

        metadata, metadata_name = load_metadata(uploaded_metadata, selected_metadata)
        validate_metadata(metadata)
        detection = detect_sheet_markers(uploaded_image.getvalue(), require_all=True)
        original = detection["image"]
        preview = draw_marker_preview(original, detection["markers"])
        destination_points = metadata["alignment"]["destination_marker_centers_px"]
        output_size = metadata["alignment"]["warp_target_size_px"]
        aligned, matrix = align_sheet(original, detection["source_centers"], destination_points, output_size)
        sheet_id = str(metadata.get("sheet_id", Path(metadata_name).stem))
        saved = save_scan_result(sheet_id, original, preview, aligned, metadata, metadata_name, detection, matrix)
        st.session_state["scan_result"] = saved
        st.success(f"Sheet aligned successfully. Detected ArUco IDs: {detection['detected_ids']}")
        original_column, marker_column, aligned_column = st.columns(3)

        with original_column:
            st.subheader("Original")
            st.image(to_rgb(original), use_container_width=True)

        with marker_column:
            st.subheader("Detected markers")
            st.image(to_rgb(preview), use_container_width=True)

        with aligned_column:
            st.subheader("Aligned sheet")
            st.image(to_rgb(aligned), use_container_width=True)

        with st.expander("Detection details"):
            st.json({"sheet_id": sheet_id, "metadata": metadata_name, "detected_ids": detection["detected_ids"], "missing_ids": detection["missing_ids"], "rejected_candidates": detection["rejected_count"], "markers": markers_to_dict(detection["markers"]), "transform_matrix": matrix.round(8).tolist()})

    except MarkerDetectionError as error:
        st.error(str(error))
        st.info("Make sure ArUco IDs 0, 1, 2, and 3 are all visible in the photograph.")

    except json.JSONDecodeError:
        st.error("The selected metadata file is not valid JSON.")

    except ValueError as error:
        st.error(str(error))

    except OSError as error:
        st.error(str(error))

    except Exception as error:
        st.exception(error)

saved_scan = st.session_state.get("scan_result")

if saved_scan and Path(saved_scan["aligned_path"]).exists():
    st.divider()
    st.subheader("Latest aligned scan")
    aligned_image = cv2.imread(saved_scan["aligned_path"], cv2.IMREAD_COLOR)
    st.caption(f"Sheet ID: `{saved_scan['sheet_id']}`")
    st.image(to_rgb(aligned_image), use_container_width=True)
    st.download_button("Download aligned image", data=Path(saved_scan["aligned_path"]).read_bytes(), file_name=Path(saved_scan["aligned_path"]).name, mime="image/png", use_container_width=True)