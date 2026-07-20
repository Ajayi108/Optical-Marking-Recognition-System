# Detects the four unique ArUco markers on an OMR answer sheet.

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


ARUCO_DICTIONARY_ID = cv2.aruco.DICT_4X4_50
REQUIRED_MARKER_IDS = (0, 1, 2, 3)
MARKER_NAMES = {0: "top_left", 1: "top_right", 2: "bottom_right", 3: "bottom_left"}
MARKER_CORNER_INDEXES = {0: 0, 1: 1, 2: 2, 3: 3}


# Represents marker detection failures.
class MarkerDetectionError(Exception):
    pass


# Checks that the installed OpenCV package supports ArUco markers.
def check_aruco_support() -> None:
    if not hasattr(cv2, "aruco"):
        raise ImportError("ArUco support is missing. Install opencv-contrib-python-headless.")


# Loads an image from a path, uploaded bytes, or an existing NumPy array.
def load_image(source: str | Path | bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    if isinstance(source, np.ndarray):
        image = source.copy()
    elif isinstance(source, (bytes, bytearray, memoryview)):
        image = cv2.imdecode(np.frombuffer(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(str(Path(source)), cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise MarkerDetectionError("The image could not be loaded.")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image


# Creates ArUco detection settings with corner refinement enabled.
def create_detector_parameters() -> Any:
    parameters = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters") else cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 40
    parameters.cornerRefinementMinAccuracy = 0.01
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.minMarkerPerimeterRate = 0.02
    parameters.maxMarkerPerimeterRate = 4.0
    return parameters


# Detects all ArUco markers in an image.
def detect_aruco_markers(image: np.ndarray, dictionary_id: int = ARUCO_DICTIONARY_ID) -> tuple[dict[int, dict[str, Any]], list[np.ndarray]]:
    check_aruco_support()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    parameters = create_detector_parameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        detected_corners, detected_ids, rejected = detector.detectMarkers(gray)
    else:
        detected_corners, detected_ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if detected_ids is None or len(detected_ids) == 0:
        return {}, rejected

    markers: dict[int, dict[str, Any]] = {}

    for raw_corners, raw_id in zip(detected_corners, detected_ids.flatten().tolist()):
        marker_id = int(raw_id)
        corners = np.asarray(raw_corners, dtype=np.float32).reshape(4, 2)
        center = np.mean(corners, axis=0)
        area = abs(float(cv2.contourArea(corners)))
        perimeter = float(cv2.arcLength(corners, True))
        marker = {"id": marker_id, "corners": corners, "center": center, "area": area, "perimeter": perimeter}

        if marker_id not in markers or area > markers[marker_id]["area"]:
            markers[marker_id] = marker

    return markers, rejected


# Returns the required marker IDs that were not detected.
def get_missing_marker_ids(markers: dict[int, dict[str, Any]], required_ids: Sequence[int] = REQUIRED_MARKER_IDS) -> list[int]:
    return [marker_id for marker_id in required_ids if marker_id not in markers]


# Returns marker centers in top-left, top-right, bottom-right, bottom-left order.
def get_ordered_marker_centers(markers: dict[int, dict[str, Any]], required_ids: Sequence[int] = REQUIRED_MARKER_IDS) -> np.ndarray:
    missing_ids = get_missing_marker_ids(markers, required_ids)

    if missing_ids:
        raise MarkerDetectionError(f"Missing ArUco marker IDs: {missing_ids}")

    return np.asarray([markers[marker_id]["center"] for marker_id in required_ids], dtype=np.float32)


# Returns the outside page corner of each ArUco marker.
def get_ordered_page_corners(markers: dict[int, dict[str, Any]], required_ids: Sequence[int] = REQUIRED_MARKER_IDS) -> np.ndarray:
    missing_ids = get_missing_marker_ids(markers, required_ids)

    if missing_ids:
        raise MarkerDetectionError(f"Missing ArUco marker IDs: {missing_ids}")

    return np.asarray([markers[marker_id]["corners"][MARKER_CORNER_INDEXES[marker_id]] for marker_id in required_ids], dtype=np.float32)


# Detects and validates the four required sheet markers.
def detect_sheet_markers(source: str | Path | bytes | bytearray | memoryview | np.ndarray, require_all: bool = True) -> dict[str, Any]:
    image = load_image(source)
    markers, rejected = detect_aruco_markers(image)
    missing_ids = get_missing_marker_ids(markers)

    if require_all and missing_ids:
        detected_ids = sorted(markers.keys())
        raise MarkerDetectionError(f"All four ArUco markers are required. Detected IDs: {detected_ids}. Missing IDs: {missing_ids}.")

    source_centers = get_ordered_marker_centers(markers) if not missing_ids else None
    source_page_corners = get_ordered_page_corners(markers) if not missing_ids else None
    return {"image": image, "markers": markers, "detected_ids": sorted(markers.keys()), "missing_ids": missing_ids, "source_centers": source_centers, "source_page_corners": source_page_corners, "rejected_count": len(rejected)}


# Draws detected marker borders, IDs, centers, and corner names.
def draw_marker_preview(image: np.ndarray, markers: dict[int, dict[str, Any]]) -> np.ndarray:
    preview = image.copy()

    for marker_id, marker in markers.items():
        corners = np.round(marker["corners"]).astype(np.int32)
        center = tuple(np.round(marker["center"]).astype(int))
        label = f"{MARKER_NAMES.get(marker_id, 'unknown')} ID {marker_id}"
        cv2.polylines(preview, [corners], True, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(preview, center, 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(preview, label, (int(corners[0][0]), int(corners[0][1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    return preview


# Converts marker information into JSON-safe values for debugging.
def markers_to_dict(markers: dict[int, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}

    for marker_id, marker in markers.items():
        output[str(marker_id)] = {"name": MARKER_NAMES.get(marker_id, "unknown"), "center": marker["center"].round(3).tolist(), "corners": marker["corners"].round(3).tolist(), "area": round(marker["area"], 3), "perimeter": round(marker["perimeter"], 3)}

    return output