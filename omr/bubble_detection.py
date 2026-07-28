# Bubble fill detection for aligned OMR sheets.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionSettings:
    # Thresholds used to classify filled bubbles.
    minimum_fill_ratio: float = 0.28
    multiple_mark_margin: float = 0.08
    inner_radius_scale: float = 0.72


class BubbleDetectionError(Exception):
    # Raised when bubble detection cannot be completed.
    pass


def settings_from_metadata(metadata: Mapping[str, Any]) -> DetectionSettings:
    # Builds detection settings from metadata defaults.
    defaults = metadata.get("detection_defaults", {})

    # Older metadata files may not have every tuning value, so each field has a default.
    return DetectionSettings(
        minimum_fill_ratio=float(defaults.get("minimum_fill_ratio", 0.28)),
        multiple_mark_margin=float(defaults.get("multiple_mark_margin", 0.08)),
    )


def threshold_marks(image: np.ndarray) -> np.ndarray:
    # Creates a binary mask of dark marks on a sheet.
    if image is None or image.size == 0:
        raise BubbleDetectionError("The aligned sheet image could not be loaded.")

    # OMR marks are detected as dark pixels, so grayscale plus inverted Otsu works well.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def _clamp_rect(rect: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    # Keeps generated ROI coordinates inside the actual aligned image bounds.
    x1, y1, x2, y2 = [int(value) for value in rect]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def measure_bubble(mask: np.ndarray, bubble: Mapping[str, Any], settings: DetectionSettings) -> dict[str, Any]:
    # Measures how much of a single bubble's inner area is dark.
    height, width = mask.shape[:2]
    roi = bubble.get("roi_px")
    center = bubble.get("center_px")

    # The generator writes both the rectangular ROI and the exact bubble center.
    if not isinstance(roi, list) or len(roi) != 4:
        raise BubbleDetectionError("Bubble metadata is missing roi_px coordinates.")

    if not isinstance(center, list) or len(center) != 2:
        raise BubbleDetectionError("Bubble metadata is missing center_px coordinates.")

    x1, y1, x2, y2 = _clamp_rect(roi, width, height)

    # A bad or out-of-frame ROI should not crash review; it becomes an empty measure.
    if x2 <= x1 or y2 <= y1:
        return {
            "choice": str(bubble.get("choice", "")),
            "fill_ratio": 0.0,
            "dark_pixels": 0,
            "area_pixels": 0,
            "center_px": center,
            "roi_px": roi,
        }

    roi_mask = mask[y1:y2, x1:x2]
    local_center = (int(round(float(center[0]) - x1)), int(round(float(center[1]) - y1)))
    radius_px = float(bubble.get("radius_px", min(x2 - x1, y2 - y1) / 2))
    inner_radius = max(3, int(round(radius_px * settings.inner_radius_scale)))
    circle_mask = np.zeros(roi_mask.shape[:2], dtype=np.uint8)
    cv2.circle(circle_mask, local_center, inner_radius, 255, -1, cv2.LINE_AA)
    area_pixels = int(np.count_nonzero(circle_mask))

    # Only the inside of the bubble is measured to avoid counting printed outlines.
    if area_pixels == 0:
        fill_ratio = 0.0
        dark_pixels = 0
    else:
        dark_pixels = int(np.count_nonzero(roi_mask[circle_mask > 0]))
        fill_ratio = dark_pixels / area_pixels

    return {
        "choice": str(bubble.get("choice", "")),
        "fill_ratio": round(float(fill_ratio), 4),
        "dark_pixels": dark_pixels,
        "area_pixels": area_pixels,
        "center_px": center,
        "roi_px": roi,
    }


def classify_question(measurements: list[dict[str, Any]], settings: DetectionSettings) -> dict[str, Any]:
    # Classifies a question as marked, blank, or multiple.
    ranked = sorted(measurements, key=lambda item: item["fill_ratio"], reverse=True)
    best = ranked[0] if ranked else None
    second = ranked[1] if len(ranked) > 1 else None

    # Anything below the minimum fill threshold is treated as intentionally blank.
    if best is None or best["fill_ratio"] < settings.minimum_fill_ratio:
        return {"selected_choice": None, "status": "blank", "confidence": 0.0}

    second_ratio = second["fill_ratio"] if second else 0.0
    # Two filled bubbles should be sent to review instead of guessed automatically.
    has_filled_second = second_ratio >= settings.minimum_fill_ratio
    # A close second is also suspicious, even if it is barely under the threshold.
    has_close_second = (
        second_ratio >= max(0.0, settings.minimum_fill_ratio - settings.multiple_mark_margin)
        and (best["fill_ratio"] - second_ratio) <= settings.multiple_mark_margin
    )

    if has_filled_second or has_close_second:
        return {
            "selected_choice": None,
            "status": "multiple",
            "confidence": round(float(best["fill_ratio"] - second_ratio), 4),
        }

    # Confidence is the separation between the strongest and second-strongest marks.
    return {
        "selected_choice": best["choice"],
        "status": "marked",
        "confidence": round(float(best["fill_ratio"] - second_ratio), 4),
    }


def detect_answers(
    aligned_image: np.ndarray,
    metadata: Mapping[str, Any],
    settings: DetectionSettings | None = None,
) -> list[dict[str, Any]]:
    # Detects answers for every question in an aligned sheet.
    questions = metadata.get("questions")

    # Bubble coordinates are the contract between the sheet generator and scanner.
    if not isinstance(questions, list) or not questions:
        raise BubbleDetectionError("The metadata does not contain question bubble coordinates.")

    active_settings = settings or settings_from_metadata(metadata)
    mask = threshold_marks(aligned_image)
    detections: list[dict[str, Any]] = []

    # Each question is scored independently so review can show per-row diagnostics.
    for question in questions:
        bubbles = question.get("bubbles", [])

        if not bubbles:
            raise BubbleDetectionError(f"Question {question.get('question')} has no bubble coordinates.")

        measurements = [measure_bubble(mask, bubble, active_settings) for bubble in bubbles]
        classification = classify_question(measurements, active_settings)
        # Keep raw bubble measurements so the UI can show fill ratios beside choices.
        detections.append(
            {
                "question": int(question["question"]),
                "selected_choice": classification["selected_choice"],
                "status": classification["status"],
                "confidence": classification["confidence"],
                "bubbles": measurements,
            }
        )

    return detections


def answers_to_mapping(detections: list[Mapping[str, Any]]) -> dict[str, str]:
    # Returns selected answers as a JSON-safe question-to-choice mapping.
    answers: dict[str, str] = {}

    for detection in detections:
        selected = detection.get("selected_choice")

        if selected:
            answers[str(detection["question"])] = str(selected)

    return answers


def statuses_to_mapping(detections: list[Mapping[str, Any]]) -> dict[str, str]:
    # Returns detection statuses as a JSON-safe question-to-status mapping.
    return {str(detection["question"]): str(detection["status"]) for detection in detections}


def summarize_detections(detections: list[Mapping[str, Any]]) -> dict[str, int]:
    # Counts marked, blank, and multiple detections.
    summary = {"marked": 0, "blank": 0, "multiple": 0}

    for detection in detections:
        status = str(detection.get("status", "blank"))
        summary[status] = summary.get(status, 0) + 1

    return summary


def draw_detection_overlay(
    image: np.ndarray,
    metadata: Mapping[str, Any],
    detections: list[Mapping[str, Any]],
    settings: DetectionSettings | None = None,
) -> np.ndarray:
    # Draws detected answer states over an aligned sheet.
    active_settings = settings or settings_from_metadata(metadata)
    overlay = image.copy()
    by_question = {int(detection["question"]): detection for detection in detections}

    # The overlay reuses generated coordinates so visual review matches detection exactly.
    for question in metadata.get("questions", []):
        question_number = int(question["question"])
        detection = by_question.get(question_number)
        selected = detection.get("selected_choice") if detection else None
        status = detection.get("status") if detection else "blank"
        measured = {item["choice"]: item for item in detection.get("bubbles", [])} if detection else {}

        for bubble in question.get("bubbles", []):
            choice = str(bubble["choice"])
            center = tuple(int(round(value)) for value in bubble["center_px"])
            radius = max(8, int(round(float(bubble["radius_px"]) * 1.25)))
            measurement = measured.get(choice, {})
            fill_ratio = float(measurement.get("fill_ratio", 0.0))
            color = (165, 165, 165)
            thickness = 2

            # Green is an accepted mark, red is a possible multiple, orange is blank.
            if selected == choice:
                color = (60, 160, 60)
                thickness = 4
            elif status == "multiple" and fill_ratio >= active_settings.minimum_fill_ratio:
                color = (40, 40, 220)
                thickness = 4
            elif status == "blank":
                color = (0, 165, 255)

            cv2.circle(overlay, center, radius, color, thickness, cv2.LINE_AA)

    return overlay
