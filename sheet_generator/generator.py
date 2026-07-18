# Generates printable A4 OMR answer sheets with four unique ArUco corner markers.

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader

POINTS_PER_INCH = 72.0
DEFAULT_CHOICES = ("A", "B", "C", "D")
ARUCO_DICTIONARY_NAME = "DICT_4X4_50"
ARUCO_MARKER_IDS = {"top_left": 0, "top_right": 1, "bottom_right": 2, "bottom_left": 3}
ARUCO_IMAGE_SIZE = 400


# Stores the generated PDF path, JSON path, and sheet ID.
@dataclass(frozen=True)
class SheetArtifacts:
    sheet_id: str
    pdf_path: Path
    metadata_path: Path


# Creates printable OMR sheets and matching coordinate metadata.
class OMRSheetGenerator:

    # Creates the output directory and loads the ArUco dictionary.
    def __init__(self, output_dir: str | Path = "generated_sheets") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICTIONARY_NAME))

    # Generates one PDF sheet and one JSON metadata file.
    def generate(self, *, exam_id: str, title: str, num_questions: int, choices: Sequence[str] = DEFAULT_CHOICES, answer_key: Mapping[int | str, str] | None = None, dpi: int = 300, sheet_id: str | None = None, filename_stem: str | None = None) -> SheetArtifacts:
        exam_id = str(exam_id).strip()
        title = str(title).strip()
        choices = tuple(str(choice).strip().upper() for choice in choices)
        self._validate(exam_id, title, num_questions, choices, dpi)
        sheet_id = str(sheet_id).strip() if sheet_id else str(uuid.uuid4())
        stem = self._slug(filename_stem or f"{exam_id}_{sheet_id[:8]}")
        pdf_path = self.output_dir / f"{stem}.pdf"
        metadata_path = self.output_dir / f"{stem}.json"
        page_width, page_height = A4
        layout = self._build_layout(page_width, page_height, num_questions, choices, dpi)
        pdf = Canvas(str(pdf_path), pagesize=A4, pageCompression=1)
        pdf.setTitle(f"{title} - OMR Answer Sheet")
        pdf.setAuthor("OMR Sheet Generator")
        pdf.setSubject(f"OMR answer sheet for exam {exam_id}")
        self._draw_sheet(pdf, exam_id, title, sheet_id, choices, layout)
        pdf.showPage()
        pdf.save()
        metadata: dict[str, Any] = {"schema_version": "2.0", "sheet_id": sheet_id, "exam": {"exam_id": exam_id, "title": title, "num_questions": num_questions, "choices": list(choices)}, "page": layout["page"], "alignment": layout["alignment"], "layout": layout["layout"], "questions": layout["questions"], "detection_defaults": {"recommended_threshold": "OTSU_BINARY_INV", "minimum_fill_ratio": 0.28, "multiple_mark_margin": 0.08}}
        if answer_key is not None:
            metadata["answer_key"] = self._normalise_answer_key(answer_key, num_questions, choices)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return SheetArtifacts(sheet_id=sheet_id, pdf_path=pdf_path, metadata_path=metadata_path)

    # Validates the exam configuration.
    @staticmethod
    def _validate(exam_id: str, title: str, num_questions: int, choices: Sequence[str], dpi: int) -> None:
        if not exam_id:
            raise ValueError("exam_id cannot be empty")
        if not title:
            raise ValueError("title cannot be empty")
        if not isinstance(num_questions, int) or num_questions < 1:
            raise ValueError("num_questions must be a positive integer")
        if not 2 <= len(choices) <= 5:
            raise ValueError("choices must contain between 2 and 5 labels")
        if len(set(choices)) != len(choices):
            raise ValueError("choice labels must be unique")
        if any(not choice or len(choice) > 3 for choice in choices):
            raise ValueError("each choice label must contain 1 to 3 characters")
        if not isinstance(dpi, int) or not 72 <= dpi <= 600:
            raise ValueError("dpi must be an integer between 72 and 600")

    # Calculates ArUco marker, column, row, and bubble positions.
    def _build_layout(self, page_width: float, page_height: float, num_questions: int, choices: Sequence[str], dpi: int) -> dict[str, Any]:
        marker_size = 12 * mm
        marker_quiet_zone = 2 * mm
        marker_inset = 7 * mm
        bubble_radius = 1.8 * mm
        bubble_roi_half = 2.8 * mm
        answer_left = 16 * mm
        answer_right = page_width - 16 * mm
        answer_top_from_top = 76 * mm
        answer_bottom_from_top = 278 * mm
        row_pitch = 5.7 * mm
        question_number_width = 9 * mm
        choice_pitch = 7.5 * mm
        column_gap = 4 * mm
        answer_height = answer_bottom_from_top - answer_top_from_top
        rows_per_column = int(math.floor(answer_height / row_pitch))
        required_columns = int(math.ceil(num_questions / rows_per_column))
        column_width = question_number_width + len(choices) * choice_pitch
        available_width = answer_right - answer_left
        max_columns = int(math.floor((available_width + column_gap) / (column_width + column_gap)))

        if required_columns > max_columns:
            raise ValueError(f"{num_questions} questions do not fit this A4 layout with {len(choices)} choices. Maximum is {rows_per_column * max_columns}.")

        used_width = required_columns * column_width + (required_columns - 1) * column_gap
        first_column_x = answer_left + (available_width - used_width) / 2
        marker_rects = {"top_left": (marker_inset + marker_quiet_zone, page_height - marker_inset - marker_quiet_zone - marker_size, marker_size, marker_size), "top_right": (page_width - marker_inset - marker_quiet_zone - marker_size, page_height - marker_inset - marker_quiet_zone - marker_size, marker_size, marker_size), "bottom_right": (page_width - marker_inset - marker_quiet_zone - marker_size, marker_inset + marker_quiet_zone, marker_size, marker_size), "bottom_left": (marker_inset + marker_quiet_zone, marker_inset + marker_quiet_zone, marker_size, marker_size)}
        marker_order = ("top_left", "top_right", "bottom_right", "bottom_left")
        markers: dict[str, Any] = {}
        destination_marker_centers_px: list[list[int]] = []

        for marker_name in marker_order:
            x, y, width, height = marker_rects[marker_name]
            marker_id = ARUCO_MARKER_IDS[marker_name]
            center_x = x + width / 2
            center_y = y + height / 2
            center_px = self._point_to_px(center_x, center_y, page_height, dpi)
            destination_marker_centers_px.append(center_px)
            markers[marker_name] = {"id": marker_id, "dictionary": ARUCO_DICTIONARY_NAME, "rect_pt": self._round_values((x, y, width, height)), "rect_px": self._rect_to_px(x, y, width, height, page_height, dpi), "center_pt": self._round_values((center_x, center_y)), "center_px": center_px}

        questions: list[dict[str, Any]] = []

        for question_index in range(num_questions):
            column_index = question_index // rows_per_column
            row_index = question_index % rows_per_column
            question_number = question_index + 1
            column_x = first_column_x + column_index * (column_width + column_gap)
            center_y = page_height - answer_top_from_top - row_index * row_pitch
            bubbles: list[dict[str, Any]] = []

            for choice_index, choice in enumerate(choices):
                center_x = column_x + question_number_width + choice_index * choice_pitch + choice_pitch / 2
                roi_x = center_x - bubble_roi_half
                roi_y = center_y - bubble_roi_half
                bubbles.append({"choice": choice, "center_pt": self._round_values((center_x, center_y)), "center_px": self._point_to_px(center_x, center_y, page_height, dpi), "radius_pt": round(bubble_radius, 3), "radius_px": round(bubble_radius / POINTS_PER_INCH * dpi, 2), "roi_pt": self._round_values((roi_x, roi_y, bubble_roi_half * 2, bubble_roi_half * 2)), "roi_px": self._rect_to_px(roi_x, roi_y, bubble_roi_half * 2, bubble_roi_half * 2, page_height, dpi)})

            questions.append({"question": question_number, "column": column_index, "row": row_index, "bubbles": bubbles})

        page_size_px = [round(page_width / POINTS_PER_INCH * dpi), round(page_height / POINTS_PER_INCH * dpi)]
        page = {"size": "A4", "width_mm": round(page_width / mm, 3), "height_mm": round(page_height / mm, 3), "width_pt": round(page_width, 3), "height_pt": round(page_height, 3), "dpi": dpi, "size_px": page_size_px, "pdf_origin": "bottom-left", "pixel_origin": "top-left"}
        alignment = {"type": "aruco", "dictionary": ARUCO_DICTIONARY_NAME, "marker_ids": ARUCO_MARKER_IDS, "marker_order": list(marker_order), "markers": markers, "destination_marker_centers_px": destination_marker_centers_px, "warp_target_size_px": page_size_px}
        layout_info = {"columns": required_columns, "rows_per_column": rows_per_column, "row_pitch_mm": round(row_pitch / mm, 3), "bubble_radius_mm": round(bubble_radius / mm, 3), "bubble_roi_size_mm": round(bubble_roi_half * 2 / mm, 3), "aruco_marker_size_mm": round(marker_size / mm, 3), "aruco_quiet_zone_mm": round(marker_quiet_zone / mm, 3)}
        drawing = {"marker_rects": marker_rects, "marker_quiet_zone": marker_quiet_zone, "bubble_radius": bubble_radius, "first_column_x": first_column_x, "column_width": column_width, "column_gap": column_gap, "question_number_width": question_number_width, "choice_pitch": choice_pitch, "answer_top_from_top": answer_top_from_top, "row_pitch": row_pitch, "rows_per_column": rows_per_column}

        return {"page": page, "alignment": alignment, "layout": layout_info, "questions": questions, "drawing": drawing}

    # Draws the complete sheet.
    def _draw_sheet(self, pdf: Canvas, exam_id: str, title: str, sheet_id: str, choices: Sequence[str], layout: Mapping[str, Any]) -> None:
        page_width = layout["page"]["width_pt"]
        page_height = layout["page"]["height_pt"]
        drawing = layout["drawing"]
        self._draw_aruco_markers(pdf, drawing["marker_rects"], drawing["marker_quiet_zone"])
        self._draw_header(pdf, page_width, page_height, exam_id, title, sheet_id)
        self._draw_answer_grid(pdf, page_height, len(layout["questions"]), choices, drawing)
        self._draw_footer(pdf, page_width, sheet_id)

    # Creates one ArUco marker image.
    def _create_aruco_marker(self, marker_id: int) -> ImageReader:
        if hasattr(cv2.aruco, "generateImageMarker"):
            marker = cv2.aruco.generateImageMarker(self.aruco_dictionary, marker_id, ARUCO_IMAGE_SIZE)
        else:
            marker = np.zeros((ARUCO_IMAGE_SIZE, ARUCO_IMAGE_SIZE), dtype=np.uint8)
            cv2.aruco.drawMarker(self.aruco_dictionary, marker_id, ARUCO_IMAGE_SIZE, marker, 1)

        encoded, png = cv2.imencode(".png", marker)

        if not encoded:
            raise RuntimeError(f"Could not encode ArUco marker {marker_id}")

        return ImageReader(BytesIO(png.tobytes()))

    # Draws four unique ArUco markers with white quiet zones.
    def _draw_aruco_markers(self, pdf: Canvas, marker_rects: Mapping[str, Sequence[float]], quiet_zone: float) -> None:
        for marker_name, rect in marker_rects.items():
            x, y, width, height = rect
            pdf.setFillColorRGB(1, 1, 1)
            pdf.rect(x - quiet_zone, y - quiet_zone, width + quiet_zone * 2, height + quiet_zone * 2, stroke=0, fill=1)
            pdf.drawImage(self._create_aruco_marker(ARUCO_MARKER_IDS[marker_name]), x, y, width=width, height=height, preserveAspectRatio=True, mask="auto")

    # Draws the title, student fields, and instructions.
    @staticmethod
    def _draw_header(pdf: Canvas, page_width: float, page_height: float, exam_id: str, title: str, sheet_id: str) -> None:
        pdf.setFillColorRGB(0, 0, 0)
        pdf.setStrokeColorRGB(0, 0, 0)
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawCentredString(page_width / 2, page_height - 19 * mm, title[:80])
        pdf.setFont("Helvetica", 8)
        pdf.drawCentredString(page_width / 2, page_height - 25 * mm, f"Exam ID: {exam_id}    Sheet ID: {sheet_id}")

        box_left = 23 * mm
        box_right = page_width - 23 * mm
        box_top = page_height - 31 * mm
        box_bottom = page_height - 52 * mm
        split_x = box_left + 112 * mm
        baseline = box_bottom + 6 * mm

        pdf.setLineWidth(0.8)
        pdf.rect(box_left, box_bottom, box_right - box_left, box_top - box_bottom, stroke=1, fill=0)
        pdf.line(split_x, box_bottom, split_x, box_top)
        pdf.setFont("Helvetica", 9)
        pdf.drawString(box_left + 3 * mm, box_top - 6 * mm, "Student name:")
        pdf.drawString(split_x + 3 * mm, box_top - 6 * mm, "Student ID:")
        pdf.setStrokeGray(0.4)
        pdf.setLineWidth(0.5)
        pdf.line(box_left + 28 * mm, baseline, split_x - 4 * mm, baseline)
        pdf.line(split_x + 25 * mm, baseline, box_right - 4 * mm, baseline)
        pdf.setStrokeGray(0)
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawString(23 * mm, page_height - 60 * mm, "Instructions")
        pdf.setFont("Helvetica", 8)
        pdf.drawString(43 * mm, page_height - 60 * mm, "Fill one bubble per question completely. Use a dark pen or pencil. Keep the four ArUco markers clear.")

    # Draws choice labels, question numbers, and bubbles.
    @staticmethod
    def _draw_answer_grid(pdf: Canvas, page_height: float, num_questions: int, choices: Sequence[str], drawing: Mapping[str, Any]) -> None:
        first_column_x = drawing["first_column_x"]
        column_width = drawing["column_width"]
        column_gap = drawing["column_gap"]
        question_number_width = drawing["question_number_width"]
        choice_pitch = drawing["choice_pitch"]
        answer_top_from_top = drawing["answer_top_from_top"]
        row_pitch = drawing["row_pitch"]
        rows_per_column = drawing["rows_per_column"]
        bubble_radius = drawing["bubble_radius"]
        columns = int(math.ceil(num_questions / rows_per_column))
        header_y = page_height - answer_top_from_top + 4.5 * mm

        pdf.setFillColorRGB(0, 0, 0)
        pdf.setStrokeColorRGB(0, 0, 0)
        pdf.setFont("Helvetica-Bold", 7)

        for column_index in range(columns):
            column_x = first_column_x + column_index * (column_width + column_gap)

            for choice_index, choice in enumerate(choices):
                center_x = column_x + question_number_width + choice_index * choice_pitch + choice_pitch / 2
                pdf.drawCentredString(center_x, header_y, choice)

        pdf.setLineWidth(0.8)

        for question_index in range(num_questions):
            column_index = question_index // rows_per_column
            row_index = question_index % rows_per_column
            question_number = question_index + 1
            column_x = first_column_x + column_index * (column_width + column_gap)
            center_y = page_height - answer_top_from_top - row_index * row_pitch
            label = f"{question_number}."
            label_width = stringWidth(label, "Helvetica-Bold", 7)

            pdf.setFillColorRGB(0, 0, 0)
            pdf.drawString(column_x + question_number_width - label_width - 1.5 * mm, center_y - 2.2, label)

            for choice_index in range(len(choices)):
                center_x = column_x + question_number_width + choice_index * choice_pitch + choice_pitch / 2
                pdf.setFillColorRGB(1, 1, 1)
                pdf.circle(center_x, center_y, bubble_radius, stroke=1, fill=1)

    # Draws the sheet ID and warning.
    @staticmethod
    def _draw_footer(pdf: Canvas, page_width: float, sheet_id: str) -> None:
        pdf.setFillColorRGB(0, 0, 0)
        pdf.setFont("Helvetica", 6.5)
        pdf.drawCentredString(page_width / 2, 10 * mm, f"OMR sheet {sheet_id} - Do not crop, fold, or cover the four ArUco markers.")

    # Validates and converts the answer key for JSON storage.
    @staticmethod
    def _normalise_answer_key(answer_key: Mapping[int | str, str], num_questions: int, choices: Sequence[str]) -> dict[str, str]:
        normalised: dict[str, str] = {}
        valid_choices = set(choices)

        for raw_question, raw_choice in answer_key.items():
            try:
                question = int(raw_question)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid answer-key question: {raw_question!r}") from error

            choice = str(raw_choice).strip().upper()

            if not 1 <= question <= num_questions:
                raise ValueError(f"Answer-key question {question} is out of range")

            if choice not in valid_choices:
                raise ValueError(f"Invalid choice {choice!r} for question {question}")

            normalised[str(question)] = choice

        return dict(sorted(normalised.items(), key=lambda item: int(item[0])))

    # Converts text into a safe filename.
    @staticmethod
    def _slug(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
        return cleaned.strip("._-") or "exam"

    # Rounds coordinate values for JSON output.
    @staticmethod
    def _round_values(values: Sequence[float]) -> list[float]:
        return [round(float(value), 3) for value in values]

    # Converts a PDF point to top-left image pixels.
    @staticmethod
    def _point_to_px(x: float, y: float, page_height: float, dpi: int) -> list[int]:
        return [round(x / POINTS_PER_INCH * dpi), round((page_height - y) / POINTS_PER_INCH * dpi)]

    # Converts a PDF rectangle to top-left image pixel bounds.
    @classmethod
    def _rect_to_px(cls, x: float, y: float, width: float, height: float, page_height: float, dpi: int) -> list[int]:
        top_left = cls._point_to_px(x, y + height, page_height, dpi)
        bottom_right = cls._point_to_px(x + width, y, page_height, dpi)
        return [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]