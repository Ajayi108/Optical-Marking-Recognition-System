# Perspective alignment helpers for OMR sheets.

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np


class AlignmentError(Exception):
    # Raised when alignment metadata or coordinates are unusable.
    pass


def validate_alignment_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    # Returns validated alignment metadata from a sheet metadata document.
    alignment = metadata.get("alignment")

    # The generator stores all warp instructions under this key.
    if not isinstance(alignment, dict):
        raise AlignmentError("The JSON file does not contain alignment metadata.")

    destination_points = alignment.get("destination_marker_centers_px")
    output_size = alignment.get("warp_target_size_px")

    # Four destination centers match the four required ArUco marker IDs.
    if not isinstance(destination_points, list) or len(destination_points) != 4:
        raise AlignmentError("The JSON file must contain four destination marker centers.")

    # The warp target is stored as [width, height] in pixels.
    if not isinstance(output_size, list) or len(output_size) != 2:
        raise AlignmentError("The JSON file must contain a two-value warp target size.")

    destination = np.asarray(destination_points, dtype=np.float32)

    # OpenCV expects exactly four x/y coordinate pairs for perspective transforms.
    if destination.shape != (4, 2):
        raise AlignmentError("Destination marker centers must be four x/y coordinate pairs.")

    width, height = int(output_size[0]), int(output_size[1])

    if width <= 0 or height <= 0:
        raise AlignmentError("Warp target width and height must be positive.")

    return alignment


def align_sheet(
    image: np.ndarray,
    source_points: Sequence[Sequence[float]] | np.ndarray,
    destination_points: Sequence[Sequence[float]] | np.ndarray,
    output_size: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    # Aligns a scanned sheet using four corresponding marker centers.
    source = np.asarray(source_points, dtype=np.float32)
    destination = np.asarray(destination_points, dtype=np.float32)

    # Source points come from detection; destination points come from metadata.
    if source.shape != (4, 2):
        raise AlignmentError("Source marker centers must be four x/y coordinate pairs.")

    if destination.shape != (4, 2):
        raise AlignmentError("Destination marker centers must be four x/y coordinate pairs.")

    width, height = int(output_size[0]), int(output_size[1])

    if width <= 0 or height <= 0:
        raise AlignmentError("Warp target width and height must be positive.")

    # The transform maps the photographed marker centers back to canonical sheet pixels.
    matrix = cv2.getPerspectiveTransform(source, destination)
    aligned = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return aligned, matrix


def align_sheet_to_metadata(
    image: np.ndarray,
    source_points: Sequence[Sequence[float]] | np.ndarray,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    # Aligns a scanned sheet using marker centers stored in sheet metadata.
    alignment = validate_alignment_metadata(metadata)
    return align_sheet(
        image,
        source_points,
        alignment["destination_marker_centers_px"],
        alignment["warp_target_size_px"],
    )
