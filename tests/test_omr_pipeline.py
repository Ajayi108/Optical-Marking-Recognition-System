from __future__ import annotations

import unittest

import cv2
import numpy as np

from omr.alignment import align_sheet
from omr.bubble_detection import DetectionSettings, detect_answers
from omr.grading import grade_answers


class OMRPipelineTests(unittest.TestCase):
    def test_alignment_identity_transform_keeps_page_size(self) -> None:
        # Identity alignment should keep a clean page at the requested output size.
        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        source = np.array([[0, 0], [199, 0], [199, 99], [0, 99]], dtype=np.float32)
        destination = source.copy()

        aligned, matrix = align_sheet(image, source, destination, [200, 100])

        self.assertEqual(aligned.shape[:2], (100, 200))
        self.assertEqual(matrix.shape, (3, 3))

    def test_bubble_detection_finds_marked_answer(self) -> None:
        # The synthetic sheet has one filled bubble and three printed outlines.
        image = np.full((120, 220, 3), 255, dtype=np.uint8)
        centers = {"A": [40, 40], "B": [80, 40], "C": [120, 40], "D": [160, 40]}

        for center in centers.values():
            # Printed outlines should not count as selected answers by themselves.
            cv2.circle(image, tuple(center), 11, (0, 0, 0), 1, cv2.LINE_AA)

        # Fill the B bubble dark enough to pass the minimum fill ratio.
        cv2.circle(image, tuple(centers["B"]), 8, (0, 0, 0), -1, cv2.LINE_AA)

        metadata = {
            "questions": [
                {
                    "question": 1,
                    "bubbles": [
                        {
                            "choice": choice,
                            "center_px": center,
                            "radius_px": 11,
                            "roi_px": [center[0] - 16, center[1] - 16, center[0] + 16, center[1] + 16],
                        }
                        for choice, center in centers.items()
                    ],
                }
            ]
        }

        detections = detect_answers(image, metadata, DetectionSettings(minimum_fill_ratio=0.25))

        self.assertEqual(detections[0]["selected_choice"], "B")
        self.assertEqual(detections[0]["status"], "marked")

    def test_bubble_detection_flags_multiple_filled_answers(self) -> None:
        # Two filled bubbles should be flagged for human review instead of guessed.
        image = np.full((120, 220, 3), 255, dtype=np.uint8)
        centers = {"A": [40, 40], "B": [80, 40], "C": [120, 40], "D": [160, 40]}

        for center in centers.values():
            cv2.circle(image, tuple(center), 11, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.circle(image, tuple(centers["A"]), 8, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, tuple(centers["C"]), 7, (0, 0, 0), -1, cv2.LINE_AA)

        metadata = {
            "questions": [
                {
                    "question": 1,
                    "bubbles": [
                        {
                            "choice": choice,
                            "center_px": center,
                            "radius_px": 11,
                            "roi_px": [center[0] - 16, center[1] - 16, center[0] + 16, center[1] + 16],
                        }
                        for choice, center in centers.items()
                    ],
                }
            ]
        }

        detections = detect_answers(image, metadata, DetectionSettings(minimum_fill_ratio=0.25))

        self.assertIsNone(detections[0]["selected_choice"])
        self.assertEqual(detections[0]["status"], "multiple")

    def test_grading_counts_correct_incorrect_blank_and_multiple(self) -> None:
        # Mixed answers should produce both a score and useful status counts.
        grading = grade_answers(
            {"1": "A", "2": "C"},
            {"1": "A", "2": "B", "3": "D", "4": "A"},
            4,
            {"4": "multiple"},
        )

        self.assertEqual(grading["score"], 1)
        self.assertEqual(grading["total"], 4)
        self.assertEqual(grading["status_counts"]["correct"], 1)
        self.assertEqual(grading["status_counts"]["incorrect"], 1)
        self.assertEqual(grading["status_counts"]["blank"], 1)
        self.assertEqual(grading["status_counts"]["multiple"], 1)


if __name__ == "__main__":
    unittest.main()
