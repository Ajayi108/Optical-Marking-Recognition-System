"""Generate printable A4 OMR answer sheets."""

from __future__ import annotations

import argparse
import json
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas


POINTS_PER_INCH = 72.0
MM_PER_INCH = 25.4
DEFAULT_CHOICES = ("A", "B", "C", "D")


@dataclass(frozen=True)
class SheetArtifacts:
    sheet_id: str
    pdf_path: Path
    metadata_path: Path


class OMRSheetGenerator:
    # The remaining generator implementation follows here.
    ...