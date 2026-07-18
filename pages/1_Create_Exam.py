"""Create an exam and generate its OMR answer sheet."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

import streamlit as st


APP_ROOT = Path(__file__).resolve().parents[1]

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from sheet_generator.generator import OMRSheetGenerator


OUTPUT_DIR = APP_ROOT / "generated_sheets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Create Exam | OMR",
    page_icon="➕",
    layout="wide",
)


def parse_choices(raw: str) -> tuple[str, ...]:
    """Parse and validate comma-separated choice labels."""

    choices = tuple(
        item.strip().upper()
        for item in raw.split(",")
        if item.strip()
    )

    if not 2 <= len(choices) <= 5:
        raise ValueError("Enter between 2 and 5 choice labels.")

    if len(set(choices)) != len(choices):
        raise ValueError("Choice labels must be unique.")

    if any(len(choice) > 3 for choice in choices):
        raise ValueError(
            "Each choice label may contain at most 3 characters."
        )

    return choices


def parse_answer_key(
    raw: str,
    num_questions: int,
    choices: Sequence[str],
) -> Mapping[int, str] | None:
    """Parse a sequential or numbered answer key.

    Accepted formats:

    A, B, C, D
    A B C D
    1:A, 2:B, 3:C, 4:D
    """

    value = raw.strip().upper()

    if not value:
        return None

    valid_choices = set(choices)

    numbered_matches = re.findall(
        r"(\d+)\s*[:=]\s*([A-Z0-9]+)",
        value,
    )

    if numbered_matches:
        # Check that all supplied text was understood.
        remainder = re.sub(
            r"\d+\s*[:=]\s*[A-Z0-9]+",
            "",
            value,
        )
        remainder = re.sub(r"[\s,;]+", "", remainder)

        if remainder:
            raise ValueError(
                "The numbered key contains unrecognised text. "
                "Use entries such as 1:A, 2:B."
            )

        answer_key: dict[int, str] = {}

        for question_text, choice in numbered_matches:
            question = int(question_text)

            if not 1 <= question <= num_questions:
                raise ValueError(
                    f"Question {question} is outside the exam range."
                )

            if question in answer_key:
                raise ValueError(
                    f"Question {question} appears more than once."
                )

            if choice not in valid_choices:
                raise ValueError(
                    f"{choice!r} is not one of the configured choices."
                )

            answer_key[question] = choice

        return answer_key

    sequential = [
        token
        for token in re.split(r"[\s,;]+", value)
        if token
    ]

    if len(sequential) != num_questions:
        raise ValueError(
            f"The sequential answer key needs {num_questions} "
            f"answers; {len(sequential)} were provided."
        )

    invalid = [
        choice
        for choice in sequential
        if choice not in valid_choices
    ]

    if invalid:
        raise ValueError(
            f"Invalid answer {invalid[0]!r}. "
            f"Valid choices are: {', '.join(choices)}."
        )

    return {
        index: choice
        for index, choice in enumerate(sequential, start=1)
    }


def remember_artifacts(
    pdf_path: Path,
    metadata_path: Path,
    sheet_id: str,
) -> None:
    """Remember generated files between Streamlit reruns."""

    st.session_state["created_exam_artifacts"] = {
        "pdf_path": str(pdf_path),
        "metadata_path": str(metadata_path),
        "sheet_id": sheet_id,
    }


st.page_link(
    "app.py",
    label="Back to home",
    icon="🏠",
)

st.title("➕ Create Exam")

st.write(
    "Enter the exam details below. The application will create "
    "one printable PDF and one JSON coordinate file."
)

with st.sidebar:
    st.header("Answer-key formats")

    st.code(
        "A, B, C, D",
        language=None,
    )

    st.code(
        "1:A, 2:B, 3:C, 4:D",
        language=None,
    )

    st.caption(
        "Leave the answer key blank when you only want "
        "a printable sheet."
    )

left, right = st.columns([3, 2])

with left:
    exam_title = st.text_input(
        "Exam title",
        placeholder="Mathematics Midterm",
    )

    exam_id = st.text_input(
        "Exam ID",
        placeholder="MATH-101-MIDTERM",
        help=(
            "Use a stable ID that can also be stored "
            "in the database later."
        ),
    )

    field_a, field_b, field_c = st.columns(3)

    with field_a:
        num_questions = st.number_input(
            "Questions",
            min_value=1,
            max_value=100,
            value=20,
            step=1,
        )

    with field_b:
        choice_text = st.text_input(
            "Choice labels",
            value="A,B,C,D",
            help="Enter 2 to 5 comma-separated labels.",
        )

    with field_c:
        dpi = st.selectbox(
            "Canonical DPI",
            options=(150, 200, 300, 400),
            index=2,
            help="300 DPI is recommended for printed sheets.",
        )

with right:
    answer_key_text = st.text_area(
        "Answer key (optional)",
        height=180,
        placeholder="A, C, B, D, A, ...",
        help=(
            "Enter one answer per question, or use numbered "
            "entries such as 1:A, 2:C."
        ),
    )

    include_key = st.checkbox(
        "Store the answer key in the JSON metadata",
        value=True,
        disabled=not bool(answer_key_text.strip()),
    )

st.divider()

generate_clicked = st.button(
    "Generate answer sheet",
    type="primary",
    use_container_width=True,
)

if generate_clicked:
    try:
        clean_title = exam_title.strip()
        clean_exam_id = exam_id.strip()

        if not clean_title:
            raise ValueError("Enter an exam title.")

        if not clean_exam_id:
            raise ValueError("Enter an exam ID.")

        choices = parse_choices(choice_text)

        answer_key = parse_answer_key(
            answer_key_text,
            int(num_questions),
            choices,
        )

        if not include_key:
            answer_key = None

        generator = OMRSheetGenerator(OUTPUT_DIR)

        artifacts = generator.generate(
            exam_id=clean_exam_id,
            title=clean_title,
            num_questions=int(num_questions),
            choices=choices,
            answer_key=answer_key,
            dpi=int(dpi),
        )

        remember_artifacts(
            artifacts.pdf_path,
            artifacts.metadata_path,
            artifacts.sheet_id,
        )

        st.success(
            "The answer sheet and coordinate metadata "
            "were generated."
        )

    except ValueError as exc:
        st.error(str(exc))

    except OSError as exc:
        st.error(
            f"The files could not be written: {exc}"
        )

    except Exception as exc:
        st.exception(exc)


saved = st.session_state.get(
    "created_exam_artifacts"
)

if saved:
    pdf_path = Path(saved["pdf_path"])
    metadata_path = Path(saved["metadata_path"])

    if pdf_path.exists() and metadata_path.exists():
        st.subheader("Generated files")

        st.caption(
            f"Sheet ID: `{saved['sheet_id']}`"
        )

        download_pdf, download_json = st.columns(2)

        with download_pdf:
            st.download_button(
                "Download answer sheet PDF",
                data=pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
                use_container_width=True,
            )

        with download_json:
            st.download_button(
                "Download coordinate JSON",
                data=metadata_path.read_bytes(),
                file_name=metadata_path.name,
                mime="application/json",
                use_container_width=True,
            )

        with st.expander(
            "Inspect generated metadata"
        ):
            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )
            st.json(metadata)

    else:
        st.warning(
            "The previously generated files "
            "are no longer available on disk."
        )