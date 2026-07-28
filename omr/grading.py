# Answer grading helpers for OMR results.

from __future__ import annotations

from typing import Any, Mapping


def normalise_mapping(values: Mapping[Any, Any] | None) -> dict[int, str]:
    # Converts question-keyed mappings to int-to-string values.
    output: dict[int, str] = {}

    if not values:
        return output

    for raw_question, raw_value in values.items():
        # Metadata JSON stores question numbers as strings; grading wants ints.
        try:
            question = int(raw_question)
        except (TypeError, ValueError):
            continue

        value = str(raw_value).strip().upper()

        if value:
            output[question] = value

    return output


def normalise_statuses(values: Mapping[Any, Any] | None) -> dict[int, str]:
    # Converts question-keyed status mappings without changing answer case semantics.
    output: dict[int, str] = {}

    if not values:
        return output

    for raw_question, raw_value in values.items():
        try:
            question = int(raw_question)
        except (TypeError, ValueError):
            continue

        value = str(raw_value).strip().lower()

        if value:
            output[question] = value

    return output


def grade_answers(
    answers: Mapping[Any, Any],
    answer_key: Mapping[Any, Any] | None,
    num_questions: int,
    answer_statuses: Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    # Grades reviewed answers against an optional answer key.
    normalised_answers = normalise_mapping(answers)
    normalised_key = normalise_mapping(answer_key)
    statuses = normalise_statuses(answer_statuses)
    rows: list[dict[str, Any]] = []
    counts = {"correct": 0, "incorrect": 0, "blank": 0, "multiple": 0, "ungraded": 0}

    # Walk every question so blanks and missing answers are counted explicitly.
    for question in range(1, int(num_questions) + 1):
        selected = normalised_answers.get(question)
        correct_answer = normalised_key.get(question)
        detection_status = statuses.get(question, "")

        if detection_status == "multiple" and not selected:
            result = "multiple"
        elif not selected:
            result = "blank" if correct_answer else "ungraded"
        elif correct_answer is None:
            result = "ungraded"
        elif selected == correct_answer:
            result = "correct"
        else:
            result = "incorrect"

        # Keep both aggregate counts and row-level detail for exports.
        counts[result] = counts.get(result, 0) + 1
        rows.append(
            {
                "question": question,
                "answer": selected or "",
                "correct_answer": correct_answer or "",
                "result": result,
            }
        )

    total = len(normalised_key)
    score = counts["correct"] if total else None
    percentage = round((score / total) * 100, 2) if total and score is not None else None

    # If there is no answer key, preserve the review but report an ungraded score.
    return {
        "has_answer_key": bool(normalised_key),
        "score": score,
        "total": total,
        "percentage": percentage,
        "status_counts": counts,
        "rows": rows,
    }
