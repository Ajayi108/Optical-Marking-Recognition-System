# Small SQLite store for exams, scans, and graded OMR results.

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DB_PATH = Path(__file__).resolve().with_name("omr.db")


def utc_now() -> str:
    # Returns a stable ISO timestamp in UTC.
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    # Opens the database and ensures the schema exists.
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    # Row objects let the UI refer to columns by name instead of tuple indexes.
    connection.row_factory = sqlite3.Row
    init_db(connection)
    return connection


def init_db(connection: sqlite3.Connection) -> None:
    # Creates all application tables if they are missing.
    # Exams describe generated sheets, scans describe aligned uploads, and results
    # store the final reviewed and graded submissions.
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS exams (
            exam_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            num_questions INTEGER NOT NULL,
            choices_json TEXT NOT NULL,
            answer_key_json TEXT,
            sheet_id TEXT,
            metadata_name TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS scans (
            scan_id TEXT PRIMARY KEY,
            sheet_id TEXT,
            exam_id TEXT,
            metadata_name TEXT,
            original_path TEXT,
            preview_path TEXT,
            aligned_path TEXT,
            scan_json_path TEXT UNIQUE,
            detected_ids_json TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS results (
            result_id TEXT PRIMARY KEY,
            scan_id TEXT,
            sheet_id TEXT,
            exam_id TEXT,
            student_name TEXT,
            student_id TEXT,
            score INTEGER,
            total INTEGER,
            percentage REAL,
            status_counts_json TEXT NOT NULL,
            answers_json TEXT NOT NULL,
            grading_json TEXT NOT NULL,
            result_json_path TEXT UNIQUE,
            reviewed_at TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    connection.commit()


def _json(data: Any) -> str:
    # Stable key ordering makes stored JSON easier to compare in tests and diffs.
    return json.dumps(data, sort_keys=True)


def _loads(value: str | None, fallback: Any) -> Any:
    # Bad JSON should not break the results page; fall back to an empty shape.
    if not value:
        return fallback

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def save_exam(metadata: dict[str, Any], metadata_name: str = "", db_path: str | Path = DB_PATH) -> None:
    # Upserts exam metadata.
    exam = metadata.get("exam", {})
    exam_id = str(exam.get("exam_id", "")).strip()

    # Some uploaded JSON files may be incomplete; only save real exams.
    if not exam_id:
        return

    now = utc_now()

    with connect(db_path) as connection:
        # Preserve original creation time while allowing regenerated metadata updates.
        existing = connection.execute("SELECT created_at FROM exams WHERE exam_id = ?", (exam_id,)).fetchone()
        created_at = existing["created_at"] if existing else now
        connection.execute(
            """
            INSERT INTO exams (
                exam_id, title, num_questions, choices_json, answer_key_json,
                sheet_id, metadata_name, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(exam_id) DO UPDATE SET
                title=excluded.title,
                num_questions=excluded.num_questions,
                choices_json=excluded.choices_json,
                answer_key_json=excluded.answer_key_json,
                sheet_id=excluded.sheet_id,
                metadata_name=excluded.metadata_name,
                updated_at=excluded.updated_at
            """,
            (
                exam_id,
                str(exam.get("title", "")),
                int(exam.get("num_questions", 0)),
                _json(exam.get("choices", [])),
                _json(metadata.get("answer_key", {})),
                str(metadata.get("sheet_id", "")),
                metadata_name,
                created_at,
                now,
            ),
        )
        connection.commit()


def save_scan(scan_data: dict[str, Any], scan_json_path: str | Path = "", db_path: str | Path = DB_PATH) -> str:
    # Upserts a scan summary and returns its scan id.
    metadata = scan_data.get("metadata", {})
    # Saving a scan also makes sure the matching exam exists in the database.
    save_exam(metadata, str(scan_data.get("metadata_name", "")), db_path)
    scan_path = Path(scan_json_path) if scan_json_path else Path(str(scan_data.get("result_path", "")))
    scan_id = scan_path.stem if scan_path.name else str(uuid.uuid4())
    exam_id = str(metadata.get("exam", {}).get("exam_id", ""))

    with connect(db_path) as connection:
        # Scans may be reprocessed, so keep the same scan_id and refresh its file paths.
        connection.execute(
            """
            INSERT INTO scans (
                scan_id, sheet_id, exam_id, metadata_name, original_path, preview_path,
                aligned_path, scan_json_path, detected_ids_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scan_id) DO UPDATE SET
                sheet_id=excluded.sheet_id,
                exam_id=excluded.exam_id,
                metadata_name=excluded.metadata_name,
                original_path=excluded.original_path,
                preview_path=excluded.preview_path,
                aligned_path=excluded.aligned_path,
                scan_json_path=excluded.scan_json_path,
                detected_ids_json=excluded.detected_ids_json
            """,
            (
                scan_id,
                str(scan_data.get("sheet_id", metadata.get("sheet_id", ""))),
                exam_id,
                str(scan_data.get("metadata_name", "")),
                str(scan_data.get("original_path", "")),
                str(scan_data.get("preview_path", "")),
                str(scan_data.get("aligned_path", "")),
                str(scan_path) if scan_path.name else "",
                _json(scan_data.get("detected_ids", [])),
                utc_now(),
            ),
        )
        connection.commit()

    return scan_id


def save_result(result: dict[str, Any], result_json_path: str | Path = "", db_path: str | Path = DB_PATH) -> str:
    # Saves a graded result summary and returns its id.
    metadata = result.get("metadata", {})
    # A result is the last step in the pipeline, but it still carries exam metadata.
    save_exam(metadata, str(result.get("metadata_name", "")), db_path)
    scan_id = str(result.get("scan_id") or Path(str(result.get("scan_json_path", ""))).stem or uuid.uuid4())
    result_id = str(result.get("result_id") or uuid.uuid4())
    grading = result.get("grading", {})
    result_path = str(result_json_path or result.get("result_json_path", ""))

    with connect(db_path) as connection:
        # Result JSON keeps full details; SQLite stores searchable summary columns too.
        connection.execute(
            """
            INSERT INTO results (
                result_id, scan_id, sheet_id, exam_id, student_name, student_id,
                score, total, percentage, status_counts_json, answers_json,
                grading_json, result_json_path, reviewed_at, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(result_id) DO UPDATE SET
                scan_id=excluded.scan_id,
                sheet_id=excluded.sheet_id,
                exam_id=excluded.exam_id,
                student_name=excluded.student_name,
                student_id=excluded.student_id,
                score=excluded.score,
                total=excluded.total,
                percentage=excluded.percentage,
                status_counts_json=excluded.status_counts_json,
                answers_json=excluded.answers_json,
                grading_json=excluded.grading_json,
                result_json_path=excluded.result_json_path,
                reviewed_at=excluded.reviewed_at
            """,
            (
                result_id,
                scan_id,
                str(result.get("sheet_id", metadata.get("sheet_id", ""))),
                str(metadata.get("exam", {}).get("exam_id", "")),
                str(result.get("student_name", "")),
                str(result.get("student_id", "")),
                grading.get("score"),
                grading.get("total"),
                grading.get("percentage"),
                _json(grading.get("status_counts", {})),
                _json(result.get("reviewed_answers", {})),
                _json(grading),
                result_path,
                str(result.get("reviewed_at", utc_now())),
                utc_now(),
            ),
        )
        connection.commit()

    return result_id


def list_results(db_path: str | Path = DB_PATH) -> list[dict[str, Any]]:
    # Returns saved result summaries ordered newest first.
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT result_id, scan_id, sheet_id, exam_id, student_name, student_id,
                   score, total, percentage, status_counts_json, answers_json,
                   grading_json, result_json_path, reviewed_at, created_at
            FROM results
            ORDER BY reviewed_at DESC, created_at DESC
            """
        ).fetchall()

    results: list[dict[str, Any]] = []

    # Convert packed JSON columns back to dictionaries for Streamlit display.
    for row in rows:
        item = dict(row)
        item["status_counts"] = _loads(item.pop("status_counts_json"), {})
        item["answers"] = _loads(item.pop("answers_json"), {})
        item["grading"] = _loads(item.pop("grading_json"), {})
        results.append(item)

    return results


def exam_summary(db_path: str | Path = DB_PATH) -> list[dict[str, Any]]:
    # Returns score aggregates grouped by exam.
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT exam_id,
                   COUNT(*) AS submissions,
                   AVG(percentage) AS average_percentage,
                   MAX(percentage) AS best_percentage,
                   MIN(percentage) AS lowest_percentage
            FROM results
            WHERE percentage IS NOT NULL
            GROUP BY exam_id
            ORDER BY exam_id
            """
        ).fetchall()

    return [dict(row) for row in rows]
