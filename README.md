# Optical Marking Recognition System

A Streamlit application for creating printable OMR answer sheets, aligning scanned sheets with ArUco markers, detecting filled answer bubbles, reviewing uncertain detections, grading against an answer key, and saving results.

## Features

- Generate A4 answer-sheet PDFs with four unique ArUco markers.
- Save matching JSON metadata with marker coordinates, bubble ROIs, and optional answer keys.
- Upload completed sheets, detect markers, and perspective-align scans.
- Detect filled bubbles from aligned scans.
- Review and manually correct detected answers.
- Grade submissions when an answer key is present.
- Save reviewed results as JSON/CSV-ready data and SQLite summaries.

## Project Layout

- `app.py` - Streamlit home page.
- `pages/1_Create_Exam.py` - exam setup and sheet generation.
- `pages/2_Scan_Sheet.py` - marker detection and sheet alignment.
- `pages/3_Review_Answers.py` - bubble detection, review, correction, and grading.
- `pages/4_Results.py` - saved results, summaries, and exports.
- `sheet_generator/generator.py` - PDF and metadata generation.
- `omr/` - alignment, marker detection, bubble detection, and grading helpers.
- `database/database.py` - local SQLite persistence.
- `generated_sheets/` - runtime PDF/JSON outputs.
- `scanned_sheets/` - runtime scan, overlay, and result outputs.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Workflow

1. Create an exam and optionally store the answer key in the metadata.
2. Print the generated PDF.
3. Scan or photograph the completed sheet.
4. Upload the scan and matching JSON metadata on the Scan Sheet page.
5. Review detected answers, correct any uncertain rows, and save the result.
6. Open Results to inspect scores and export summaries.

