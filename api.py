"""
api.py
By: Natiq Ghafoor

FastAPI entrypoint for the gym pose estimation project.

This file exposes a small backend API so a client can:
- check if the server is running
- upload a video
- run the analysis pipeline
- run a demo analysis on a bundled sample video
- get a JSON response with saved artifact paths and summary data
- view saved history from the database
- ask follow-up questions about one saved analysis
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from database import (
    get_all_analyses,
    get_analysis_by_id,
    get_rep_results_by_analysis_id,
    init_db,
    insert_analysis,
    insert_rep_results,
)
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel



app = FastAPI(title="Gym Pose API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://gym-pose-estimation-webapp.vercel.app",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
RUNS_DIR = BASE_DIR / "api_outputs"
DEMO_VIDEO_PATH = BASE_DIR / "curl.mov"

UPLOADS_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)

app.mount("/api_outputs", StaticFiles(directory=RUNS_DIR), name="api_outputs")


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")


class ChatMessage(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    analysis_id: int
    messages: List[ChatMessage]


def get_run_analysis():
    """
    Imports the heavy analysis pipeline only when live video analysis needs it.
    """
    from analysis_service import run_analysis
    return run_analysis


def load_json_file(path):
    """
    Loads a JSON file from disk.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_demo_reps_csv(path):
    """
    Loads precomputed demo rep rows from CSV into the format expected by the app.
    """
    reps = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rep = {
                "rep_index": int(float(row.get("rep_index") or len(reps) + 1)),
                "start_idx": int(float(row.get("start_idx") or row.get("start_frame") or row.get("start") or 0)),
                "end_idx": int(float(row.get("end_idx") or row.get("end_frame") or row.get("end") or 0)),
                "duration": float(row.get("duration") or row.get("rep_duration") or 0.0),
                "label": str(row.get("label") or "fail").strip().lower(),
                "reason": str(row.get("reason") or row.get("fail_reason") or "").strip(),
                "rom": float(row.get("rom") or row.get("ROM") or 0.0),
            }
            reps.append(rep)

    return reps


def to_public_url(path_value):
    """
    Converts a saved local file path into a URL the frontend can open.
    """
    if not path_value:
        return None

    path_obj = Path(path_value)
    if not path_obj.exists():
        return None

    try:
        return "/" + str(path_obj.relative_to(BASE_DIR)).replace("\\", "/")
    except ValueError:
        return None


def find_matching_file(output_dir, suffix):
    """
    Looks for the first file inside the run folder that ends with the given suffix.
    """
    if not output_dir:
        return None

    out_dir = Path(output_dir)
    if not out_dir.exists():
        return None

    matches = sorted(out_dir.glob(f"*{suffix}"))
    if not matches:
        return None

    return matches[0]


def build_artifact_urls(summary_json_path, reps_csv_path, output_dir=None):
    """
    Converts saved local file paths into URLs the frontend can open.
    """
    artifact_urls = {
        "summary_json": None,
        "reps_csv": None,
        "annotated_video": None,
        "angles_csv": None,
        "reps_json": None,
        "joint_angles_plot": None,
        "rep_timeline_plot": None,
        "rep_rom_plot": None,
        "rep_duration_plot": None,
        "rep_outcomes_plot": None,
        "rep_metrics_json": None,
    }

    artifact_urls["summary_json"] = to_public_url(summary_json_path)
    artifact_urls["reps_csv"] = to_public_url(reps_csv_path)

    output_file_map = {
        "annotated_video": "_annotated.mp4",
        "angles_csv": "_angles.csv",
        "reps_json": "_reps.json",
        "joint_angles_plot": "joint_angles.png",
        "rep_timeline_plot": "rep_timeline.png",
        "rep_rom_plot": "rep_rom.png",
        "rep_duration_plot": "rep_duration.png",
        "rep_outcomes_plot": "rep_outcomes.png",
        "rep_metrics_json": "rep_metrics.json",
    }

    for key, suffix in output_file_map.items():
        match = find_matching_file(output_dir, suffix)
        artifact_urls[key] = to_public_url(match)

    return artifact_urls


def build_feedback(analysis, reps):
    """
    Builds simple coaching feedback from the saved analysis and rep rows.
    """
    if not reps:
        return {
            "headline": "No rep data found",
            "summary": (
                "This session does not have rep rows yet, so there is not enough "
                "information to generate feedback."
            ),
            "bullets": [],
            "highlights": [],
        }

    exercise = str(analysis.get("exercise", "exercise")).capitalize()
    pass_count = int(analysis.get("pass_count", 0) or 0)
    fail_count = int(analysis.get("fail_count", 0) or 0)
    rep_count = int(analysis.get("rep_count", len(reps)) or len(reps))

    roms = [float(rep.get("rom", 0) or 0) for rep in reps]
    durations = [float(rep.get("duration", 0) or 0) for rep in reps]

    avg_rom = sum(roms) / len(roms) if roms else 0.0
    avg_duration = sum(durations) / len(durations) if durations else 0.0

    max_rom = max(roms) if roms else 0.0
    min_rom = min(roms) if roms else 0.0
    rom_spread = max_rom - min_rom

    max_duration = max(durations) if durations else 0.0
    min_duration = min(durations) if durations else 0.0
    duration_spread = max_duration - min_duration

    best_rep = max(reps, key=lambda rep: float(rep.get("rom", 0) or 0))
    weakest_rep = min(reps, key=lambda rep: float(rep.get("rom", 0) or 0))

    bullets = []
    highlights = []

    if fail_count == 0:
        bullets.append(
            f"All {rep_count} reps passed, so the set was consistently accepted by the current rules."
        )
    else:
        bullets.append(
            f"{pass_count} out of {rep_count} reps passed, while {fail_count} were flagged."
        )

    if rom_spread <= 8:
        bullets.append(
            f"ROM stayed fairly steady across the set. The spread from best to lowest rep was only {rom_spread:.2f} degrees."
        )
    elif rom_spread <= 18:
        bullets.append(
            f"ROM was decent overall, but there was some drop-off across reps. The spread was {rom_spread:.2f} degrees."
        )
    else:
        bullets.append(
            f"ROM changed a lot across the set. The spread was {rom_spread:.2f} degrees, which suggests some inconsistency from rep to rep."
        )

    if duration_spread <= 0.4:
        bullets.append(
            f"Tempo looked pretty controlled. Rep durations stayed within {duration_spread:.2f} seconds of each other."
        )
    else:
        bullets.append(
            f"Tempo changed a bit during the set. Rep durations varied by {duration_spread:.2f} seconds."
        )

    if int(best_rep.get("rep_index", 0) or 0) != int(weakest_rep.get("rep_index", 0) or 0):
        highlights.append(
            f"Best ROM: rep {best_rep.get('rep_index')} at {float(best_rep.get('rom', 0) or 0):.2f} degrees."
        )
        highlights.append(
            f"Lowest ROM: rep {weakest_rep.get('rep_index')} at {float(weakest_rep.get('rom', 0) or 0):.2f} degrees."
        )

    if len(roms) >= 2:
        split_idx = (len(roms) + 1) // 2
        first_half = roms[:split_idx]
        second_half = roms[split_idx:]

        first_half_avg = sum(first_half) / len(first_half) if first_half else 0.0
        second_half_avg = sum(second_half) / len(second_half) if second_half else 0.0

        if second_half and (second_half_avg + 4 < first_half_avg):
            highlights.append(
                "Later reps had noticeably lower ROM than the earlier reps, which may suggest fatigue or a loss of consistency."
            )
        elif second_half and (second_half_avg > first_half_avg + 4):
            highlights.append(
                "Later reps actually improved compared to the earlier reps, which may mean the set got cleaner as it went on."
            )
        else:
            highlights.append("ROM stayed fairly even from the start of the set to the end.")

    if fail_count == 0:
        highlights.append(
            "Since everything passed, the next improvement is less about fixing errors and more about keeping the same quality across different sessions."
        )
    else:
        fail_reasons = [
            str(rep.get("reason", "Form issue") or "Form issue")
            for rep in reps
            if str(rep.get("label", "")).lower() != "pass"
        ]

        reason_counts = {}
        for reason in fail_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        if reason_counts:
            top_reason = max(reason_counts, key=reason_counts.get)
            highlights.append(f'The most common issue in this set was "{top_reason}".')

    return {
        "headline": f"{exercise} set feedback",
        "summary": (
            f"Average ROM was {avg_rom:.2f} degrees and average rep duration "
            f"was {avg_duration:.2f} seconds."
        ),
        "bullets": bullets,
        "highlights": highlights,
    }


def build_rule_based_chat_reply(message, analysis, reps, feedback):
    """
    Fallback reply when OpenAI is not configured.
    """
    text = (message or "").strip().lower()

    if not text:
        return "Ask a question about this workout and I’ll answer from the saved session data."

    if not reps:
        return "This analysis does not have rep rows yet, so there is not enough data to answer follow-up questions."

    rep_count = int(analysis.get("rep_count", len(reps)) or len(reps))
    pass_count = int(analysis.get("pass_count", 0) or 0)
    fail_count = int(analysis.get("fail_count", 0) or 0)
    exercise = str(analysis.get("exercise", "exercise")).capitalize()

    roms = [float(rep.get("rom", 0) or 0) for rep in reps]
    durations = [float(rep.get("duration", 0) or 0) for rep in reps]

    best_rep = max(reps, key=lambda rep: float(rep.get("rom", 0) or 0))
    weakest_rep = min(reps, key=lambda rep: float(rep.get("rom", 0) or 0))

    if "summary" in text or "overall" in text or "set" in text:
        return (
            f"This {exercise.lower()} session had {rep_count} reps total. "
            f"{pass_count} passed and {fail_count} failed. "
            f"The average ROM was {sum(roms) / len(roms):.2f} degrees and the average rep duration was "
            f"{sum(durations) / len(durations):.2f} seconds."
        )

    if "best" in text and "rep" in text:
        return (
            f"Your best rep by ROM was rep {best_rep.get('rep_index')} at "
            f"{float(best_rep.get('rom', 0) or 0):.2f} degrees."
        )

    if "worst" in text or "weakest" in text or ("lowest" in text and "rom" in text):
        return (
            f"Your lowest-ROM rep was rep {weakest_rep.get('rep_index')} at "
            f"{float(weakest_rep.get('rom', 0) or 0):.2f} degrees."
        )

    if "rom" in text or "range of motion" in text:
        rom_spread = max(roms) - min(roms)
        return (
            f"ROM ranged from {min(roms):.2f} to {max(roms):.2f} degrees, "
            f"for a spread of {rom_spread:.2f} degrees across the set."
        )

    if "duration" in text or "tempo" in text or "speed" in text:
        duration_spread = max(durations) - min(durations)
        return (
            f"Rep duration ranged from {min(durations):.2f} to {max(durations):.2f} seconds. "
            f"The total spread was {duration_spread:.2f} seconds."
        )

    if "fail" in text or "failed" in text or "issue" in text:
        failed_reps = [
            rep for rep in reps if str(rep.get("label", "")).lower() != "pass"
        ]

        if not failed_reps:
            return "None of the reps failed in this session. Everything passed under the current rules."

        fail_lines = []
        for rep in failed_reps:
            fail_lines.append(
                f"rep {rep.get('rep_index')} failed because of {rep.get('reason', 'Form issue')}"
            )

        return "The flagged reps were: " + "; ".join(fail_lines) + "."

    if "pass" in text or "passed" in text:
        return f"{pass_count} out of {rep_count} reps passed in this session."

    if "later" in text or "fatigue" in text or "drop" in text:
        split_idx = (len(roms) + 1) // 2
        first_half = roms[:split_idx]
        second_half = roms[split_idx:]

        first_half_avg = sum(first_half) / len(first_half) if first_half else 0.0
        second_half_avg = sum(second_half) / len(second_half) if second_half else 0.0

        if not second_half:
            return "There were not enough later reps in this session to compare early reps versus late reps."

        if second_half_avg + 4 < first_half_avg:
            return (
                "Yes, later reps dropped off compared to the earlier reps. "
                f"The first-half average ROM was {first_half_avg:.2f} degrees, while the second-half average was {second_half_avg:.2f}."
            )

        if second_half_avg > first_half_avg + 4:
            return (
                "The later reps actually improved a bit. "
                f"The first-half average ROM was {first_half_avg:.2f} degrees and the second-half average was {second_half_avg:.2f}."
            )

        return (
            "There was not a major drop-off from early reps to later reps. "
            f"The first-half average ROM was {first_half_avg:.2f} degrees and the second-half average was {second_half_avg:.2f}."
        )

    if "improve" in text or "better" in text or "work on" in text:
        notes = feedback.get("highlights", []) if isinstance(feedback, dict) else []
        if notes:
            return (
                "The main thing to focus on next is keeping the strongest parts of the set consistent. "
                + " ".join(notes[:2])
            )
        return "The next improvement would be keeping your ROM and tempo as consistent as possible across the full set."

    return (
        "I can answer questions about this workout’s summary, best rep, weakest rep, ROM, tempo, pass/fail results, and whether later reps dropped off."
    )


def openai_is_available():
    """
    Checks whether OpenAI is configured on the backend.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        return False

    if api_key == "your_api_key_here":
        return False

    return True


def build_context_block(analysis, reps, feedback) -> str:
    """
    Converts the workout data into a compact text block for the model.
    """
    lines = []
    lines.append("WORKOUT SUMMARY")
    lines.append(f"Exercise: {analysis.get('exercise')}")
    lines.append(f"Rep count: {analysis.get('rep_count')}")
    lines.append(f"Pass count: {analysis.get('pass_count')}")
    lines.append(f"Fail count: {analysis.get('fail_count')}")
    lines.append(f"Average ROM: {analysis.get('avg_rom')}")
    lines.append(f"Average duration: {analysis.get('avg_duration')}")
    lines.append("")

    lines.append("COACH FEEDBACK")
    lines.append(f"Headline: {feedback.get('headline', '')}")
    lines.append(f"Summary: {feedback.get('summary', '')}")

    bullets = feedback.get("bullets", []) or []
    if bullets:
        lines.append("Bullets:")
        for item in bullets:
            lines.append(f"- {item}")

    highlights = feedback.get("highlights", []) or []
    if highlights:
        lines.append("Highlights:")
        for item in highlights:
            lines.append(f"- {item}")

    lines.append("")
    lines.append("REPS")
    for rep in reps:
        lines.append(
            f"Rep {rep.get('rep_index')}: "
            f"start={rep.get('start_idx')}, "
            f"end={rep.get('end_idx')}, "
            f"duration={rep.get('duration')}, "
            f"rom={rep.get('rom')}, "
            f"label={rep.get('label')}, "
            f"reason={rep.get('reason')}"
        )

    return "\n".join(lines)


def build_history_block(messages: List[ChatMessage]) -> str:
    """
    Builds a simple conversation transcript for the model.
    """
    history_lines = []

    for message in messages[-10:]:
        role = (message.role or "").strip().lower()
        text = (message.text or "").strip()
        if not text:
            continue

        if role == "user":
            history_lines.append(f"User: {text}")
        else:
            history_lines.append(f"Assistant: {text}")

    return "\n".join(history_lines)


def build_llm_chat_reply(messages, analysis, reps, feedback):
    """
    Uses OpenAI to answer questions about a saved workout.
    """
    from openai import OpenAI

    client = OpenAI(timeout=20.0)

    context_block = build_context_block(analysis, reps, feedback)
    history_block = build_history_block(messages)

    latest_user_message = ""
    for item in reversed(messages):
        role = ""
        text = ""

        if isinstance(item, dict):
            role = str(item.get("role", "")).strip().lower()
            text = str(item.get("text", "")).strip()
        else:
            role = str(getattr(item, "role", "")).strip().lower()
            text = str(getattr(item, "text", "")).strip()

        if role == "user" and text:
            latest_user_message = text
            break

    system_prompt = """
You are an experienced strength coach analyzing a workout.

Speak clearly and naturally like a real coach. Avoid markdown formatting.
Do not use symbols like **, bullet formatting, or report-style headings.

Your reply should:
- stay under 5 sentences
- start with a short overall assessment
- mention 1 main strength
- mention 1 main thing to improve
- give practical advice the user can apply next session
- use numbers only when they help the explanation

Tone:
- direct
- supportive
- practical
- natural

Do not:
- sound robotic
- dump raw stats
- repeat every metric
- invent details that are not in the workout data
""".strip()

    prompt = f"""
Workout data:
{context_block}

Conversation so far:
{history_block}

User question:
{latest_user_message}

Write the next assistant reply.
""".strip()

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    text = (response.output_text or "").strip()
    if not text:
        return "I couldn't generate a reply for that workout question."

    return text


def save_analysis_result(
    *,
    exercise: str,
    original_filename: str,
    uploaded_file_path: str,
    output_dir: str,
    result: dict,
) -> dict:
    """
    Saves one completed analysis into the database and returns response payload pieces.
    """
    summary = result.get("summary", {})
    artifacts = result.get("artifacts", {})
    reps = result.get("reps", [])

    rom_values = [rep.get("rom") for rep in reps if rep.get("rom") is not None]
    duration_values = [rep.get("duration") for rep in reps if rep.get("duration") is not None]

    avg_rom = sum(rom_values) / len(rom_values) if rom_values else 0.0
    avg_duration = sum(duration_values) / len(duration_values) if duration_values else 0.0

    analysis_id = insert_analysis(
        created_at=datetime.now().isoformat(timespec="seconds"),
        exercise=summary.get("exercise", exercise),
        original_filename=original_filename,
        rep_count=summary.get("rep_count", 0),
        pass_count=summary.get("pass_count", 0),
        fail_count=summary.get("fail_count", 0),
        avg_rom=avg_rom,
        avg_duration=avg_duration,
        uploaded_file_path=uploaded_file_path,
        output_dir=output_dir,
        summary_json_path=artifacts.get("summary_json"),
        reps_csv_path=artifacts.get("reps_csv"),
    )

    insert_rep_results(analysis_id, reps)

    saved_analysis = get_analysis_by_id(analysis_id)
    saved_reps = get_rep_results_by_analysis_id(analysis_id)
    feedback = build_feedback(saved_analysis, saved_reps)

    return {
        "analysis_id": analysis_id,
        "feedback": feedback,
        "artifact_urls": build_artifact_urls(
            artifacts.get("summary_json"),
            artifacts.get("reps_csv"),
            output_dir,
        ),
    }


@app.get("/health")
def health_check() -> dict:
    """
    Simple route to confirm the backend is running.
    """
    return {"status": "ok"}


@app.get("/analyses")
def list_analyses() -> dict:
    """
    Returns all saved workout sessions.
    """
    init_db()
    analyses = get_all_analyses()

    for analysis in analyses:
        analysis["artifact_urls"] = build_artifact_urls(
            analysis.get("summary_json_path"),
            analysis.get("reps_csv_path"),
            analysis.get("output_dir"),
        )

    return {"analyses": analyses}


@app.get("/analyses/{analysis_id}")
def get_analysis_detail(analysis_id: int) -> dict:
    """
    Returns one saved workout plus its rep rows and feedback.
    """
    init_db()

    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    reps = get_rep_results_by_analysis_id(analysis_id)
    feedback = build_feedback(analysis, reps)

    return {
        "analysis": analysis,
        "reps": reps,
        "feedback": feedback,
        "artifact_urls": build_artifact_urls(
            analysis.get("summary_json_path"),
            analysis.get("reps_csv_path"),
            analysis.get("output_dir"),
        ),
    }


@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    exercise: str = Form(...),
    calibration_path: str = Form("calibration_easy.json"),
    save_video: bool = Form(False),
    save_plots: bool = Form(False),
    save_angle_csv: bool = Form(False),
    save_reps_json: bool = Form(False),
) -> dict:
    """
    Accepts an uploaded video and runs the analysis pipeline.
    """
    exercise = exercise.strip().lower()

    if exercise not in {"curl", "bench", "squat"}:
        raise HTTPException(status_code=400, detail="Exercise must be curl, bench, or squat.")

    calib_file = BASE_DIR / calibration_path
    if not calib_file.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Calibration file not found: {calibration_path}",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a filename.")

    run_id = str(uuid.uuid4())[:8]
    upload_path = UPLOADS_DIR / f"{run_id}_{file.filename}"
    output_dir = RUNS_DIR / f"{exercise}_{run_id}"
    output_dir.mkdir(exist_ok=True)

    try:
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        init_db()

        run_analysis = get_run_analysis()

        result = run_analysis(
            video_path=str(upload_path),
            exercise=exercise,
            calibration_path=str(calib_file),
            output_dir=str(output_dir),
            save_video=save_video,
            save_plots=save_plots,
            save_angle_csv=save_angle_csv,
            save_reps_json=save_reps_json,
        )

        saved = save_analysis_result(
            exercise=exercise,
            original_filename=file.filename,
            uploaded_file_path=str(upload_path),
            output_dir=str(output_dir),
            result=result,
        )

        return {
            "message": "Analysis completed successfully.",
            "run_id": run_id,
            "analysis_id": saved["analysis_id"],
            "uploaded_file": str(upload_path),
            "output_dir": str(output_dir),
            "result": result,
            "feedback": saved["feedback"],
            "artifact_urls": saved["artifact_urls"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            file.file.close()
        except Exception:
            pass


@app.post("/analyze-demo")
def analyze_demo_video(
    exercise: str = Form("curl"),
    calibration_path: str = Form("calibration_easy.json"),
    save_video: bool = Form(True),
    save_plots: bool = Form(True),
    save_angle_csv: bool = Form(False),
    save_reps_json: bool = Form(False),
) -> dict:
    """
    Loads precomputed demo results instead of running live YOLO inference.

    This keeps the hosted demo reliable on small cloud instances while the full
    pipeline remains available through /analyze or local runs.
    """
    exercise = exercise.strip().lower()

    if exercise != "curl":
        raise HTTPException(status_code=400, detail="Demo mode currently supports curl only.")

    demo_dir = BASE_DIR / "demo_outputs"
    if not demo_dir.exists():
        raise HTTPException(
            status_code=500,
            detail="demo_outputs folder not found. Add precomputed demo artifacts first.",
        )

    summary_src = demo_dir / "summary.json"
    reps_src = demo_dir / "reps.csv"

    # Support your current generated filenames too, so you do not have to rename
    # files locally unless you want to.
    if not summary_src.exists():
        summary_src = demo_dir / "curl_summary.json"

    if not reps_src.exists():
        reps_src = demo_dir / "curl_reps.csv"

    if not summary_src.exists():
        raise HTTPException(
            status_code=500,
            detail="Missing demo summary file. Expected demo_outputs/summary.json or demo_outputs/curl_summary.json.",
        )

    if not reps_src.exists():
        raise HTTPException(
            status_code=500,
            detail="Missing demo reps file. Expected demo_outputs/reps.csv or demo_outputs/curl_reps.csv.",
        )

    run_id = str(uuid.uuid4())[:8]
    output_dir = RUNS_DIR / f"{exercise}_demo_{run_id}"
    output_dir.mkdir(exist_ok=True)

    try:
        for item in demo_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, output_dir / item.name)

        summary_path = output_dir / summary_src.name
        reps_csv_path = output_dir / reps_src.name

        # Also create generic copies so artifact links are predictable.
        generic_summary_path = output_dir / "summary.json"
        generic_reps_path = output_dir / "reps.csv"

        if summary_path.name != "summary.json":
            shutil.copy2(summary_path, generic_summary_path)
        else:
            generic_summary_path = summary_path

        if reps_csv_path.name != "reps.csv":
            shutil.copy2(reps_csv_path, generic_reps_path)
        else:
            generic_reps_path = reps_csv_path

        summary = load_json_file(generic_summary_path)
        reps = load_demo_reps_csv(generic_reps_path)

        rep_count = int(summary.get("rep_count", len(reps)) or len(reps))
        pass_count = int(
            summary.get(
                "pass_count",
                sum(1 for rep in reps if str(rep.get("label", "")).lower() == "pass"),
            )
            or 0
        )
        fail_count = int(
            summary.get(
                "fail_count",
                sum(1 for rep in reps if str(rep.get("label", "")).lower() != "pass"),
            )
            or 0
        )

        result = {
            "summary": {
                "exercise": summary.get("exercise", exercise),
                "joint_name": summary.get("joint_name", "right_elbow"),
                "rep_count": rep_count,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "calibration_file": summary.get("calibration_file", calibration_path),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
            "artifacts": {
                "summary_json": str(generic_summary_path),
                "reps_csv": str(generic_reps_path),
            },
            "reps": reps,
        }

        init_db()

        saved = save_analysis_result(
            exercise=exercise,
            original_filename="curl_precomputed_demo",
            uploaded_file_path=str(DEMO_VIDEO_PATH),
            output_dir=str(output_dir),
            result=result,
        )

        return {
            "message": "Demo analysis loaded from precomputed results.",
            "run_id": run_id,
            "analysis_id": saved["analysis_id"],
            "uploaded_file": str(DEMO_VIDEO_PATH),
            "output_dir": str(output_dir),
            "result": result,
            "feedback": saved["feedback"],
            "artifact_urls": saved["artifact_urls"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat_about_analysis(payload: ChatRequest) -> dict:
    """
    Returns a reply about one saved analysis.
    Uses OpenAI when configured, otherwise falls back to rule-based chat.
    """
    init_db()

    analysis = get_analysis_by_id(payload.analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    reps = get_rep_results_by_analysis_id(payload.analysis_id)
    feedback = build_feedback(analysis, reps)

    latest_user_message = ""
    for item in reversed(payload.messages):
        role = str(getattr(item, "role", "")).strip().lower()
        text = str(getattr(item, "text", "")).strip()
        if role == "user" and text:
            latest_user_message = text
            break

    try:
        if openai_is_available():
            reply = build_llm_chat_reply(payload.messages, analysis, reps, feedback)
            llm_enabled = True
            model_name = OPENAI_MODEL
        else:
            reply = build_rule_based_chat_reply(
                latest_user_message,
                analysis,
                reps,
                feedback,
            )
            llm_enabled = False
            model_name = None

    except Exception:
        reply = build_rule_based_chat_reply(
            latest_user_message,
            analysis,
            reps,
            feedback,
        )
        llm_enabled = False
        model_name = None

    return {
        "reply": reply,
        "llm_enabled": llm_enabled,
        "model": model_name,
    }
