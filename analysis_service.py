"""
analysis_service.py
By: Natiq Ghafoor

Runs the full exercise analysis pipeline as a callable Python service
instead of only through the CLI.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from main import (
    ROOT_DIR,
    build_rep_feature_list,
    best_side_angle,
    coco_pose_dict,
    default_limb_pairs,
    get_joint_cfg,
    joint_plotter,
    load_json,
    load_plugin,
    overlay_renderer,
    pick_weights_path,
    rep_metrics_plotter,
    rep_segmenter,
    safe_fps,
    save_angles_csv,
    save_reps_csv,
    smooth_angles,
    timeline_plotter,
    try_open_file,
)


def _find_ffmpeg() -> Optional[str]:
    """
    Looks for ffmpeg on the system path.
    """
    return shutil.which("ffmpeg")


def _convert_video_for_browser(input_path: str, output_path: str, debug: bool = False) -> bool:
    """
    Converts a saved video into a browser-friendly H.264 mp4.

    Returns True if conversion worked, otherwise False.
    """
    ffmpeg_path = _find_ffmpeg()
    if not ffmpeg_path:
        if debug:
            print("[WARN] ffmpeg was not found on PATH. Keeping original mp4.")
        return False

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            if debug:
                print("[WARN] ffmpeg conversion failed.")
                print(result.stderr)
            return False

        return os.path.exists(output_path)

    except Exception as e:
        if debug:
            print(f"[WARN] ffmpeg conversion error: {e}")
        return False


def run_analysis(
    video_path: str,
    exercise: str,
    calibration_path: str,
    output_dir: str = "outputs",
    save_video: bool = False,
    save_plots: bool = False,
    save_angle_csv: bool = False,
    save_reps_json: bool = False,
    weights: str = "",
    device: str = "",
    imgsz: int = 640,
    conf: float = 0.25,
    prefix: str = "",
    min_joint_conf: float = 0.25,
    overlay_scale: float = 0.7,
    open_video: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Runs the full exercise analysis pipeline and returns structured results.

    Args:
        video_path: Input video path.
        exercise: Exercise name ("bench", "curl", or "squat").
        calibration_path: Calibration JSON path.
        output_dir: Folder where outputs will be saved.
        save_video: Whether to save annotated MP4.
        save_plots: Whether to save plot PNGs/metrics.
        save_angle_csv: Whether to save per-frame angle CSV.
        save_reps_json: Whether to save reps JSON.
        weights: Optional pose model weights path.
        device: Optional CUDA device id (e.g. "0").
        imgsz: YOLO inference image size.
        conf: YOLO detection confidence threshold.
        prefix: Optional filename prefix inside output_dir.
        min_joint_conf: Minimum keypoint confidence for angle extraction / overlay.
        overlay_scale: Font scale for overlay text.
        open_video: Whether to open saved video on Windows after render.
        debug: Whether to print debug logs.

    Returns:
        Dictionary containing summary info, artifact paths, and rep results.
    """
    exercise = exercise.strip().lower()
    os.makedirs(output_dir, exist_ok=True)

    # Load calibration and exercise config.
    calib = load_json(calibration_path)
    joint_name, joint_cfg = get_joint_cfg(calib, exercise)

    # Limb pairs from plugin if available, otherwise fallback.
    limb_pairs: List[Tuple[str, str]] = []
    try:
        plugin = load_plugin(exercise)
        if hasattr(plugin, "get_limb_pairs"):
            limb_pairs = plugin.get_limb_pairs()  # type: ignore[attr-defined]
    except Exception:
        limb_pairs = []

    if not limb_pairs:
        limb_pairs = default_limb_pairs()

    # Open input video.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    fps = safe_fps(cap)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Load pose model.
    weights_path = pick_weights_path(weights)
    model = YOLO(weights_path)

    if device:
        try:
            model.to(f"cuda:{device}")
        except Exception:
            pass

    if debug:
        print(f"[DEBUG] exercise={exercise}")
        print(f"[DEBUG] joint_name={joint_name}")
        print(f"[DEBUG] weights={weights_path}")
        print(f"[DEBUG] fps={fps:.2f} size={width}x{height}")
        print(f"[DEBUG] limb_pairs={len(limb_pairs)}")

    # Pass 1: pose inference + angle extraction.
    pose_sequence: List[Dict[str, Tuple[float, float, float]]] = []
    angle_sequence: List[Optional[float]] = []
    prev_angle: Optional[float] = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        results = model(frame, verbose=False, conf=float(conf), imgsz=int(imgsz))
        if not results:
            pose_sequence.append({})
            angle_sequence.append(None)
            continue

        r0 = results[0]
        if r0.keypoints is None:
            pose_sequence.append({})
            angle_sequence.append(None)
            continue

        try:
            xy = r0.keypoints.xy.cpu().numpy()
            cf = None
            if getattr(r0.keypoints, "conf", None) is not None:
                cf = r0.keypoints.conf.cpu().numpy()
        except Exception:
            pose_sequence.append({})
            angle_sequence.append(None)
            continue

        if xy is None or len(xy) == 0:
            pose_sequence.append({})
            angle_sequence.append(None)
            continue

        pose = coco_pose_dict(
            xy[0],
            cf[0] if cf is not None and len(cf) > 0 else None,
        )
        pose_sequence.append(pose)

        # Extract the exercise signal angle.
        ang: Optional[float] = None

        if "elbow" in joint_name:
            ang = best_side_angle(
                pose=pose,
                left_triplet=("left_shoulder", "left_elbow", "left_wrist"),
                right_triplet=("right_shoulder", "right_elbow", "right_wrist"),
                min_conf=float(min_joint_conf),
                prev_angle=prev_angle,
                max_jump=60.0 if exercise == "bench" else 45.0,
            )
        elif "knee" in joint_name:
            ang = best_side_angle(
                pose=pose,
                left_triplet=("left_hip", "left_knee", "left_ankle"),
                right_triplet=("right_hip", "right_knee", "right_ankle"),
                min_conf=float(min_joint_conf),
                prev_angle=prev_angle,
                max_jump=45.0,
            )

        if ang is not None:
            prev_angle = ang

        angle_sequence.append(ang)

    cap.release()

    # Smooth using calibration setting.
    smooth_w = int(joint_cfg.get("smooth_window", 1) or 1)
    if smooth_w > 1:
        angle_sequence = smooth_angles(angle_sequence, smooth_w)

    if debug:
        clean = [a for a in angle_sequence if a is not None and np.isfinite(a)]
        if clean:
            print(
                f"[DEBUG] angle_range: min={min(clean):.1f}° "
                f"max={max(clean):.1f}° (after smoothing w={smooth_w})"
            )
        print(f"[DEBUG] frames={len(angle_sequence)}")

    # Segment reps.
    reps = rep_segmenter.segment_reps_from_angle(
        angle_sequence=angle_sequence,
        calibration=joint_cfg,
        fps=fps,
        debug=debug,
        exercise=exercise,
    )

    # Output paths.
    prefix = (prefix.strip() + "_") if prefix.strip() else ""
    out_csv = os.path.join(output_dir, f"{prefix}{exercise}_reps.csv")
    out_summary = os.path.join(output_dir, f"{prefix}{exercise}_summary.json")
    out_video = os.path.join(output_dir, f"{prefix}{exercise}_annotated.mp4")
    out_angles = os.path.join(output_dir, f"{prefix}{exercise}_angles.csv")
    out_reps_json = os.path.join(output_dir, f"{prefix}{exercise}_reps.json")

    # Temp video path used before browser conversion.
    temp_video = os.path.join(output_dir, f"{prefix}{exercise}_annotated_raw.mp4")

    # Always-save artifacts.
    save_reps_csv(out_csv, reps)

    summary = {
        "exercise": exercise,
        "joint_name": joint_name,
        "rep_count": len(reps or []),
        "pass_count": sum(
            1 for r in (reps or []) if str(r.get("label", "")).lower() == "pass"
        ),
        "fail_count": sum(
            1 for r in (reps or []) if str(r.get("label", "")).lower() != "pass"
        ),
        "calibration_file": os.path.basename(calibration_path),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if save_angle_csv:
        save_angles_csv(out_angles, angle_sequence)

    if save_reps_json:
        with open(out_reps_json, "w", encoding="utf-8") as f:
            json.dump(reps, f, indent=2)

    # Plot artifacts.
    if save_plots:
        rep_feature_list = build_rep_feature_list(reps)

        try:
            if hasattr(joint_plotter, "plot_joint_angles"):
                joint_plotter.plot_joint_angles(
                    angle_sequence,
                    rep_feature_list,
                    output_dir,
                )
        except Exception as e:
            print(f"[WARN] joint_plotter failed: {e}")

        try:
            if hasattr(timeline_plotter, "plot_rep_timeline"):
                timeline_plotter.plot_rep_timeline(
                    rep_feature_list,
                    fps=float(fps),
                    output_dir=output_dir,
                )
        except Exception as e:
            print(f"[WARN] timeline_plotter failed: {e}")

        try:
            if hasattr(rep_metrics_plotter, "plot_rep_metrics"):
                rep_metrics_plotter.plot_rep_metrics(
                    rep_feature_list,
                    output_dir=output_dir,
                )
        except Exception as e:
            print(f"[WARN] rep_metrics_plotter failed: {e}")

    # Annotated video render.
    if save_video:
        rep_map = overlay_renderer.build_rep_map(reps)

        end_frames = sorted(
            [
                int(r.get("end_idx", r.get("end_frame", r.get("end", -1))))
                for r in reps
                if r.get("end_idx", r.get("end_frame", r.get("end", None))) is not None
            ]
        )
        end_ptr = 0
        rep_count = 0

        cap2 = cv2.VideoCapture(video_path)
        if not cap2.isOpened():
            raise RuntimeError(f"Could not re-open input video: {video_path}")

        # Write a raw mp4 first, then convert it into a browser-friendly one.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            temp_video,
            fourcc,
            float(fps),
            (int(width), int(height)),
        )

        frame_i = 0
        while True:
            ok, frame = cap2.read()
            if not ok or frame is None:
                break

            pose = pose_sequence[frame_i] if frame_i < len(pose_sequence) else {}

            while end_ptr < len(end_frames) and end_frames[end_ptr] == frame_i:
                rep_count += 1
                end_ptr += 1

            overlay_renderer.draw_skeleton(
                frame,
                pose,
                limb_pairs=limb_pairs,
                min_conf=float(min_joint_conf),
            )
            overlay_renderer.annotate_frame(
                frame=frame,
                frame_index=frame_i,
                exercise_name=exercise,
                rep_count=rep_count,
                rep_map=rep_map,
                fps=float(fps),
                font_scale=float(overlay_scale),
            )

            writer.write(frame)
            frame_i += 1

        cap2.release()
        writer.release()

        # Convert to H.264 for browser playback.
        converted = _convert_video_for_browser(
            input_path=temp_video,
            output_path=out_video,
            debug=debug,
        )

        # If conversion failed, keep the raw file as the final output.
        if not converted:
            if os.path.exists(out_video):
                try:
                    os.remove(out_video)
                except Exception:
                    pass

            if os.path.exists(temp_video):
                os.replace(temp_video, out_video)
        else:
            try:
                if os.path.exists(temp_video):
                    os.remove(temp_video)
            except Exception:
                pass

        if open_video:
            try_open_file(out_video)

    if debug:
        print(f"[INFO] CSV saved: {out_csv}")
        print(f"[INFO] Summary saved: {out_summary}")

    artifacts = {
        "summary_json": out_summary if os.path.exists(out_summary) else None,
        "reps_csv": out_csv if os.path.exists(out_csv) else None,
        "annotated_video": out_video if save_video and os.path.exists(out_video) else None,
        "angles_csv": out_angles if save_angle_csv and os.path.exists(out_angles) else None,
        "reps_json": out_reps_json if save_reps_json and os.path.exists(out_reps_json) else None,
    }

    return {
        "summary": summary,
        "reps": reps,
        "artifacts": artifacts,
        "metadata": {
            "video_path": video_path,
            "output_dir": output_dir,
            "fps": float(fps),
            "frame_count": len(angle_sequence),
            "width": int(width),
            "height": int(height),
            "exercise": exercise,
            "joint_name": joint_name,
        },
    }