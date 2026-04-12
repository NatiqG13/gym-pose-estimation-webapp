"""
main.py
By: Natiq Ghafoor

CLI wrapper + shared helpers for the gym pose estimation pipeline.

Notes:
- The actual end-to-end pipeline now lives in analysis_service.py
- This file keeps shared helpers/import logic so existing modules still work
- CLI usage remains the same
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def _add_path(p: str) -> None:
    """
    Adds a directory to sys.path if it exists and isn't already present.
    """
    if p and os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


# Ensure local modules can be imported regardless of where the script is run from.
_add_path(ROOT_DIR)
_add_path(os.path.join(ROOT_DIR, "src"))
_add_path(os.path.join(ROOT_DIR, "visualization"))
_add_path(os.path.join(ROOT_DIR, "exercise_modules"))


def _import_from_file(module_name: str, file_path: str):
    """
    Imports a module directly from a file path.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Could not load {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def import_any(module_paths: List[str], file_candidates: List[str]):
    """
    Imports a module by trying:
      1) normal imports (module paths)
      2) direct file imports (file_candidates)

    This keeps the repo resilient to refactors like moving files under src/ or visualization/.
    """
    last_err: Optional[Exception] = None

    for mp in module_paths:
        try:
            return importlib.import_module(mp)
        except Exception as e:
            last_err = e

    for fp in file_candidates:
        if os.path.exists(fp):
            try:
                name = os.path.splitext(os.path.basename(fp))[0]
                return _import_from_file(name, fp)
            except Exception as e:
                last_err = e

    raise ModuleNotFoundError(
        "Import failed. Tried:\n"
        f"  modules: {module_paths}\n"
        f"  files:   {file_candidates}\n"
        f"Last error: {last_err}"
    )


# Core modules (robust import paths)
rep_segmenter = import_any(
    module_paths=["rep_segmenter", "src.rep_segmenter"],
    file_candidates=[
        os.path.join(ROOT_DIR, "rep_segmenter.py"),
        os.path.join(ROOT_DIR, "src", "rep_segmenter.py"),
    ],
)

overlay_renderer = import_any(
    module_paths=["visualization.overlay_renderer", "overlay_renderer", "src.overlay_renderer"],
    file_candidates=[
        os.path.join(ROOT_DIR, "overlay_renderer.py"),
        os.path.join(ROOT_DIR, "visualization", "overlay_renderer.py"),
        os.path.join(ROOT_DIR, "src", "overlay_renderer.py"),
    ],
)

# Plotters (optional utilities)
joint_plotter = import_any(
    module_paths=["visualization.joint_plotter", "joint_plotter"],
    file_candidates=[
        os.path.join(ROOT_DIR, "joint_plotter.py"),
        os.path.join(ROOT_DIR, "visualization", "joint_plotter.py"),
    ],
)

timeline_plotter = import_any(
    module_paths=["visualization.timeline_plotter", "timeline_plotter"],
    file_candidates=[
        os.path.join(ROOT_DIR, "timeline_plotter.py"),
        os.path.join(ROOT_DIR, "visualization", "timeline_plotter.py"),
    ],
)

rep_metrics_plotter = import_any(
    module_paths=["visualization.rep_metrics_plotter", "rep_metrics_plotter"],
    file_candidates=[
        os.path.join(ROOT_DIR, "rep_metrics_plotter.py"),
        os.path.join(ROOT_DIR, "visualization", "rep_metrics_plotter.py"),
    ],
)

# COCO keypoint names (Ultralytics uses COCO order)
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def parse_args() -> argparse.Namespace:
    """
    CLI options for running the pipeline and saving artifacts.
    """
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True, help="Video file path")
    p.add_argument("--exercise", required=True, choices=["bench", "curl", "squat"], help="Exercise")
    p.add_argument("--calibration", required=True, help="Calibration JSON path")

    # YOLO options (aliases supported)
    p.add_argument("--weights", "--model", default="", help="Pose model weights path")
    p.add_argument("--device", default="", help="CUDA device id (e.g. 0) or empty for auto")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--conf", "--yolo-conf", type=float, default=0.25, help="YOLO detection confidence")

    # Outputs
    p.add_argument("--output-dir", "--output_dir", dest="output_dir", default="outputs", help="Output directory")
    p.add_argument("--prefix", default="", help="Optional filename prefix inside output dir")

    p.add_argument("--save-video", action="store_true", help="Save annotated MP4")
    p.add_argument("--open-video", action="store_true", help="Open MP4 after saving (Windows)")
    p.add_argument("--save-plots", action="store_true", help="Save all plots (PNG)")
    p.add_argument("--save-angle-csv", action="store_true", help="Save per-frame angle CSV")
    p.add_argument("--save-reps-json", action="store_true", help="Save reps list as JSON")
    p.add_argument(
        "--save-all",
        action="store_true",
        help="Saves video + plots + angle CSV + reps JSON (CSV + summary always saved).",
    )

    # Overlay / filtering
    p.add_argument("--min-joint-conf", "--min_joint_conf", dest="min_joint_conf", type=float, default=0.25)
    p.add_argument("--overlay-scale", "--overlay_scale", dest="overlay_scale", type=float, default=0.7)
    p.add_argument("--debug", action="store_true")

    return p.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    """
    Loads a JSON file into a dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_fps(cap: cv2.VideoCapture) -> float:
    """
    Returns a usable FPS value, defaulting to 30 when metadata is missing.
    """
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return fps if fps > 1e-6 else 30.0


def coco_pose_dict(
    kps_xy: np.ndarray,
    kps_conf: Optional[np.ndarray],
) -> Dict[str, Tuple[float, float, float]]:
    """
    Converts YOLO keypoints arrays into a pose dict: joint_name -> (x, y, conf).
    """
    pose: Dict[str, Tuple[float, float, float]] = {}
    if kps_xy is None or len(kps_xy) != 17:
        return pose

    if kps_conf is None or len(kps_conf) != 17:
        kps_conf = np.ones((17,), dtype=np.float32)

    for i, name in enumerate(COCO_KEYPOINT_NAMES):
        x = float(kps_xy[i, 0])
        y = float(kps_xy[i, 1])
        c = float(kps_conf[i])
        pose[name] = (x, y, c)

    return pose


def angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Returns angle ABC in degrees.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by

    mag1 = (bax * bax + bay * bay) ** 0.5
    mag2 = (bcx * bcx + bcy * bcy) ** 0.5
    if mag1 < 1e-6 or mag2 < 1e-6:
        return float("nan")

    dot = bax * bcx + bay * bcy
    cosang = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return float(np.degrees(np.arccos(cosang)))


def best_side_angle(
    pose: Dict[str, Tuple[float, float, float]],
    left_triplet: Tuple[str, str, str],
    right_triplet: Tuple[str, str, str],
    min_conf: float,
    prev_angle: Optional[float],
    max_jump: float,
) -> Optional[float]:
    """
    Computes an angle from left/right triplets and chooses the best candidate by confidence.

    Additional filter:
    - If the chosen angle jumps too far from the previous frame, drop it (or fall back
      to the second-best candidate if it is more consistent).
    """
    candidates: List[Tuple[float, float]] = []  # (angle, conf_sum)

    la = pose.get(left_triplet[0]); lb = pose.get(left_triplet[1]); lc = pose.get(left_triplet[2])
    if la and lb and lc:
        ax, ay, ac = la; bx, by, bc = lb; cx, cy, cc = lc
        if min(ac, bc, cc) >= min_conf:
            a = angle_deg((ax, ay), (bx, by), (cx, cy))
            if np.isfinite(a):
                candidates.append((float(a), float(ac + bc + cc)))

    ra = pose.get(right_triplet[0]); rb = pose.get(right_triplet[1]); rc = pose.get(right_triplet[2])
    if ra and rb and rc:
        ax, ay, ac = ra; bx, by, bc = rb; cx, cy, cc = rc
        if min(ac, bc, cc) >= min_conf:
            a = angle_deg((ax, ay), (bx, by), (cx, cy))
            if np.isfinite(a):
                candidates.append((float(a), float(ac + bc + cc)))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[1], reverse=True)
    chosen = candidates[0][0]

    if prev_angle is not None and np.isfinite(prev_angle):
        if abs(chosen - prev_angle) > max_jump:
            if len(candidates) > 1:
                alt = candidates[1][0]
                if abs(alt - prev_angle) <= max_jump:
                    return alt
            return None

    return chosen


def smooth_angles(angle_sequence: List[Optional[float]], window: int) -> List[Optional[float]]:
    """
    Simple moving-average smoothing for optional jitter reduction.
    """
    if window <= 1:
        return angle_sequence[:]

    n = len(angle_sequence)
    half = window // 2
    out: List[Optional[float]] = [None] * n

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        vals = [v for v in angle_sequence[lo:hi] if v is not None and np.isfinite(v)]
        out[i] = float(sum(vals) / len(vals)) if vals else None

    return out


def default_limb_pairs() -> List[Tuple[str, str]]:
    """
    Fallback skeleton edges when the exercise plugin does not provide limb pairs.
    """
    return [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]


def load_plugin(exercise: str):
    """
    Loads the exercise module (bench/curl/squat).
    """
    return import_any(
        module_paths=[exercise, f"exercise_modules.{exercise}"],
        file_candidates=[
            os.path.join(ROOT_DIR, f"{exercise}.py"),
            os.path.join(ROOT_DIR, "exercise_modules", f"{exercise}.py"),
        ],
    )


def pick_weights_path(arg_path: str) -> str:
    """
    Chooses a weights file based on CLI or local defaults.
    """
    if arg_path and os.path.exists(arg_path):
        return arg_path

    cand1 = os.path.join(ROOT_DIR, "yolov8n-pose.pt")
    cand2 = os.path.join(ROOT_DIR, "yolo11n-pose.pt")

    if os.path.exists(cand1):
        return cand1
    if os.path.exists(cand2):
        return cand2

    return "yolov8n-pose.pt"


def try_open_file(path: str) -> None:
    """
    Best-effort helper for Windows to open a file after it is written.
    """
    try:
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
    except Exception:
        pass


def get_joint_cfg(calib: Dict[str, Any], exercise: str) -> Tuple[str, Dict[str, Any]]:
    """
    Reads the calibration JSON for the requested exercise and returns (joint_name, joint_cfg).
    """
    ex_cfg = calib.get(exercise, {})
    if not isinstance(ex_cfg, dict) or not ex_cfg:
        raise RuntimeError(f"No calibration found for exercise='{exercise}'")

    joint_name = list(ex_cfg.keys())[0]
    joint_cfg = ex_cfg.get(joint_name, {})
    if not isinstance(joint_cfg, dict):
        joint_cfg = {}
    return joint_name, joint_cfg


def build_rep_feature_list(reps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalizes segmented reps into the format expected by plotters.
    """
    rep_feature_list: List[Dict[str, Any]] = []
    for i, r in enumerate(reps or [], start=1):
        rep_feature_list.append(
            {
                "rep_index": i,
                "start": int(r.get("start_idx", r.get("start", 0))),
                "end": int(r.get("end_idx", r.get("end", 0))),
                "label": str(r.get("label", "fail")).lower(),
                "reason": str(r.get("reason", r.get("fail_reason", ""))),
                "rom": float(r.get("rom", 0.0)),
                "duration": float(r.get("duration", r.get("rep_duration", 0.0))),
            }
        )
    return rep_feature_list


def save_reps_csv(path: str, reps: List[Dict[str, Any]]) -> None:
    """
    Writes a compact reps CSV intended for quick inspection and README assets.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("rep_index,start_idx,end_idx,duration,label,reason,rom\n")
        for i, rep in enumerate(reps or [], start=1):
            f.write(
                f"{i},"
                f"{rep.get('start_idx','')},"
                f"{rep.get('end_idx','')},"
                f"{rep.get('duration','')},"
                f"{rep.get('label','')},"
                f"\"{rep.get('reason','')}\""
                f","
                f"{rep.get('rom','')}\n"
            )


def save_angles_csv(path: str, angle_sequence: List[Optional[float]]) -> None:
    """
    Writes per-frame angle values to CSV (useful for debugging and plots).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("frame_index,angle_deg\n")
        for i, a in enumerate(angle_sequence):
            if a is None or not np.isfinite(a):
                f.write(f"{i},\n")
            else:
                f.write(f"{i},{float(a):.4f}\n")


def main() -> None:
    args = parse_args()

    # Convenience flag: produce all artifacts in one run.
    if args.save_all:
        args.save_video = True
        args.save_plots = True
        args.save_angle_csv = True
        args.save_reps_json = True

    # Bridge alias so analysis_service can import shared helpers from "main"
    # even when this file is executed as __main__.
    sys.modules.setdefault("main", sys.modules[__name__])

    from analysis_service import run_analysis

    result = run_analysis(
        video_path=args.input,
        exercise=args.exercise,
        calibration_path=args.calibration,
        output_dir=args.output_dir,
        save_video=args.save_video,
        save_plots=args.save_plots,
        save_angle_csv=args.save_angle_csv,
        save_reps_json=args.save_reps_json,
        weights=args.weights,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        prefix=args.prefix,
        min_joint_conf=args.min_joint_conf,
        overlay_scale=args.overlay_scale,
        open_video=args.open_video,
        debug=args.debug,
    )

    artifacts = result.get("artifacts", {})
    print(f"[INFO] CSV saved: {artifacts.get('reps_csv')}")
    print(f"[INFO] Summary saved: {artifacts.get('summary_json')}")

    if artifacts.get("angles_csv"):
        print(f"[INFO] Angles CSV saved: {artifacts.get('angles_csv')}")
    if artifacts.get("reps_json"):
        print(f"[INFO] Reps JSON saved: {artifacts.get('reps_json')}")
    if args.save_plots:
        print(f"[INFO] Plots saved under: {args.output_dir}")
    if artifacts.get("annotated_video"):
        print(f"[INFO] Video saved: {artifacts.get('annotated_video')}")


if __name__ == "__main__":
    main()