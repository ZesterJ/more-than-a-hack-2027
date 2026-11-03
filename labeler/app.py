"""Barca Vision Model Analyzer: inspect a possession model against ground truth.

Flask backend that serves 25 fps tracking data (byte-offset frame index), loads model
submission CSVs (frame,label), and analyses them against the public ground truth with the
official scorer (evaluation/possession_metrics.py). It does not label anything by hand.
"""

import io
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, request, render_template, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "evaluation"))

from games import GAMES, GAME  # noqa: E402
import possession_metrics as pm  # noqa: E402

app = Flask(__name__, template_folder=os.path.join(HERE, "templates"))
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # a submission CSV is ~2 MB

PITCH_LEN, PITCH_WID = GAMES[GAME]["pitch"]  # 105.3m, 68.0m
FPS = 25
TOTAL_FRAMES = GAMES[GAME]["n_video_frames"]  # 199319

# Candidate locations for data files (supports flexible directory setups on Mac and Windows)
ATD_CANDIDATES = [
    os.path.join(REPO_ROOT, "data", "tactical", "BAR-ATH_tactical_atd.txt"),
    os.path.join(REPO_ROOT, "data", "BAR-ATH_tactical_atd.txt"),
    os.path.join(REPO_ROOT, "..", "data", "tactical", "BAR-ATH_tactical_atd.txt"),
    os.path.join(REPO_ROOT, "BAR-ATH_tactical_atd.txt"),
]
META_CANDIDATES = [p.replace("_atd.txt", "_atd.xml") for p in ATD_CANDIDATES]

VIDEO_CANDIDATES = [
    os.path.join(REPO_ROOT, "data", "broadcast", "BAR-ATH_broadcast.mp4"),
    os.path.join(REPO_ROOT, "data", "BAR-ATH_broadcast.mp4"),
    os.path.join(REPO_ROOT, "..", "BAR-ATH_broadcast.mp4"),
    os.path.join(REPO_ROOT, "BAR-ATH_broadcast.mp4"),
    os.path.join(REPO_ROOT, "data", "tactical", "BAR-ATH_tactical.mp4"),
    os.path.join(REPO_ROOT, "data", "BAR-ATH_tactical.mp4"),
    os.path.join(REPO_ROOT, "BAR-ATH_tactical.mp4"),
]

RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation", "results")
GT_PUBLIC_PATH = os.path.join(REPO_ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

LETTERS = pm.LABELS  # ("H", "A", "D", "X") -> codes 0..3
X_CODE = 3
HANDOVER_MARGIN = 50  # frames (2 s): an error this close to a true change of possession is a handover error
BALL_MISSING_SHARE = 0.5  # an error segment is "ball missing" if at least this share of its frames have no ball


def resolve_existing_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        abs_p = os.path.abspath(p)
        if os.path.isfile(abs_p):
            return abs_p
    return None


RAW_ATD_PATH = resolve_existing_path(ATD_CANDIDATES) or ATD_CANDIDATES[0]
META_PATH = resolve_existing_path(META_CANDIDATES)

# Global state
BYTE_OFFSETS: List[int] = []
BALL_MISSING = np.zeros(TOTAL_FRAMES, dtype=bool)
SLOT_TEAM: List[int] = []  # per track slot: 0 home, 1 away, from the ATD metadata XML
GT_CODES = np.full(TOTAL_FRAMES, X_CODE, dtype=np.int8)
MODELS: Dict[str, np.ndarray] = {}  # name -> codes 0..3 per video frame


def init_slot_teams():
    """Which team each track slot belongs to, read from the metadata XML (not hard-coded)."""
    global SLOT_TEAM
    if META_PATH is None:
        print("[Analyzer] Warning: ATD metadata XML not found; assuming slots 1-24 home")
        SLOT_TEAM = [0] * 24 + [1] * 22
        return
    from starter.baseline import team_of_each_track
    SLOT_TEAM = [int(t) for t in team_of_each_track(META_PATH)]


def init_byte_offsets():
    """Index byte offsets for O(1) frame access and note which frames have no ball."""
    global BYTE_OFFSETS
    if BYTE_OFFSETS:
        return
    t0 = time.time()
    offsets = []
    if os.path.exists(RAW_ATD_PATH):
        with open(RAW_ATD_PATH, "rb") as f:
            offset = 0
            for i, line in enumerate(f):
                offsets.append(offset)
                offset += len(line)
                if i < TOTAL_FRAMES:
                    BALL_MISSING[i] = line.rstrip().rsplit(b":", 1)[-1][:1] == b"N"
        BYTE_OFFSETS = offsets
        print(f"[Analyzer] Indexed {len(BYTE_OFFSETS)} frames in {time.time() - t0:.2f}s")
    else:
        print(f"[Analyzer] Warning: Raw tracking file not found at {RAW_ATD_PATH}")


def init_reference_gt():
    """Load the public ground truth (first 5 minutes of play; everything else is X)."""
    global GT_CODES
    if os.path.exists(GT_PUBLIC_PATH):
        frame = pm.read_label_csv(GT_PUBLIC_PATH)
        GT_CODES = pm.label_codes(frame)
        scored = int((GT_CODES != X_CODE).sum())
        print(f"[Analyzer] Loaded public ground truth: {scored} graded frames")
    else:
        print(f"[Analyzer] Warning: ground truth not found at {GT_PUBLIC_PATH}")


def gt_span() -> Optional[Tuple[int, int]]:
    scored = np.flatnonzero(GT_CODES != X_CODE)
    return (int(scored[0]), int(scored[-1])) if len(scored) else None


def parse_raw_line(line_str: str) -> Tuple[int, List[dict], Optional[dict]]:
    """Parse raw ATD line string into structured players and ball dict."""
    parts = line_str.strip().split(":")
    frame_id = int(parts[0])
    players_raw = parts[1].split(";")
    players = []

    for slot_idx, p in enumerate(players_raw):
        if not p or p.startswith("NaN"):
            continue
        vals = p.split(",")
        try:
            x_norm = float(vals[0])
            y_norm = float(vals[1])
            s = float(vals[2]) if len(vals) > 2 else 0.0
            if math.isnan(s):
                s = 0.0
            is_home = slot_idx < len(SLOT_TEAM) and SLOT_TEAM[slot_idx] == 0
            players.append({
                "slot": slot_idx + 1,
                "team": "H" if is_home else "A",
                "x": x_norm,
                "y": y_norm,
                "s": round(s, 2)
            })
        except (ValueError, IndexError):
            continue

    ball = None
    if len(parts) > 2 and not parts[2].startswith("NaN"):
        bvals = parts[2].split(",")
        try:
            bx = float(bvals[0])
            by = float(bvals[1])
            bz = float(bvals[2]) if len(bvals) > 2 else 0.0
            bs = float(bvals[3]) if len(bvals) > 3 else 0.0
            if not (math.isnan(bx) or math.isnan(by)):
                ball = {
                    "x": bx,
                    "y": by,
                    "z": 0.0 if math.isnan(bz) else round(bz, 2),
                    "s": 0.0 if math.isnan(bs) else round(bs, 2)
                }
        except (ValueError, IndexError):
            ball = None

    return frame_id, players, ball


def compute_frame_telemetry(players: List[dict], ball: Optional[dict]) -> dict:
    """Compute physical metrics and ball proximity for one frame."""
    h_players = [p for p in players if p["team"] == "H"]
    a_players = [p for p in players if p["team"] == "A"]

    h_cx = sum(p["x"] for p in h_players) / len(h_players) if h_players else 0.5
    h_cy = sum(p["y"] for p in h_players) / len(h_players) if h_players else 0.5
    a_cx = sum(p["x"] for p in a_players) / len(a_players) if a_players else 0.5
    a_cy = sum(p["y"] for p in a_players) / len(a_players) if a_players else 0.5

    h_speed = sum(p["s"] for p in h_players) / len(h_players) if h_players else 0.0
    a_speed = sum(p["s"] for p in a_players) / len(a_players) if a_players else 0.0
    overall_speed = sum(p["s"] for p in players) / len(players) if players else 0.0

    nearest_h = None
    nearest_a = None
    min_dh = 999.0
    min_da = 999.0

    if ball is not None:
        bx_m = ball["x"] * PITCH_LEN
        by_m = ball["y"] * PITCH_WID

        for p in players:
            px_m = p["x"] * PITCH_LEN
            py_m = p["y"] * PITCH_WID
            dist = math.hypot(px_m - bx_m, py_m - by_m)
            p["dist_ball"] = round(dist, 2)

            if p["team"] == "H" and dist < min_dh:
                min_dh = dist
                nearest_h = {"slot": p["slot"], "dist": round(dist, 2)}
            elif p["team"] == "A" and dist < min_da:
                min_da = dist
                nearest_a = {"slot": p["slot"], "dist": round(dist, 2)}

        margin = round(min_da - min_dh, 2)
    else:
        for p in players:
            p["dist_ball"] = None
        margin = 0.0

    return {
        "nearest_home": nearest_h,
        "nearest_away": nearest_a,
        "margin": margin,
        "home_centroid": {"x": round(h_cx, 3), "y": round(h_cy, 3)},
        "away_centroid": {"x": round(a_cx, 3), "y": round(a_cy, 3)},
        "home_speed": round(h_speed, 2),
        "away_speed": round(a_speed, 2),
        "overall_speed": round(overall_speed, 2),
    }


def get_single_frame(video_frame: int) -> Optional[dict]:
    """Retrieve full data for a single video frame."""
    if video_frame < 0 or video_frame >= len(BYTE_OFFSETS):
        return None

    offset = BYTE_OFFSETS[video_frame]
    with open(RAW_ATD_PATH, "r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        line = f.readline()

    frame_id, players, ball = parse_raw_line(line)
    telemetry = compute_frame_telemetry(players, ball)

    seconds_total = video_frame / float(FPS)
    minutes = int(seconds_total // 60)
    seconds = int(seconds_total % 60)
    subframes = int((seconds_total - int(seconds_total)) * 100)
    timestamp = f"{minutes:02d}:{seconds:02d}.{subframes:02d}"

    half = 1 if video_frame < 100000 else 2

    return {
        "video_frame": video_frame,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "half": half,
        "players": players,
        "ball": ball,
        "telemetry": telemetry,
        "gt_label": LETTERS[GT_CODES[video_frame]],
        "preds": {name: LETTERS[codes[video_frame]] for name, codes in MODELS.items()},
    }


# ===================== MODEL LOADING & ANALYSIS =====================

def register_model(name: str, source) -> Optional[str]:
    """Read + validate a frame,label CSV and store it. Returns an error string or None."""
    try:
        frame = pm.read_label_csv(source)
    except Exception as exc:  # unreadable / wrong columns
        return f"could not read CSV: {exc}"
    reason = pm.validate_submission(frame, TOTAL_FRAMES)
    if reason:
        return reason
    MODELS[name] = pm.label_codes(frame)
    return None


def available_csvs() -> List[str]:
    if not os.path.isdir(RESULTS_DIR):
        return []
    return sorted(f for f in os.listdir(RESULTS_DIR) if f.lower().endswith(".csv"))


def runs_of(codes: np.ndarray) -> List[list]:
    """Run-length encode a code array as [start, end_inclusive, code]."""
    if not len(codes):
        return []
    change = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change - 1, [len(codes) - 1]))
    return [[int(s), int(e), int(codes[s])] for s, e in zip(starts, ends)]


def error_mask(pred: np.ndarray) -> np.ndarray:
    """True where the ground truth is scored and the model disagrees (an X counts as a miss)."""
    return (GT_CODES != X_CODE) & (pred != GT_CODES)


def analyse(pred: np.ndarray) -> dict:
    """Official metrics plus a breakdown of where the model's errors fall."""
    res = pm.evaluate(GT_CODES, pred)
    per_class = {k: {"f1": v["f1"], "precision": v["precision"], "recall": v["recall"], "support": v["support"]}
                 for k, v in res["per_class"].items()}

    err = error_mask(pred)
    graded = int((GT_CODES != X_CODE).sum())
    gt16 = GT_CODES.astype(np.int16)
    gt_change = np.flatnonzero(np.diff(gt16) != 0) + 1
    gt_change = gt_change[(GT_CODES[gt_change] != X_CODE) & (GT_CODES[gt_change - 1] != X_CODE)]

    # error segments: contiguous stretches with the same (truth, prediction) pair
    pair = np.where(err, gt16 * 4 + pred, -1)
    segments = []
    for start, end, key in runs_of(pair):
        if key < 0:
            continue
        missing = float(BALL_MISSING[start:end + 1].mean())
        lo = np.searchsorted(gt_change, start - HANDOVER_MARGIN)
        hi = np.searchsorted(gt_change, end + HANDOVER_MARGIN, side="right")
        segments.append({
            "start": start, "end": end, "frames": end - start + 1,
            "seconds": round((end - start + 1) / FPS, 2),
            "truth": LETTERS[key // 4], "pred": LETTERS[key % 4],
            "ball_missing": round(missing, 2),
            "handover": bool(hi > lo),
        })

    def frames_where(cond):
        return int(sum(s["frames"] for s in segments if cond(s)))

    breakdown = {
        "error_frames": int(err.sum()),
        "graded_frames": graded,
        "ball_missing": frames_where(lambda s: s["ball_missing"] >= BALL_MISSING_SHARE),
        "ball_visible": frames_where(lambda s: s["ball_missing"] < BALL_MISSING_SHARE),
        "handover": frames_where(lambda s: s["handover"]),
        "mid_possession": frames_where(lambda s: not s["handover"]),
        "by_pair": {},
    }
    for s in segments:
        key = f"{s['truth']}>{s['pred']}"
        breakdown["by_pair"][key] = breakdown["by_pair"].get(key, 0) + s["frames"]

    return {
        "metrics": {
            "spell_f1": res["spell_f1_mean"],
            "macro_f1": res["macro_f1"],
            "accuracy": res["accuracy"],
            "n_gt_spells": res["n_gt_spells"],
            "n_pred_spells": res["n_pred_spells"],
            "mean_spell_s": res["mean_spell_duration_s"],
            "gt_mean_spell_s": res["gt_mean_spell_duration_s"],
            "n_abstain_on_graded": res["n_abstain_on_graded"],
            "per_class": per_class,
        },
        "breakdown": breakdown,
        "segments": segments,
    }


# ===================== FLASK ROUTES =====================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config")
def api_config():
    """Return match dimensions, frame bounds, and initial state."""
    video_path = resolve_existing_path(VIDEO_CANDIDATES)
    span = gt_span()
    return jsonify({
        "game": GAME,
        "total_frames": TOTAL_FRAMES,
        "fps": FPS,
        "pitch": [PITCH_LEN, PITCH_WID],
        "default_kickoff": 11185,
        "default_video_offset": 4013,  # Calibrated kickoff alignment offset
        "video_detected": video_path is not None,
        "video_name": os.path.basename(video_path) if video_path else None,
        "gt_span": list(span) if span else None,
    })


@app.route("/api/frame/<int:video_frame>")
def api_frame(video_frame):
    frame_data = get_single_frame(video_frame)
    if frame_data is None:
        return jsonify({"error": "Frame out of bounds"}), 404
    return jsonify(frame_data)


@app.route("/api/chunk")
def api_chunk():
    """Return a contiguous chunk of frames for smooth frontend playback."""
    start = int(request.args.get("start", 11185))
    length = int(request.args.get("length", 100))
    length = min(max(length, 1), 300)

    frames = []
    end = min(start + length, len(BYTE_OFFSETS))
    for vf in range(start, end):
        fd = get_single_frame(vf)
        if fd:
            frames.append(fd)
    return jsonify({"start": start, "end": end, "count": len(frames), "frames": frames})


@app.route("/api/models")
def api_models():
    return jsonify({
        "loaded": sorted(MODELS),
        "available": [f for f in available_csvs() if os.path.splitext(f)[0] not in MODELS],
    })


@app.route("/api/models/load", methods=["POST"])
def api_models_load():
    """Load a CSV from evaluation/results/ by file name."""
    data = request.get_json(force=True, silent=True) or {}
    fname = os.path.basename(str(data.get("file", "")))  # basename: never read outside results/
    path = os.path.join(RESULTS_DIR, fname)
    if not fname.lower().endswith(".csv") or not os.path.isfile(path):
        return jsonify({"error": f"{fname or 'file'} not found in evaluation/results/"}), 404
    name = os.path.splitext(fname)[0]
    err = register_model(name, path)
    if err:
        return jsonify({"error": f"{fname}: {err}"}), 400
    return jsonify({"success": True, "name": name})


@app.route("/api/models/upload", methods=["POST"])
def api_models_upload():
    """Load a CSV picked in the browser. Kept in memory only."""
    up = request.files.get("file")
    if up is None or not up.filename:
        return jsonify({"error": "no file"}), 400
    name = os.path.splitext(os.path.basename(up.filename))[0]
    err = register_model(name, io.BytesIO(up.read()))
    if err:
        return jsonify({"error": f"{up.filename}: {err}"}), 400
    return jsonify({"success": True, "name": name})


@app.route("/api/models/remove", methods=["POST"])
def api_models_remove():
    data = request.get_json(force=True, silent=True) or {}
    MODELS.pop(str(data.get("name", "")), None)
    return jsonify({"success": True})


@app.route("/api/ribbon")
def api_ribbon():
    """Run-length encoded ground truth, each model, and each model's error stretches."""
    return jsonify({
        "letters": list(LETTERS),
        "gt": runs_of(GT_CODES),
        "models": {n: runs_of(c) for n, c in MODELS.items()},
        "errors": {n: [r for r in runs_of(error_mask(c).astype(np.int8)) if r[2] == 1]
                   for n, c in MODELS.items()},
    })


@app.route("/api/analysis")
def api_analysis():
    name = request.args.get("name", "")
    if name not in MODELS:
        return jsonify({"error": "model not loaded"}), 404
    if not (GT_CODES != X_CODE).any():
        return jsonify({"error": "no ground truth available"}), 400
    return jsonify(analyse(MODELS[name]))


@app.route("/api/video")
def serve_video():
    """Stream match video if available in workspace or data directories."""
    video_path = resolve_existing_path(VIDEO_CANDIDATES)
    if video_path:
        return send_from_directory(os.path.dirname(video_path), os.path.basename(video_path), conditional=True)
    return jsonify({"error": "Video file not found on server. Use client-side file picker."}), 404


def initialize():
    """Initialize index and datasets on boot."""
    init_slot_teams()
    init_byte_offsets()
    init_reference_gt()
    # Pre-load a committed baseline so the UI has something to show on a fresh start.
    path = os.path.join(RESULTS_DIR, "baseline_smooth50.csv")
    if os.path.isfile(path):
        register_model("baseline_smooth50", path)


if __name__ == "__main__":
    initialize()
    print("[Analyzer] Starting server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
