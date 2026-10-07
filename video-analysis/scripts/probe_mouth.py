#!/usr/bin/env python3
"""Task 30 第二部分：嘴部可行性探测（独立脚本，不进 pipeline、不进测试）。

背景：控制器 brief 实测本组（文化4-4.28）就座框高中位数 769px（城市2组
442px），推算头部约 192px、嘴部约 38px——接近人脸检测下限（通常要求人脸
≥80px），"边缘可行"。但现有就座帧间隔中位 17.5 秒，唇动检测需要 ≥10fps，
必须新抽帧才能验证。本脚本只做这一件事：抽一段新帧、跑人脸检测、报告
可行性结论，不产出任何功能代码，不修改 pipeline 任何模块。

用法：
    .venv/bin/python scripts/probe_mouth.py [--order 11] [--duration 20]

产出：
  - 新帧写到 work/文化4-4.28/mouth_probe/frames/（不碰既有的 frames/ 目录）
  - 人脸检测权重缓存在 work/文化4-4.28/mouth_probe/models/（HF 镜像下载，
    首次运行后复用，不重复下载）
  - 打印一份可行性报告到 stdout

人脸检测器选型：优先复用项目已有依赖——ultralytics 本身已是本项目依赖，
且能直接加载任意 YOLO 兼容权重，不需要新增 mediapipe/insightface 之类的
新包。选用 HuggingFace 上的 `arnabdhar/YOLOv8-Face-Detection`（yolov8n 微
调的纯人脸检测器，权重约 6MB，114 赞、非孤儿项目），走
HF_ENDPOINT=https://hf-mirror.com 下载，不产生除该权重文件外的新增体积。
该模型只出人脸框、不出关键点，嘴部区域用「人脸框下 35% 高、中间 50% 宽」
的几何比例估算——足以回答"嘴部区域有多少像素"这个可行性问题，不需要为
一次性探测再引入关键点模型。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config  # noqa: E402
from pipeline.detect import load_detections  # noqa: E402
from pipeline.schedule import Segment, Presenter  # noqa: E402

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
PROBE_DIR = config.WORK / "mouth_probe"
FRAMES_DIR = PROBE_DIR / "frames"
MODEL_DIR = PROBE_DIR / "models"
FACE_MODEL_PATH = MODEL_DIR / "model.pt"
FACE_MODEL_REPO = "arnabdhar/YOLOv8-Face-Detection"
FACE_MODEL_FILE = "model.pt"

PROBE_FPS = 10
FACE_DETECT_RATE_MIN = 0.30   # 低于此判「不可行」
MOUTH_SIZE_MIN_PX = 30.0      # 嘴部区域任一边低于此判「不可行」


def _load_segment(order: int) -> Segment:
    raw = json.loads((config.WORK / "segments.json").read_text(encoding="utf-8"))
    for s in raw["segments"]:
        if s["order"] == order:
            presenters = tuple(Presenter(**p) for p in s["presenters"])
            return Segment(s["order"], s["company"], s["start"], s["end"],
                            s["presenter_num"], presenters, s.get("dta_order", 0))
    raise SystemExit(f"segments.json 里找不到 order={order}")


def _confirmed_seat_x_range(order: int, frame_w: int = 1440) -> tuple[float, float] | None:
    """已确认座位在画面里的左右边界（cx 的 min/max，各留 15% 边距）。

    只用来在探测帧里圈定「大概率是创业者」的区域，不是精确匹配——探测脚本
    本身不跑跟踪，没有 track_id 可用来对齐 seats.json 里的座位。
    """
    seats_raw = json.loads((config.WORK / "seats.json").read_text(encoding="utf-8"))
    seats = seats_raw.get(str(order), {})
    if not seats:
        return None
    dets = load_detections(config.WORK / "detections.json")
    tids = {int(t) for t in seats}
    cxs = [d.cx for d in dets
           if d.seg_order == order and d.kind == "seated" and d.track_id in tids]
    if not cxs:
        return None
    margin = 0.15 * frame_w
    return (min(cxs) - margin, max(cxs) + margin)


def _seated_window_midpoint(order: int) -> float:
    """本段就座窗口（跳过段首/段尾各自的 margin）里靠中间的一个绝对秒数。"""
    seg = _load_segment(order)
    lo = seg.start + config.SEATED_HEAD_MARGIN
    hi = seg.end - config.SEATED_TAIL_MARGIN
    if hi <= lo:
        raise SystemExit(f"order={order} 就座窗口太短，无法探测：{lo}~{hi}")
    return (lo + hi) / 2.0


def extract_probe_frames(video: Path, t_start: float, duration: float) -> list[Path]:
    """用 ffmpeg 以 PROBE_FPS 抽 duration 秒的帧，写到 FRAMES_DIR（先清空旧的）。"""
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAMES_DIR.glob("*.jpg"):
        old.unlink()
    pattern = str(FRAMES_DIR / "f_%05d.jpg")
    cmd = [
        FFMPEG, "-y", "-ss", f"{t_start:.3f}", "-t", f"{duration:.3f}",
        "-i", str(video), "-vf", f"fps={PROBE_FPS}",
        "-qscale:v", "2", pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg 抽帧失败:\n{proc.stderr[-2000:]}")
    return sorted(FRAMES_DIR.glob("*.jpg"))


def ensure_face_model() -> Path:
    if FACE_MODEL_PATH.exists():
        return FACE_MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(repo_id=FACE_MODEL_REPO, filename=FACE_MODEL_FILE,
                                  local_dir=str(MODEL_DIR))
    downloaded = Path(downloaded)
    if downloaded != FACE_MODEL_PATH:
        downloaded.rename(FACE_MODEL_PATH)
    return FACE_MODEL_PATH


def run_probe(order: int, duration: float) -> dict:
    seg = _load_segment(order)
    t_mid = _seated_window_midpoint(order)
    t_start = t_mid - duration / 2.0
    print(f"order={order} 「{seg.company}」就座窗口 {seg.start}~{seg.end}，"
          f"探测起点 t={t_start:.1f}s，时长 {duration:.0f}s，{PROBE_FPS}fps")

    frames = extract_probe_frames(config.VIDEO, t_start, duration)
    print(f"抽帧完成：{len(frames)} 张 -> {FRAMES_DIR}")
    if not frames:
        raise SystemExit("没有抽到任何帧，中止")

    x_range = _confirmed_seat_x_range(order)
    print(f"已确认座位 x 范围（用于圈定候选区域）: {x_range}")

    from ultralytics import YOLO
    person_model = YOLO(config.YOLO_MODEL)
    face_model_path = ensure_face_model()
    face_model = YOLO(str(face_model_path))
    print(f"人脸模型: {face_model_path}（{face_model.names}）")

    person_crops = 0
    face_hits = 0
    face_sizes: list[tuple[float, float]] = []   # (w, h)
    mouth_sizes: list[tuple[float, float]] = []  # (w, h)

    for fp in frames:
        res = person_model.predict(str(fp), classes=[0], conf=config.YOLO_CONF,
                                    imgsz=config.YOLO_IMGSZ, verbose=False)[0]
        if res.boxes is None:
            continue
        import numpy as np
        from PIL import Image
        img = np.asarray(Image.open(fp).convert("RGB"))
        h_img, w_img = img.shape[:2]

        for box in res.boxes.xyxy.tolist():
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            box_h = y2 - y1
            if box_h < config.SEAT_MIN_H_RATIO * h_img:
                continue   # 明显不是完整人形（沿用 seats.py 同一把尺子）
            if x_range and not (x_range[0] <= cx <= x_range[1]):
                continue   # 不在已确认座位区域附近，跳过（评委/工作人员/路人）

            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(w_img, int(x2)), min(h_img, int(y2))
            if ix2 - ix1 < 10 or iy2 - iy1 < 10:
                continue
            crop = img[iy1:iy2, ix1:ix2]
            person_crops += 1

            fres = face_model.predict(crop, conf=0.25, verbose=False)[0]
            if fres.boxes is None or len(fres.boxes) == 0:
                continue
            # 取置信度最高的人脸框
            confs = fres.boxes.conf.tolist()
            best_i = max(range(len(confs)), key=lambda i: confs[i])
            fx1, fy1, fx2, fy2 = fres.boxes.xyxy.tolist()[best_i]
            fw, fh = fx2 - fx1, fy2 - fy1
            if fw <= 0 or fh <= 0:
                continue
            face_hits += 1
            face_sizes.append((fw, fh))
            # 嘴部区域几何估算：人脸框下 35% 高、中间 50% 宽
            mouth_w = fw * 0.5
            mouth_h = fh * 0.35
            mouth_sizes.append((mouth_w, mouth_h))

    def _median(vals: list[float]) -> float:
        s = sorted(vals)
        n = len(s)
        if n == 0:
            return 0.0
        mid = n // 2
        return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0

    detect_rate = face_hits / person_crops if person_crops else 0.0
    face_w_med = _median([w for w, _ in face_sizes])
    face_h_med = _median([h for _, h in face_sizes])
    mouth_w_med = _median([w for w, _ in mouth_sizes])
    mouth_h_med = _median([h for _, h in mouth_sizes])

    feasible = detect_rate >= FACE_DETECT_RATE_MIN and min(mouth_w_med, mouth_h_med) >= MOUTH_SIZE_MIN_PX

    report = {
        "order": order,
        "frames_probed": len(frames),
        "person_crops": person_crops,
        "face_hits": face_hits,
        "face_detect_rate": detect_rate,
        "face_w_median_px": face_w_med,
        "face_h_median_px": face_h_med,
        "mouth_w_median_px": mouth_w_med,
        "mouth_h_median_px": mouth_h_med,
        "feasible": feasible,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--order", type=int, default=11,
                     help="探测哪个 order（默认 11：3 位创业者，就座框最大）")
    ap.add_argument("--duration", type=float, default=20.0, help="抽帧时长（秒）")
    args = ap.parse_args()

    report = run_probe(args.order, args.duration)

    print("\n=== 嘴部可行性探测报告 ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    print()
    if report["feasible"]:
        print("结论：可行（人脸检出率 >= 30% 且嘴部区域中位尺寸 >= 30px）。")
    else:
        reasons = []
        if report["face_detect_rate"] < FACE_DETECT_RATE_MIN:
            reasons.append(f"人脸检出率 {report['face_detect_rate']:.0%} < {FACE_DETECT_RATE_MIN:.0%}")
        if min(report["mouth_w_median_px"], report["mouth_h_median_px"]) < MOUTH_SIZE_MIN_PX:
            reasons.append(
                f"嘴部区域中位尺寸 {report['mouth_w_median_px']:.0f}x"
                f"{report['mouth_h_median_px']:.0f}px 有一边 < {MOUTH_SIZE_MIN_PX:.0f}px"
            )
        print("结论：不可行 —— " + "；".join(reasons))


if __name__ == "__main__":
    main()
