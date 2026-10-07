"""阶段 ③：YOLO 人物检测 + BoT-SORT 跟踪。"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from pipeline import config


@dataclass(frozen=True)
class Detection:
    """一帧图像中一个人的检测框 + 跟踪身份。

    track_id 的作用域是 (seg_order, kind)，不是全局、也不是仅 seg_order：
    跟踪器在每个 (seg_order, kind) 窗口开始时都会重新从 1 编号（见
    detect_frames / _reset_tracker）。因此同一 seg_order 内，"entry" 窗口的
    track_id=5 和 "seated" 窗口的 track_id=5 不是同一个人；跨 seg_order 的
    同号更不是。任何按 track_id 关联同一个人的下游代码，都必须同时用
    (seg_order, kind, track_id) 三元组做键，仅用 track_id 或仅用
    (seg_order, track_id) 关联会把不同的人错认成同一个人，而且不会报错。
    """

    seg_order: int
    kind: str
    t: float
    frame_path: Path
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def bottom_y(self) -> float:
        return self.y2

    @property
    def height(self) -> float:
        return self.y2 - self.y1


def passes_size_filter(box_h: float, frame_h: int) -> bool:
    """bbox 高度是否达到入选门槛。

    过小的框多半是背景里的路人或误检，不足以支撑服装颜色测量。
    """
    return box_h >= config.MIN_BBOX_H_RATIO * frame_h


def _probe_devices() -> tuple[bool, bool]:
    """(有没有 CUDA, 有没有 MPS)。单独一个函数是为了测试能替换掉它——
    本机恒为 (False, True)，服务器上是 (True, False)，都不该写死。"""
    try:
        import torch
        return bool(torch.cuda.is_available()), bool(torch.backends.mps.is_available())
    except Exception:                                   # pragma: no cover - 没装 torch
        return False, False


def _device_and_quantize() -> tuple[str, int | None]:
    """推理跑在哪、用什么精度。

    **根因记录（用户 2026-09-08：「这个效率太低了，有没有可能优化」）**：
    ultralytics 的 `select_device('')` 在这台机器上返回 **cpu**——MPS 可用却
    从没被选中，而一次要跑两万多帧。它不报错、不打日志，只是慢五倍。

    实测同一批 150 帧（城市1 order 6 entry）：

        cpu fp32   0.955 秒/帧   → 22877 帧约 6.1 小时
        mps fp32   0.444 秒/帧   → 约 2.8 小时
        mps fp16   0.192 秒/帧   → 约 1.2 小时

    三者检测框数量一致（763 个）、逐框坐标差约 0.1px（fp16 最大 1.0px）、
    置信度差 ≤0.05；唯一的结构差别是 fp16 下 track 编号整体平移一位，而
    track 编号本就跨重检测不稳定（Task 46 起标注一律以 stable_id 为准，
    重检测后按视频内容重新对齐）。

    多进程并行**没有用**：MPS 上 1/2/3/4 个进程的总吞吐都在 1.9 帧/秒左右
    （4 个反而更慢），GPU 已被单进程占满。`stream=True` 传一串路径会让
    ultralytics 攒成一个大 batch，直接 MPS OOM。两条路都试过，别再试。

    CPU 上不开半精度：那只会更慢。两个环境变量都可覆盖（服务器上取值不同，
    与 `frames.DRESS_FFMPEG` 同一条路子）。
    """
    dev = os.environ.get("DRESS_YOLO_DEVICE")
    if not dev:
        has_cuda, has_mps = _probe_devices()
        dev = "0" if has_cuda else ("mps" if has_mps else "cpu")
    q_env = os.environ.get("DRESS_YOLO_QUANTIZE")
    if q_env:
        quant = 16 if q_env.strip() == "16" else None
    else:
        quant = None if dev == "cpu" else 16
    return dev, quant


_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(config.YOLO_MODEL)
    return _model


def _reset_tracker(model) -> None:
    """在切换窗口时重置跟踪器，避免 track_id 跨段串号。

    detect_frames 只在某个 (seg_order, kind) 组至少推理过一帧之后才调用本函数，
    此时 model.predictor.trackers 理应已经存在——ultralytics 的
    on_predict_start 回调会在首次调用 track()/predict() 时把它挂载好。

    因此"找不到 trackers"不是一个可以静默忽略的正常分支：一旦出现（比如未来
    某个 ultralytics 版本改了 predictor/trackers 的属性名或结构），意味着重置
    根本没有发生，而下一组的跟踪状态会继承上一组的 track_id，产生跨窗口串号——
    且不会抛任何异常，下游会拿到一份格式完整、数字看起来合理、但把不同的人
    张冠李戴的结果表。这个后果比让 pipeline 在此处硬失败更严重，所以选择直接
    抛异常中止，而不是打日志后继续跑。
    """
    trackers = getattr(getattr(model, "predictor", None), "trackers", None)
    if not trackers:
        raise RuntimeError(
            "跟踪器重置失败：model.predictor.trackers 不存在或为空，"
            "无法在切换 (seg_order, kind) 窗口时重置 BoT-SORT 状态。"
            "继续跑下去会导致 track_id 跨窗口串号且不会报错，已主动中止。"
            "请检查当前 ultralytics 版本的 predictor/trackers API 是否有变化。"
        )
    trackers[0].reset()


def detect_frames(frames, frame_h: int = 1080) -> list[Detection]:
    """按 (seg_order, kind) 分组顺序推理，使跟踪器在窗口内保持 track_id 连续。

    每处理完一组就调用 _reset_tracker，让跟踪器在下一组从头开始编号。
    这意味着返回的 Detection.track_id 只在同一个 (seg_order, kind) 组内有效：
    见 Detection 类文档字符串——不能只用 track_id（或 seg_order+track_id）
    跨 kind / 跨 seg_order 关联同一个人。
    """
    model = _get_model()
    device, quantize = _device_and_quantize()
    groups: dict[tuple[int, str], list] = {}
    for f in frames:
        groups.setdefault((f.seg_order, f.kind), []).append(f)

    out: list[Detection] = []
    # 跑在哪、用什么精度必须打出来：这次「慢五倍」正是因为它一直悄悄跑在 CPU
    # 上，输出里看不出任何异常。同理逐组打进度——上一轮跑了七个小时，期间
    # 一行输出都没有，无从判断它是在干活还是卡住了。
    print(f"   推理设备 {device}"
          f"{'（半精度 fp16）' if quantize == 16 else '（全精度）'}，"
          f"共 {len(frames)} 帧 / {len(groups)} 个窗口", flush=True)

    for i, ((order, kind), fs) in enumerate(groups.items(), 1):
        fs = sorted(fs, key=lambda x: x.t)
        print(f"   [{i}/{len(groups)}] order={order} {kind} {len(fs)} 帧", flush=True)
        for f in fs:
            res = model.track(
                str(f.path),
                persist=True,
                classes=[0],                 # 仅 person
                conf=config.YOLO_CONF,
                imgsz=config.YOLO_IMGSZ,
                tracker="botsort.yaml",
                verbose=False,
                device=device,
                quantize=quantize,
            )[0]
            if res.boxes is None or res.boxes.id is None:
                continue
            for box, tid, cf in zip(
                res.boxes.xyxy.tolist(),
                res.boxes.id.int().tolist(),
                res.boxes.conf.tolist(),
            ):
                x1, y1, x2, y2 = box
                if not passes_size_filter(y2 - y1, frame_h):
                    continue
                out.append(Detection(order, kind, f.t, f.path, int(tid),
                                     x1, y1, x2, y2, float(cf)))
        _reset_tracker(model)   # 换窗口时重置，避免 id 串段
    return out


def frames_group_digests(frames) -> dict[tuple[int, str], str]:
    """每个 `(seg_order, kind)` 组一份帧集合摘要，用来判断这一组变没变。

    摘要覆盖组内每一帧的时间与路径，与顺序无关（先排序再算）——索引是可以被
    多次拼接的，同一批帧换个顺序不该被判成「变了」而白跑一遍推理。
    """
    groups: dict[tuple[int, str], list[str]] = {}
    for f in frames:
        groups.setdefault((int(f.seg_order), str(f.kind)), []).append(
            f"{float(f.t):.3f}|{f.path}")
    return {k: hashlib.sha256("\n".join(sorted(v)).encode("utf-8")).hexdigest()[:16]
            for k, v in groups.items()}


def detect_frames_incremental(frames, prior_dets, prior_digests,
                              frame_h: int = 1080):
    """只对**帧集合变了的那几组**重跑检测，其余组原样留用上一轮的结果。

    返回 `(dets, digests, reused_groups, redone_groups)`。

    **为什么这是可证明等价的**（用户 2026-09-09：「能只对 order 16 重抽吗，
    不抽其他项目」）：`detect_frames` 按 (seg_order, kind) 分组顺序推理，每组
    跑完调 `_reset_tracker`，而 ultralytics 的 `BYTETracker.reset()` 连全局
    track 编号计数器一起归零（`BaseTrack.reset_id`）。因此每一组的检测结果
    只取决于**这一组自己的帧**，与别的组跑没跑过、跑的顺序都无关。
    tests/test_detect.py 里有一条逐条比对增量与全量结果的断言钉住这一点——
    这条等价性一旦不成立，省下来的时间买到的是一份和全量跑不一样的检测结果，
    而它看起来完全正常。

    没有上一轮摘要（旧产物）时全量重跑，**不猜**哪几组还能用。

    留用的那部分**原样搬运上一轮的 Detection**，不做任何重算：重算等于把
    「等价」变成「但愿等价」。
    """
    digests = frames_group_digests(frames)
    if not prior_digests:
        return detect_frames(frames, frame_h), digests, [], sorted(digests)

    reused_keys = [k for k, d in digests.items()
                   if prior_digests.get(k) == d]
    redone_keys = sorted(k for k in digests if k not in set(reused_keys))
    reuse = {k for k in reused_keys}
    kept = [d for d in (prior_dets or [])
            if (int(d.seg_order), str(d.kind)) in reuse]
    fresh_frames = [f for f in frames
                    if (int(f.seg_order), str(f.kind)) not in reuse]
    fresh = detect_frames(fresh_frames, frame_h) if fresh_frames else []
    out = kept + fresh
    # 排序口径与全量跑一致（按组、组内按时间），这样两条路产出的列表逐条可比。
    order = {k: i for i, k in enumerate(
        dict.fromkeys((int(f.seg_order), str(f.kind)) for f in frames))}
    out.sort(key=lambda d: (order.get((int(d.seg_order), str(d.kind)), 1 << 30),
                            d.t, d.track_id, d.x1))
    return out, digests, sorted(reused_keys), redone_keys


def group_digests_path(det_path: Path) -> Path:
    """帧集合摘要存在 detections.json **旁边**，不塞进它自己。

    塞进去会改变 `detections_fingerprint`（它是整份 detections.json 的
    sha256），而那个指纹是标注文件「是不是这一份检测的」的唯一判据——为了
    一个加速功能去动它，等于让所有既有标注凭空作废一次。
    """
    return Path(det_path).with_name(Path(det_path).stem + "_groups.json")


def save_group_digests(digests: dict, path: Path) -> None:
    Path(path).write_text(
        json.dumps({f"{o}|{k}": v for (o, k), v in sorted(digests.items())},
                   ensure_ascii=False, indent=1), encoding="utf-8")


def load_group_digests(path: Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    out = {}
    for key, v in (raw or {}).items():
        order, _, kind = str(key).partition("|")
        try:
            out[(int(order), kind)] = str(v)
        except ValueError:
            continue
    return out or None


def save_detections(dets: list[Detection], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"seg_order": d.seg_order, "kind": d.kind, "t": d.t, "frame_path": str(d.frame_path),
         "track_id": d.track_id, "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2, "conf": d.conf}
        for d in dets
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_detections(path: Path) -> list[Detection]:
    return [
        Detection(d["seg_order"], d["kind"], d["t"], Path(d["frame_path"]), d["track_id"],
                  d["x1"], d["y1"], d["x2"], d["y2"], d["conf"])
        for d in json.loads(path.read_text(encoding="utf-8"))
    ]
