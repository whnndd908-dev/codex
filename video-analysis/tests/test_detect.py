from pathlib import Path
import pytest
from pipeline import config
from pipeline.frames import load_index
from pipeline.detect import (
    Detection, detect_frames, save_detections, load_detections, passes_size_filter,
    _reset_tracker,
)


def test_detection_geometry_properties():
    d = Detection(1, "seated", 0.0, Path("x.jpg"), 3, 100.0, 200.0, 300.0, 600.0, 0.9)
    assert d.cx == 200.0
    assert d.bottom_y == 600.0
    assert d.height == 400.0


def test_passes_size_filter_boundary_cases():
    frame_h = 1080
    threshold = config.MIN_BBOX_H_RATIO * frame_h

    # 恰好等于阈值：应通过（判断用 >=）
    assert passes_size_filter(threshold, frame_h) is True

    # 略低于阈值：应拒绝
    assert passes_size_filter(threshold - 1.0, frame_h) is False

    # 明显高于阈值：应通过
    assert passes_size_filter(threshold * 3, frame_h) is True


class _FakeTracker:
    def __init__(self):
        self.reset_called = False

    def reset(self):
        self.reset_called = True


class _FakePredictorWithTrackers:
    def __init__(self):
        self.trackers = [_FakeTracker()]


class _FakeModelWithTrackers:
    def __init__(self):
        self.predictor = _FakePredictorWithTrackers()


class _FakeModelEmptyTrackers:
    class predictor:
        trackers = []


class _FakeModelNoPredictor:
    predictor = None


def test_reset_tracker_calls_reset_when_trackers_present():
    model = _FakeModelWithTrackers()
    _reset_tracker(model)
    assert model.predictor.trackers[0].reset_called is True


def test_reset_tracker_raises_when_trackers_list_is_empty():
    # trackers 属性存在但是空列表：说明 on_predict_start 没有正常挂载跟踪器。
    # 这必须是可观测的失败（抛异常），而不是静默跳过——否则 track_id 会跨
    # (seg_order, kind) 窗口串号且没有任何报错信号。
    with pytest.raises(RuntimeError, match="跟踪器重置失败"):
        _reset_tracker(_FakeModelEmptyTrackers())


@pytest.mark.skipif(
    not (config.WORK / "frames_index_pilot.json").exists(),
    reason="该断言依赖某一组特有的 pilot 抽帧产物；换视频组后不存在，跳过而非硬失败",
)
def test_reset_tracker_raises_when_predictor_missing():
    # predictor 从未被赋值（None）的情况同样必须报错，而不是被 getattr 吞掉。
    with pytest.raises(RuntimeError, match="跟踪器重置失败"):
        _reset_tracker(_FakeModelNoPredictor())


@pytest.mark.slow
@pytest.mark.skipif(
    not (config.WORK / "frames_index_pilot.json").exists(),
    reason="该断言依赖某一组特有的 pilot 抽帧产物；换视频组后不存在，跳过而非硬失败",
)
def test_detects_people_in_known_seated_frame():
    frames = [f for f in load_index(config.WORK / "frames_index_pilot.json")
              if f.seg_order == 2 and f.kind == "seated"][:5]
    dets = detect_frames(frames)
    # 北斗羲和段：左侧评委席 4-6 人 + 右侧 3 位创业者 + 计时员
    per_frame = {}
    for d in dets:
        per_frame.setdefault(d.frame_path, []).append(d)
    assert all(len(v) >= 5 for v in per_frame.values()), "每帧至少应检出评委席若干人"


@pytest.mark.slow
@pytest.mark.skipif(
    not (config.WORK / "frames_index_pilot.json").exists(),
    reason="该断言依赖某一组特有的 pilot 抽帧产物；换视频组后不存在，跳过而非硬失败",
)
def test_right_region_has_detections_in_pilot_frames():
    frames = [f for f in load_index(config.WORK / "frames_index_pilot.json")
              if f.seg_order == 2 and f.kind == "seated"][:5]
    dets = detect_frames(frames)
    right = [d for d in dets if d.cx > config.RIGHT_REGION_X * 1440]
    assert right, "右半区未检出任何人，RIGHT_REGION_X 需要重新校准"


def test_roundtrip(tmp_path):
    dets = [Detection(1, "seated", 0.0, Path("x.jpg"), 3, 1.0, 2.0, 3.0, 4.0, 0.9)]
    p = tmp_path / "d.json"
    save_detections(dets, p)
    assert load_detections(p) == dets


# ---------------------------------------------------------------------------
# 推理设备与精度（用户 2026-09-08：「这个效率太低了，有没有可能优化」）
#
# 根因是 ultralytics 的 `select_device('')` 在这台机器上返回 **cpu** —— MPS
# 可用却从没被选中，而整条流水线一次要跑两万多帧。实测同一批 150 帧：
#   cpu fp32   0.955 秒/帧
#   mps fp32   0.444 秒/帧
#   mps fp16   0.192 秒/帧
# 三者检测框数量一致（763），坐标差 ~0.1px（fp16 最大 1.0px），置信度差
# ≤0.05；唯一的结构差别是 fp16 下 track 编号整体平移了一位，而 track 编号
# 本就跨重检测不稳定（Task 46 起标注一律以 stable_id 为准）。
# ---------------------------------------------------------------------------

def test_device_defaults_to_the_gpu_when_one_is_available(monkeypatch):
    from pipeline import detect as _d
    monkeypatch.delenv("DRESS_YOLO_DEVICE", raising=False)
    monkeypatch.delenv("DRESS_YOLO_QUANTIZE", raising=False)
    monkeypatch.setattr(_d, "_probe_devices", lambda: (False, True))   # cuda 无、mps 有
    assert _d._device_and_quantize() == ("mps", 16)
    monkeypatch.setattr(_d, "_probe_devices", lambda: (True, False))   # cuda 有
    assert _d._device_and_quantize() == ("0", 16)


def test_device_falls_back_to_cpu_in_full_precision(monkeypatch):
    """没有 GPU 时回到 cpu，且**不开 fp16**——CPU 上的半精度只会更慢。"""
    from pipeline import detect as _d
    monkeypatch.delenv("DRESS_YOLO_DEVICE", raising=False)
    monkeypatch.delenv("DRESS_YOLO_QUANTIZE", raising=False)
    monkeypatch.setattr(_d, "_probe_devices", lambda: (False, False))
    assert _d._device_and_quantize() == ("cpu", None)


def test_device_and_precision_can_be_overridden_by_env(monkeypatch):
    """服务器上的取值与本机不同（用户要把流水线搬上服务器），必须能覆盖，
    与 `frames.DRESS_FFMPEG` 同一条路子。"""
    from pipeline import detect as _d
    monkeypatch.setattr(_d, "_probe_devices", lambda: (False, True))
    monkeypatch.setenv("DRESS_YOLO_DEVICE", "cpu")
    monkeypatch.delenv("DRESS_YOLO_QUANTIZE", raising=False)
    assert _d._device_and_quantize() == ("cpu", None)
    monkeypatch.setenv("DRESS_YOLO_DEVICE", "0")
    monkeypatch.setenv("DRESS_YOLO_QUANTIZE", "32")
    assert _d._device_and_quantize() == ("0", None)
    monkeypatch.setenv("DRESS_YOLO_QUANTIZE", "16")
    assert _d._device_and_quantize() == ("0", 16)


def test_detect_frames_passes_device_and_precision_to_the_tracker(monkeypatch, tmp_path):
    """选出来的设备/精度必须真的传给 `model.track`。

    这条钉的正是这次的根因：设备选对了、却没传下去，跑起来一切正常，只是
    慢五倍，而输出里看不出任何异常。
    """
    from pipeline import detect as _d
    seen = []

    class _Boxes:
        id = None
    class _Res:
        boxes = _Boxes()
    class _FakeModel:
        predictor = _FakePredictorWithTrackers()
        def track(self, path, **kw):
            seen.append(kw)
            return [_Res()]

    monkeypatch.setattr(_d, "_get_model", lambda: _FakeModel())
    monkeypatch.setattr(_d, "_device_and_quantize", lambda: ("mps", 16))
    frame = _d.FrameRef if hasattr(_d, "FrameRef") else None
    from pipeline.frames import FrameRef
    _d.detect_frames([FrameRef(1, "entry", 0.0, tmp_path / "a.jpg")])
    assert seen and seen[0].get("device") == "mps", seen
    assert seen[0].get("quantize") == 16, seen


# ---------------------------------------------------------------------------
# 增量检测（用户 2026-09-09：「能只对 order 16 重抽吗，不抽其他项目，
# 是不是就可以省时间了」）。
#
# 能，而且是**可证明等价**的，不是近似：`detect_frames` 按 (seg_order, kind)
# 分组顺序推理，每组跑完调 `_reset_tracker`，而 ultralytics 的
# `BYTETracker.reset()` 连全局 track 编号计数器一起归零（`BaseTrack.reset_id`）。
# 因此每一组的检测结果只取决于**这一组自己的帧**，与别的组跑没跑过无关——
# 只重跑帧集合变了的那几组，其余原样留用，结果与全量重跑逐条相同。
#
# 帧集合变没变，用每组一份摘要判断，存在 detections.json 旁边的小文件里；
# 摘要缺失（旧产物）时退回全量，不猜。
# ---------------------------------------------------------------------------

def _fr(order, kind, t, tmp_path):
    from pipeline.frames import FrameRef
    return FrameRef(order, kind, t, tmp_path / f"{order:02d}" / kind / f"{t:09.3f}.jpg")


def test_group_digest_changes_only_when_that_groups_frames_change(tmp_path):
    from pipeline.detect import frames_group_digests
    a = [_fr(1, "entry", 0.0, tmp_path), _fr(1, "entry", 0.2, tmp_path),
         _fr(2, "exit", 5.0, tmp_path)]
    d1 = frames_group_digests(a)
    assert set(d1) == {(1, "entry"), (2, "exit")}
    # 顺序无关
    assert frames_group_digests(list(reversed(a))) == d1
    # 只给第 1 组加一帧：只有它的摘要变
    b = a + [_fr(1, "entry", 0.4, tmp_path)]
    d2 = frames_group_digests(b)
    assert d2[(2, "exit")] == d1[(2, "exit")]
    assert d2[(1, "entry")] != d1[(1, "entry")]


def test_incremental_detection_equals_a_full_rerun(monkeypatch, tmp_path):
    """只跑变了的那几组，结果必须与全量重跑**逐条相同**。

    这是这条优化唯一站得住的理由：跟踪器每组开头归零（含 track 编号计数器），
    所以每组只取决于自己的帧。等价性一旦不成立，省下来的时间买到的是一份
    和全量跑不一样的检测结果，而它看起来完全正常。
    """
    from pipeline import detect as _d
    from pipeline.detect import (detect_frames, detect_frames_incremental,
                                 frames_group_digests)

    calls = []

    class _Boxes:
        def __init__(self, n): self._n = n
        id = None
    class _Res:
        boxes = None
    class _FakeModel:
        predictor = _FakePredictorWithTrackers()
        def track(self, path, **kw):
            calls.append(path)
            return [_Res()]

    monkeypatch.setattr(_d, "_get_model", lambda: _FakeModel())
    monkeypatch.setattr(_d, "_device_and_quantize", lambda: ("cpu", None))

    old_frames = [_fr(1, "entry", 0.0, tmp_path), _fr(2, "exit", 5.0, tmp_path)]
    prior = detect_frames(old_frames)
    digests = frames_group_digests(old_frames)

    new_frames = old_frames + [_fr(2, "exit", 5.2, tmp_path)]
    calls.clear()
    got, new_digests, reused, redone = detect_frames_incremental(
        new_frames, prior, digests)
    assert redone == [(2, "exit")], redone
    assert reused == [(1, "entry")], reused
    assert all("/01/" not in c for c in calls), (
        f"没变的那一组不该再跑一遍推理：{calls}")

    calls.clear()
    full = detect_frames(new_frames)
    assert got == full, "增量的结果必须与全量逐条相同"
    assert new_digests == frames_group_digests(new_frames)


def test_incremental_falls_back_to_full_without_prior_digests(monkeypatch, tmp_path):
    """没有上一轮的摘要（旧产物）就全量重跑，**不猜**哪几组还能用。"""
    from pipeline import detect as _d
    from pipeline.detect import detect_frames_incremental

    class _Res:
        boxes = None
    class _FakeModel:
        predictor = _FakePredictorWithTrackers()
        def track(self, path, **kw):
            return [_Res()]

    monkeypatch.setattr(_d, "_get_model", lambda: _FakeModel())
    monkeypatch.setattr(_d, "_device_and_quantize", lambda: ("cpu", None))
    frames = [_fr(1, "entry", 0.0, tmp_path), _fr(2, "exit", 5.0, tmp_path)]
    _got, _dig, reused, redone = detect_frames_incremental(frames, [], None)
    assert reused == [] and sorted(redone) == [(1, "entry"), (2, "exit")]
