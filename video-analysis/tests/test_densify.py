"""加密抽帧的 CLI（`pipeline/densify.py`）。"""
import json
import subprocess

import pytest

from pipeline import paths
from pipeline.frames import FFMPEG


def _tiny_video(path, seconds=4, rate=50):
    subprocess.run(
        [FFMPEG, "-y", "-f", "lavfi",
         "-i", f"testsrc=duration={seconds}:size=320x240:rate={rate}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-loglevel", "error", str(path)],
        check=True,
    )


def _work_with_a_named_moment(tmp_path, sheet="组"):
    """一个只有一段、且该段座位 1 填了重抽时刻的 work 目录。"""
    w = tmp_path / "grp"
    w.mkdir()
    (w / "segments.json").write_text(json.dumps({"segments": [
        {"order": 1, "company": "甲", "start": 1.0, "end": 3.0,
         "presenter_num": 1, "presenters": [], "dta_order": 1},
    ]}, ensure_ascii=False), encoding="utf-8")
    lp = paths.labels_path(w, sheet)
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps({
        "_meta": {},
        "no_candidate_seats": {"1": [1]},
        "no_candidate_notes": {"1": {"1": {"at": "2.0"}}},
    }, ensure_ascii=False), encoding="utf-8")
    return w


def test_the_named_moment_interval_uses_the_content_rate(tmp_path, monkeypatch,
                                                         capsys):
    """指名时刻的间隔按**独立内容帧率**算，不按容器标称的帧率。

    这组视频标称 50fps、实际只有 25fps 的独立内容（每帧成对出现）。按标称算得
    0.02 秒，抽出来一半是重复画面——2026-09-18 那次加密 16000 张里约 8000 张重复，
    而检测把它们全跑了一遍，41 分钟白烧一半。
    """
    from pipeline import densify, frames

    video = tmp_path / "v.mp4"
    _tiny_video(video)
    w = _work_with_a_named_moment(tmp_path)

    calls = []
    monkeypatch.setattr(frames, "content_fps",
                        lambda v: calls.append("content") or 25.0)
    monkeypatch.setattr(frames, "probe_fps",
                        lambda v: calls.append("nominal") or 50.0)

    densify.main(["--work", str(w), "--sheet", "组", "--video", str(video),
                  "--duration", "4", "--dry-run"])
    out = capsys.readouterr().out
    assert "content" in calls, f"没有按独立内容帧率算间隔：{out}"
    assert "nominal" not in calls, f"还在用容器标称的帧率：{out}"
    assert "间隔 0.040s" in out, out


def test_an_explicit_interval_still_wins(tmp_path, monkeypatch, capsys):
    """`--interval` 显式给了就用它——自动探测只是缺省值，不是强制。"""
    from pipeline import densify, frames

    video = tmp_path / "v.mp4"
    _tiny_video(video)
    w = _work_with_a_named_moment(tmp_path)
    monkeypatch.setattr(frames, "content_fps", lambda v: 25.0)

    densify.main(["--work", str(w), "--sheet", "组", "--video", str(video),
                  "--duration", "4", "--interval", "0.1", "--dry-run"])
    assert "间隔 0.100s" in capsys.readouterr().out


def test_spots_only_densifies_just_the_named_moment(tmp_path, monkeypatch, capsys):
    """`--spots-only`：只抽人工指名的那几处，**不**顺带加密整段的预设窗口。

    用户 2026-09-26 电子10 order 6 座位 4：勾了「无可用画面」并填了重抽时刻
    2:01:56。默认路径会在指名那 20 秒之外，把这一段的就座窗口也从 20 秒一帧
    加密到 10 秒一帧——那不是他要的，而且重检测就座那一路会让这一段的几何
    基线重新分座，他已经标好的座位跟着挪。

    重采样的代价与动了多少成正比（`iterating.md` 那张表：一段一个窗口
    51/51 存活，三段加密 44/50）。人已经告诉你人在哪儿了，就只抽那儿。
    """
    from pipeline import densify, frames

    video = tmp_path / "v.mp4"
    _tiny_video(video)
    w = _work_with_a_named_moment(tmp_path)
    monkeypatch.setattr(frames, "content_fps", lambda v: 25.0)

    densify.main(["--work", str(w), "--sheet", "组", "--video", str(video),
                  "--duration", "4", "--spots-only", "--dry-run"])
    out = capsys.readouterr().out
    assert "指定重抽时刻" in out, f"指名那一处必须照抽：{out}"
    assert "seated" not in out, f"就座窗口不该被顺带加密：{out}"
    assert "order 1 entry" not in out and "order 1 exit" not in out, (
        f"预设的进出场窗口也不该动：{out}")


def test_spots_only_without_any_named_moment_does_nothing(tmp_path, monkeypatch, capsys):
    """给了 `--spots-only` 却一处时刻都没填：一帧都不抽，并说清楚为什么。

    不能悄悄退回整段加密——那正是这个开关要避免的事。
    """
    from pipeline import densify, frames

    video = tmp_path / "v.mp4"
    _tiny_video(video)
    w = _work_with_a_named_moment(tmp_path)
    lp = paths.labels_path(w, "组")
    d = json.loads(lp.read_text(encoding="utf-8"))
    d["no_candidate_notes"] = {}
    lp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(frames, "content_fps", lambda v: 25.0)

    rc = densify.main(["--work", str(w), "--sheet", "组", "--video", str(video),
                       "--duration", "4", "--spots-only", "--dry-run"])
    out = capsys.readouterr().out
    assert "seated" not in out and "order 1 entry" not in out, out
    assert "没有" in out and "时刻" in out, f"要说明是因为没填时刻：{out}"
    assert rc == 0
