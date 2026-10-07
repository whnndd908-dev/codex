import pytest
from pipeline.run_all import parse_args, STAGES


def test_stage_numbers_follow_execution_order():
    """阶段编号 = 执行顺序，且**付费那一步排在人工核实之后**。

    用户 2026-08-29：「重新对各个阶段进行梳理排序，VLM 打标应放到最后吧」。
    旧编号是历史遗留：⑤ 是 VLM 打标、⑥ 是出挑选页——而挑选页正是人工核实
    的入口，必须在打标**之前**。编号与执行顺序相反，于是有了 `prepare` 这个
    「跑 1,2,3,4,6 跳过 5」的补丁命令，每次都要解释一遍。

    新编号里 5 之后是人工核实这条分界线，6 起全部依赖人工结果。
    """
    assert list(STAGES) == ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    names = list(STAGES.values())
    assert "挑选页" in names[4], f"阶段 5 该是出挑选页，实得 {names[4]}"
    assert "登记表" in names[5], f"阶段 6 该是座位登记表，实得 {names[5]}"
    assert "打标" in names[6], f"阶段 7 该是 VLM 打标，实得 {names[6]}"
    i_review = next(i for i, n in enumerate(names) if "挑选页" in n)
    i_vlm = next(i for i, n in enumerate(names) if "打标" in n)
    assert i_review < i_vlm, (
        "出挑选页（人工核实的入口）必须排在 VLM 打标之前——拿一份没核实的"
        "座位表去打标是花钱买错数据，且错在哪一处都不会报错")


def test_parse_stages_and_orders():
    ns = parse_args(["--stages", "1,2,3", "--orders", "2,7,10"])
    assert ns.stages == ["1", "2", "3"]
    assert ns.orders == [2, 7, 10]


def test_default_runs_all_stages_and_all_orders():
    ns = parse_args([])
    assert ns.stages == ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    assert ns.orders is None


def test_rejects_unknown_stage():
    # 用 "10"：重排之后 9 是合法阶段（项目排名），拿它当"不存在的号"会让这条
    # 测试恒绿而无人察觉。
    with pytest.raises(SystemExit):
        parse_args(["--stages", "10"])
    with pytest.raises(SystemExit):
        parse_args(["--stages", "0"])


def test_no_vlm_flag_parses():
    ns = parse_args(["--no-vlm"])
    assert ns.no_vlm is True


def test_no_vlm_defaults_false():
    assert parse_args([]).no_vlm is False


def test_missing_upstream_artifact_exits_with_guidance(tmp_path, monkeypatch):
    from pipeline import config as cfg
    from pipeline.run_all import run
    monkeypatch.setattr(cfg, "WORK", tmp_path / "empty")
    with pytest.raises(SystemExit) as e:
        run(["4"], None)
    assert "阶段" in str(e.value), "报错信息应指明先跑哪个阶段"


# --- staff-net 开关（Ruling R51：默认必须关闭） ---

def test_staff_net_flag_defaults_to_config_value():
    """未显式传 --staff-net 时，取 config.STAFF_NET_ENABLED（当前为 False）。"""
    from pipeline import config as cfg
    ns = parse_args([])
    assert ns.staff_net == cfg.STAFF_NET_ENABLED
    assert ns.staff_net is False, "Ruling R51：安全网必须默认关闭"


def test_staff_net_flag_can_be_explicitly_enabled():
    ns = parse_args(["--staff-net"])
    assert ns.staff_net is True


def test_missing_annotations_before_stage_six_exits_with_guidance(tmp_path, monkeypatch):
    """阶段 8/9 依赖 annotations.json；缺失时必须指明先跑阶段 7，而不是让
    json.loads 或 FileNotFoundError 冒出一条对用户无意义的堆栈。
    """
    from pipeline import config as cfg
    from pipeline.schedule import save_segments, load_segments
    from pipeline.run_all import run

    work = tmp_path / "work"
    monkeypatch.setattr(cfg, "WORK", work)
    work.mkdir(parents=True)
    # 伪造一份最小 segments.json，让 run() 走到阶段 8 之前不因阶段 1 缺失而提前退出
    import json as _json
    from dataclasses import asdict
    from pipeline.schedule import Segment, Presenter
    seg = Segment(1, "测试公司", 0, 100, 1, (Presenter(1, 0, 1, 0),))
    (work / "segments.json").write_text(
        _json.dumps({"video": "x", "sheet": "x", "panel": "x",
                     "segments": [asdict(seg)]}, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as e:
        run(["8"], None)
    assert "阶段" in str(e.value)


# ---------------------------------------------------------------------------
# Task 24 · 缺陷：阶段 8 把阶段 7 的证据图丢了。
#
# 阶段 7 结尾把 `_evidence_png`（二进制证据图字节）从写盘的 annotations.json
# 里弹出——这是对的，二进制绝不能进 JSON（Ruling R13）。但阶段 8 开头
# **无条件**从磁盘重新加载 anns，把内存里还带着 `_evidence_png` 的对象整个
# 覆盖掉，导致 `export_keyframes` 永远拿不到证据图。实测全量 16 段 34 人，
# 只跑出 0 张（此前 5 张是更早一次 pilot 跑剩下的）。
# ---------------------------------------------------------------------------

def _write_stage7_fixture(tmp_path, monkeypatch):
    """搭一份最小的阶段 1-4 产物，让 run(["7"]) 能跳过阶段 1-4 直接起步。

    一段一人：order=2、seat=1、track_id=8，配一张纯色合成图（含白平衡参照
    块，仿 test_assemble.py 的 `_write_solid`），使 measure_person_color /
    build_evidence_sheet 能真正跑通，不需要 mock 掉整条像素轨。
    """
    from pipeline import config as cfg
    from pipeline.schedule import Segment, Presenter
    from pipeline.detect import Detection, save_detections
    import json as _json
    from dataclasses import asdict
    from PIL import Image
    import numpy as np

    work = tmp_path / "work"
    monkeypatch.setattr(cfg, "WORK", work)
    work.mkdir(parents=True)

    seg = Segment(2, "公司2", 960, 1836, 1, (Presenter(1, 0, 1, 0),))
    (work / "segments.json").write_text(
        _json.dumps({"video": "x", "sheet": "x", "panel": "x",
                     "segments": [asdict(seg)]}, ensure_ascii=False),
        encoding="utf-8",
    )

    frames_dir = work / "frames"
    frames_dir.mkdir()
    p = frames_dir / "f2.jpg"
    arr = np.zeros((300, 400, 3), dtype=np.uint8)
    arr[:, :] = (20, 30, 90)
    arr[170:270, 20:220] = (205, 203, 200)   # 白平衡参照物
    Image.fromarray(arr).save(p)

    dets = [Detection(2, "seated", float(t), p, 8, 240, 10, 390, 140, 0.9)
            for t in range(3)]
    save_detections(dets, work / "detections.json")

    (work / "seats.json").write_text(_json.dumps({"2": {"8": 1}}), encoding="utf-8")
    return work


def test_stage8_keeps_evidence_png_when_stage7_ran_in_same_call(tmp_path, monkeypatch):
    """现行实现下必红：阶段 8 开头无条件重载 annotations.json，会把阶段 5
    刚装配出的、带 `_evidence_png` 的内存对象覆盖掉，证据图导不出来。
    """
    import pipeline.assemble as assemble_mod
    from pipeline.run_all import run

    work = _write_stage7_fixture(tmp_path, monkeypatch)

    def _fake_ask_vlm(sheet_png, context):
        return {"evidence_quality": "good", "confidence": 0.9,
                "outer_hue_class": "蓝", "formality_1_5": 4}
    monkeypatch.setattr(assemble_mod, "ask_vlm", _fake_ask_vlm)

    run(["7", "8"], orders=[2])

    matches = list((work / "keyframes").glob("*/seat1/evidence_sent_to_model.png"))
    assert len(matches) == 1, (
        "阶段 7→8 同一次调用内必须保留证据图，不能被阶段 8 的 JSON 重载丢弃"
    )
    assert matches[0].stat().st_size > 0, "证据图不能是空文件"


def test_stage8_alone_still_works_without_evidence_png(tmp_path, monkeypatch):
    """annotations.json 已在磁盘上（不含 `_evidence_png`，这是刻意设计）时
    单独跑 `--stages 8`：不能抛异常，画廊与 CSV 要正常产出。锁住单独运行
    这条路径没有被"只在同批次保留证据图"的修法改坏。
    """
    from pipeline.run_all import run

    work = _write_stage7_fixture(tmp_path, monkeypatch)
    run(["7"], orders=[2], no_vlm=True)   # 先落一份不含证据图的 annotations.json

    run(["8"], orders=[2])   # 单独重跑阶段 8：annotations.json 必须从磁盘重载

    from pipeline import config as cfg2
    safe_sheet = cfg2.sanitize_filename_component(cfg2.SHEET)
    assert (work / "gallery.html").exists()
    assert (work / f"result_{safe_sheet}.csv").exists()
    matches = list((work / "keyframes").glob("*/seat1/evidence_sent_to_model.png"))
    assert matches == [], (
        "no_vlm 落盘的 annotations.json 本就没有证据图，单独跑阶段 8 不应凭空生出"
    )


def test_annotations_json_never_contains_evidence_png(tmp_path, monkeypatch):
    """护栏：不许为了"修好"阶段 8 的重载问题，反过来把二进制证据图字节
    写进 annotations.json——二进制绝不能进 JSON/CSV（Ruling R13，Task 15
    既有测试也锁着这条）。
    """
    import pipeline.assemble as assemble_mod
    from pipeline.run_all import run

    work = _write_stage7_fixture(tmp_path, monkeypatch)

    def _fake_ask_vlm(sheet_png, context):
        return {"evidence_quality": "good", "confidence": 0.9, "outer_hue_class": "蓝"}
    monkeypatch.setattr(assemble_mod, "ask_vlm", _fake_ask_vlm)

    run(["7"], orders=[2])

    raw = (work / "annotations.json").read_text(encoding="utf-8")
    assert "_evidence_png" not in raw, "二进制证据图绝不能落进 annotations.json"


# ---------------------------------------------------------------------------
# Task 26 · 需求二 —— 阶段 8 除长表外，还要产出一企业一行的宽表
# （用户明确要求）。Task 33 起两者文件名都带组名（见下方 Task 33 小节）。
# ---------------------------------------------------------------------------

def test_stage8_also_writes_wide_csv(tmp_path, monkeypatch):
    from pipeline.run_all import run
    from pipeline import config as cfg

    work = _write_stage7_fixture(tmp_path, monkeypatch)
    run(["7", "8"], orders=[2], no_vlm=True)

    safe_sheet = cfg.sanitize_filename_component(cfg.SHEET)
    result_path = work / f"result_{safe_sheet}.csv"
    wide_path = work / f"result_wide_{safe_sheet}.csv"
    assert result_path.exists(), "长表不能因为新增宽表而消失"
    assert wide_path.exists(), "阶段 8 必须产出宽表"

    import csv as csv_mod
    with wide_path.open(encoding="utf-8-sig") as fh:
        rows = list(csv_mod.DictReader(fh))
    assert len(rows) == 1, "本 fixture 只有 order=2 一个企业"
    assert rows[0]["company"] == "公司2"
    assert "seat1_formality_1_5" in rows[0]


# ---------------------------------------------------------------------------
# Task 33（用户追加需求）—— 阶段 8/9 写盘的全部 CSV 都带组名，且组名要经过
# 与 export_keyframes 相同的净化，不裸拼进文件名。
# ---------------------------------------------------------------------------

def test_output_csv_filenames_contain_sheet_name(tmp_path, monkeypatch):
    """result.csv / result_wide.csv / project_ranking.csv 都要带上
    净化后的 config.SHEET，三组视频的产物才不会互相覆盖、分不清谁是谁。"""
    from pipeline.run_all import run
    from pipeline import config as cfg

    work = _write_stage7_fixture(tmp_path, monkeypatch)
    run(["7", "8", "9"], orders=[2], no_vlm=True)

    safe_sheet = cfg.sanitize_filename_component(cfg.SHEET)
    assert (work / f"result_{safe_sheet}.csv").exists()
    assert (work / f"result_wide_{safe_sheet}.csv").exists()
    assert (work / f"project_ranking_{safe_sheet}.csv").exists()
    # 护栏：不能两套都写（新旧文件名并存也是一种"没真改对"的失败模式）。
    assert not (work / "result.csv").exists()
    assert not (work / "result_wide.csv").exists()
    assert not (work / "project_ranking.csv").exists()


# ---------------------------------------------------------------------------
# Task 40 · 阈值体检——阶段④必须打印覆盖全部"实际生效"阈值的体检表，不能
# 只打印其中几个而遗漏其余（那正是变异 (e) 要试探的失败模式）。
# ---------------------------------------------------------------------------

def test_stage4_report_covers_every_live_threshold(tmp_path, monkeypatch, capsys):
    """构造一段有若干就座 track（含左区评委、右区若干创业者候选）的最小
    fixture，跑 `run(["4"], ...)`，标准输出必须包含表格里全部六个阈值的
    名字——哪怕某个阈值样本不足给不出自信结论，名字本身也必须出现在报告
    里，否则将来新增阈值时很容易漏掉某一个而不被察觉（这正是本测试要防的
    坏味道，见 brief 变异清单 (e)）。"""
    from pathlib import Path
    from pipeline import config as cfg
    from pipeline.run_all import run
    from pipeline.schedule import Segment
    from pipeline.detect import Detection, save_detections

    work = tmp_path / "work"
    monkeypatch.setattr(cfg, "WORK", work)
    work.mkdir(parents=True)

    seg = Segment(2, "公司2", 0, 100, 2, ())
    import json as _json
    from dataclasses import asdict
    (work / "segments.json").write_text(
        _json.dumps({"video": "x", "sheet": "x", "panel": "x",
                     "segments": [asdict(seg)]}, ensure_ascii=False),
        encoding="utf-8",
    )

    frame_path = Path("/tmp/f.jpg")
    dets = []
    # 左区评委（cx 远小于 RIGHT_REGION_X*frame_w），不参与右区判据取样，
    # 但计入 RIGHT_REGION_X 的全体就座 track 样本。
    for tid, cx in [(50, 100.0), (51, 200.0), (52, 300.0)]:
        for t in range(6):
            dets.append(Detection(2, "seated", float(t), frame_path, tid,
                                   cx - 40, 500, cx + 40, 700, 0.9))
    # 右区若干创业者候选，框高、位置各不相同，制造非退化的样本分布。
    for i, (tid, cx, by, h) in enumerate([
        (1, 1300, 900, 420), (2, 1310, 850, 400), (3, 1320, 800, 380),
        (4, 1250, 750, 200), (5, 1260, 700, 190),
    ]):
        for t in range(6):
            dets.append(Detection(2, "seated", float(t), frame_path, tid,
                                   cx - h * 0.3, by - h, cx + h * 0.3, by, 0.9))
    save_detections(dets, work / "detections.json")

    run(["4"], orders=[2])

    out = capsys.readouterr().out
    for name in [
        "RIGHT_REGION_X", "SEAT_MIN_H_RATIO", "SEAT_MIN_H_VS_TALLEST",
        "SEAT_MIN_FRAME_COVERAGE", "SEAT_DUP_CONTAIN", "SEAT_PARTIAL_REL",
    ]:
        assert name in out, f"阶段④输出应包含阈值 {name} 的体检结果，实际输出：\n{out}"


def test_sheet_name_is_sanitised_in_filename(tmp_path, monkeypatch):
    """SHEET 含 '/' 等文件系统非法字符时，不能被裸拼进路径意外创建子目录
    ——净化行为必须与 export_keyframes 复用的同一个
    `config.sanitize_filename_component` 一致。"""
    from pipeline.run_all import run
    from pipeline import config as cfg

    work = _write_stage7_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(cfg, "SHEET", "某组/含:非法*字符")
    run(["7", "8", "9"], orders=[2], no_vlm=True)

    safe_sheet = cfg.sanitize_filename_component(cfg.SHEET)
    assert "/" not in safe_sheet and "\\" not in safe_sheet
    csv_files = list(work.glob("result_*.csv")) + list(work.glob("project_ranking_*.csv"))
    assert any(safe_sheet in p.name for p in csv_files), (
        f"净化后的组名应出现在产物文件名里，实得 {[p.name for p in csv_files]}"
    )
    # 恰好一层 work/ 目录下的文件，没有因为 SHEET 里的 '/' 被当成路径分隔符
    # 意外拆出子目录。
    for p in csv_files:
        assert p.parent == work


# ---------------------------------------------------------------------------
# Task 44：进出场递补必须可见——不能让用户以为某人的衣着来自就座画面，
# 其实来自进出场画面。阶段 7 要报告本次到底递补了几人。
# ---------------------------------------------------------------------------

def test_stage7_reports_backfill_count(tmp_path, monkeypatch, capsys):
    """搭一个 2 人段：座位1有就座轨迹，座位2就座阶段缺席、靠
    entry_labels_*.json 里的一条『创业者』标注从进出场候选递补回来。
    阶段 7 的标准输出必须报告本次递补了 1 人，不能静默完成。
    """
    from pipeline import config as cfg
    from pipeline.schedule import Segment, Presenter
    from pipeline.detect import Detection, save_detections
    from pipeline.run_all import run
    import json as _json
    from dataclasses import asdict
    from PIL import Image
    import numpy as np

    work = tmp_path / "work"
    monkeypatch.setattr(cfg, "WORK", work)
    monkeypatch.setattr(cfg, "SHEET", "测试组task44")
    work.mkdir(parents=True)

    seg = Segment(2, "公司2", 960, 1836, 2,
                  (Presenter(1, 0, 1, 0), Presenter(2, 0, 0, 0)))
    (work / "segments.json").write_text(
        _json.dumps({"video": "x", "sheet": "x", "panel": "x",
                     "segments": [asdict(seg)]}, ensure_ascii=False),
        encoding="utf-8",
    )

    frames_dir = work / "frames"
    frames_dir.mkdir()
    p_seated = frames_dir / "f2.jpg"
    arr = np.zeros((300, 400, 3), dtype=np.uint8)
    arr[:, :] = (20, 30, 90)
    arr[170:270, 20:220] = (205, 203, 200)   # 白平衡参照物
    Image.fromarray(arr).save(p_seated)

    p_entry = frames_dir / "entry2.jpg"
    arr2 = np.zeros((1300, 1300, 3), dtype=np.uint8)
    arr2[:, :] = (20, 30, 90)
    arr2[600:700, 20:220] = (205, 203, 200)
    Image.fromarray(arr2).save(p_entry)

    dets = [Detection(2, "seated", float(t), p_seated, 8, 240, 10, 390, 140, 0.9)
            for t in range(3)]
    cx, h, w, bottom = 900.0, 500.0, 180.0, 1200.0
    half = w / 2.0
    dets += [
        Detection(2, "entry", float(t), p_entry, 70, cx - half, bottom - h,
                  cx + half, bottom, 0.9)
        for t in range(6)
    ]
    save_detections(dets, work / "detections.json")

    # 座位1已占用（tid=8），座位2就座阶段本就空缺。
    (work / "seats.json").write_text(_json.dumps({"2": {"8": 1}}), encoding="utf-8")

    # Task 46：标注文件用稳定标识作键，并带上这份检测结果的指纹。
    from pipeline.entryexit import detections_fingerprint, entryexit_candidates
    from pipeline.schedule import load_segments_json
    from pipeline.detect import load_detections

    seg_loaded = load_segments_json(work / "segments.json")[0]
    cands = entryexit_candidates(load_detections(work / "detections.json"), seg_loaded)
    entry_cand = next(c for c in cands if c["track_id"] == 70)
    safe_sheet = cfg.sanitize_filename_component(cfg.SHEET)
    (work / f"entry_labels_{safe_sheet}.json").write_text(
        _json.dumps({
            "_meta": {
                "detections_fingerprint": detections_fingerprint(work / "detections.json"),
                "sheet": cfg.SHEET, "n_candidates": len(cands),
            },
            "2": {entry_cand["stable_id"]: {"role": "创业者", "seat": 2,
                                             "n_people": 1, "note": "",
                                             "track_id": 70}},
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    run(["7"], orders=[2], no_vlm=True)
    out = capsys.readouterr().out
    assert "进出场递补" in out, f"阶段 5 必须打印递补相关信息，实得：\n{out}"
    assert "1 人" in out, f"应报告本次递补了 1 人，实得：\n{out}"


def test_stage7_reports_zero_backfill_when_no_labels_file(tmp_path, monkeypatch, capsys):
    """entry_labels_*.json 不存在时，阶段 5 不能静默——必须明确说明递补
    未启用（否则用户无从判断"没有递补"到底是没有缺口还是标注文件没找到）。
    """
    from pipeline.run_all import run

    _write_stage7_fixture(tmp_path, monkeypatch)
    run(["7"], orders=[2], no_vlm=True)
    out = capsys.readouterr().out
    assert "进出场递补" in out
    assert "未启用" in out


def test_export_override_comes_after_original_definition(tmp_path):
    """注入的 exportLabels 必须排在页面原定义之后，否则原版会胜出。

    现行实现是在 run_all 里对 entry_review.html 做后处理，把升级版
    `window.exportLabels = ...` 追加到文件末尾，靠「后者覆盖前者」生效。
    这个约束是隐式的：若有人把注入点挪到页面 <script> 之前（看起来更"整洁"），
    覆盖就会失效——而页面照常渲染、按钮照常可点，只是又变回不能选路径的
    老行为，不报任何错。本项目已两次因「改了但没生效且不报错」给用户过时
    产物，这条把顺序锁死。
    """
    from pipeline.run_all import upgrade_entry_review_html

    html = (
        '<html><body><div id="main"></div>'
        "<script>function exportLabels(){/*原版*/}</script>"
        "</body></html>"
    )
    out, _ = upgrade_entry_review_html(html, tmp_path / "entry_labels_x.json", None)
    i_orig = out.index("function exportLabels()")
    i_new = out.index("window.exportLabels")
    assert i_new > i_orig, "升级版必须在原定义之后，否则不生效"


def test_export_upgrade_keeps_download_fallback(tmp_path):
    """必须保留 <a download> 回退——Safari/Firefox 没有 showSaveFilePicker，
    只走选择器会让那些浏览器完全导不出。"""
    from pipeline.run_all import upgrade_entry_review_html

    out, _ = upgrade_entry_review_html(
        '<div id="main"></div><script>function exportLabels(){}</script>',
        tmp_path / "entry_labels_x.json", None)
    assert "showSaveFilePicker" in out
    assert "a.download" in out, "缺少 <a download> 回退分支"
    assert "AbortError" in out, "用户取消应被静默处理，不弹错误"


# ---------------------------------------------------------------------------
# Task 46：阶段 5 必须核对标注文件是不是「这一份检测结果」的。
# 事故背景：企业行序变动后重抽帧 + 重检测，track_id 全部改变，用户此前标的
# `entry:19` 大部分指向不存在的 track、少数碰巧编号还在但已是另一个人——
# 错配会把别人的衣着编进这个人的行里且不报错。
# ---------------------------------------------------------------------------

def test_stage7_voids_labels_from_another_detection(tmp_path, monkeypatch, capsys):
    """旧格式（`entry:70`，无 _meta 指纹）的标注文件必须整份作废并提示重标，
    绝不能照常拿去递补——那正是会错配的情形。"""
    from pipeline import config as cfg
    from pipeline.run_all import run
    import json as _json

    _write_stage7_fixture(tmp_path, monkeypatch)
    work = cfg.WORK
    safe_sheet = cfg.sanitize_filename_component(cfg.SHEET)
    (work / f"entry_labels_{safe_sheet}.json").write_text(
        _json.dumps({"2": {"entry:70": {"role": "创业者", "seat": 2}}},
                    ensure_ascii=False), encoding="utf-8")

    run(["7"], orders=[2], no_vlm=True)
    out = capsys.readouterr().out
    assert "作废" in out and "重新标注" in out, (
        f"阶段 5 必须醒目提示标注已作废，不得静默，实得：\n{out}"
    )
    assert "进出场递补：0 人" in out, (
        f"作废之后不许再拿这份标注递补任何人，实得：\n{out}"
    )


def test_export_upgrade_reuses_page_payload_builder(tmp_path):
    """注入的「另存为」升级版只接管保存方式，导出内容一律调用页面里的
    `buildLabelsPayload()`——两处各拼一份 JSON 必然日后走样（本项目刚吃过
    「两个页面各写一套 CSS、无一致性测试」的亏）。"""
    from pipeline.run_all import upgrade_entry_review_html

    out, _ = upgrade_entry_review_html(
        '<div id="main"></div><script>function exportLabels(){}</script>',
        tmp_path / "entry_labels_x.json", None)
    injected = out.split('<div id="main"></div>', 1)[1]
    assert "buildLabelsPayload()" in injected, "升级版必须复用页面的 payload 构造函数"
    assert "out[segKey]" not in injected, (
        "升级版不得自己再拼一遍导出内容——那会与页面里的定义各走各的"
    )
    assert "_meta" not in injected, "_meta 只该由 buildLabelsPayload 一处拼出"


def test_stage5_preloads_labels_realigned_after_redetection(tmp_path, monkeypatch, capsys):
    """阶段 6 重新生成挑选页时，用户此前的标注要按视频内容重新对齐后预填。

    重检测后 track_id 全变、框位置微抖：若预填还按标注文件里的旧键走，
    页面上一条也对不上当前候选，用户会以为"标注不见了"而重标一遍——
    用户要标 60 多组，这个代价不可接受。
    """
    from pipeline import config as cfg
    from pipeline.schedule import Segment, Presenter
    from pipeline.detect import Detection, save_detections, load_detections
    from pipeline.schedule import load_segments_json
    from pipeline.entryexit import entryexit_candidates, stable_candidate_id
    from pipeline.run_all import run
    import json as _json
    from dataclasses import asdict
    from PIL import Image
    import numpy as np

    work = tmp_path / "work"
    monkeypatch.setattr(cfg, "WORK", work)
    monkeypatch.setattr(cfg, "SHEET", "测试组task46")
    work.mkdir(parents=True)

    seg = Segment(2, "公司2", 960, 1836, 2,
                  (Presenter(1, 0, 1, 0), Presenter(2, 0, 0, 0)))
    (work / "segments.json").write_text(
        _json.dumps({"video": "x", "sheet": "x", "panel": "x",
                     "segments": [asdict(seg)]}, ensure_ascii=False),
        encoding="utf-8")

    frames_dir = work / "frames"
    frames_dir.mkdir()
    p_seated = frames_dir / "f2.jpg"
    arr = np.zeros((300, 400, 3), dtype=np.uint8)
    arr[:, :] = (20, 30, 90)
    arr[170:270, 20:220] = (205, 203, 200)
    Image.fromarray(arr).save(p_seated)
    p_entry = frames_dir / "entry2.jpg"
    arr2 = np.zeros((1300, 1300, 3), dtype=np.uint8)
    arr2[:, :] = (20, 30, 90)
    arr2[600:700, 20:220] = (205, 203, 200)
    Image.fromarray(arr2).save(p_entry)

    def write_dets(track_id, dcx):
        dets = [Detection(2, "seated", float(t), p_seated, 8, 240, 10, 390, 140, 0.9)
                for t in range(3)]
        cx, h, w, bottom = 900.0 + dcx, 500.0, 180.0, 1200.0
        dets += [Detection(2, "entry", float(t), p_entry, track_id, cx - w / 2,
                           bottom - h, cx + w / 2, bottom, 0.9) for t in range(6)]
        save_detections(dets, work / "detections.json")

    (work / "seats.json").write_text(_json.dumps({"2": {"8": 1}}), encoding="utf-8")

    # --- 第一份检测：用户在这份上标注 ---
    write_dets(70, 0.0)
    seg_loaded = load_segments_json(work / "segments.json")[0]
    cands_v1 = entryexit_candidates(load_detections(work / "detections.json"), seg_loaded)
    old_sid = next(c["stable_id"] for c in cands_v1 if c["track_id"] == 70)
    safe_sheet = cfg.sanitize_filename_component(cfg.SHEET)
    (work / f"entry_labels_{safe_sheet}.json").write_text(
        _json.dumps({"_meta": {"detections_fingerprint": "指纹属于第一份检测",
                               "sheet": cfg.SHEET, "n_candidates": len(cands_v1)},
                     "2": {old_sid: {"role": "创业者", "seat": 2, "n_people": 1,
                                     "note": "", "track_id": 70}}},
                    ensure_ascii=False), encoding="utf-8")

    # --- 重检测：track_id 70→5070，框位置抖 15px ---
    write_dets(5070, 15.0)
    cands_v2 = entryexit_candidates(load_detections(work / "detections.json"), seg_loaded)
    new_sid = next(c["stable_id"] for c in cands_v2 if c["track_id"] == 5070)
    assert new_sid != old_sid, "夹具要真的造出重检测后标识改变的情形"

    # 挑选页归阶段 5。重排之前它挤在画廊那一阶段里，所以这条测试原本要
    # 连打标一起跑——那正是「一个阶段同时干着人工前与人工后两件事」。
    run(["5"], orders=[2])
    out = capsys.readouterr().out
    # Task 56G：挑选页归 work/Output/<组>/
    from pipeline import paths
    html = (paths.output_dir(work) / "entry_review.html").read_text(encoding="utf-8")

    # 只认注入的 PRELOAD 那份——页面 SEGMENTS 里当然含当前候选的稳定标识，
    # 对整页做 `in` 断言会恒真、测不到预填这条路径（第一版就这么写的）。
    import re as _re
    m = _re.search(r"var PRELOAD=(\{.*?\});", html, _re.DOTALL)
    assert m is not None, f"阶段 6 应把既有标注预填进页面，实得输出：\n{out}"
    preload = _json.loads(m.group(1))
    assert list(preload) == [f"2::{new_sid}"], (
        f"预填必须挂在**当前**候选的稳定标识上（期望 2::{new_sid}），"
        f"实得 {list(preload)}\n{out}"
    )
    assert preload[f"2::{new_sid}"]["seat"] == "2"
    # 重新对齐这件事本身也要在输出里说出来，否则用户无从判断预填的是不是他
    # 那一份。原先这里断言的是「递补 1 人」——那是**打标阶段**的输出，混进
    # 一条测挑选页的测试里，是挑选页曾经与打标挤在同一阶段的残留。
    assert "重新对齐" in out and "1 条匹配上" in out, (
        f"重对齐的结果必须打印出来，实得：\n{out}")


def test_injected_scripts_are_valid_javascript(tmp_path):
    """注入挑选页的每个 <script> 都必须是**能通过语法检查**的 JavaScript。

    这条是查 Task 46 的排版/导出改动时撞出来的真缺陷：Task 45 注入的
    「另存为」升级版 exportLabels 是一串**不含任何换行**的字符串拼接，
    里面却有一句 `// 其他异常…` 行注释——行注释吃掉了它后面的全部代码
    （整段脚本就一行），花括号不配对，浏览器对这个 <script> 直接抛
    SyntaxError，`window.exportLabels` 从未被赋值。页面照常渲染、按钮照常
    可点（回落到页面原版导出），于是「改了但没生效且不报错」——Task 45 的
    测试全是对字符串做 `in` 断言，没人真去解析过它。
    """
    import re as _re
    import shutil as _shutil
    import subprocess as _subprocess

    node = _shutil.which("node")
    assert node, "本测试需要 node（本仓库其它 JS 测试同样依赖它）"

    from pipeline.run_all import upgrade_entry_review_html

    html, _ = upgrade_entry_review_html(
        '<div id="main"></div><script>function exportLabels(){}</script>',
        tmp_path / "entry_labels_x.json",
        {"2": {"entry@100#900,1000": {"role": "创业者", "seat": 2,
                                      "n_people": 1, "note": ""}}},
    )
    injected = html.split("<script>function exportLabels(){}</script>", 1)[1]
    blocks = _re.findall(r"<script>(.*?)</script>", injected, _re.DOTALL)
    assert len(blocks) == 2, f"应注入导出升级与预填两个脚本，实得 {len(blocks)}"
    for i, body in enumerate(blocks):
        f = tmp_path / f"injected_{i}.js"
        f.write_text(body, encoding="utf-8")
        r = _subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, (
            f"注入的第 {i} 个脚本不是合法 JavaScript：\n{r.stderr}\n----\n{body}"
        )


def test_stage3_reuses_unchanged_groups_and_says_so(tmp_path, monkeypatch, capsys):
    """阶段 3 只重跑帧集合变了的那几组，并把「留用几组 / 重跑几组」打出来。

    用户 2026-09-09：「能只对 order 16 重抽吗，不抽其他项目，是不是就可以省
    时间了」。能——每组的检测只取决于自己的帧（跟踪器每组开头归零，含 track
    编号计数器），所以这是可证明等价的加速，不是近似。

    凡有留用必须可见：不打印的话，用户无从知道这一轮到底跑了什么，而「本该
    重跑却留用了」正是那种看起来完全正常的错。
    """
    from pipeline import run_all as _r
    from pipeline import detect as _d

    seen = []

    class _Res:
        boxes = None
    class _Tracker:
        def reset(self): pass
    class _Pred:
        trackers = [_Tracker()]
    class _FakeModel:
        predictor = _Pred()
        def track(self, path, **kw):
            seen.append(str(path))
            return [_Res()]

    monkeypatch.setattr(_d, "_get_model", lambda: _FakeModel())
    monkeypatch.setattr(_d, "_device_and_quantize", lambda: ("cpu", None))

    from pipeline.frames import FrameRef, save_index
    from pipeline.detect import (detect_frames, save_detections,
                                 frames_group_digests, save_group_digests,
                                 group_digests_path)
    work = tmp_path / "work"; work.mkdir()
    idx = work / "frames_index.json"
    det = work / "detections.json"
    old = [FrameRef(1, "entry", 0.0, tmp_path / "a.jpg"),
           FrameRef(2, "exit", 5.0, tmp_path / "b.jpg")]
    save_index(old, idx)
    save_detections(detect_frames(old), det)
    save_group_digests(frames_group_digests(old), group_digests_path(det))

    new = old + [FrameRef(2, "exit", 5.2, tmp_path / "c.jpg")]
    save_index(new, idx)
    seen.clear()
    dets, digests, reused, redone = _d.detect_frames_incremental(
        new, _d.load_detections(det), _d.load_group_digests(group_digests_path(det)))
    assert reused == [(1, "entry")] and redone == [(2, "exit")]
    assert all("a.jpg" not in s for s in seen), f"没变的组不该再推理一遍：{seen}"


def test_stage_seven_also_codes_the_judges(tmp_path, monkeypatch, capsys):
    """阶段 7 跑完创业者之后，接着给评审席打标并单独写盘（用户 2026-09-19）。

    写成**另一个文件**：`annotations.json` 按 (段, 座位) 组织、每行都带企业与
    座位号，而评审人是组级的。混进去就得给每个下游消费者加一条「这一行
    seg_order=0，特殊对待」，而漏加的那一处会把评审人当成某家企业的创业者
    统计进去，且不报错。
    """
    import json
    from pipeline import paths, run_all as R

    work = tmp_path / "grp"
    work.mkdir()
    reg = {"judges": {"n_expected": 5, "n_resolved": 1, "unresolved": [],
                      "extra_candidates": [],
                      "slots": {"1": {"judge": 1, "stable_id": "judge#1,1",
                                      "frames": [{"frame_path": "a.jpg", "t": 1.0,
                                                  "box": [1, 2, 3, 4]}]},
                                **{str(i): None for i in range(2, 6)}}}}
    paths.registry_path(work, "组").parent.mkdir(parents=True, exist_ok=True)
    paths.registry_path(work, "组").write_text(json.dumps(reg), encoding="utf-8")

    calls = []
    monkeypatch.setattr(R, "_code_the_judges",
                        lambda w, s, use_vlm: calls.append((w, s, use_vlm))
                        or [{"judge": 1, "vlm": {"formality_1_5": 4}}],
                        raising=False)
    out = R._write_judge_annotations(work, "组", [{"judge": 1,
                                                   "vlm": {"formality_1_5": 4}}])
    assert out == paths.judges_path(work, "组")
    assert json.loads(out.read_text(encoding="utf-8"))[0]["judge"] == 1


def test_the_judge_file_is_a_deliverable(tmp_path):
    """评审席那份要进交付物清单——不进清单的文件会被永远留在旧位置。"""
    from pipeline import paths
    names = paths.deliverable_names("组")
    assert f"judges_{paths._safe('组')}.json" in names, names
