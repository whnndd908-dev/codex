"""Task 52 需求六：`python -m pipeline.prepare` —— 只做取帧与人工辅助定位。"""
import json
import re
import subprocess
from pathlib import Path

import pytest

from pipeline import config
from pipeline.prepare import PREPARE_STAGES, SKIPPED_STAGES, parse_args, prepare


def test_prepare_covers_exactly_the_pre_human_stages():
    """prepare = 人工核实**之前**的全部阶段，一个不多一个不少。

    重排之后（用户 2026-08-29）这条终于说得干脆：1..5 是人工之前，6 起全部
    依赖人工结果。旧编号下 prepare 得写成「1,2,3,4,6 跳过 5」，那串数字本身
    就是编号错位的产物。
    """
    assert list(PREPARE_STAGES) == ["1", "2", "3", "4", "5"]
    assert set(SKIPPED_STAGES) == {"6", "7", "8", "9"}
    assert not set(PREPARE_STAGES) & set(SKIPPED_STAGES)


def test_prepare_cli_rejects_post_human_stages():
    """显式要跑人工之后那几个阶段时报错并指明改用 run_all，不是默默跑掉。"""
    for stage in ("6", "7", "8", "9"):
        with pytest.raises(SystemExit):
            parse_args(["--stages", stage])


def _fake_work(tmp_path: Path, with_annotations: bool) -> Path:
    """一个只有产物、没有真实视频的组目录。"""
    work = tmp_path / "组"
    work.mkdir()
    (work / "segments.json").write_text(
        json.dumps({"video": "v.mp4", "sheet": "S", "panel": "P", "segments": []}),
        encoding="utf-8")
    (work / "detections.json").write_text("[]", encoding="utf-8")
    (work / "seats.json").write_text("{}", encoding="utf-8")
    if with_annotations:
        (work / "annotations.json").write_text("[]", encoding="utf-8")
    return work


def test_prepare_skips_vlm_and_ranking(tmp_path, monkeypatch, capsys):
    """prepare 跑 1,2,3,4 与 6，**绝不**把 5 或 7 交给 run_all。

    断言打在真正传给 `run_all.run` 的阶段列表上，不是打在打印文案上——
    文案对了而实际把 5 传下去，VLM 一样会被调用（本项目吃过「断言打在
    自己写的说明上」的亏，见 Task 46 报告）。
    """
    import pipeline.run_all as run_all

    work = _fake_work(tmp_path, with_annotations=True)
    monkeypatch.setattr(config, "WORK", work)
    seen: list[list[str]] = []
    monkeypatch.setattr(run_all, "run", lambda stages, orders, **kw: seen.append(list(stages)))

    prepare()

    flat = [s for call in seen for s in call]
    assert flat == ["1", "2", "3", "4", "5"], f"实际交给 run_all 的阶段：{seen}"
    for post in ("6", "7", "8", "9"):
        assert post not in flat, (
            f"prepare 绝不能跑阶段 {post}——6 起全部依赖人工核实结果")


def test_prepare_refuses_post_human_stage_when_called_directly(tmp_path, monkeypatch):
    """绕开 CLI 直接调函数，同样不许跑人工之后那几个阶段。"""
    monkeypatch.setattr(config, "WORK", _fake_work(tmp_path, with_annotations=True))
    with pytest.raises(ValueError, match="7"):
        prepare(stages=["1", "7"])


def test_prepare_works_on_a_group_that_has_never_been_annotated(tmp_path, monkeypatch):
    """新一组视频（没有 annotations.json）照常跑通——这是 prepare 的主用场。

    重排之前挑选页与画廊挤在同一个阶段里，那个阶段开头 `_require(
    annotations.json)`，新组必然缺它；prepare 只好另走一条「自己把挑选页建
    出来、并说破本次没有画廊」的特例分支。阶段 5 拆出来之后它本就不读打标
    结果，特例分支随之删除——**这条测试钉的就是「不再需要特例」**：prepare
    把阶段原样交给 run_all，不自己另建页面。
    """
    import pipeline.run_all as run_all

    work = _fake_work(tmp_path, with_annotations=False)
    monkeypatch.setattr(config, "WORK", work)
    seen: list[list[str]] = []
    monkeypatch.setattr(run_all, "run",
                        lambda stages, orders, **kw: seen.append(list(stages)))
    built: list[tuple] = []
    monkeypatch.setattr(run_all, "build_entry_review_page",
                        lambda *a: (built.append(a), work / "x.html")[1])

    prepare()

    assert seen == [["1", "2", "3", "4", "5"]], (
        f"没有打标结果也要把整串阶段原样交给 run_all，实得 {seen}")
    assert not built, (
        "prepare 不该再自己建挑选页——那是阶段 5 的事。自己建一份就是"
        "「两处各拼一份」，改了一处必然给用户一张过时的页面且不报错")


def test_prepare_builds_no_page_of_its_own(tmp_path, monkeypatch):
    """prepare **一张页面都不自己建**，整串阶段交给 run_all。

    原先这条要求的是「复用 `run_all.build_entry_review_page`，别自己拼一份」
    ——那已经比各拼一份好，但 prepare 里终究还留着一条建页面的代码路径。
    2026-08-29 阶段重排把挑选页拆成独立的阶段 5 之后，那条路径整段删掉了：
    不存在的代码不会走样。

    两个都不许出现：`render_entry_review`（绕过预填与导出升级，页面看起来
    正常但少半截功能，本项目发生过两次），以及 `build_entry_review_page`
    （自己调等于又开了第二条建页路径）。
    """
    import inspect

    import pipeline.prepare as prep

    body = "\n".join(
        ln for ln in inspect.getsource(prep).split("\n")
        if not ln.lstrip().startswith("#"))
    for name in ("render_entry_review", "build_entry_review_page"):
        assert name not in body, (
            f"prepare 不该再自己建挑选页（出现了 {name}）——那是阶段 5 的事")


def test_prepare_reports_review_path_and_shortfall(tmp_path, monkeypatch, capsys):
    """跑完打印挑选页路径 + 该组还缺几人（逐段列出缺口）。"""
    import pipeline.run_all as run_all
    from pipeline.schedule import Segment

    work = _fake_work(tmp_path, with_annotations=True)
    monkeypatch.setattr(config, "WORK", work)
    monkeypatch.setattr(run_all, "run", lambda stages, orders, **kw: None)
    # Task 56G：挑选页归 work/Output/<组>/，组目录只留原始与过程文件。
    from pipeline import paths
    page = paths.output_dir(work) / "entry_review.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text("<html>", encoding="utf-8")

    segs = [Segment(order=1, company="甲公司", start=0, end=10, presenter_num=3, presenters=()),
            Segment(order=2, company="乙公司", start=10, end=20, presenter_num=2, presenters=())]
    monkeypatch.setattr("pipeline.schedule.load_segments_json", lambda p: segs)
    # 就座阶段：甲定了 1 人（缺 2），乙定了 2 人（不缺）
    (work / "seats.json").write_text(
        json.dumps({"1": {"10": 1}, "2": {"20": 1, "21": 2}}), encoding="utf-8")

    prepare()
    text = capsys.readouterr().out

    assert str(page) in text, f"必须打印挑选页路径：{text}"
    assert "3/5" in text, f"应打印「已定 3/5 个座位」，实得：{text}"
    assert "还缺 2 人" in text, f"应打印总缺口，实得：{text}"
    assert "甲公司 缺 2 人" in text, f"应逐段列出缺口，实得：{text}"
    assert "乙公司" not in text, "不缺人的段不该混进缺口清单"


def test_prepare_unknown_group_lists_existing(tmp_path, monkeypatch):
    """--group 打错时报错并列出现有组，不新建空目录（新建只会让后面每一步
    都说「缺少上游产物」，看不出根因）。"""
    root = tmp_path / "work"
    (root / "甲组").mkdir(parents=True)
    monkeypatch.setattr(config, "WORK", root / "甲组")
    with pytest.raises(SystemExit) as e:
        prepare(group="打错的组名")
    assert "甲组" in str(e.value), "报错要列出现有的组"
    assert not (root / "打错的组名").exists(), "不许新建空的组目录"


# ===========================================================================
# Task 52 需求五：导出时记住上次保存的位置（IndexedDB 存
# FileSystemFileHandle）
#
# 全部在 node 里跑真实注入的那段 JS，用 shim 顶掉 indexedDB /
# showSaveFilePicker——断言打在「到底写到哪里去了」上，不是打在源码文本上。
# ===========================================================================

def _export_script() -> str:
    from pipeline.run_all import _build_export_upgrade

    _banner, script = _build_export_upgrade(Path("/tmp/work/entry_labels_x.json"))
    m = re.search(r"<script>(.*)</script>", script, re.DOTALL)
    assert m, "导出升级必须是一个 <script> 块"
    return m.group(1)


_HARNESS_PRELUDE = """
const EXPORT_FILENAME = 'entry_labels_x.json';
function buildLabelsPayload(){ return {ok: 1}; }
const LOG = [];
const els = {};
const document = {
  getElementById: id => (els[id] = els[id] || {textContent:'', style:{}, onclick:null}),
  createElement: () => ({ set href(v){}, set download(v){}, click(){ LOG.push(['anchor']); } }),
};
const URL = { createObjectURL: () => 'blob:x' };
function Blob(parts){ this.parts = parts; }
// 一个够用的内存版 IndexedDB：只实现本段脚本真正用到的那几个调用。
function makeIdb(opts){
  const store = new Map();
  if (opts && opts.seed) store.set('entry_labels_x.json', opts.seed);
  return {
    open(){
      const req = {};
      setTimeout(() => {
        if (opts && opts.openFails) { req.onerror && req.onerror(); return; }
        req.result = {
          transaction(){
            return { objectStore(){ return {
              get(k){ const r = {}; setTimeout(() => { r.result = store.get(k); r.onsuccess && r.onsuccess(); }, 0); return r; },
              put(v, k){ store.set(k, v); const r = {}; setTimeout(() => { r.onsuccess && r.onsuccess(); }, 0); return r; },
              delete(k){ store.delete(k); const r = {}; setTimeout(() => { r.onsuccess && r.onsuccess(); }, 0); return r; },
            }; } };
          },
        };
        req.onsuccess && req.onsuccess();
      }, 0);
      return req;
    },
  };
}
function makeHandle(name, perm){
  return {
    name: name,
    _perm: perm,
    queryPermission(){ return Promise.resolve(this._perm); },
    requestPermission(){ LOG.push(['requestPermission']); this._perm = opts_grantOnRequest ? 'granted' : 'denied'; return Promise.resolve(this._perm); },
    createWritable(){ return Promise.resolve({
      write(d){ LOG.push(['write', name, d]); return Promise.resolve(); },
      close(){ return Promise.resolve(); },
    }); },
  };
}
"""


def _run_export(scenario: str) -> list:
    """在 node 里跑真实的导出脚本，返回它实际做了哪些事（LOG）。"""
    script = _export_script()
    harness = _HARNESS_PRELUDE + scenario + "\n" + script + """
(async function(){
  await window.exportLabels();
  await new Promise(r => setTimeout(r, 20));
  console.log(JSON.stringify(LOG));
})();
"""
    r = subprocess.run(["node", "-e", harness], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, f"Node 执行失败：{r.stderr}"
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_export_reuses_saved_handle_when_permitted():
    """记住的句柄权限仍在 → 直接写，**不弹选择器**。"""
    log = _run_export("""
var opts_grantOnRequest = true;
const saved = makeHandle('记住的.json', 'granted');
const indexedDB = makeIdb({seed: saved});
const window = { indexedDB: indexedDB,
  showSaveFilePicker(){ LOG.push(['picker']); return Promise.reject(new Error('不该被调用')); } };
""")
    assert ["write", "记住的.json", '{"ok":1}'] in log, f"应直接写进记住的位置，实得 {log}"
    assert ["picker"] not in log, f"权限还在时不该再弹选择器，实得 {log}"
    assert ["anchor"] not in log, f"不该退到 <a download>，实得 {log}"


def test_export_requests_permission_when_expired():
    """句柄还在但权限退回 prompt → 重新申请，同意后直接写，仍不弹选择器。

    这条防「记住了却因为权限过期而白记一场」：不重新申请的话
    `createWritable()` 只会抛异常，用户每次还是得重选一遍。
    """
    log = _run_export("""
var opts_grantOnRequest = true;
const saved = makeHandle('记住的.json', 'prompt');
const indexedDB = makeIdb({seed: saved});
const window = { indexedDB: indexedDB,
  showSaveFilePicker(){ LOG.push(['picker']); return Promise.reject(new Error('不该被调用')); } };
""")
    assert ["requestPermission"] in log, f"权限过期必须重新申请，实得 {log}"
    assert ["write", "记住的.json", '{"ok":1}'] in log, f"同意后应直接写，实得 {log}"
    assert ["picker"] not in log, f"不该退回选择器，实得 {log}"


def test_export_falls_back_when_permission_denied():
    """用户拒绝续权 → 退回选择器，选完照常写，并把新句柄记下来。"""
    log = _run_export("""
var opts_grantOnRequest = false;
const saved = makeHandle('旧的.json', 'prompt');
const indexedDB = makeIdb({seed: saved});
const picked = makeHandle('新选的.json', 'granted');
const window = { indexedDB: indexedDB,
  showSaveFilePicker(){ LOG.push(['picker']); return Promise.resolve(picked); } };
""")
    assert ["requestPermission"] in log
    assert ["picker"] in log, f"拒绝之后必须退回选择器，实得 {log}"
    assert ["write", "新选的.json", '{"ok":1}'] in log, f"选完照常写，实得 {log}"
    assert not any(e[0] == "write" and e[1] == "旧的.json" for e in log), (
        f"用户拒绝的那个位置不许再写，实得 {log}")


def test_export_survives_indexeddb_failure():
    """IndexedDB 打不开（隐私模式）→ 导出照常走选择器，绝不整个失效。

    Task 45 有过前科：一处 JS 出错导致导出全程无效，而页面照常渲染、按钮
    照常可点，无人发现。
    """
    log = _run_export("""
var opts_grantOnRequest = true;
const picked = makeHandle('新选的.json', 'granted');
const window = { indexedDB: { open(){ throw new Error('隐私模式禁用 IndexedDB'); } },
  showSaveFilePicker(){ LOG.push(['picker']); return Promise.resolve(picked); } };
""")
    assert ["picker"] in log, f"IndexedDB 挂了也要能走选择器，实得 {log}"
    assert ["write", "新选的.json", '{"ok":1}'] in log, f"导出必须仍然成功，实得 {log}"


def test_export_falls_back_to_anchor_without_file_system_access():
    """浏览器压根不支持 File System Access（Safari/Firefox）→ <a download>。"""
    log = _run_export("""
var opts_grantOnRequest = true;
const window = {};
const indexedDB = null;
""")
    assert ["anchor"] in log, f"不支持时必须退到 <a download>，实得 {log}"


def test_export_script_passes_node_check(tmp_path):
    """新加的这段 JS 要过 `node --check`（Task 45 的护栏，一行注释吃掉整段
    代码那次之后立的规矩）。"""
    f = tmp_path / "export.js"
    f.write_text("const EXPORT_FILENAME='x';const window={};"
                 "const document={getElementById:()=>null};"
                 "function buildLabelsPayload(){return {};}\n" + _export_script(),
                 encoding="utf-8")
    r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, f"导出升级脚本不是合法 JavaScript：\n{r.stderr}"


def test_export_banner_shows_saved_target_and_forget_button():
    """页面上要显示「上次保存到：…」并给一个「改存到别处」的按钮。"""
    from pipeline.run_all import _build_export_upgrade

    banner, script = _build_export_upgrade(Path("/tmp/work/entry_labels_x.json"))
    assert 'id="saved-target"' in banner, "缺少「上次保存到」的显示位"
    assert 'id="forget-target"' in banner and "改存到别处" in banner, "缺少清除按钮"
    assert "上次保存到" in script, "脚本要把上次的位置填进去"
    assert "forgetHandle" in script, "「改存到别处」要真的清掉记住的句柄"
