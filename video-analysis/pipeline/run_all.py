"""阶段编排 CLI。

每个阶段读上一阶段落盘的产物，故可单独重跑：
    python -m pipeline.run_all --stages 2 --orders 2,7,10

本文件同时承载 Task 14（阶段 1–4 编排）与 Task 15（阶段 5–7 装配/画廊/排名）
的产出——Task 14 原计划的落地任务在本仓库从未真正提交（既无 run_all.py、
也无 tests/test_run_all.py），故本次一并按 task-14-brief.md 的既定接口补齐，
再叠加 Task 15 要求的改动。
"""
from __future__ import annotations

import argparse
import html as html_lib
import json
import sys
from pathlib import Path

from pipeline import config, paths

# 阶段编号 = 执行顺序，**付费那一步排在人工核实之后**（用户 2026-08-29：
# 「重新对各个阶段进行梳理排序，VLM 打标应放到最后吧」）。
#
# 旧编号是历史遗留：⑤ 打标、⑥ 出挑选页——而挑选页正是人工核实的入口，必须在
# 打标之前。编号与执行顺序相反，于是长出了 `prepare`「跑 1,2,3,4,6 跳过 5」
# 这个补丁命令，以及阶段 6 里「没有 annotations 就只出挑选页」那条特例分支
# （一个阶段同时干着人工前与人工后两件事）。重排把两者都消掉了。
#
# 旧号 → 新号：5→7、6→5(挑选页)+8(画廊)、7→9；6 是新增的登记表那一步
# （此前是 `python -m pipeline.registry` 这个无编号的旁路命令，最容易被忘）。
STAGES = {
    "1": "读时间表与人工标注",
    "2": "抽帧",
    "3": "人物检测与跟踪",
    "4": "座位分配（基线，仅供人工修正）",
    "5": "生成挑选页",
    # ——————— 以上无需人工输入；以下全部依赖人工核实结果 ———————
    "6": "座位登记表与核对页",
    "7": "打标",
    "8": "生成画廊",
    "9": "项目排名",
}

# 人工核实这条分界线的位置：`prepare` 与文档都从这里取，不各写一份数字。
LAST_PRE_HUMAN_STAGE = "5"


def _write_judge_annotations(work, sheet: str, rows: list) -> "Path":
    """把评审席的打标结果写进 `judges_<组>.json`，返回路径。"""
    import json as _json
    from pipeline import paths
    out = paths.judges_path(work, sheet)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(rows, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    return out


def _code_the_judges(work, sheet: str, use_vlm: bool):
    """读登记表里的评审席，打标并返回记录；没有登记表就返回 None（不报错）。

    增量：上一轮的 `judges_<组>.json` 作为 `prior` 传进去，帧摘要没变的那几位
    直接复用，不重复烧钱——与创业者那一路同一条口径。
    """
    import json as _json
    from pipeline import paths
    from pipeline.judgevlm import assemble_judges

    reg_path = paths.registry_path(work, sheet)
    if not reg_path.exists():
        print("   ⚠ 没有座位登记表，评审席这一步跳过（先跑阶段 6）")
        return None
    try:
        registry = _json.loads(reg_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        print(f"   ⚠ 登记表读不出（{exc}），评审席这一步跳过")
        return None
    prior = None
    jp = paths.judges_path(work, sheet)
    if jp.exists():
        try:
            prior = _json.loads(jp.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            prior = None
    return assemble_judges(registry, sheet=sheet, use_vlm=use_vlm, prior=prior)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="路演着装识别流水线")
    p.add_argument("--stages", default=",".join(STAGES),
                   help="要运行的阶段，逗号分隔，如 1,2,3")
    p.add_argument("--orders", default=None,
                   help="只处理这些 order，逗号分隔；省略则全部 16 个")
    p.add_argument("--no-vlm", action="store_true", default=False,
                   help="不调用 VLM API，只跑像素轨（离线调试用）")
    p.add_argument("--staff-net", action="store_true", default=config.STAFF_NET_ENABLED,
                   help="开启工作人员跨片段聚类安全网（默认关闭，见 Ruling R51："
                        "实测精确率 0/5，会把每一位真实创业者都标成疑似工作人员）")
    p.add_argument("--force-vlm", action="store_true", default=False,
                   help="忽略 Task 45 的增量打标缓存（config.VLM_INCREMENTAL），"
                        "强制对全部候选人重新调用 VLM")
    ns = p.parse_args(argv)
    ns.stages = [s.strip() for s in ns.stages.split(",") if s.strip()]
    for s in ns.stages:
        if s not in STAGES:
            p.error(f"未知阶段 {s}，可选 {list(STAGES)}")
    ns.orders = ([int(x) for x in ns.orders.split(",")] if ns.orders else None)
    return ns


def _require(path: Path, stage: str) -> None:
    """上游产物缺失时给出明确指引而非晦涩的 FileNotFoundError（Ruling R7）。"""
    if not path.exists():
        raise SystemExit(f"缺少 {path.name}，请先运行阶段 {stage}：\n"
                         f"  .venv/bin/python -m pipeline.run_all --stages {stage}")


# ---------------------------------------------------------------------------
# Task 45 · 改动二 + 追加改动三 —— entry_review.html 的自动升级后处理。
#
# entry_review.html 本身（pipeline/entryreview.py）不在本任务可改文件列表
# 内，且控制器正并行改动该文件的其它部分（图片尺寸/导出按钮悬浮），直接
# 编辑会与那份并行工作冲突。改为对 `render_entry_review` 产出的 HTML 字符
# 串做后处理：追加新的 <script>，覆盖同名的全局函数/变量——浏览器里同一
# 文档内多个 classic <script> 标签共享全局作用域，后声明的同名
# `function`/顶层 `const` 会覆盖先声明的，这是标准行为，不是取巧。
#
# 两件事：
#   1. 导出升级（总是做）：exportLabels() 优先用 File System Access API
#      直接"另存为"到 work 目录，不支持/用户取消/抛异常时回退到原有的
#      <a download> 方案；页面上写明目标路径全文。
#   2. 预填已有标注（entry_labels_*.json 存在时才做）：把文件内容转换成
#      页面内部 `store` 用的 key 格式，与 localStorage 现有值合并——
#      **文件优先**（同 key 时文件覆盖 localStorage），localStorage 独有
#      的 key（尚未导出的新标注）保留不动。
# ---------------------------------------------------------------------------

_ENTRY_REVIEW_MAIN_MARKER = '<div id="main"></div>'


def _entry_labels_to_store(entry_labels: dict) -> dict:
    """把 `entry_labels_*.json` 的结构转成 entry_review.html 内部 `store`
    用的 key 格式（候选走 `f"{seg_order}::{稳定标识}"`，与页面 JS 里的
    `candKey()` 同一套拼法；座位否决走 `f"seatovr::{seg_order}::{seat}"`，
    与页面 JS 里的 `seatOvrKey()` 同一套拼法）。

    座位号/人数在页面里都是 `<select>` 的值，恒为字符串；文件里是 `int`/
    `None`。`None` 必须转成空串（对应"不确定"选项），不能让 JS 侧
    `String(null)`（="null"）与任何座位号选项字符串比较失败，导致页面上
    看起来"什么都没选中"。

    Task 49：`seat_overrides`（座位否决/替代）能原样回填页面，因为它的键是
    (段号, 座位号)，重检测不会改变；`position_exclusions` 记的是位置
    (cx, y2) 而不是候选，回填需要先把位置匹配回某个候选，本任务不做——它
    只存在 localStorage 里，换浏览器后勾选框看起来是空的（**文件里的排除
    仍然生效**，`entryexit_fill_plan` 读的是文件）。
    """
    from pipeline.entryexit import LABEL_RESERVED_KEYS

    store: dict[str, dict] = {}
    for seg_order, seg_labels in (entry_labels or {}).items():
        # 顶层保留键（_meta / seat_overrides / position_exclusions）不是段：
        # 当成段号展开会得到一堆对不上任何候选的预填项，且不报错。
        if seg_order in LABEL_RESERVED_KEYS or not isinstance(seg_labels, dict):
            continue
        for local_key, rec in seg_labels.items():
            if not isinstance(rec, dict):
                continue
            seat = rec.get("seat")
            store[f"{seg_order}::{local_key}"] = {
                "role": rec.get("role", ""),
                "seat": "" if seat is None else str(seat),
                "n_people": str(rec.get("n_people", 1)),
                "note": rec.get("note", ""),
            }
    for seg_order, seats in ((entry_labels or {}).get("seat_overrides") or {}).items():
        if not isinstance(seats, dict):
            continue
        for seat, rec in seats.items():
            if not isinstance(rec, dict):
                continue
            store[f"seatovr::{seg_order}::{seat}"] = {
                "rejected": bool(rec.get("rejected")),
                "replacement_stable_id": rec.get("replacement_stable_id") or "",
                "note": rec.get("note", ""),
            }
    return store


def _build_export_upgrade(target_labels_path: Path) -> tuple[str, str]:
    """返回 (banner_html, script_html)：导出升级——优先"另存为"到指定路径，
    不支持/失败/取消时回退原有 `<a download>`。

    JS 里复用页面已声明的全局 `buildLabelsPayload()`/`EXPORT_FILENAME`
    （entry_review.html 模板里的既有全局函数/常量），**不重新拼一遍导出
    内容**——两处各写一份必然日后走样（Ruling R39 的教训；Task 46 就是在
    这里踩到的：稳定标识与 `_meta` 指纹只改了模板一侧，这里照常导出老格式
    且不报错）。本函数只负责「怎么存」。

    Task 52 需求五（用户 2026-08-23：每次导出都要重新选目录）：把用户选中
    的 `FileSystemFileHandle` **存进 IndexedDB**（该对象可结构化克隆，是
    目前唯一能跨页面刷新留存的办法；localStorage 只存字符串，存不了它）。
    下次导出的降级链，从上到下逐级回退：

      1. 取回句柄 → `queryPermission({mode:'readwrite'})` 为 granted →
         直接写，不弹任何选择器；
      2. 权限过期（prompt）→ `requestPermission({mode:'readwrite'})`，
         用户同意 → 直接写；
      3. 用户拒绝 / 没有句柄 / 句柄失效 → `showSaveFilePicker()` 重新选，
         选完再存回 IndexedDB；
      4. 浏览器不支持 File System Access（Safari、Firefox）/ 上面任何一步
         抛异常 → `<a download>`。

    **每一级都包在 try/catch 里**，任何一环出问题都只是降到下一级，绝不能
    让导出整个失效：IndexedDB 在隐私模式下会直接抛异常，Safari 的实现也
    时有差异。Task 45 曾因一行注释吃掉整段 JS 导致导出全程失效而无人发现
    （页面照常渲染、按钮照常可点），此后加了 `node --check` 护栏——新代码
    同样要过那道检查，所以脚本按真实换行拼，不挤成一行。
    """
    escaped_path = html_lib.escape(str(target_labels_path))
    banner = (
        '<div id="export-path-banner" style="background:#eef4fb;'
        'border-left:4px solid #4a7fb5;padding:8px 12px;margin:8px 0;'
        'font-size:13px">'
        '若浏览器支持“另存为”（Chrome/Edge），点击下方“导出标注 JSON”可'
        f'直接选择保存到：<code>{escaped_path}</code>；'
        '否则文件会存入浏览器默认下载目录，请手动移动到该文件夹'
        '（或等流水线下次运行时自动从下载目录收取，见阶段⑤输出）。'
        # Task 52 需求五：记住上次保存的位置。这两个元素由下面的脚本填内容
        # ——用户要能看见「这次会写到哪」，也要能一键改到别处。
        '<div id="saved-target" style="margin-top:6px"></div>'
        '<button type="button" id="forget-target" style="margin-top:6px;'
        'padding:4px 12px;font-size:13px;background:#fff;color:#333;'
        'border:1px solid #888;border-radius:6px;cursor:pointer">'
        '改存到别处</button>'
        '</div>'
    )
    # 用真实换行拼（`\n`.join / 三引号），不要把整段脚本挤成不含换行的
    # 一行——里面只要有一句 `//` 行注释，它就会吃掉后面的全部代码，浏览器
    # 抛 SyntaxError、`window.exportLabels` 从未被赋值，而页面照常渲染、
    # 按钮照常可点（回落到原版导出），"改了但没生效且不报错"。这不是假想：
    # Task 45 交付的就是那样一行，一直是坏的，由
    # `test_injected_scripts_are_valid_javascript`（node --check）查出。
    script = """<script>
// ---- Task 52 需求五：把上次选中的保存位置记住 ----
// FileSystemFileHandle 可结构化克隆，因此能存进 IndexedDB；localStorage
// 只存字符串，存不了它。整段的每一处 await 都包了 try/catch：隐私模式下
// IndexedDB 直接抛异常，Safari 的实现也时有差异，任何一环出问题都必须
// 只降一级，而不是让导出整个失效。
const HANDLE_DB = 'dress_entry_review_handles';
const HANDLE_STORE = 'handles';
const HANDLE_KEY = EXPORT_FILENAME;

function openHandleDb(){
  return new Promise(function(resolve, reject){
    try {
      if (!window.indexedDB) { resolve(null); return; }
      const req = window.indexedDB.open(HANDLE_DB, 1);
      req.onupgradeneeded = function(){
        try { req.result.createObjectStore(HANDLE_STORE); } catch (e) { /* 已存在 */ }
      };
      req.onsuccess = function(){ resolve(req.result); };
      req.onerror = function(){ resolve(null); };
      req.onblocked = function(){ resolve(null); };
    } catch (err) { resolve(null); }
  });
}

function handleDbOp(mode, fn){
  // 统一的一次事务；任何异常都收敛成 null，调用方据此降级。
  return openHandleDb().then(function(db){
    if (!db) return null;
    return new Promise(function(resolve){
      try {
        const tx = db.transaction(HANDLE_STORE, mode);
        const req = fn(tx.objectStore(HANDLE_STORE));
        req.onsuccess = function(){ resolve(req.result === undefined ? true : req.result); };
        req.onerror = function(){ resolve(null); };
      } catch (err) { resolve(null); }
    });
  }).catch(function(){ return null; });
}

async function loadSavedHandle(){
  try { return await handleDbOp('readonly', function(s){ return s.get(HANDLE_KEY); }); }
  catch (err) { return null; }
}

async function rememberHandle(handle){
  try { await handleDbOp('readwrite', function(s){ return s.put(handle, HANDLE_KEY); }); }
  catch (err) { /* 记不住只是下次要再选一次，不影响本次导出 */ }
}

async function forgetHandle(){
  try { await handleDbOp('readwrite', function(s){ return s.delete(HANDLE_KEY); }); }
  catch (err) { /* 同上 */ }
  await showSavedTarget();
}

async function ensureWritable(handle){
  // 权限可能在下次打开页面时退回 'prompt'——**必须重新申请**，不能因为
  // 「上次给过」就直接写（那样只会静默抛异常、退回选择器，用户白选一次）。
  try {
    if (!handle || !handle.queryPermission) return false;
    const opts = {mode: 'readwrite'};
    if (await handle.queryPermission(opts) === 'granted') return true;
    return (await handle.requestPermission(opts)) === 'granted';
  } catch (err) { return false; }
}

async function writeTo(handle, dataStr){
  try {
    const writable = await handle.createWritable();
    await writable.write(dataStr);
    await writable.close();
    return true;
  } catch (err) { return false; }
}

async function showSavedTarget(){
  const el = document.getElementById('saved-target');
  const btn = document.getElementById('forget-target');
  if (!el) return;
  let name = '';
  try {
    const h = await loadSavedHandle();
    name = (h && h.name) ? h.name : '';
  } catch (err) { name = ''; }
  el.textContent = name ? ('上次保存到：' + name + '（再次导出会直接写这里）')
                        : '尚未记住保存位置，首次导出会让你选一次。';
  if (btn) btn.style.display = name ? '' : 'none';
}

// Task 53 需求五：导出成功后给出页内提示（用户原话：「再次导出的时候，
// 没有任何提示，应该弹出提示」）。三条路径落地的位置完全不同，`where` 让
// 页面能写出「到底存到哪儿了」——提示条本身与文案都由页面的
// `showExportToast` 负责（entryreview.py），这里只报告「哪条路径、什么
// 名字」，两处各写一份文案必然日后走样。
//
// 整个包在 try 里、并且允许页面根本没有这个函数（旧页面 + 新 run_all）：
// 提示是锦上添花，一个 TypeError 把导出打断就是 Task 45 那次「改了但没
// 生效且不报错」的重演。
function notifyExport(where, name, dataStr){
  try {
    if (window.showExportToast) {
      window.showExportToast(where, name, JSON.parse(dataStr));
    }
  } catch (err) { /* 文件已经写成功了，提示没弹出来不影响这件事 */ }
}

window.exportLabels = async function(){
  const dataStr = JSON.stringify(buildLabelsPayload());
  if (window.showSaveFilePicker) {
    // ① 记住的句柄，权限还在（或用户当场同意续期）→ 直接写，不弹选择器
    try {
      const saved = await loadSavedHandle();
      if (saved && await ensureWritable(saved) && await writeTo(saved, dataStr)) {
        await showSavedTarget();
        // 这一条最需要提示：连选择器都不弹，屏幕上什么都不动，用户完全
        // 无法分辨「写成功了」与「按钮坏了」。
        notifyExport('handle', saved.name || EXPORT_FILENAME, dataStr);
        return;
      }
    } catch (err) { /* 降到 ② */ }
    // ② 重新选一次，并把新句柄记下来
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName: EXPORT_FILENAME,
        types: [{description:'JSON', accept:{'application/json':['.json']}}],
      });
      if (await writeTo(handle, dataStr)) {
        await rememberHandle(handle);
        await showSavedTarget();
        notifyExport('picker', handle.name || EXPORT_FILENAME, dataStr);
        return;
      }
    } catch (err) {
      if (err && err.name === 'AbortError') return;
      // 其他异常（权限拒绝等）：回退到 <a download>，不弹错误提示
    }
  }
  // ③ 兜底：浏览器不支持 File System Access，或上面每一级都没成
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([dataStr],{type:'application/json'}));
  a.download=EXPORT_FILENAME; a.click();
  notifyExport('download', EXPORT_FILENAME, dataStr);
};

(function(){
  try {
    const btn = document.getElementById('forget-target');
    if (btn) btn.onclick = forgetHandle;
    showSavedTarget();
  } catch (err) { /* 提示条是锦上添花，坏了也不能拖垮导出 */ }
})();
</script>"""
    return banner, script


def _build_preload_script(store: dict) -> tuple[str, str]:
    """返回 (banner_html, script_html)：把已有标注文件预填进页面。

    合并规则（用户明确要求，也原样写进页面供核对）：**文件里的标注优先**
    （那是用户明确导出过的），仅存在于本机浏览器 localStorage、尚未导出到
    文件的新标注予以保留。`Object.assign(store, PRELOAD)` 天然实现这条
    规则：同 key 时后者（文件）覆盖前者（localStorage 已加载进 store 的
    值），文件没有的 key 保持 localStorage 原值不动。
    """
    banner = (
        '<div id="preload-banner" style="background:#e8f4ea;'
        'border-left:4px solid #2e7d32;padding:8px 12px;margin:8px 0;'
        'font-size:13px">'
        '检测到已有标注文件，已自动载入。'
        '合并规则：文件里的标注优先（已明确导出过的），仅存在于本机浏览器 '
        'localStorage、尚未导出到文件的新标注予以保留。'
        '</div>'
    )
    script = (
        "<script>(function(){"
        f"var PRELOAD={json.dumps(store, ensure_ascii=False)};"
        # 用 mergePreload 而不是直接 Object.assign：用户明确「清除本条标注」
        # 过的那些不许被文件里那份盖回来（用户 2026-09-09：「清除了标注也没
        # 生效」）。页面里没有这个函数（旧页面）时退回原样合并。
        "Object.assign(store, (typeof mergePreload==='function')"
        " ? mergePreload(PRELOAD) : PRELOAD);"
        "localStorage.setItem(KEY, JSON.stringify(store));"
        "render();"
        "})();</script>"
    )
    return banner, script


def upgrade_entry_review_html(
    html: str, target_labels_path: Path, entry_labels: dict | None,
) -> tuple[str, int]:
    """对 `render_entry_review` 产出的 HTML 做后处理（Task 45）。

    总是加导出升级（File System Access API 优先 + 回退 + 目标路径提示）；
    `entry_labels` 非空时额外加预填。依赖 `<div id="main"></div>` 这个锚点
    ——它是页面 JS `render()` 的挂载点 id，属于功能性标记，不太可能被样式
    改动动到；找不到时直接报错而不是静默跳过注入，避免"代码改了但产物
    没跟着更新，且不报错"这个本任务原本要修的同一类缺陷在这里重演。

    返回 (新 html, 预填的标注条数)。
    """
    if _ENTRY_REVIEW_MAIN_MARKER not in html:
        raise RuntimeError(
            'entry_review.html 缺少 <div id="main"></div> 锚点，无法注入'
            '自动升级——entryreview.py 的模板可能已改动，请检查 '
            'pipeline.run_all.upgrade_entry_review_html 的锚点假设'
        )

    banners = []
    scripts = []

    export_banner, export_script = _build_export_upgrade(target_labels_path)
    banners.append(export_banner)
    scripts.append(export_script)

    n_preload = 0
    if entry_labels:
        store = _entry_labels_to_store(entry_labels)
        n_preload = len(store)
        preload_banner, preload_script = _build_preload_script(store)
        banners.append(preload_banner)
        scripts.append(preload_script)

    new_html = html.replace(
        _ENTRY_REVIEW_MAIN_MARKER,
        _ENTRY_REVIEW_MAIN_MARKER + "".join(banners),
        1,
    )
    new_html += "".join(scripts)
    return new_html, n_preload


def build_entry_review_page(segments, all_dets, det_path: Path,
                             seats_path: Path, safe_sheet: str) -> Path:
    """生成（并就地升级）人工挑选页 `entry_review.html`，返回它的路径。

    从阶段 6 里原样提出来的一段，**行为逐字未变**——提出来是为了让
    `pipeline.prepare`（Task 52 需求六：只跑取帧与人工辅助定位、不跑 VLM）
    能复用同一份逻辑。两处各拼一份的后果在本项目发生过不止一次：一边改了
    一边没改，页面照常生成、看不出旧在哪里（Ruling R39）。

    只读 `entry_labels_*.json`，**从不写**——那是用户的劳动成果。
    """
    import json as _json

    from pipeline.entryreview import render_entry_review
    from pipeline.entryexit import (
        detections_fingerprint as _detections_fingerprint,
        candidates_for_labels as _candidates_for_labels,
        load_entry_labels as _load_entry_labels,
    )

    review_seats_by_order: dict[int, dict[int, int]] = {}
    if seats_path.exists():
        seats_raw_for_review = _json.loads(seats_path.read_text(encoding="utf-8"))
        review_seats_by_order = {
            int(o): {int(t): s for t, s in m.items()}
            for o, m in seats_raw_for_review.items()
        }
    # Task 56G：交付物写到 work/Output/<组>/，组目录只留原始与过程文件。
    entry_review_path = paths.output_dir(config.WORK) / "entry_review.html"
    entry_review_path.parent.mkdir(parents=True, exist_ok=True)
    # Task 46：页面导出的标注要带上这份检测结果的指纹，下游据此判断
    # 「这份标注是不是这份检测的」。
    review_det_fp = _detections_fingerprint(det_path)
    render_entry_review(segments, all_dets, entry_review_path,
                         seats_by_order=review_seats_by_order,
                         detections_fp=review_det_fp)

    # 已有标注文件绝不能被覆盖（用户的劳动成果）——这里只读，从不写。
    # 存在时把内容预填进页面（并在页面上说明合并规则），让用户重新
    # 打开页面不会以为标注不见了。
    entry_labels_path_for_review = paths.labels_path(config.WORK, config.SHEET)
    existing_labels = None
    if entry_labels_path_for_review.exists():
        try:
            # 走与阶段 5 同一套失效校验（Task 46）：指纹一致直接预填；
            # 不一致但用的是稳定标识时按视频内容重新对齐后预填——预填的
            # 键必须是**当前**候选的稳定标识，否则页面上对不上任何候选，
            # 用户会以为标注不见了；旧格式/匹配率过低则整份作废、不预填。
            review_cands_by_order = {
                # 展示 ∪ 折叠：重检测后按视频内容重新对齐时，标在折叠区
                # 的那几条同样要对得上，否则它们会被当成失配丢掉。
                seg.order: _candidates_for_labels(all_dets, seg) for seg in segments
            }
            existing_labels = _load_entry_labels(
                entry_labels_path_for_review,
                detections_fp=review_det_fp,
                candidates_by_order=review_cands_by_order,
            ) or None
        except (ValueError, OSError) as exc:
            print(f"   ⚠ 读取既有 {entry_labels_path_for_review.name} 失败，"
                  f"挑选页不预填：{exc}")

    review_html = entry_review_path.read_text(encoding="utf-8")
    review_html, n_preload = upgrade_entry_review_html(
        review_html, entry_labels_path_for_review, existing_labels,
    )
    entry_review_path.write_text(review_html, encoding="utf-8")
    if existing_labels:
        print(f"   检测到已有标注文件 {entry_labels_path_for_review.name}，"
              f"已自动载入 {n_preload} 条既有标注到 {entry_review_path.name}")
    return entry_review_path


def run(stages: list[str], orders: list[int] | None,
        no_vlm: bool = False, staff_net: bool | None = None,
        force_vlm: bool = False) -> None:
    from pipeline.schedule import load_segments, save_segments, load_segments_json
    from pipeline.windows import compute_windows
    from pipeline.frames import extract_all, save_index, load_index, probe_duration
    from pipeline.detect import (detect_frames, detect_frames_incremental,
                                 group_digests_path, load_detections,
                                 load_group_digests, save_detections,
                                 save_group_digests)

    if staff_net is None:
        staff_net = config.STAFF_NET_ENABLED

    config.WORK.mkdir(parents=True, exist_ok=True)
    seg_path = config.WORK / "segments.json"

    if "1" in stages:
        print("① 读时间表与人工标注")
        save_segments(load_segments(), seg_path)

    _require(seg_path, "1")
    segments = load_segments_json(seg_path)
    if orders:
        segments = [s for s in segments if s.order in orders]
    print(f"   处理 {len(segments)} 个项目，共 {sum(s.presenter_num for s in segments)} 位创业者")

    idx_path = config.WORK / "frames_index.json"
    if "2" in stages:
        print("② 抽帧")
        ws = compute_windows(segments, probe_duration(config.VIDEO))
        save_index(extract_all(config.VIDEO, ws, config.WORK / "frames"), idx_path)

    # 帧与时间表的一致性校验（用户 2026-08-22 发现的真实错配）。
    #
    # 文化4组的 Book1 从 13 家变成 14 家（补了一家原先缺时间戳的），新增那家
    # 排在第 8 位，其后所有企业的行号后移一位；而帧是按旧编号抽的、一直存在
    # 盘上没重抽，于是 seg 8-13 的衣着/颜色/性别判定全部取自隔壁公司。整条
    # 流水线照常跑完、表照常生成、图照常出，**没有任何一处报错**——是用户
    # 肉眼看出「华韵的图是中科直线的」才发现的。
    #
    # 放在检测与座位分配之前硬失败，而不是打印警告：这类错配一旦流到下游，
    # 产出的每一个数字都是错的且看不出来，继续跑没有任何意义。
    if ("3" in stages or "4" in stages) and (config.WORK / "frames").is_dir():
        from pipeline.frames import segment_frame_mismatches
        bad = segment_frame_mismatches(segments, config.WORK / "frames")
        out_of_window = [b for b in bad if b["reason"] == "out_of_window"]
        if out_of_window:
            lines = "\n".join(
                f"    order{b['order']:>2} {b['company']}：{b['detail']}"
                for b in out_of_window)
            raise RuntimeError(
                f"帧与时间表不一致（{len(out_of_window)} 段），拒绝继续——"
                f"下游产出的衣着/颜色/性别会取自别家企业且不会报错：\n{lines}\n"
                f"  修法：重跑阶段 2 重抽帧（企业行序变动后必须重抽）。"
            )
        no_frames = [b for b in bad if b["reason"] == "no_frames"]
        if no_frames:
            print(f"   ⚠ {len(no_frames)} 段没有帧："
                  + "、".join(f"order{b['order']}" for b in no_frames))

    det_path = config.WORK / "detections.json"
    if "3" in stages:
        _require(idx_path, "2")
        print("③ 人物检测与跟踪")
        # 只重跑**帧集合变了**的那几个 (段号, 窗口) 组，其余原样留用上一轮的
        # 结果。这是可证明等价的加速，不是近似：跟踪器每组开头归零（连 track
        # 编号计数器一起），所以每组只取决于自己的帧——见
        # `detect.detect_frames_incremental` 的 docstring 与那条逐条比对
        # 增量与全量结果的测试。
        #
        # 用户 2026-09-09：「能只对 order 16 重抽吗，不抽其他项目，是不是就可以
        # 省时间了」。加密一段之后，全量重跑要一个小时，实际变的只有一两组。
        _all_frames = load_index(idx_path)
        _prior_digests = load_group_digests(group_digests_path(det_path))
        _prior = load_detections(det_path) if det_path.exists() else []
        if _prior_digests and _prior:
            dets, _digests, _reused, _redone = detect_frames_incremental(
                _all_frames, _prior, _prior_digests)
            # 凡有留用必须可见：不说的话用户无从知道这一轮到底跑了什么，
            # 而「本该重跑却留用了」正是那种看起来完全正常的错。
            print(f"   增量：留用 {len(_reused)} 组、重跑 {len(_redone)} 组"
                  + (f"（{'、'.join(f'{o}/{k}' for o, k in _redone)}）"
                     if _redone else "（无变化）"), flush=True)
        else:
            why = "没有上一轮的分组摘要" if not _prior_digests else "没有上一轮的检测结果"
            print(f"   {why}，本次全量检测", flush=True)
            dets = detect_frames(_all_frames)
            from pipeline.detect import frames_group_digests as _fgd
            _digests = _fgd(_all_frames)
        save_detections(dets, det_path)
        save_group_digests(_digests, group_digests_path(det_path))

    seats_path = config.WORK / "seats.json"
    if "4" in stages:
        _require(det_path, "3")
        print("④ 座位分配")
        from pipeline.seats import (
            assign_segment_seats, depth_gap_report, threshold_samples,
            gender_match_candidates,
        )
        from pipeline.seatmatch import gender_match_report
        from pipeline.audit import threshold_health, audit_report
        dets = load_detections(det_path)

        # Task 33：性别匹配用的是阶段 5（打标）才产出的 `gender_judged`，而
        # 阶段 5 又要先有本阶段的座位号才能跑——先后顺序天然决定了性别匹配
        # 只能在**重跑**阶段 4 时才真正生效（读到上一轮 annotations.json 里
        # 已经产出的 gender_judged，按 (seg_order, track_id) 对上）。首次
        # 跑（annotations.json 还不存在，或读取失败）时退化为无信息可用，
        # `gender_match=True` 与 `gender_match=False` 结果完全一致——这是
        # 刻意设计（不能循环依赖），不是遗漏，见 seats.assign_segment_seats
        # 的 docstring。
        ann_path_for_gender = config.WORK / "annotations.json"
        gender_by_seg_track: dict[int, dict[int, str]] = {}
        if ann_path_for_gender.exists():
            try:
                prior = json.loads(ann_path_for_gender.read_text(encoding="utf-8"))
                for row in prior:
                    tid = (row.get("pixel") or {}).get("track_id")
                    seg_order = row.get("seg_order")
                    judged = (row.get("vlm") or {}).get("gender_judged")
                    if tid is not None and seg_order is not None:
                        gender_by_seg_track.setdefault(int(seg_order), {})[int(tid)] = (
                            judged or "不确定"
                        )
            except (ValueError, OSError) as exc:
                print(f"   ⚠ 读取上一轮 annotations.json 失败，性别匹配退化为无信息可用：{exc}")

        out = {}
        for s in segments:
            seats = assign_segment_seats(
                dets, s, gender_match=True,
                gender_judged_by_track=gender_by_seg_track.get(s.order),
            )
            out[str(s.order)] = {str(k): v for k, v in seats.items()}
            flag = "" if len(seats) == s.presenter_num else "  ← 人数不符"
            print(f"   order {s.order:>2} {s.company:<30} {len(seats)}/{s.presenter_num}{flag}")
        seats_path.write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

        # 性别匹配标定报告（Task 33）：每段一行，标注序列 vs VLM 序列、
        # 是否重排、是否冲突——不用等人工逐个企业核对完才发现某段性别对不上。
        gender_segments_data = [
            {
                "seg_order": s.order,
                "people": gender_match_candidates(
                    dets, s, gender_judged_by_track=gender_by_seg_track.get(s.order),
                ),
                "annotated_genders": [p.gender_female for p in s.presenters],
            }
            for s in segments if s.presenters
        ]
        for row in gender_match_report(gender_segments_data):
            conflict = "冲突" if row["gender_conflict"] else "正常"
            rearranged = "是" if row["rearranged"] else "否"
            print(f"   性别匹配 order {row['seg_order']:>2}：标注={row['annotated']} "
                  f"VLM={row['vlm']} 重排={rearranged} {conflict}")

        # Task 40：阈值体检——原先阶段 4 只打印深度断层那一条标定摘要，
        # 且是一套独立于 `pipeline.audit.threshold_health` 的手搓判据（分位数
        # + below/above 目测），换一个阈值就要重新手搓一遍解读逻辑。现在改为
        # 打印覆盖全部「实际起作用」阈值的完整体检表：每个阈值配的样本是它
        # 在生产代码里真正比较的那批数值（见 `seats.threshold_samples`
        # docstring），用同一套 `threshold_health` 判据统一给出「真空/稀疏/
        # 密集/样本不足」结论，`audit_report` 把最需要人工核对的（密集的）
        # 排在最前面。
        samples = threshold_samples(dets, segments)
        live_thresholds = {
            "RIGHT_REGION_X": config.RIGHT_REGION_X,
            "SEAT_MIN_H_RATIO": config.SEAT_MIN_H_RATIO,
            "SEAT_MIN_H_VS_TALLEST": config.SEAT_MIN_H_VS_TALLEST,
            "SEAT_MIN_FRAME_COVERAGE": config.SEAT_MIN_FRAME_COVERAGE,
            "SEAT_DUP_CONTAIN": config.SEAT_DUP_CONTAIN,
            "SEAT_PARTIAL_REL": config.SEAT_PARTIAL_REL,
        }
        checks = [
            threshold_health(samples.get(name, []), value, name=name)
            for name, value in live_thresholds.items()
        ]
        print("   阈值体检（阶段④实际生效的六个阈值，密集的排最前——最需要人工核对）：")
        for line in audit_report(checks).splitlines():
            print(f"     {line}")

        # 深度断层诊断（Task 29 新增，Task 31 起该机制默认停用，Task 40 起
        # `cut_at_depth_gap` 改为显式空操作，见其 docstring）：这条机制不在
        # 上面六个「实际生效」阈值之列，但 `depth_gap_report` 产出的 y2/height
        # 比值分布仍然是换组时评估「要不要恢复这条机制」的参考起点，继续
        # 复用同一套 `threshold_health` 判据打印，不再另起一套百分位数 +
        # below/above 的手搓解读。0.52 是 Task 31 停用前用两组视频标定出的
        # 历史经验值，仅供本次诊断参考，不影响 `cut_at_depth_gap` 的实际
        # 行为（它现在恒不截断）。
        _DEPTH_GAP_HISTORICAL_REFERENCE = 0.52
        report = depth_gap_report(dets, segments)
        ratios = [r["ratio"] for r in report if r["ratio"] != float("inf")]
        if ratios:
            depth_check = threshold_health(
                ratios, _DEPTH_GAP_HISTORICAL_REFERENCE,
                name="深度断层比值（历史参考 0.52，机制已停用）",
            )
            print(f"   {depth_check['detail']}")

    safe_sheet = config.sanitize_filename_component(config.SHEET)

    if "5" in stages:
        # 挑选页：人工核实的入口。**不读 annotations.json**——它是人工之前的
        # 产物，此前和画廊挤在同一个阶段里，于是新组第一次跑必然缺打标结果，
        # `prepare` 只好为此写一条特例分支。拆开之后两边各自干净。
        _require(seg_path, "1")
        _require(det_path, "3")
        print("⑤ 生成挑选页")
        segments_for_review = load_segments_json(seg_path)
        entry_review_path = build_entry_review_page(
            segments_for_review, load_detections(det_path),
            det_path, seats_path, safe_sheet)
        print(f"   已写 {entry_review_path}")

    if "6" in stages:
        # 座位登记表 + 核对页 + 交付包。此前是 `python -m pipeline.registry`
        # 与 `pipeline.bundle` 两个**无编号的旁路命令**，最容易被漏掉——漏了
        # 的表现是「人核对的是一张模型没看过的图」，本项目最忌讳的那一类。
        _require(det_path, "3")
        print("⑥ 座位登记表与核对页")
        from pipeline import bundle as _bundle, registry as _registry
        n_resolved = _registry.main(["--preview", "--work", str(config.WORK),
                                     "--sheet", config.SHEET])
        if n_resolved != 0:
            print("   ⚠ 登记表未能产出，后续阶段会拿不到人工核实结果")
        else:
            _bundle.main(["--group", config.WORK.name, "--sheet", config.SHEET])

    ann_path = config.WORK / "annotations.json"
    if "7" in stages:
        _require(det_path, "3")
        _require(seats_path, "4")
        print("⑦ 打标装配")
        import json as _json
        from dataclasses import asdict
        from pipeline.assemble import assemble_annotations
        from pipeline.entryexit import (
            load_entry_labels, adopt_downloaded_labels,
            default_downloads_dir, _downloaded_label_candidates,
            candidates_for_labels, detections_fingerprint,
        )

        # 追加改动三（用户 2026-08-17）：先把用户从浏览器导出、落在下载
        # 目录的 entry_labels_*.json 收进 work 目录——浏览器不允许网页写
        # 任意路径，entry_review.html 侧的"另存为"优先路径在很多浏览器/
        # 场景下不可用（Safari/Firefox 不支持 File System Access API、
        # 用户也可能取消），这一步才是主要保障。凡有文件被复制进来必须
        # 可见——三种情形（收取了/未发现/已是最新）各打印一行，不能让用户
        # 以为流水线读的是他刚导出的，其实读的是几天前那份。
        adopted = adopt_downloaded_labels(config.WORK, config.SHEET)
        if adopted is not None:
            print(f"   进出场标注：已从下载目录收取更新文件 {adopted.name} → {config.WORK}/")
        else:
            _downloads_dir = default_downloads_dir()
            if _downloaded_label_candidates(_downloads_dir, config.SHEET):
                print("   进出场标注：下载目录里有同名文件，但不比 work 目录里的更新，未收取")
            else:
                print(f"   进出场标注：下载目录（{_downloads_dir}）未发现 "
                      "entry_labels_*.json，未收取")

        # Task 45：增量打标——读上一轮 annotations.json（若存在）供
        # assemble_annotations 判断哪些人可以复用、跳过真正的 VLM 调用。
        # 必须在写盘覆盖之前读（下面第 284 行左右会把本轮结果写回同一个
        # 文件），且读失败（文件损坏/非法 JSON）不能让整条流水线崩溃——
        # 退化为"当作没有上一轮"，即全部重新调用，仍是正确但更慢的结果。
        prior_annotations = None
        if ann_path.exists():
            try:
                prior_annotations = _json.loads(ann_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                print(f"   ⚠ 读取上一轮 annotations.json 失败，增量打标退化为全量"
                      f"重跑：{exc}")

        seats_raw = _json.loads(seats_path.read_text(encoding="utf-8"))
        seats_by_order = {
            int(o): {int(t): s for t, s in m.items()} for o, m in seats_raw.items()
        }
        all_dets = load_detections(det_path)

        # Task 16：进出场轨迹配到座位（Ruling R18）。配不上的座位拿到 None，
        # assemble 会让 best_frame_entry 留空，而不是按 track_id 猜一个错的人。
        from pipeline.pairing import pair_transitions
        pairs_by_order = {
            seg.order: pair_transitions(all_dets, seg, seats_by_order.get(seg.order, {}))
            for seg in segments
        }

        # Task 44：进出场递补——剔除最里侧专家后从进出场候选补回一位真人。
        # entry_labels_*.json 由用户在 entry_review.html 上挑完候选后导出，
        # 文件不存在时传 None——assemble_annotations 的行为与本任务之前
        # 逐字段相同（硬要求，见其 docstring）。凡有递补必须可见：这里先
        # 打印一行说明本次是否启用，免得用户以为某人的衣着来自就座画面，
        # 而实际来自进出场画面。
        entry_labels_path = paths.labels_path(config.WORK, config.SHEET)
        if entry_labels_path.exists():
            # Task 46：标注必须核对是不是「这一份检测结果」的。企业行序变动
            # 后重抽帧+重检测，track_id 全部改变——旧标注大部分指向不存在的
            # track，少数碰巧编号还在但已是另一个人，错配不报错。指纹一致
            # 直接用；不一致但标注用的是稳定标识时按视频内容重新对齐并报告
            # 匹配率；匹配率过低或旧格式一律整份作废（宁可全废也不要错配）。
            det_fp = detections_fingerprint(det_path)
            cands_by_order = {
                seg.order: candidates_for_labels(all_dets, seg) for seg in segments
            }
            entry_labels = load_entry_labels(
                entry_labels_path, detections_fp=det_fp,
                candidates_by_order=cands_by_order,
            )
            if entry_labels:
                print(f"   进出场递补：已启用（读取 {entry_labels_path.name}）")
            else:
                # 作废等同于"没有可用标注"——退回 None，让 assemble 走它
                # 逐字段保证过的"不传 entry_labels"那条路径。
                entry_labels = None
                print(f"   进出场递补：{entry_labels_path.name} 未通过检测指纹"
                      "校验，本次不递补（见上方提示）")
        else:
            entry_labels = None
            print(f"   进出场递补：未启用（未找到 {entry_labels_path.name}，"
                  "如需递补请先在 entry_review.html 上标注并导出）")

        anns = assemble_annotations(
            segments, all_dets, seats_by_order,
            use_vlm=not no_vlm, pairs_by_order=pairs_by_order,
            entry_labels=entry_labels,
            prior_annotations=prior_annotations, force_vlm=force_vlm,
        )

        # 递补人数必须可见（凡有递补必须可见，本项目一贯要求）——不能让
        # 用户以为某人的衣着来自就座画面，其实来自进出场画面。
        backfilled_n = sum(1 for a in anns if a.source == "entryexit")
        print(f"   进出场递补：{backfilled_n} 人的衣着证据来自进出场画面（非就座画面）")

        # Task 17：跨片段外观聚类安全网（Ruling R19）。只置标志、不删任何一行。
        # 放在 assemble 之后单独跑：需要先有座位表才知道哪些人入了表。
        # 默认关闭（Ruling R51，参见 config.STAFF_NET_ENABLED 上的实测记录）。
        if staff_net:
            from pipeline.staff import person_crops, embed_crops, flag_suspect_staff
            persons = []
            for a in anns:
                tid = a.pixel.get("track_id")
                if tid is None:
                    continue
                track = [d for d in all_dets if d.seg_order == a.seg_order
                         and d.kind == "seated" and d.track_id == tid]
                persons.append((a.person_id, a.seg_order, embed_crops(person_crops(track))))
            flagged = flag_suspect_staff(persons)
            for a in anns:
                a.suspect_staff = int(a.person_id in flagged)
            print(f"   安全网标记 {len(flagged)} 人为疑似工作人员（仅标记，未删除）")

        # 装配阶段还会把每人的证据图字节暂存在 a.pixel["_evidence_png"]，
        # 供阶段 6 导出关键帧使用；写盘前先弹出，绝不能让二进制内容随
        # asdict 序列化进 annotations.json。
        payload = []
        for a in anns:
            d = asdict(a)
            d.get("pixel", {}).pop("_evidence_png", None)
            payload.append(d)
        ann_path.write_text(
            _json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8",
        )
        print(f"   装配 {len(anns)} 位创业者")

        # 评审席（用户 2026-09-19）。单独一份产出：`annotations.json` 按
        # (段, 座位) 组织、每行都带企业与座位号，而评审人是**组级**的，没有
        # 段也没有企业。混进去就得给每个下游消费者加一条「这一行 seg_order=0，
        # 特殊对待」，而漏加的那一处会把评审人当成某家企业的创业者统计进去。
        judge_rows = _code_the_judges(config.WORK, config.SHEET,
                                      use_vlm=not no_vlm)
        if judge_rows is not None:
            jp = _write_judge_annotations(config.WORK, config.SHEET, judge_rows)
            n_err = sum(1 for r in judge_rows if r.get("vlm_error"))
            print(f"   装配 {len(judge_rows)} 位评审人 → {jp.name}"
                  + (f"（{n_err} 位调用失败，已记 vlm_error）" if n_err else ""))

    if "8" in stages or "9" in stages:
        _require(ann_path, "7")
        import json as _json
        from pipeline.annotate import PersonAnnotation
        from pipeline.validate import check_all
        # 只有本次调用没有跑阶段 5 时才从磁盘重新加载。阶段 5 结尾把
        # `_evidence_png`（每人那张交给模型的 2×2 证据图字节）从写盘的
        # annotations.json 里弹掉了——二进制绝不能进 JSON（Ruling R13），
        # 但内存里的 `anns` 仍带着它。若在 `--stages 5,6` 这种同一次调用里
        # 无条件重新 `PersonAnnotation(**d)`，就会用没有证据图的磁盘副本
        # 把内存对象整个覆盖掉，导致阶段 6 的 export_keyframes 永远拿不到
        # 证据图（Task 24 缺陷：全量 34 人实测一张没导出）。
        # `--stages 8` 单独运行时（"7" not in stages）本来就没有内存态的
        # anns，也就无证据图可留，必须、也只能从磁盘重载；此时导出的
        # 关键帧目录里不会有 evidence_sent_to_model.png——这是刻意设计，
        # 不是遗漏。想要证据图就把阶段 7、8 放进同一次 --stages 调用
        # （如 --stages 7,8），不要分两次单独跑。
        if "7" not in stages:
            anns = [PersonAnnotation(**d)
                    for d in _json.loads(ann_path.read_text(encoding="utf-8"))]
        violations = check_all(anns, segments)
        for v in violations:
            print(f"   ⚠ {v.rule} order={v.seg_order} {v.person_id} {v.detail}")

    # Task 33（用户追加需求）：阶段 8/9 写盘的全部 CSV 都带组名——三组视频
    # 的产物此前都叫 result.csv/result_wide.csv/project_ranking.csv，下载到
    # 同一个文件夹会互相覆盖、也分不清谁是谁。净化规则与画廊人工校对版
    # 导出文件名、`export_keyframes` 的企业名净化共用同一份
    # `config.sanitize_filename_component`，不另写一份（`safe_sheet` 在阶段 5
    # 之前就算好了，那里也要用）。
    if "8" in stages:
        print("⑧ 生成画廊")
        from pipeline.gallery import render_gallery
        gallery_path = config.WORK / "gallery.html"
        render_gallery(anns, violations, gallery_path)
        print(f"   已写 {gallery_path.name}")
        from pipeline.assemble import write_result_csv, write_wide_csv
        result_csv_path = config.WORK / f"result_{safe_sheet}.csv"
        result_wide_path = config.WORK / f"result_wide_{safe_sheet}.csv"
        write_result_csv(anns, result_csv_path)
        # Task 26 · 需求二：一企业一行的宽表，用户原话「每个企业为一行，
        # 依次输出座位1、座位2…的相关值到不同列」。不改动长表本身的列/行为。
        write_wide_csv(anns, result_wide_path)
        print(f"   已写 {result_csv_path.name}、{result_wide_path.name}")

        # 按企业/座位导出全分辨率关键帧供人工核验（Ruling R13，用户明确要求）。
        # 原始帧目录 work/城市2-4.26/frames/ 一律保留，本步骤只做导出。
        from pipeline.assemble import export_keyframes
        all_dets_for_export = load_detections(det_path)
        export_keyframes(anns, all_dets_for_export, config.WORK / "keyframes")
        companies = {a.company for a in anns}
        print(f"   关键帧已导出：{len(companies)} 个企业，{len(anns)} 人")

    if "9" in stages:
        print("⑨ 项目排名")
        from pipeline.rank import rank_projects
        from pipeline.assemble import write_ranking_csv
        rows = rank_projects(anns, violations)
        ranking_csv_path = config.WORK / f"project_ranking_{safe_sheet}.csv"
        write_ranking_csv(rows, ranking_csv_path)
        print(f"   已写 {ranking_csv_path.name}")
        if rows:
            print(f"   推荐项目：{rows[0]['company']}（{rows[0]['top_reason']}）")


if __name__ == "__main__":
    ns = parse_args(sys.argv[1:])
    run(ns.stages, ns.orders, no_vlm=ns.no_vlm, staff_net=ns.staff_net,
        force_vlm=ns.force_vlm)
