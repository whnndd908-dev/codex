import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline import config
from pipeline import entryreview
from pipeline.detect import Detection
from pipeline.schedule import Segment
from pipeline.entryreview import render_entry_review

W = 1440
H = W * 0.75
RIGHT_CX = 0.55 * W
STANDING_H = 0.55 * H


def _write_frame(tmp_path: Path, name: str = "frame.jpg") -> Path:
    """1440x1080——与本文件里 Detection 的坐标（W×H）同尺度，确保裁剪框落在
    图内（build_evidence_sheet 直接按检测框坐标裁剪，图太小会越界报错）。"""
    p = tmp_path / name
    Image.fromarray(np.full((int(H), W, 3), 80, dtype=np.uint8)).save(p)
    return p


def mkseg(order, presenter_num, company="测试企业"):
    return Segment(order=order, company=company, start=0, end=1000,
                   presenter_num=presenter_num, presenters=())


def det(tid, cx, h, frame_path, kind="entry", order=2, t=0.0, bottom=1000.0, width=120.0):
    half = width / 2.0
    return Detection(order, kind, t, frame_path, tid,
                      cx - half, bottom - h, cx + half, bottom, 0.9)


def track(tid, cx, h, n_frames, frame_path, kind="entry", order=2, bottom=1000.0, width=120.0):
    return [det(tid, cx, h, frame_path, kind=kind, order=order, t=float(i), bottom=bottom, width=width)
            for i in range(n_frames)]


def _extract_script(html: str) -> str:
    m = re.search(r"<script>(.*)</script>", html, re.DOTALL)
    assert m is not None, "未在 HTML 里找到 <script> 区块"
    return m.group(1)


def _segments_blob_from_html(html: str) -> list:
    m = re.search(r"const SEGMENTS=(.*?);\nconst KEY", html, re.DOTALL)
    assert m is not None, "未在 HTML 里找到 SEGMENTS 赋值"
    return json.loads(m.group(1))


def _run_js(html: str, expr: str):
    """在 Node 里跑一遍生成 HTML 里真实的 <script> 源码（最小 shim 顶掉
    document/localStorage，其余一字不改），返回执行 `expr` 并 JSON 序列化
    后的结果。"""
    script = _extract_script(html)
    harness = (
        "const localStorage = { getItem: () => null, setItem: () => {} };\n"
        "const document = { getElementById: () => ({ innerHTML: '', textContent: '' }) };\n"
        + script
        + f"\nconsole.log(JSON.stringify({expr}));\n"
    )
    result = subprocess.run(["node", "-e", harness], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, f"Node 执行失败：{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# 4) 段落一律按 order 升序（Task 52 需求一，取代 Task 39 的「缺人的段排最前」）
#
# 用户 2026-08-23：「既然目录已用颜色区分标注完成状态，段落就按 order 正常
# 排序更好找」。缺人优先那一版的排序键是「标注进度」，而标注进度会随着用户
# 边标边变——同一家企业今天在第 2 位、标完两个候选就跑到第 9 位，用户按位置
# 记住的路线全废。order 是唯一不随标注变化的稳定锚。缺人状态改由目录的颜色
# + ⛔/⚠/✓ 标记承担（那一套本就存在且不受排序影响）。
# ---------------------------------------------------------------------------

def test_segments_rendered_in_order(tmp_path):
    """正文里的段落按 order 升序渲染，**不是**缺人优先。

    断言打在真实渲染出来的正文（跑一遍页面 JS 取 h2 的锚点 id）上，不是
    SEGMENTS 数组——数组排对了而 render() 自己又排一次同样会走样。
    """
    frame = _write_frame(tmp_path)
    # order 1 已满、order 5 缺 2 人、order 9 缺 1 人：缺人优先会排成 5,9,1。
    segs = [mkseg(order=1, presenter_num=1, company="甲已满"),
            mkseg(order=5, presenter_num=3, company="乙缺两人"),
            mkseg(order=9, presenter_num=2, company="丙缺一人")]
    dets = []
    for i, s in enumerate(segs):
        dets += track(10 + i, RIGHT_CX + 200, STANDING_H, 6, frame, order=s.order)
    seats_by_order = {1: {1: 1}, 5: {2: 1}, 9: {3: 1}}

    out = tmp_path / "entry_review.html"
    # 传入顺序刻意打乱：正确的实现按 order 排，不是照抄入参顺序。
    render_entry_review([segs[2], segs[0], segs[1]], dets, out,
                        seats_by_order=seats_by_order)
    html = out.read_text(encoding="utf-8")

    missing = {s["order"]: s["missing"] for s in _segments_blob_from_html(html)}
    assert missing == {1: 0, 5: 2, 9: 1}, (
        f"夹具前提：三段缺口必须各不相同，否则「缺人优先」与「按 order」"
        f"排出来一样、这条测试测不到差异，实得 {missing}"
    )

    main = _run_page(html)["main"]
    rendered = [int(m) for m in re.findall(r'<h2 id="seg-(\d+)"', main)]
    assert rendered == [1, 5, 9], (
        f"段落必须按 order 升序渲染（缺人优先会排成 [5, 9, 1]），实得 {rendered}"
    )


# ---------------------------------------------------------------------------
# 5) 未标注候选：四个单选默认全部不选，座位号下拉默认「不确定」
# ---------------------------------------------------------------------------

def test_unlabeled_candidate_has_no_preselected_role(tmp_path):
    frame = _write_frame(tmp_path)
    seg = mkseg(order=2, presenter_num=3)
    dets = track(5, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    out = tmp_path / "entry_review.html"
    render_entry_review([seg], dets, out)
    html = out.read_text(encoding="utf-8")

    segs = _segments_blob_from_html(html)
    seg_json = segs[0]
    cand_json = seg_json["candidates"][0]
    card_html = _run_js(html, f"candidateCard({json.dumps(seg_json)}, {json.dumps(cand_json)})")

    # 只看**属性位**上的 checked：Task 49 起卡片里有 `onchange="…this.checked)"`
    # 这样的事件代码，它出现在标签内部，裸着找 "checked" 会误判（而且是
    # 「测试恒失败」这种一眼看不出原因的失效）。先把属性值整体挖空再找。
    bare = re.sub(r'="[^"]*"', '=""', card_html)
    assert "checked" not in bare, (
        "未标注的候选，六个角色单选必须全部不选中——预填等于把'没判定'伪装成"
        f"'已判定'。实得：{bare}"
    )
    m = re.search(r'<option value="" (selected)?>不确定</option>', card_html)
    assert m is not None and m.group(1) == "selected", (
        "座位号下拉默认必须选中「不确定」这一项"
    )


# ---------------------------------------------------------------------------
# 6) 座位号下拉选项恰为 1..presenter_num 加「不确定」
# ---------------------------------------------------------------------------

def test_seat_dropdown_offers_only_valid_seats(tmp_path):
    frame = _write_frame(tmp_path)
    seg = mkseg(order=2, presenter_num=4)
    dets = track(5, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    out = tmp_path / "entry_review.html"
    render_entry_review([seg], dets, out)
    html = out.read_text(encoding="utf-8")

    segs = _segments_blob_from_html(html)
    seg_json = segs[0]
    cand_json = seg_json["candidates"][0]
    card_html = _run_js(html, f"candidateCard({json.dumps(seg_json)}, {json.dumps(cand_json)})")

    # 卡片里现在还有一个「人数」下拉（Task 41 改动三），也含 1/2/3 的
    # option——必须只在「座位号」这个 <select> 范围内数选项，不能被人数
    # 下拉的选项混进来。
    m = re.search(r'座位号.*?<select[^>]*>(.*?)</select>', card_html, re.DOTALL)
    assert m is not None, "未找到座位号下拉"
    values = re.findall(r'<option value="([^"]*)"', m.group(1))
    assert values == ["", "1", "2", "3", "4"], (
        f"座位号选项应恰为 1..presenter_num(4) 加「不确定」，不多不少，实得 {values}"
    )


# ---------------------------------------------------------------------------
# 候选按框高排序展示，每段每个窗口最多 config.ENTRYEXIT_TOP_N 个——这是
# 展示上限，不是判据。超出上限时页面必须如实告知截断了多少（凡有截断必须
# 可见，本项目一贯要求）。
# ---------------------------------------------------------------------------

def test_truncation_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENTRYEXIT_TOP_N", 2)
    frame = _write_frame(tmp_path)
    seg = mkseg(order=5, presenter_num=2)
    # 4 条互不合并的 entry 轨迹（cx 间隔远超合并容差），超过 TOP_N=2。
    dets = []
    for i in range(4):
        dets += track(300 + i, RIGHT_CX + 100 + i * 300, STANDING_H, 6, frame, order=seg.order)

    out = tmp_path / "entry_review.html"
    render_entry_review([seg], dets, out)
    html = out.read_text(encoding="utf-8")

    segs = _segments_blob_from_html(html)
    seg_json = segs[0]
    assert seg_json["entry_total"] == 4, "截断前的真实候选总数必须原样带到页面数据里"
    assert seg_json["top_n"] == 2
    assert len(seg_json["candidates"]) == 2, "页面只应展示前 TOP_N 个候选"

    notice_html = _run_js(html, f"truncationNotice({json.dumps(seg_json)})")
    assert "4" in notice_html and "2" in notice_html, (
        f"截断提示必须同时出现真实总数(4)与展示数(2)，实得 {notice_html!r}"
    )
    assert notice_html.strip() != "", "候选被截断时必须显示提示，不能悄悄隐藏"


def test_no_truncation_notice_when_not_truncated(tmp_path):
    """护栏：候选数没超过 TOP_N 时不应显示截断提示（否则用户会被无意义的
    警告淹没，真正的截断反而不显眼）。"""
    frame = _write_frame(tmp_path)
    seg = mkseg(order=5, presenter_num=1)
    dets = track(5, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    out = tmp_path / "entry_review.html"
    render_entry_review([seg], dets, out)
    html = out.read_text(encoding="utf-8")

    segs = _segments_blob_from_html(html)
    seg_json = segs[0]
    notice_html = _run_js(html, f"truncationNotice({json.dumps(seg_json)})")
    assert notice_html.strip() == "", "候选没被截断时不应显示截断提示"


# ---------------------------------------------------------------------------
# 11) 企业名转义
# ---------------------------------------------------------------------------

def test_html_escapes_company_name(tmp_path):
    frame = _write_frame(tmp_path)
    seg = mkseg(order=3, presenter_num=1, company='坏企业<script>alert(1)</script>')
    dets = track(5, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    out = tmp_path / "entry_review.html"
    render_entry_review([seg], dets, out)
    html = out.read_text(encoding="utf-8")

    segs = _segments_blob_from_html(html)
    seg_json = segs[0]
    assert seg_json["company"] == '坏企业<script>alert(1)</script>', (
        "嵌入的 JSON 数据本身应保留原始企业名（转义发生在渲染阶段，不是数据阶段）"
    )

    header_html = _run_js(html, f"segHeader({json.dumps(seg_json)})")
    assert "&lt;script&gt;" in header_html, "企业名含 <script> 时渲染出的标题必须转义"
    assert "<script>alert(1)</script>" not in header_html, (
        "渲染出的标题不能包含未转义的原始 <script> 标签"
    )


# ---------------------------------------------------------------------------
# Task 41 改动一：被长宽比挡掉的候选不静默丢弃，页面末尾用可展开的
# <details> 区块展示，数量必须如实告知（凡有过滤必须可见）。
# ---------------------------------------------------------------------------

def test_filtered_by_aspect_are_disclosed_not_dropped(tmp_path):
    frame = _write_frame(tmp_path)
    seg = mkseg(order=10, presenter_num=1)
    # 一个正常候选（长宽比达标，width=120 走默认值）+ 一个方框（h/w=1.0，
    # 应被判为疑似局部检测/手，不进入正常候选列表）。
    # 方框取 300px 高（>0.15*1080=162）使它**只**被长宽比挡下——原来的 80px
    # 在 Task 52 之后两条规则都会挡，长宽比这一条改坏了也测不出来。
    dets = (track(1, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
            + track(2, RIGHT_CX + 500, 300.0, 6, frame, order=seg.order, width=300.0))
    out = tmp_path / "entry_review.html"
    render_entry_review([seg], dets, out)
    html = out.read_text(encoding="utf-8")

    segs = _segments_blob_from_html(html)
    seg_json = segs[0]
    assert seg_json["filtered_count"] == 1, "被长宽比挡掉的候选数量必须原样带到页面数据里"
    assert all(c["track_id"] != 2 for c in seg_json["candidates"]), (
        "被挡掉的候选不应混进正常候选列表"
    )
    assert any(c["track_id"] == 2 for c in seg_json["filtered"]), (
        "被挡掉的候选本身必须仍能在页面数据里查到，不能被静默丢弃"
    )

    section_html = _run_js(html, f"filteredSection({json.dumps(seg_json)})")
    assert "<details" in section_html, (
        "折叠区必须是可展开的 <details> 元素，不能恒常展示或恒常隐藏"
    )
    assert "1" in section_html, "折叠区提示必须写明被挡掉的数量"

    # 护栏：没有候选被长宽比挡掉时，折叠区不应出现（避免用无意义的空区块
    # 淹没真正的提示）。
    seg2 = mkseg(order=11, presenter_num=1)
    dets2 = track(3, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg2.order)
    out2 = tmp_path / "entry_review2.html"
    render_entry_review([seg2], dets2, out2)
    html2 = out2.read_text(encoding="utf-8")
    seg2_json = _segments_blob_from_html(html2)[0]
    assert seg2_json["filtered_count"] == 0
    empty_section = _run_js(html2, f"filteredSection({json.dumps(seg2_json)})")
    assert empty_section.strip() == "", "没有候选被挡掉时不应渲染折叠区块"


# ---------------------------------------------------------------------------
# Task 41 改动二：角色单选恰为六个，新增「上一组创业者」「下一组创业者」。
# ---------------------------------------------------------------------------

def test_six_role_options_present(tmp_path):
    """六个角色都在，且**相邻段存在时**两个方向角色都渲染得出来。

    夹具必须给出前后两段：2026-08-26 起，指向不存在那一段的方向角色会被藏起来
    （选了也只会记一条「没有第 N 段」的未落位）。只传一段的话两个方向角色都不
    显示，这条会误报——它要验的正是「相邻段在时它们出得来」。
    """
    frame = _write_frame(tmp_path)
    prev = mkseg(order=1, presenter_num=1, company="上一家")
    seg = mkseg(order=2, presenter_num=2)
    nxt = mkseg(order=3, presenter_num=1, company="下一家")
    dets = track(5, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    out = tmp_path / "entry_review.html"
    render_entry_review([prev, seg, nxt], dets, out)
    html = out.read_text(encoding="utf-8")

    roles = _run_js(html, "ROLES")
    assert roles == ['创业者', '工作人员', '专家', '不确定', '上一组创业者', '下一组创业者'], (
        f"角色选项必须恰为六个（含新增的上一组/下一组创业者），实得 {roles}"
    )

    segs = _segments_blob_from_html(html)
    seg_json = next(s for s in segs if s["order"] == 2)
    cand_json = seg_json["candidates"][0]
    card_html = _run_js(html, f"candidateCard({json.dumps(seg_json)}, {json.dumps(cand_json)})")
    # 查**六个都在**，不只查新增那两个。只查方向角色的话，「把非方向角色全过滤
    # 掉」这种错照样能过（变异测试实测存活）——那会让卡片上连「创业者」都没有。
    vals = re.findall(r'<input type="radio"[^>]*value="([^"]*)"', card_html)
    assert vals == roles, (
        f"相邻段都在时，六个角色单选一个不少、顺序与 ROLES 一致，实得 {vals}")


# ---------------------------------------------------------------------------
# Task 41 改动三：人数下拉默认选中「1」——这是真实默认值，不是「未标注」
# 的伪装，与角色/座位号的「默认不选」规则不同。
# ---------------------------------------------------------------------------

def test_people_count_dropdown_defaults_to_one(tmp_path):
    frame = _write_frame(tmp_path)
    seg = mkseg(order=2, presenter_num=2)
    dets = track(5, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    out = tmp_path / "entry_review.html"
    render_entry_review([seg], dets, out)
    html = out.read_text(encoding="utf-8")

    segs = _segments_blob_from_html(html)
    seg_json = segs[0]
    cand_json = seg_json["candidates"][0]
    card_html = _run_js(html, f"candidateCard({json.dumps(seg_json)}, {json.dumps(cand_json)})")

    # 卡片里还有一个「座位号」下拉，其选项也含 value="1"——必须只在「人数」
    # 这个 <select> 范围内断言，不能被座位号下拉的选项混进来。
    pm = re.search(r'人数.*?<select[^>]*>(.*?)</select>', card_html, re.DOTALL)
    assert pm is not None, "未找到人数下拉"
    people_html = pm.group(1)

    m1 = re.search(r'<option value="1" (selected)?>1</option>', people_html)
    assert m1 is not None and m1.group(1) == "selected", "人数下拉默认必须选中「1」"
    m2 = re.search(r'<option value="2" (selected)?>2</option>', people_html)
    assert m2 is not None and m2.group(1) is None, "未标注多人时「2」选项不应被预选"
    m3 = re.search(r'<option value="3" (selected)?>3</option>', people_html)
    assert m3 is not None and m3.group(1) is None, "未标注多人时「3」选项不应被预选"
    assert "multiHint" not in card_html, "人数为默认值 1 时不应显示多人备注提示"


# ---------------------------------------------------------------------------
# Task 41 改动四：进场/退场区块各写一句朝向提示——进场大概率背影、退场
# 大概率正面，措辞必须是「大概率」而非绝对判据。
# ---------------------------------------------------------------------------

def test_orientation_hint_differs_between_entry_and_exit(tmp_path):
    frame = _write_frame(tmp_path)
    seg = mkseg(order=2, presenter_num=1)
    dets = track(5, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    out = tmp_path / "entry_review.html"
    render_entry_review([seg], dets, out)
    html = out.read_text(encoding="utf-8")

    entry_hint = _run_js(html, "orientationHint('entry')")
    exit_hint = _run_js(html, "orientationHint('exit')")

    assert entry_hint != exit_hint, "进场与退场的朝向提示必须不同"
    assert "背影" in entry_hint and "背影" not in exit_hint, (
        "进场提示必须写背影线索，退场提示不能也出现「背影」二字"
    )
    assert "正面" in exit_hint and "正面" not in entry_hint, (
        "退场提示必须写正面线索，进场提示不能也出现「正面」二字"
    )
    assert "大概率" in entry_hint and "大概率" in exit_hint, (
        "措辞必须写成「大概率」——这是概率线索，不是绝对判据"
    )


def test_candidate_image_is_not_capped_to_a_fixed_pixel_width(tmp_path, monkeypatch):
    """候选图不得被写死的像素上限压小（用户 2026-08-17 反馈「图又变得非常小」）。

    原实现 `.cand img{width:calc(var(--img-scale) * 320px)}` 把图限死 320px，
    乘上 0.8 的缩放后只有 256px——而证据图本身是 800×800，等于压到三分之一。
    画廊那边（Task 36）的口径是「原尺寸 × GALLERY_IMG_SCALE」，两处应当一致：
    人是靠这张图认人的，压到 256px 就看不出谁是谁，而这个页面的全部价值
    就在于让用户能一眼认出候选是谁。

    同时 `.noimg` 占位块原为 200×150（非方形），会与方形证据图的版式打架。
    """
    import re
    frame = _write_frame(tmp_path)
    seg = mkseg(order=2, presenter_num=3)
    dets = track(5, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    out = tmp_path / "entry_review.html"
    render_entry_review([seg], dets, out)
    html = out.read_text(encoding="utf-8")
    css = html[:html.index("</style>")] if "</style>" in html else html

    m = re.search(r"\.cand\s+img\{([^}]*)\}", css)
    assert m, "未找到候选图的 CSS 规则"
    rule = m.group(1)
    assert "320px" not in rule, f"候选图仍被限死 320px：{rule}"
    # 应当按百分比缩放原尺寸，与画廊口径一致
    assert "%" in rule, f"应按百分比缩放原尺寸，实为：{rule}"


def test_candidate_image_scale_comes_from_config(tmp_path, monkeypatch):
    """缩放比例必须读 config.GALLERY_IMG_SCALE，不是写死。"""
    from pipeline import config
    monkeypatch.setattr(config, "GALLERY_IMG_SCALE", 0.55)
    frame = _write_frame(tmp_path)
    seg = mkseg(order=2, presenter_num=3)
    dets = track(5, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    out = tmp_path / "entry_review.html"
    render_entry_review([seg], dets, out)
    assert "0.55" in out.read_text(encoding="utf-8"), "改 GALLERY_IMG_SCALE 后页面应跟着变"


def test_export_button_is_sticky(tmp_path):
    """导出按钮须常驻可见，不能只在页尾（用户 2026-08-17 反馈需翻到最后一页）。

    候选按段分组、缺人段在前，整页往往很长；标到一半想先导出一份，得一路
    滚到底。做成悬浮（position:fixed）后随时可导，也降低标了半天丢失的风险
    ——标注只存在 localStorage 里，浏览器一清就没了。
    """
    frame = _write_frame(tmp_path)
    seg = mkseg(order=2, presenter_num=3)
    dets = track(5, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    out = tmp_path / "entry_review.html"
    render_entry_review([seg], dets, out)
    html = out.read_text(encoding="utf-8")
    css = html[:html.index("</style>")] if "</style>" in html else html

    import re
    m = re.search(r"\.(?:export|floatbar|stickybar)[A-Za-z-]*\{([^}]*)\}", css)
    assert m, "未找到导出按钮的样式类"
    assert "fixed" in m.group(1), f"导出按钮应 position:fixed 常驻，实为：{m.group(1)}"


def test_export_bar_shows_progress(tmp_path):
    """悬浮条上要显示已标进度——用户随时能看到还剩多少，不必回顶部。"""
    frame = _write_frame(tmp_path)
    seg = mkseg(order=2, presenter_num=3)
    dets = track(5, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    out = tmp_path / "entry_review.html"
    render_entry_review([seg], dets, out)
    html = out.read_text(encoding="utf-8")
    assert "已标" in html, "悬浮条应显示已标注进度"


# ===========================================================================
# Task 46：导出稳定标识 + 检测指纹；角色/人数/座位号选项改为一行一个
# ===========================================================================

def _run_js_with_store(html: str, store: dict, expr: str):
    """同 `_run_js`，但让 localStorage 里已有一份标注（模拟用户标过之后
    点导出）——默认 shim 的 getItem 恒返回 null，标注为空时导出的自然是
    空对象，测不到「键长什么样」这件事。"""
    script = _extract_script(html)
    harness = (
        "const SAVED = " + json.dumps(json.dumps(store, ensure_ascii=False)) + ";\n"
        "const localStorage = { getItem: () => SAVED, setItem: () => {} };\n"
        "const document = { getElementById: () => ({ innerHTML: '', textContent: '' }) };\n"
        + script
        + f"\nconsole.log(JSON.stringify({expr}));\n"
    )
    result = subprocess.run(["node", "-e", harness], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, f"Node 执行失败：{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def _css_decls(html: str, selector: str) -> str:
    """页面 CSS 里 `selector` 那条规则的声明块。

    先剥掉 `/* ... */` 注释再找——写这条测试时第一版没剥，结果匹配到了
    改动说明注释里引用的旧规则 `.roles label{margin-right:14px}`，测试
    对着注释断言、看起来"改了没生效"。
    """
    css = re.sub(r"/\*.*?\*/", "", html, flags=re.DOTALL)
    m = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert m is not None, f"未在页面 CSS 里找到规则 {selector}"
    return m.group(1)


def _render_one(tmp_path, order=2, presenter_num=2, track_id=5):
    frame = _write_frame(tmp_path)
    seg = mkseg(order=order, presenter_num=presenter_num)
    dets = track(track_id, RIGHT_CX + 200, STANDING_H, 6, frame, order=order)
    out = tmp_path / "entry_review.html"
    render_entry_review([seg], dets, out, detections_fp="FP-TEST-123")
    return out.read_text(encoding="utf-8")


def test_export_writes_stable_ids(tmp_path):
    """导出的 JSON 必须用稳定 id 作键（由视频内容导出），track_id 只作
    参考信息存进值里——重检测后 track_id 全部改变，拿它当键等于让用户
    标的 60 多组随时作废。"""
    html = _render_one(tmp_path)
    segs = _segments_blob_from_html(html)
    cand = segs[0]["candidates"][0]
    sid = cand["stable_id"]
    assert "@" in sid and "#" in sid, f"候选必须带稳定 id，实得 {sid!r}"

    ck = f"{segs[0]['order']}::{sid}"
    payload = _run_js_with_store(
        html, {ck: {"role": "创业者", "seat": "2", "n_people": "1"}},
        "buildLabelsPayload()")

    body = {k: v for k, v in payload.items() if k != "_meta"}
    assert list(body) == ["2"]
    assert list(body["2"]) == [sid], (
        f"导出必须以稳定 id 作键，实得 {list(body['2'])}"
    )
    rec = body["2"][sid]
    assert rec["role"] == "创业者" and rec["seat"] == 2
    assert rec["track_id"] == cand["track_id"], (
        "track_id 要作为人工排查用的参考信息一并写入值里"
    )


def test_export_writes_detections_fingerprint(tmp_path):
    """导出的 `_meta` 里必须带检测指纹，否则下游无从判断这份标注属不属于
    当前这份检测结果（本任务修的就是这个：帧与企业错配、流水线照常跑完
    没有一处报错）。"""
    html = _render_one(tmp_path)
    payload = _run_js_with_store(html, {}, "buildLabelsPayload()")
    meta = payload["_meta"]
    assert meta["detections_fingerprint"] == "FP-TEST-123"
    assert meta["sheet"]
    assert meta["n_candidates"] == 1


def test_role_options_are_one_per_line(tmp_path):
    """六个角色单选改为竖排、每个选项独占一行——图片区已占 62% 宽、
    高度足够，右侧竖排更好读（用户 2026-08-21）。"""
    html = _render_one(tmp_path)
    decls = _css_decls(html, ".roles label")
    assert "display:block" in decls.replace(" ", ""), (
        f"角色选项必须每个独占一行（display:block），实得 {decls!r}"
    )
    assert "margin-right" not in decls, (
        f"横排残留：竖排后不该再靠 margin-right 拉开左右间距，实得 {decls!r}"
    )


def test_seat_and_people_controls_are_one_per_line(tmp_path):
    """「人数」与「座位号」下拉与角色区风格一致：各自独占一行、标签左对齐。"""
    html = _render_one(tmp_path)
    for selector in (".seatWrap label", ".peopleWrap label"):
        decls = _css_decls(html, selector).replace(" ", "")
        assert "display:block" in decls, f"{selector} 必须独占一行，实得 {decls!r}"
        assert "text-align:left" in decls, f"{selector} 的标签必须左对齐，实得 {decls!r}"


def test_option_font_size_stays_readable(tmp_path):
    """守住 Task 36 的成果：选项区字号一律 ≥14px，排版改动不许把字缩小。"""
    html = _render_one(tmp_path)
    for selector in (".roles label", ".seatWrap label", ".seatWrap select",
                     ".peopleWrap label", ".peopleWrap select"):
        decls = _css_decls(html, selector)
        for size in re.findall(r"font-size:\s*(\d+(?:\.\d+)?)px", decls):
            assert float(size) >= 14, (
                f"{selector} 的字号 {size}px 低于 14px（Task 36 的下限）"
            )


# ===========================================================================
# Task 47：企业目录（左侧固定栏）——定位 + 一眼看出哪些企业缺人
#
# 用户 2026-08-21：「是否能给各个企业名加个目录，且带颜色，一点滑到对应的
# 标记处，方便定位和知道有哪些企业缺人」。实测文化4组 14 家里 11 家缺人、
# 共缺 12 人，候选按段分组后整页很长，没有导航很难定位；用户还要标 60+ 组。
# ===========================================================================

def _render_toc_page(tmp_path, specs, seats_by_order=None):
    """按 `specs`（[(order, presenter_num, company, n_cands), …]）渲染一页。

    每段的候选都落在右区（`exclude_judges` 的右侧边界之内，左区是评委席、
    会被整体剔除），横坐标彼此相距 220px——远大于
    `seats.merge_tracks_by_position` 的合并阈值，保证是 n_cands 个不同的
    候选、各有各的稳定标识。
    """
    frame = _write_frame(tmp_path)
    segs, dets = [], []
    tid = 100
    for order, presenter_num, company, n_cands in specs:
        segs.append(mkseg(order=order, presenter_num=presenter_num, company=company))
        for k in range(n_cands):
            tid += 1
            dets += track(tid, RIGHT_CX + 40 + 220 * k, STANDING_H, 6, frame, order=order)
    out = tmp_path / "entry_review.html"
    render_entry_review(segs, dets, out, seats_by_order=seats_by_order or {},
                        detections_fp="FP-TOC")
    return out.read_text(encoding="utf-8")


def _run_page(html: str, store: dict | None = None, after: str = "") -> dict:
    """在 node 里跑一遍页面里**真实的** <script>（含顶层 `render()`），
    返回各挂载点被写入的内容。

    shim 只顶掉 document/localStorage：`querySelectorAll` 返回空数组（node
    里没有 DOM），`IntersectionObserver` **故意不提供**——页面 JS 必须自己
    做特性检测，否则这里就会 ReferenceError、node 退出码非 0。

    `after`：首轮 `render()` 之后再执行的一段 JS（例如
    `clearLabel("5::exit@1#2,3")`）。挂载点内容在它跑完之后才读取，所以
    「点了按钮页面有没有跟着重算」这类断言不必刷新页面就能验（Task 52
    需求二：清除标注后目录计数与变绿判定必须回退）。

    返回值另带两把钥匙：
      - `store`：页面 JS 内存里的那个 store 对象；
      - `written`：最后一次 `localStorage.setItem` 写进去的内容（已解析）。
    两者都要——「内存里删了、写回磁盘的还是旧的」同样是静默失效。
    """
    script = _extract_script(html)
    saved = json.dumps(json.dumps(store or {}, ensure_ascii=False))
    harness = (
        "const SAVED = " + saved + ";\n"
        "let WROTE = null;\n"
        "const els = {};\n"
        "const el = id => (els[id] = els[id] || {innerHTML:'', textContent:'',\n"
        "                   classList:{add(){}, remove(){}}});\n"
        "const localStorage = { getItem: () => SAVED, setItem: (k,v) => { WROTE = v; } };\n"
        "const document = { getElementById: el, querySelectorAll: () => [] };\n"
        + script
        + "\n" + (after or "") + "\n"
        + "\nconsole.log(JSON.stringify({main: el('main').innerHTML,\n"
          "  toc: el('toc-list').innerHTML,\n"
          "  progress: el('progress-bar').textContent,\n"
          "  store: store,\n"
          "  written: WROTE === null ? null : JSON.parse(WROTE)}));\n"
    )
    result = subprocess.run(["node", "-e", harness], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, f"Node 执行失败：{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def _toc_rows(toc_html: str) -> list[dict]:
    """把目录 HTML 拆成一行一条：类名、锚点、纯文本、以及图标记号。

    纯文本按用户实际看到的样子取（剥标签、压空白），避免断言被 span 边界
    绊住；图标记号单独取出（`<span class="toc-mark">…</span>` 的内容），
    这样「除颜色外另有非视觉标记」这条能直接断言，不必从整行文本里猜。
    """
    rows = []
    for m in re.finditer(r"<a\b([^>]*)>(.*?)</a>", toc_html, re.DOTALL):
        attrs, inner = m.group(1), m.group(2)
        cls_m = re.search(r'class="([^"]*)"', attrs)
        href_m = re.search(r'href="([^"]*)"', attrs)
        mark_m = re.search(r'<span class="toc-mark">(.*?)</span>', inner, re.DOTALL)
        rows.append({
            "cls": cls_m.group(1) if cls_m else "",
            "href": href_m.group(1) if href_m else None,
            "mark": mark_m.group(1).strip() if mark_m else "",
            "text": re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", inner)).strip(),
        })
    return rows


def test_toc_lists_every_segment(tmp_path):
    """目录必须列出**每一家**企业，包括不缺人的——不缺人的段也可能混进了
    工作人员，用户要能跳过去纠正；而且目录一旦只列一部分，就没法用它回答
    「一共有哪些企业」。"""
    html = _render_toc_page(
        tmp_path,
        [(1, 1, "甲已满", 1), (5, 3, "乙缺两人", 2), (9, 2, "丙缺一人", 1)],
        seats_by_order={1: {1: 1}, 5: {1: 1}, 9: {1: 1}},
    )
    rows = _toc_rows(_run_page(html)["toc"])
    hrefs = [r["href"] for r in rows]
    assert sorted(hrefs) == ["#seg-1", "#seg-5", "#seg-9"], (
        f"目录须逐段列全，实得 {hrefs}"
    )
    for company in ("甲已满", "乙缺两人", "丙缺一人"):
        assert any(company in r["text"] for r in rows), (
            f"目录漏了企业「{company}」：{[r['text'] for r in rows]}"
        )


def test_toc_rows_in_order(tmp_path):
    """目录行同样按 order 升序（Task 52 需求一），且与正文同序。

    目录与正文各排各的，是「点了跳到别处」这类不报错的缺陷——两侧必须
    共用同一个顺序，这里连正文一起断言。
    """
    html = _render_toc_page(
        tmp_path,
        [(1, 1, "甲已满", 1), (5, 3, "乙缺两人", 2), (9, 2, "丙缺一人", 1)],
        seats_by_order={1: {1: 1}, 5: {1: 1}, 9: {1: 1}},
    )
    page = _run_page(html)
    hrefs = [r["href"] for r in _toc_rows(page["toc"])]
    assert hrefs == ["#seg-1", "#seg-5", "#seg-9"], (
        f"目录必须按 order 升序（缺人优先会排成 5,9,1），实得 {hrefs}"
    )
    body = ["#seg-" + m for m in re.findall(r'<h2 id="seg-(\d+)"', page["main"])]
    assert body == hrefs, f"目录与正文必须同序，目录 {hrefs} 正文 {body}"


def test_toc_shows_missing_count(tmp_path):
    """目录每行显示缺几人，且必须等于「本段人数 − 就座已分配座位数」，
    不是 presenter_num 本身。"""
    html = _render_toc_page(
        tmp_path,
        [(5, 3, "缺两人", 2), (9, 2, "已满", 1)],
        seats_by_order={5: {11: 1}, 9: {21: 1, 22: 2}},
    )
    rows = {r["href"]: r["text"] for r in _toc_rows(_run_page(html)["toc"])}
    assert "需3人" in rows["#seg-5"] and "已定1人" in rows["#seg-5"] \
        and "缺2人" in rows["#seg-5"], rows["#seg-5"]
    assert "需2人" in rows["#seg-9"] and "已定2人" in rows["#seg-9"], rows["#seg-9"]
    assert "缺" not in rows["#seg-9"].replace("不缺人", ""), (
        f"座位已满的段不该显示缺人：{rows['#seg-9']!r}"
    )


def test_toc_anchor_matches_section_id(tmp_path):
    """目录链接的 href 与区块 id 必须**逐字一致**——差一个字，点了不跳、
    浏览器也不报错，用户只会觉得「这个目录时灵时不灵」。"""
    html = _render_toc_page(
        tmp_path, [(3, 2, "甲", 1), (8, 1, "乙", 1)], seats_by_order={})
    out = _run_page(html)
    hrefs = [r["href"] for r in _toc_rows(out["toc"])]
    section_ids = re.findall(r'<h2[^>]*\bid="([^"]+)"', out["main"])
    assert section_ids, f"每段的 <h2> 必须带 id 作为锚点：{out['main'][:400]}"
    assert [h.lstrip("#") for h in hrefs] == section_ids, (
        f"href {hrefs} 与区块 id {section_ids} 不是逐字一致"
    )


def test_toc_anchor_id_is_order_derived_not_company_name(tmp_path):
    """锚点 id 由 order 派生，不能用企业名——真实企业名含全角括号、
    圆点、斜杠，塞进 id 里既不合法也跳不动，而且照样不报错。"""
    company = "华韵（北京）传媒 · A/B #1"
    html = _render_toc_page(tmp_path, [(8, 2, company, 1)], seats_by_order={})
    out = _run_page(html)
    ids = re.findall(r'<h2[^>]*\bid="([^"]+)"', out["main"])
    hrefs = [r["href"] for r in _toc_rows(out["toc"])]
    assert ids == ["seg-8"] and hrefs == ["#seg-8"], (
        f"锚点须为 order 派生的 seg-8，实得 id={ids} href={hrefs}"
    )
    for token in ("华韵", "北京", "传媒"):
        for anchor in ids + hrefs:
            assert token not in anchor, f"锚点 {anchor!r} 里混进了企业名片段 {token!r}"
    assert any("华韵（北京）传媒" in r["text"] for r in _toc_rows(out["toc"])), (
        "企业名仍要作为目录行的可见文字显示（只是不进 id）"
    )


def test_toc_colour_has_nonvisual_marker(tmp_path):
    """三档缺人状态除了颜色，必须各有**非颜色**的区分（图标/文字）。

    只靠颜色区分对色觉障碍者不可用，打印或截图成灰度后也全部失效——而这
    页的用途正是边看边标、看完常要截图存证。
    """
    html = _render_toc_page(
        tmp_path,
        [(1, 1, "已满", 1), (5, 3, "缺两人", 2), (9, 2, "缺一人", 1)],
        seats_by_order={1: {1: 1}, 5: {11: 1}, 9: {21: 1}},
    )
    rows = {r["href"]: r for r in _toc_rows(_run_page(html)["toc"])}
    css_stripped = re.sub(r"/\*.*?\*/", "", html[:html.index("</style>")], flags=re.DOTALL)

    seen_classes, seen_marks = set(), set()
    # seg-1 座位全部确定，文案为「座位已全部确定」——比「不缺人」更准：
    # 不缺人只说明就座阶段够数，全部确定还含人工补位与否决的净结果。
    for href, expect in (("#seg-5", "缺2人"), ("#seg-9", "缺1人"),
                         ("#seg-1", "座位已全部确定")):
        row = rows[href]
        state_cls = [c for c in row["cls"].split() if c != "toc-row"]
        assert len(state_cls) == 1, f"{href} 应恰有一个状态类，实得 {row['cls']!r}"
        cls = state_cls[0]
        seen_classes.add(cls)
        decls = _css_decls(html, "." + cls)
        # 不限定必须用 color:——背景色 + 左边框同样是语义色，在长列表里
        # 比文字色更醒目。真正要守的是「三种状态样式互不相同」+「另有
        # 非视觉标记」，而非具体用哪个 CSS 属性表达。
        assert re.search(r"(color|background|border)\s*:", decls), \
            f".{cls} 应有可区分的视觉样式，实得 {decls!r}"

        assert expect in row["text"], f"{href} 行应有文字标记「{expect}」，实得 {row['text']!r}"
        # 文字之外还要有一个图标记号，且三档互不相同（灰度打印后仍可分辨）
        assert row["mark"], f"{href} 行除颜色外还应有图标记号，实得 {row['text']!r}"
        assert not re.match(r"[0-9A-Za-z\u4e00-\u9fff]", row["mark"]), (
            f"{href} 的记号 {row['mark']!r} 看着像普通文字，应是可辨识的图标"
        )
        seen_marks.add(row["mark"])

    assert len(seen_classes) == 3, f"三档状态必须用三个不同的类，实得 {seen_classes}"
    assert len(seen_marks) == 3, f"三档状态的图标必须互不相同，实得 {seen_marks}"
    # 颜色之外的第三重区分：字重。缺人的行要更粗。
    assert re.search(r"font-weight:\s*[67]00", css_stripped), (
        "缺人的行应同时以字重区分（打印成灰度后颜色全失效）"
    )


def test_toc_distinguishes_seats_assigned_from_labels_made(tmp_path):
    """「就座已定 M 人」（阶段 4 分配的座位数，静态）与「已标 X 个候选」
    （用户在本页标注的进度，来自 localStorage）是两个数，页面上必须分开
    显示。

    混为一谈的后果很实在：用户标了 3 个候选就以为这家企业补齐了，可那
    3 个可能全是工作人员，座位一个都没补上。
    """
    html = _render_toc_page(tmp_path, [(5, 3, "缺两人", 3)], seats_by_order={5: {11: 1}})
    segs = _segments_blob_from_html(html)
    sids = [c["stable_id"] for c in segs[0]["candidates"]]
    assert len(sids) == 3, f"这段应有 3 个候选，实得 {len(sids)}"
    store = {f"5::{sids[0]}": {"role": "创业者", "seat": "2"},
             f"5::{sids[1]}": {"role": "工作人员"}}

    row = _toc_rows(_run_page(html, store)["toc"])[0]["text"]
    assert "已定1人" in row, f"「已定」须来自座位分配（1 人），实得 {row!r}"
    assert "缺2人" in row, f"缺口不随标注进度变（仍缺 2 人），实得 {row!r}"
    assert re.search(r"已标\s*2\s*/\s*3", row), f"「已标」须是本页标注进度 2/3，实得 {row!r}"

    # 一个都没标时，「已定」不受影响——两个字段确实各算各的
    row0 = _toc_rows(_run_page(html, {})["toc"])[0]["text"]
    assert "已定1人" in row0 and re.search(r"已标\s*0\s*/\s*3", row0), row0


def test_toc_does_not_collide_with_export_bar(tmp_path):
    """目录与右下角悬浮导出条都是 position:fixed，必须锚在不同的边、
    互不遮挡；目录还要给正文让出等宽的位置，否则它会盖住候选图。"""
    html = _render_toc_page(tmp_path, [(5, 2, "甲", 1)], seats_by_order={})
    toc = _css_decls(html, ".toc").replace(" ", "")
    bar = _css_decls(html, ".exportbar").replace(" ", "")

    # 只看定位属性本身：`border-right` 之类的简写里也有 "right"，
    # 拿裸子串判断会误伤（也会让断言变成看运气）。
    def anchored(decls, prop):
        # 只看定位属性（right:/left: 后紧跟值），不做裸子串匹配——
        # 原实现会把 border-right 误判成「锚定右边」，那是与定位无关的边框。
        return re.search(r"(?:^|;)" + re.escape(prop), decls) is not None

    assert "position:fixed" in toc and "position:fixed" in bar
    assert anchored(toc, "left:0") and not re.search(r"(?<![a-z-])right\s*:", toc), (
        f"目录须只锚左边（left），不能同时占右边，实得 {toc!r}"
    )
    assert re.search(r"(?<![a-z-])right\s*:", bar) \
        and not re.search(r"(?<![a-z-])left\s*:", bar), (
        f"导出条须只锚右边，实得 {bar!r}"
    )
    m = re.search(r"width:(\d+)px", toc)
    assert m, f"目录必须有确定宽度，否则占多宽算不出来：{toc!r}"
    toc_w = int(m.group(1))

    body_rules = re.findall(r"\bbody\s*\{([^}]*)\}",
                            re.sub(r"/\*.*?\*/", "", html[:html.index("</style>")],
                                   flags=re.DOTALL))
    pads = [int(p) for decls in body_rules
            for p in re.findall(r"padding-left:\s*(\d+)px", decls)]
    assert pads, "正文必须留出 padding-left，否则固定目录会盖住候选图"
    assert max(pads) >= toc_w, f"正文留出的 {max(pads)}px 小于目录宽度 {toc_w}px"


def test_intersection_observer_absence_degrades_silently(tmp_path):
    """滚动高亮用 IntersectionObserver；浏览器不支持时静默降级为不高亮，
    不许抛错——一个 ReferenceError 会打断同一个 <script> 里后面的全部代码
    （包括 render 与导出），页面直接白屏或按钮失灵。"""
    html = _render_toc_page(tmp_path, [(5, 2, "甲", 1)], seats_by_order={})
    script = _extract_script(html)
    use = script.index("new IntersectionObserver")
    guard = re.search(r"typeof\s+IntersectionObserver|['\"]IntersectionObserver['\"]\s*in\s+\w+",
                      script[:use])
    assert guard, "调用 IntersectionObserver 之前必须先做特性检测"
    # 真跑一遍：node 里没有 IntersectionObserver，能跑通才算真降级
    out = _run_page(html)
    assert out["toc"], "降级后目录仍要正常渲染（只是不高亮当前项）"


def test_injected_scripts_pass_node_check(tmp_path):
    """页面里的每个 <script> 都要过 `node --check`（沿用 Task 46 的护栏）。

    Task 45 交付的注入脚本是不含换行的一整行，里面一句 `//` 行注释吃掉了
    后面全部代码，浏览器抛 SyntaxError、导出函数从未被赋值，而页面照常
    渲染、按钮照常可点——「改了但没生效且不报错」，全程无人发现。目录这
    次又往同一个 <script> 里塞了几十行 JS，护栏必须跟上。
    """
    import shutil as _shutil

    node = _shutil.which("node")
    assert node, "本测试需要 node（本仓库其它 JS 测试同样依赖它）"
    pages = [
        ("toc", _render_toc_page(tmp_path, [(5, 2, "甲", 1)], seats_by_order={})),
        # Task 48 又往同一个 <script> 里塞了「已定座位」区块的 JS，护栏跟上
        ("filled", _render_toc_page(tmp_path, [(5, 2, "甲", 1)],
                                    seats_by_order={5: {11: 1}})),
        # Task 49 再塞了座位否决/替代的 JS，护栏继续跟上
        ("task49", _filled_page_with_candidates(tmp_path)),
        # Task 53 又塞了「缺座位号」提示、目录跟随滚动、导出提示条三段 JS
        ("task53", _render_toc_page(tmp_path, [(1, 2, "甲", 2), (2, 1, "乙", 1)],
                                    seats_by_order={})),
        # Task 56C 再塞了「本座位无可用画面」的勾选与目录三态，护栏继续跟上
        ("task56c", _nc_page(tmp_path)),
    ]
    for tag, html in pages:
        blocks = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
        assert blocks, "页面里应至少有一个 <script>"
        for i, body in enumerate(blocks):
            f = tmp_path / f"page_{tag}_{i}.js"
            f.write_text(body, encoding="utf-8")
            r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
            assert r.returncode == 0, (
                f"{tag} 页第 {i} 个 <script> 不是合法 JavaScript：\n{r.stderr}"
            )


# ===========================================================================
# Task 48：每段顶部展示「已定座位」+ 座位号下拉只列尚空的座位
#
# 用户 2026-08-22：「html 是不是可以把已经标定的座位几的图像显示出来，就不用
# 再标了，只要标缺失的座位就好了」。实测文化4组 34 个座位里就座阶段已定 22
# 个、缺 12 个——把已定的那 22 个连人像一起摆出来，用户的判断量少三分之二，
# 而且看着已定者的长相去找缺失者比凭空判断准得多（同一家企业的人往往同时
# 入场、衣着风格相近）。
# ===========================================================================

def _seat_dir(keyframes: Path, order: int, company: str, seat: int) -> Path:
    """关键帧目录一律由 `assemble.person_keyframe_dir` 给出——测试侧也不许
    另拼一份，否则净化规则一改，测试与实现会一起走样却仍然「通过」。"""
    from pipeline.assemble import person_keyframe_dir

    d = person_keyframe_dir(keyframes, order, company, seat)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_jpg(path: Path, colour: int) -> bytes:
    Image.fromarray(np.full((60, 40, 3), colour, dtype=np.uint8)).save(path, quality=95)
    return path.read_bytes()


def _render_filled_page(tmp_path, seg, dets, seats_by_order, keyframes_dir=None,
                        segs=None):
    # keyframes_dir 只为老调用点保留形参，已不再传给 render_entry_review
    # （2026-09-20 已定座位卡片去掉了那张就座单图）。
    out = tmp_path / "entry_review.html"
    render_entry_review(segs or [seg], dets, out, seats_by_order=seats_by_order,
                        detections_fp="FP-48")
    return out.read_text(encoding="utf-8")


def _filled_cards(main_html: str) -> list[dict]:
    """从渲染出的正文里取出「已定座位」卡片：座位号、图片来源、纯文本。

    用 data- 属性定位而不是靠中文文案切片——卡片里嵌着 <img>/<div>，正则切
    嵌套标签必然出错，而出错的表现是「断言恒真」这类看不出来的失效。
    """
    # 按卡片起始标签切，不用「嵌套 div 数到第几层」的正则：卡片的层数会随
    # 布局改动而变（2026-09-19 改成与候选同尺寸时就多了一层 imgwrap），而
    # 数错层的表现是这个函数返回空列表、依赖它的断言全部恒真。
    OPEN = '<div class="cand filledSeat"'
    cards = []
    chunks = main_html.split(OPEN)[1:]
    for chunk in chunks:
        head, _, rest = chunk.partition(">")
        body = rest.split(OPEN)[0]
        # 到本段「已定座位」区块结束为止（下一个 h3/h2 或该 box 的收尾）
        for stop in ('<h3 class="kindTitle"', '</div></div>'):
            if stop in body:
                body = body[:body.index(stop)]
        seat_m = re.search(r'data-seat="(\d+)"', head)
        src_m = re.search(r'data-imgsrc="([^"]*)"', head)
        img_m = re.search(r'<img[^>]*src="([^"]*)"', body)
        cards.append({
            "seat": int(seat_m.group(1)) if seat_m else None,
            "imgsrc": src_m.group(1) if src_m else "",
            "img": img_m.group(1) if img_m else "",
            "text": re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)).strip(),
        })
    return cards


def _vacancy_line(main_html: str) -> str | None:
    """「还缺：…」那句话，没有则 None。

    2026-09-20 修：原先找的是 `<div class="vacancy">`，而缺口早已改成写在
    「本段已定座位」那一行里，页面上根本没有这个 class——于是它**恒返回
    None**，依赖它的两条断言（一条要求非 None、一条要求 None）一条恒假、
    一条恒真。恒假那条藏在一个被同名函数遮蔽的测试里，所以一直没人发现。
    """
    m = re.search(r"还缺：(.*?)(?:<div|</div>)", main_html, re.DOTALL)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m.group(1))).strip() if m else None


def _seat_option_values(card_html: str) -> list[str]:
    """只在「座位号」这个 <select> 范围内取选项值——卡片里还有一个「人数」
    下拉，它也含 1/2/3 的 option（沿用 test_seat_dropdown_offers_only_valid_seats
    的取法）。"""
    m = re.search(r'座位号.*?<select[^>]*>(.*?)</select>', card_html, re.DOTALL)
    assert m is not None, "未找到座位号下拉"
    return re.findall(r'<option value="([^"]*)"', m.group(1))


def _card_html_for(html: str, seg_idx: int = 0, cand_idx: int = 0) -> str:
    segs = _segments_blob_from_html(html)
    seg_json = segs[seg_idx]
    cand_json = seg_json["candidates"][cand_idx]
    return _run_js(html, f"candidateCard({json.dumps(seg_json)}, {json.dumps(cand_json)})")


def test_filled_seats_section_lists_each_seat_and_comes_first(tmp_path):
    """就座阶段已定的座位逐个列出（座位号正确），且排在候选之前——用户要
    先看「已经是谁」，再去判断「还缺的那几个是谁」。

    2026-09-20：这份测试此前叫 `..._lists_each_occupied_seat`，与 1190 行那份
    **同名**，Python 后定义的覆盖前面的，所以它从来没有运行过；里面的断言
    也停在改版之前（「已定，无需标注」这句文案早就不是这么写的了）。改成
    唯一的名字并修好断言——它比遮蔽它的那份多测两件事（页面上的座位号、
    「已定座位」区块排在候选之前），那份因此删掉。
    """
    frame = _write_frame(tmp_path)
    seg = mkseg(order=3, presenter_num=4, company="甲企业（北京）")
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    kf = tmp_path / "keyframes"
    for seat in (1, 4):
        _write_jpg(_seat_dir(kf, seg.order, seg.company, seat)
                   / f"crop_seated_01_t{seat}.0.jpg", 60 + seat * 20)
    html = _render_filled_page(tmp_path, seg, dets,
                               {seg.order: {11: 1, 44: 4}}, kf)

    blob = _segments_blob_from_html(html)[0]
    assert [f["seat"] for f in blob["filled_seats"]] == [1, 4], (
        f"已定座位应为 1、4，实得 {[f['seat'] for f in blob['filled_seats']]}"
    )

    main = _run_page(html)["main"]
    cards = _filled_cards(main)
    assert [c["seat"] for c in cards] == [1, 4], (
        f"页面上应逐个列出已定座位 1、4，实得 {[c['seat'] for c in cards]}"
    )
    for c in cards:
        assert "已定" in c["text"], (
            f"座位 {c['seat']} 必须写明已定、无需标注，实得文案：{c['text']}"
        )
    i_filled = main.index("本段已定座位")
    i_cands = main.index("进场候选")
    assert i_filled < i_cands, "「已定座位」区块必须排在候选列表之前"




def test_missing_seat_numbers_are_listed_on_the_page(tmp_path):
    """明示还缺哪几个座位号，且与 seats_by_order 一致（不是把 1..N 全列一遍）。

    2026-09-20：这份与下文同名，被后定义的那份遮蔽，从未运行过。改成唯一
    名字后它才真正生效；被它遮蔽的那份只断言 `vacant_seats`，是这份的真
    子集（连下拉都没看），已删除。
    """
    frame = _write_frame(tmp_path)
    seg = mkseg(order=4, presenter_num=4, company="丁企业")
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    kf = tmp_path / "keyframes"
    for seat in (1, 4):
        _write_jpg(_seat_dir(kf, seg.order, seg.company, seat)
                   / "crop_seated_01_t1.0.jpg", 70 + seat)

    html = _render_filled_page(tmp_path, seg, dets,
                               {seg.order: {11: 1, 44: 4}}, kf)
    blob = _segments_blob_from_html(html)[0]
    assert blob["vacant_seats"] == [2, 3], (
        f"还缺的应恰为座位 2、3，实得 {blob['vacant_seats']}"
    )

    line = _vacancy_line(_run_page(html)["main"])
    assert line is not None, "缺座位时必须在页面上明示还缺哪几个"
    assert "2" in line and "3" in line, f"缺口文案应点名座位 2、3，实得：{line}"
    assert "座位 1" not in line and "座位 4" not in line, (
        f"已定的座位不许出现在「还缺」里，实得：{line}"
    )


def test_seat_dropdown_lists_only_vacant_seat_numbers(tmp_path):
    """座位号下拉只列尚空的座位号 +「不确定」，不列已被就座阶段占用的。

    这条直接消掉一类无声错误：用户把候选指到一个已占座位上，
    `entryexit_fill_plan` 会跳过它（不覆盖已占座位），于是标了没生效、
    且没有任何提示。

    2026-09-20：这份与下文同名，被后定义的那份遮蔽，从未运行过。改成唯一
    名字后它才真正生效；被它遮蔽的那份只断言 `vacant_seats`，是这份的真
    子集（连下拉都没看），已删除。
    """
    frame = _write_frame(tmp_path)
    seg = mkseg(order=4, presenter_num=4, company="戊企业")
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    kf = tmp_path / "keyframes"
    html = _render_filled_page(tmp_path, seg, dets,
                               {seg.order: {11: 1, 44: 4}}, kf)

    values = _seat_option_values(_card_html_for(html))
    assert values == ["", "2", "3"], (
        f"座位号选项应恰为尚空的 2、3 加「不确定」，实得 {values}"
    )


def test_seat_dropdown_still_offers_uncertain(tmp_path):
    """「不确定」始终在选项里——用户可能判断不出这个人坐哪个座位，
    没有这一项就只能被迫瞎指一个。"""
    frame = _write_frame(tmp_path)
    seg = mkseg(order=4, presenter_num=2, company="己企业")
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    kf = tmp_path / "keyframes"
    for seats in ({}, {seg.order: {11: 1}}, {seg.order: {11: 1, 22: 2}}):
        html = _render_filled_page(tmp_path, seg, dets, seats, kf)
        values = _seat_option_values(_card_html_for(html))
        assert values[0] == "", (
            f"「不确定」必须始终是第一个选项，seats={seats} 时实得 {values}"
        )
        labels = re.findall(r'<option value="[^"]*"[^>]*>([^<]*)</option>',
                            re.search(r'座位号.*?<select[^>]*>(.*?)</select>',
                                      _card_html_for(html), re.DOTALL).group(1))
        assert labels[0].strip() == "不确定", f"实得选项文案 {labels}"


def test_all_seats_filled_shows_no_vacancy_prompt(tmp_path):
    """全部已定的段：不显示「还缺」，座位号下拉只剩「不确定」。"""
    frame = _write_frame(tmp_path)
    seg = mkseg(order=9, presenter_num=2, company="庚企业")
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    kf = tmp_path / "keyframes"
    for seat in (1, 2):
        _write_jpg(_seat_dir(kf, seg.order, seg.company, seat)
                   / "crop_seated_01_t1.0.jpg", 40 + seat * 30)

    html = _render_filled_page(tmp_path, seg, dets,
                               {seg.order: {11: 1, 22: 2}}, kf)
    blob = _segments_blob_from_html(html)[0]
    assert blob["vacant_seats"] == []

    main = _run_page(html)["main"]
    assert _vacancy_line(main) is None, "座位全满时不该显示「还缺」"
    assert "还缺：" not in main, "座位全满时不该出现「还缺：」字样"
    assert len(_filled_cards(main)) == 2, "两个座位都要照常展示出来"
    assert _seat_option_values(_card_html_for(html)) == [""], (
        "没有空位时座位号下拉只剩「不确定」"
    )



def test_filled_seat_images_share_the_image_folder(tmp_path, monkeypatch):
    """已定座位图与候选图走**同一条**出图路径，不另起一套。

    原先这条测的是「同一份体积预算」；图改成写文件之后预算没有了（用户
    2026-09-18），但它守的那件事没变：两类图必须一起交给
    `_write_review_images`。分两套必然日后走样，而走样的表现是某一类图静默
    变成叉。
    """
    import pipeline.entryreview as er

    real = er._write_review_images
    seen: list[list] = []

    def spy(out_path, refs, cands):
        seen.append([c.get("data") for c in cands])
        return real(out_path, refs, cands)

    monkeypatch.setattr(er, "_write_review_images", spy)

    frame = _write_frame(tmp_path)
    seg = mkseg(order=2, presenter_num=2, company="壬企业")
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    kf = tmp_path / "keyframes"
    _write_jpg(_seat_dir(kf, seg.order, seg.company, 1)
               / "crop_seated_01_t1.0.jpg", 170)
    # 座位 1 那一位的就座帧——已定座位的 2×2 由它们拼成
    dets = dets + track(11, RIGHT_CX, STANDING_H, 6, frame, kind="seated",
                        order=seg.order)

    _render_filled_page(tmp_path, seg, dets, {seg.order: {11: 1}})

    assert len(seen) == 1, (
        f"出图只能有一条路径（_write_review_images 恰调用一次），实得 {len(seen)} 次"
    )
    datas = seen[0]
    assert sum(1 for d in datas if d) >= 2, (
        "已定座位的 2×2 与候选图必须在同一批里交出去（否则这条恒真）")
    names = {f.name for f in (tmp_path / er.REVIEW_IMG_DIR).iterdir()}
    assert len(names) >= 2, names


# --- 已定座位展示（用户 2026-08-23：只标缺失的座位）---

def _seg_with_keyframes(tmp_path, order, pnum, company, seats, with_crop=True):
    """造一段，并按 person_keyframe_dir 的口径铺好关键帧。"""
    from pipeline.assemble import person_keyframe_dir
    kf = tmp_path / "keyframes"
    for seat in seats.values():
        d = person_keyframe_dir(kf, order, company, seat)
        d.mkdir(parents=True, exist_ok=True)
        _write_frame(d, name=(f"crop_seated_01_t100.jpg" if with_crop
                              else "context_full_frame.jpg"))
    return kf





def test_all_seats_filled_leaves_no_vacancy(tmp_path):
    """全部已定的段：没有空座位号。"""
    company = "乙公司"
    seats = {21: 1, 22: 2}
    kf = _seg_with_keyframes(tmp_path, 6, 2, company, seats)
    frame = _write_frame(tmp_path)
    seg = mkseg(order=6, presenter_num=2, company=company)
    dets = track(9, RIGHT_CX + 200, STANDING_H, 6, frame, order=6)
    out = tmp_path / "entry_review.html"
    render_entry_review([seg], dets, out, seats_by_order={6: seats},)
    blob = _segments_blob_from_html(out.read_text(encoding="utf-8"))[0]
    assert blob["vacant_seats"] == []
    assert len(blob["filled_seats"]) == 2



# ===========================================================================
# Task 49（用户 2026-08-23 三条需求）
#   ① 2×2 证据图给每格编号，页面写明格位约定，备注按格位号写；
#   ② 已定座位可人工否决 / 指定替代候选；
#   ③ 组级「本组该位置恒为工作人员/专家」的位置排除。
# ===========================================================================

def _quadrant_of_colour(arr, colour):
    """颜色 `colour` 落在 2×2 拼图的哪个象限，返回 (col, row)。"""
    h, w = arr.shape[:2]
    ch, cw = h // 2, w // 2
    hits = []
    for col in (0, 1):
        for row in (0, 1):
            px = arr[row * ch + ch // 2, col * cw + cw // 2]
            if tuple(int(v) for v in px) == colour:
                hits.append((col, row))
    assert len(hits) == 1, f"颜色 {colour} 应恰好落在一个象限，实得 {hits}"
    return hits[0]


def _badge_mask_at(arr, col, row):
    h, w = arr.shape[:2]
    ch, cw = h // 2, w // 2
    patch = arr[row * ch: row * ch + 100, col * cw: col * cw + 100]
    return np.all(patch > 240, axis=-1)


def test_cell_order_matches_documented_convention(tmp_path):
    """页面写的格位约定必须与 `build_evidence_sheet` **实际**的绘制顺序
    一致。

    这条测试不看代码注释、也不照抄常量：先用四张纯色图实测出「第 i 张图
    落在哪个象限」，再用同一个字形绘制函数把 1/2/3/4 各画一遍作参照、认出
    每个象限徽标上是几号，最后拿页面上那句约定去核对。约定与实际反过来
    （变异 b），用户会按错误的编号写备注、重新切分时张冠李戴。
    """
    import io as _io

    from PIL import Image as _Image, ImageDraw as _ImageDraw

    from pipeline.annotate import build_evidence_sheet, _draw_cell_label

    colours = [(10, 120, 200), (20, 200, 60), (200, 180, 10), (150, 20, 190)]
    paths = []
    for i, c in enumerate(colours):
        p = tmp_path / f"cell{i}.png"
        _Image.fromarray(np.full((200, 200, 3), c, dtype=np.uint8)).save(p)
        paths.append(p)
    png = build_evidence_sheet(paths, [(10, 10, 190, 190)] * 4, label_cells=True)
    arr = np.asarray(_Image.open(_io.BytesIO(png)).convert("RGB"))

    # 参照字形：同一个绘制函数，画在与拼图相同的背景上
    refs = {}
    for n in range(1, 5):
        im = _Image.new("RGB", (100, 100), (32, 32, 32))
        _draw_cell_label(_ImageDraw.Draw(im), 0, 0, n)
        refs[n] = np.all(np.asarray(im) > 240, axis=-1)

    names = {(0, 0): "左上", (1, 0): "右上", (0, 1): "左下", (1, 1): "右下"}
    measured = {}
    for i, colour in enumerate(colours):
        col, row = _quadrant_of_colour(arr, colour)
        mask = _badge_mask_at(arr, col, row)
        digits = [n for n, ref in refs.items() if np.array_equal(mask, ref)]
        assert len(digits) == 1, (
            f"{names[(col, row)]} 格的徽标认不出是哪个数字（匹配到 {digits}）"
        )
        assert digits[0] == i + 1, (
            f"第 {i + 1} 张图落在{names[(col, row)]}格，但那格画的编号是 {digits[0]}"
        )
        measured[names[(col, row)]] = digits[0]

    html = _render_one(tmp_path)
    main = _run_page(html)["main"]
    stated = {}
    for name in ("左上", "右上", "左下", "右下"):
        m = re.search(name + r"\s*=\s*(\d)", main)
        assert m, f"候选卡片上必须写明格位约定，缺「{name}=?」：{main[:600]}"
        stated[name] = int(m.group(1))
    assert stated == measured, (
        f"页面写的格位约定 {stated} 与实际绘制顺序 {measured} 不一致"
    )


def test_note_placeholder_tells_user_to_use_cell_numbers(tmp_path):
    """备注输入框要提示按格位号写——不提示，用户只会写「左边那个是创业者」，
    重新切分时无从对应。"""
    html = _render_one(tmp_path)
    card = _run_js_with_store(
        html,
        {f"2::{_segments_blob_from_html(html)[0]['candidates'][0]['stable_id']}":
            {"role": "创业者", "n_people": "2"}},
        "candidateCard(SEGMENTS[0], SEGMENTS[0].candidates[0])")
    m = re.search(r'<textarea[^>]*placeholder="([^"]*)"', card)
    assert m, f"多人候选的备注框应有 placeholder：{card}"
    assert "号格" in m.group(1), (
        f"placeholder 要提示按格位号写（例：1号格是创业者），实得 {m.group(1)!r}"
    )


def _filled_page_with_candidates(tmp_path, seats_by_order=None):
    """一段、两个候选、座位 1 已由就座阶段判定——用于座位否决/替代的测试。"""
    frame = _write_frame(tmp_path)
    seg = mkseg(order=5, presenter_num=2)
    dets = (track(11, RIGHT_CX + 100, STANDING_H, 6, frame, order=5)
            + track(12, RIGHT_CX + 400, STANDING_H, 6, frame, order=5))
    out = tmp_path / "entry_review.html"
    render_entry_review([seg], dets, out,
                        seats_by_order=seats_by_order or {5: {99: 1}},
                        detections_fp="FP-49")
    return out.read_text(encoding="utf-8")


def _seat_override_key(section_html: str) -> str:
    m = re.search(r"save\('([^']*)','rejected'", section_html)
    assert m, f"已定座位上必须有「是否替换」的复选框：{section_html}"
    return m.group(1)


def test_filled_seat_offers_the_replace_switch(tmp_path):
    """每个已定座位都要能勾「是否替换」，把它当作空缺交给下面的候选去补。

    用户原话（2026-08-23）：「已标注的座位可能是错的…或者画面没有我看到的
    清楚，舍弃掉用我指定的画面」。

    这条只钉**开关接到了 `rejected` 字段**。旁边那个「改用我指定的候选」
    下拉 2026-08-24 已删——用户：「哪个选项也是冗余的，根据我标注的进出场
    等进行选择即可」。同一件事在两个地方各说一遍，两处不一致时还得再定一套
    优先级；冗余本身就是缺陷来源。勾选项那句文案由
    test_seat_override_checkbox_is_labelled_replace 单独钉，两处各钉一半。
    """
    html = _filled_page_with_candidates(tmp_path)
    section = _run_js(html, "filledSeatsSection(SEGMENTS[0])")
    key = _seat_override_key(section)
    assert key.startswith("seatovr::"), (
        f"键要用 (order, seat) 那把——座位号跨重检测稳定，track_id 不是："
        f"{key}")
    assert "replacement_stable_id" not in section, (
        f"替换下拉已删，不该还往那个字段里写：{section}")


def test_export_writes_seat_overrides(tmp_path):
    """座位否决/替代要写进导出的 JSON（`_meta` 同级的 seat_overrides）。"""
    html = _filled_page_with_candidates(tmp_path)
    section = _run_js(html, "filledSeatsSection(SEGMENTS[0])")
    key = _seat_override_key(section)
    sid = _segments_blob_from_html(html)[0]["candidates"][1]["stable_id"]

    payload = _run_js_with_store(
        html, {key: {"rejected": True, "replacement_stable_id": sid,
                     "note": "就座那张糊了"}},
        "buildLabelsPayload()")
    assert payload["seat_overrides"] == {
        "5": {"1": {"rejected": True, "replacement_stable_id": sid,
                    "note": "就座那张糊了"}}
    }, payload.get("seat_overrides")


def test_preloaded_seat_override_key_matches_the_page(tmp_path):
    """阶段 6 预填座位否决时用的 store 键，必须与页面 `seatOvrKey()` 拼出来
    的**逐字一致**；`seat_overrides`/`position_exclusions` 这两个顶层键不得
    被当成段号展开。

    对不上时的表现是「预填了但页面上看不出来」——用户会以为自己的否决没
    保存，再点一遍；展开成段号的表现是页面里多出一堆对不上任何候选的条目，
    同样不报错。
    """
    from pipeline.run_all import _entry_labels_to_store

    html = _filled_page_with_candidates(tmp_path)
    section = _run_js(html, "filledSeatsSection(SEGMENTS[0])")
    page_key = _seat_override_key(section)
    sid = _segments_blob_from_html(html)[0]["candidates"][0]["stable_id"]

    store = _entry_labels_to_store({
        "5": {sid: {"role": "创业者", "seat": 2, "n_people": 1, "note": ""}},
        "seat_overrides": {"5": {"1": {"rejected": True,
                                        "replacement_stable_id": sid,
                                        "note": "用退场那张"}}},
        "position_exclusions": [{"cx": 1204, "y2": 1079,
                                  "role": "工作人员", "note": ""}],
    })
    assert page_key in store, (
        f"预填的座位否决键 {sorted(store)} 里没有页面用的 {page_key!r}"
    )
    assert store[page_key]["rejected"] is True
    assert store[page_key]["replacement_stable_id"] == sid
    assert set(store) == {f"5::{sid}", page_key}, (
        f"顶层的 seat_overrides/position_exclusions 不得被当成段号展开，实得 {sorted(store)}"
    )


# --- 全部确定变绿 + 座位性别提示（用户 2026-08-23）---

def _seg_with_genders(order, pnum, genders, company="某公司"):
    """genders: [0/1, ...] 按座位顺序，1=女。"""
    from pipeline.schedule import Presenter
    ps = tuple(Presenter(i + 1, g, 1 if i == 0 else 0, 0)
               for i, g in enumerate(genders))
    return Segment(order=order, company=company, start=0, end=1000,
                   presenter_num=pnum, presenters=ps)


def test_toc_shows_gender_per_seat(tmp_path):
    """目录里逐座位标出人工标注的性别，缺的座位标「应为X」。

    用户标注时据此能直接排掉一半候选——知道缺的那位是男是女，候选池立刻减半。
    """
    frame = _write_frame(tmp_path)
    seg = _seg_with_genders(5, 3, [0, 1, 0])      # 男 女 男
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=5)
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={5: {11: 1, 12: 3}})
    blob = _segments_blob_from_html(out.read_text(encoding="utf-8"))[0]
    gm = blob.get("seat_genders")
    assert gm, "应给出各座位的人工标注性别"
    assert gm.get("1") == "男" and gm.get("2") == "女" and gm.get("3") == "男"


def test_gender_is_labelled_as_human_annotation(tmp_path):
    """页面必须写明性别来自人工标注，不是模型判定。

    本项目吃过亏：PROMPT 里把性别当「已知事实」喂给 VLM 后，它的
    gender_judged 失去独立性，据此做的一批校验结论全部作废。人工标注与
    模型判定在页面上混起来，会让用户重蹈同一类误判。
    """
    frame = _write_frame(tmp_path)
    seg = _seg_with_genders(5, 2, [0, 1])
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=5)
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={5: {11: 1}})
    html = out.read_text(encoding="utf-8")
    assert "人工标注" in html


def test_no_presenters_degrades_gracefully(tmp_path):
    """presenters 为空时不显示性别，也不报错。"""
    frame = _write_frame(tmp_path)
    seg = mkseg(order=6, presenter_num=2)          # presenters=()
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=6)
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={6: {11: 1}})
    blob = _segments_blob_from_html(out.read_text(encoding="utf-8"))[0]
    assert blob.get("seat_genders") == {}


def test_segment_turns_green_when_all_seats_determined(tmp_path):
    """座位全部确定时目录行变绿，且绿色之外另有 ✓ 标记。"""
    frame = _write_frame(tmp_path)
    seg = _seg_with_genders(5, 2, [0, 1])
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=5)
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={5: {11: 1, 12: 2}})
    rows = _toc_rows(_run_page(out.read_text(encoding="utf-8"))["toc"])
    assert "toc-done-all" in rows[0]["cls"], f"应变绿，实得 {rows[0]['cls']!r}"
    assert "✓" in rows[0]["text"], "绿色之外必须另有非视觉标记"


def test_candidate_without_seat_number_does_not_count(tmp_path):
    """标了创业者但没指定座位号的不计入「已确定」。

    用户真实标注里就有这种（order9 标了创业者、seat 留空）——它进的是
    needs_seat、根本补不进去。若算作已确定，用户会以为标完了而那人丢了。
    """
    frame = _write_frame(tmp_path)
    seg = _seg_with_genders(5, 2, [0, 1])
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=5)
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={5: {11: 1}})
    html = out.read_text(encoding="utf-8")
    blob = _segments_blob_from_html(html)[0]
    ck = f"{blob['order']}::{blob['candidates'][0]['stable_id']}"
    store = {ck: {"role": "创业者"}}          # 有角色、无 seat
    rows = _toc_rows(_run_page(html, store)["toc"])
    assert "toc-done-all" not in rows[0]["cls"], (
        f"未指定座位号不应算已确定，实得 {rows[0]['cls']!r}")


def test_green_updates_without_reload(tmp_path):
    """勾选后重算即变绿，不需刷新页面。

    只断言初次渲染是不够的——变绿若只在首次计算，用户标完不会有任何反馈。
    这里跑两次真实 JS：空 store 不绿，补上座位号后变绿。
    """
    frame = _write_frame(tmp_path)
    seg = _seg_with_genders(5, 2, [0, 1])
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=5)
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={5: {11: 1}})
    html = out.read_text(encoding="utf-8")
    blob = _segments_blob_from_html(html)[0]
    ck = f"{blob['order']}::{blob['candidates'][0]['stable_id']}"

    before = _toc_rows(_run_page(html)["toc"])[0]["cls"]
    after = _toc_rows(_run_page(html, {ck: {"role": "创业者", "seat": "2"}})["toc"])[0]["cls"]
    assert "toc-done-all" not in before
    assert "toc-done-all" in after, f"补齐座位号后应变绿，实得 {after!r}"


def test_duplicate_seat_number_counts_once_really(tmp_path):
    """同一座位号被两个候选各标一次，只能算一个人。

    补于变异测试：把去重的 Set 换成允许重复的容器后，原测试仍全绿——
    「只算一次」这条只有实现、没有测试真正验到。用户在进场和退场各标同
    一个人是常态（同一人当然在两个窗口都出现），若重复计数，一个人会把
    两个座位都"填满"，页面提前变绿而实际还缺人。
    """
    frame = _write_frame(tmp_path)
    # 需 3 人：座位1 已定 + 两个候选都指座位2 → 只算 2 个座位，仍缺 1 人。
    # 若写成需 2 人，{1,2} 本就够数，重复计数与否都会变绿，测不到差异。
    seg = _seg_with_genders(5, 3, [0, 1, 0])
    dets = (track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=5)
            + track(8, RIGHT_CX + 260, STANDING_H, 6, frame, order=5, kind="exit"))
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={5: {11: 1}})
    html = out.read_text(encoding="utf-8")
    blob = _segments_blob_from_html(html)[0]
    assert len(blob["candidates"]) >= 2, "夹具需两个候选才测得到重复计数"
    store = {f"{blob['order']}::{c['stable_id']}": {"role": "创业者", "seat": "2"}
             for c in blob["candidates"][:2]}          # 两个候选都指向座位 2
    rows = _toc_rows(_run_page(html, store)["toc"])
    assert "toc-done-all" not in rows[0]["cls"], (
        f"两个候选都指座位2，只应算 1 人，仍缺 1 人不该变绿：{rows[0]['cls']!r}")


def test_rejected_seat_reduces_the_count_really(tmp_path):
    """被人工否决的已定座位要从「已确定」里减掉。

    补于变异测试：把 rejectedSeatsOf 直接 return 空后仍全绿。用户说了
    「这个座位判错了」，若它仍算已确定，页面会显示已满、用户不再去补，
    而那个座位实际是空的。
    """
    frame = _write_frame(tmp_path)
    seg = _seg_with_genders(5, 2, [0, 1])
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=5)
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={5: {11: 1, 12: 2}})
    html = out.read_text(encoding="utf-8")
    blob = _segments_blob_from_html(html)[0]

    before = _toc_rows(_run_page(html)["toc"])[0]["cls"]
    assert "toc-done-all" in before, "两座位都已定，初始应为绿"

    ovr = {f"seatovr::{blob['order']}::1": {"rejected": True}}
    after = _toc_rows(_run_page(html, ovr)["toc"])[0]["cls"]
    assert "toc-done-all" not in after, (
        f"否决座位1后应不再是绿（还缺 1 人），实得 {after!r}")


def test_vacant_seat_shows_expected_gender_in_section(tmp_path):
    """已定座位栏里的「还缺」要标出应有性别。

    补于变异测试：去掉「（应为X）」后仍全绿——这条提示只有实现、没有
    测试验到。它是用户标注时最直接的减半线索。
    """
    frame = _write_frame(tmp_path)
    seg = _seg_with_genders(5, 3, [0, 1, 0])       # 座位2 应为女
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=5)
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={5: {11: 1, 12: 3}})
    main = _run_page(out.read_text(encoding="utf-8"))["main"]
    assert "座位 2（应为女）" in main, f"缺失座位应标应有性别，实得片段：{main[:300]}"


# ===========================================================================
# Task 52 · 需求二：取消已做的标注
#
# 用户 2026-08-23：「如果标了创业者之后，目前无法取消标注了，如果下面有更好
# 的图，希望取消先看到的图标注，给个取消标注的选择」。
#
# 六个角色是单选（radio），选中之后 HTML 没有任何办法回到「全部未选」的初始
# 态。清除必须把该候选在 store 里的记录**整条删掉**，不是置空字符串——空串
# 与「没标过」在导出与统计里是两回事（`buildLabelsPayload` 用 `!rec.role`
# 判断跳过，一个 role:'' 的记录看起来是「标过、角色为空」）。
# ===========================================================================

def _clear_fixture(tmp_path, presenter_num=2, seats=None):
    """一段两座位、座位 1 已由就座阶段定下、一个候选——用于清除标注的测试。

    座位 1 已定 + 候选指向座位 2 = 全段齐活（目录变绿）；清除之后必须退回
    「缺 1 人」。刻意不把 presenter_num 设成 1：那样座位 1 本就够数，清不
    清除都是绿的，测不到差异（本项目栽过这个跟头，见 Task 51 报告）。
    """
    frame = _write_frame(tmp_path)
    seg = _seg_with_genders(5, presenter_num, [0] * presenter_num)
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=5)
    out = tmp_path / "clear.html"
    render_entry_review([seg], dets, out,
                        seats_by_order=seats if seats is not None else {5: {11: 1}})
    html = out.read_text(encoding="utf-8")
    blob = _segments_blob_from_html(html)[0]
    ck = f"{blob['order']}::{blob['candidates'][0]['stable_id']}"
    return html, blob, ck


def _card_by_ck(main_html: str, ck: str) -> str:
    """正文里 data-ck 等于 `ck` 的那张候选卡片（到下一张卡片/折叠区为止）。"""
    i = main_html.find(f'data-ck="{ck}"')
    assert i >= 0, f"正文里找不到候选卡片 {ck!r}：{main_html[:400]}"
    j = main_html.find('<div class="cand"', i + 1)
    return main_html[i: j if j > 0 else len(main_html)]


def test_clear_removes_record_entirely(tmp_path):
    """清除标注把 store 里那条记录整条删掉，不是留一个空字符串。

    内存里的 store 与写回 localStorage 的内容都要验：只删内存、写回旧值
    的表现是「点了有反应，刷新一下又回来了」。
    """
    html, _blob, ck = _clear_fixture(tmp_path)
    labelled = {ck: {"role": "创业者", "seat": "2", "n_people": "1", "note": "x"}}

    before = _run_page(html, labelled)
    assert ck in before["store"] and before["store"][ck].get("role") == "创业者", (
        "夹具前提：清除之前这条记录必须真的在 store 里，否则清除路径根本没走到"
    )

    after = _run_page(html, labelled, after=f"clearLabel({json.dumps(ck)});")
    assert ck not in after["store"], (
        f"清除后该候选的记录必须从 store 里整条删除（不是置空），实得 "
        f"{after['store'].get(ck)!r}"
    )
    assert after["written"] is not None, "清除必须写回 localStorage，否则刷新即复活"
    assert ck not in after["written"], (
        f"写回 localStorage 的内容里也不能再有这条记录，实得 "
        f"{after['written'].get(ck)!r}"
    )


def test_clear_resets_role_radios(tmp_path):
    """清除之后六个角色单选全部回到未选中态。"""
    html, _blob, ck = _clear_fixture(tmp_path)
    labelled = {ck: {"role": "创业者", "seat": "2", "n_people": "1"}}

    before = _card_by_ck(_run_page(html, labelled)["main"], ck)
    assert before.count("checked") >= 1, (
        f"夹具前提：清除之前必须有一个角色是选中的，实得卡片 {before[:400]}"
    )

    after = _card_by_ck(
        _run_page(html, labelled, after=f"clearLabel({json.dumps(ck)});")["main"], ck)
    roles = re.search(r'<div class="roles">(.*?)</div>', after, re.DOTALL)
    assert roles, f"卡片里应有角色单选区：{after[:400]}"
    assert "checked" not in roles.group(1), (
        f"清除后不得再有任何角色处于选中态，实得 {roles.group(1)}"
    )
    # 不钉死「六个」：指向不存在那一段的方向角色本就不显示（本夹具只有一段，
    # 两个方向都藏）。这条要验的是**清除只回到未选、不把控件删掉**。
    before_n = re.search(r'<div class="roles">(.*?)</div>', before,
                         re.DOTALL).group(1).count("<input")
    assert roles.group(1).count("<input") == before_n, (
        f"角色单选一个都不能少（清除只回到未选，不是把控件删掉）："
        f"清除前 {before_n} 个、清除后 {roles.group(1).count('<input')} 个"
    )


def test_clear_updates_toc_and_green(tmp_path):
    """清除之后目录的计数与变绿判定必须跟着回退。

    这条防的是「删了但页面没重算」：`save()` 走的是 render()，删除若走另一
    条路径而忘了重渲染，用户会看到已经取消的标注仍然算在「已确定」里、
    目录仍是绿的——他会以为这段标完了。
    """
    html, _blob, ck = _clear_fixture(tmp_path)
    labelled = {ck: {"role": "创业者", "seat": "2", "n_people": "1"}}

    before = _run_page(html, labelled)
    row_before = _toc_rows(before["toc"])[0]
    assert "toc-done-all" in row_before["cls"], (
        f"夹具前提：座位1已定 + 候选指座位2 应先变绿，实得 {row_before['cls']!r}"
    )
    assert "已标 1/1" in row_before["text"], (
        f"夹具前提：清除前目录应显示已标 1 个，实得 {row_before['text']!r}"
    )
    assert "已标 1 / 共 1" in before["progress"], before["progress"]

    after = _run_page(html, labelled, after=f"clearLabel({json.dumps(ck)});")
    row_after = _toc_rows(after["toc"])[0]
    assert "toc-done-all" not in row_after["cls"], (
        f"清除后不该还是绿的（座位 2 又空了），实得 {row_after['cls']!r}"
    )
    assert "已标 0/1" in row_after["text"], (
        f"清除后目录的已标计数必须回退，实得 {row_after['text']!r}"
    )
    assert "已标 0 / 共 1" in after["progress"], (
        f"底部进度同样要回退，实得 {after['progress']!r}"
    )


def test_clear_button_is_offered_on_every_candidate(tmp_path):
    """每个候选都要有「清除本条标注」的入口——只在已标注的候选上给按钮，
    用户就得先随便点一个角色才能取消，等于绕圈子。"""
    html, _blob, ck = _clear_fixture(tmp_path)
    card = _card_by_ck(_run_page(html)["main"], ck)
    assert "clearLabel(" in card and "清除" in card, (
        f"未标注的候选上也必须有清除入口，实得卡片 {card[:500]}"
    )


# ===========================================================================
# Task 52 · 需求三：折叠区要标明候选被**哪条**规则挡下
# ===========================================================================

def test_filtered_tiny_are_disclosed_with_reason(tmp_path):
    """被新的框高下限挡掉的候选与长宽比一样进折叠区，且逐条标明原因。

    「凡有过滤必须可见」在这里要再进一步：用户看折叠区是为了核实「这条
    规则有没有挡错人」，只看到一堆图而不知道每张是被哪条规则挡的，他判断
    不了该调哪个阈值。
    """
    frame = _write_frame(tmp_path)
    seg = mkseg(order=10, presenter_num=1)
    dets = (track(1, RIGHT_CX + 100, STANDING_H, 6, frame, order=10)                    # 正常
            + track(2, RIGHT_CX + 400, 0.10 * H, 6, frame, order=10, width=40.0)        # 只矮
            + track(3, RIGHT_CX + 700, 0.40 * H, 6, frame, order=10, width=400.0))      # 只方
    out = tmp_path / "entry_review.html"
    render_entry_review([seg], dets, out)
    html = out.read_text(encoding="utf-8")

    seg_json = _segments_blob_from_html(html)[0]
    assert {c["track_id"] for c in seg_json["candidates"]} == {1}, (
        f"矮框与方框都不该出现在正常候选里，实得 {seg_json['candidates']}"
    )
    assert seg_json["filtered_count"] == 2, (
        f"两个被挡的候选都必须进折叠区（不静默丢弃），实得 {seg_json['filtered']}"
    )

    section = _run_js(html, f"filteredSection({json.dumps(seg_json)})")
    plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", section))
    assert "长宽比" in plain and "过小" in plain, (
        f"折叠区必须分别说明两条规则，实得：{plain}"
    )

    # 选择器是 `filteredCand`（不再带 `cand`）：2026-08-25 起折叠区的卡片是
    # 一层外壳套一张**完整可标注**的候选卡片，`cand` 那个类归里层。这条测试
    # 钉的是「每张都标明自己被哪条规则挡的」，与外壳叫什么无关。
    cards = re.findall(r'<div class="filteredCand"[^>]*data-reasons="([^"]*)"',
                       section)
    assert sorted(cards) == ["aspect", "height"], (
        f"每张被挡的图都要标出自己是被哪条规则挡的，实得 {cards}"
    )
    per_card = re.split(r'(?=<div class="filteredCand")', section)
    tiny = [c for c in per_card if 'data-reasons="height"' in c]
    assert tiny and "过小" in re.sub(r"<[^>]+>", " ", tiny[0]), (
        f"被高度挡掉的那张，卡片上要直接写明原因，实得 {tiny}"
    )


# ===========================================================================
# Task 52 · 需求四：备注框旁给出可解析的写法示例
# ===========================================================================


# ===========================================================================
# Task 53 · 需求二：标了创业者却没指定座位号，要在页面上醒目提示
#
# 实测文化4组有 3 个这样的候选（`entryexit_fill_plan` 把它们收进
# `needs_seat`，一个也补不进去），页面上却看不出任何异常——用户以为标完了，
# 那几个座位一直空着。`determinedCount` 早就正确地**不把它们算作已确定**，
# 但「不算数」这件事只体现在计数上，用户看不见是哪一条出了问题。
# ===========================================================================

def _seat_missing_page(tmp_path):
    """一段两个候选、需 2 人、就座一人未定——两个座位都可选，所以「没选
    座位号」只可能是用户漏了，不是下拉里没得选。"""
    frame = _write_frame(tmp_path)
    seg = mkseg(order=6, presenter_num=2, company="甲公司")
    dets = (track(21, RIGHT_CX + 100, STANDING_H, 6, frame, order=6)
            + track(22, RIGHT_CX + 500, STANDING_H, 6, frame, order=6))
    out = tmp_path / "entry_review.html"
    render_entry_review([seg], dets, out, seats_by_order={}, detections_fp="FP-53")
    html = out.read_text(encoding="utf-8")
    sids = [c["stable_id"] for c in _segments_blob_from_html(html)[0]["candidates"]]
    return html, sids


def test_presenter_without_seat_is_flagged_on_page(tmp_path):
    """标了创业者但座位号留空的候选卡片上必须有醒目提示，目录该段也计入。

    三条一起断言，缺一条这个功能就等于没做：
      ① 出问题的那张卡片上有提示，且写明「不会补位」（后果，不只是现象）；
      ② **没出问题的卡片上没有**（否则提示就成了背景噪音）；
      ③ 目录该段带上条数——整页很长，不进目录用户根本扫不到。
    """
    html, sids = _seat_missing_page(tmp_path)
    store = {
        f"6::{sids[0]}": {"role": "创业者", "seat": ""},     # 漏了座位号
        f"6::{sids[1]}": {"role": "创业者", "seat": "2"},    # 正常
    }
    out = _run_page(html, store)

    cards = re.findall(r'<div class="cand" data-ck="([^"]*)"(.*?)(?=<div class="cand"|$)',
                       out["main"], re.DOTALL)
    by_ck = {ck: body for ck, body in cards}
    assert len(by_ck) == 2, f"应渲染两张候选卡片，实得 {sorted(by_ck)}"

    bad = by_ck[f"6::{sids[0]}"]
    good = by_ck[f"6::{sids[1]}"]
    assert "seatMissing" in bad, f"漏座位号的卡片上必须有提示元素：{bad}"
    plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", bad))
    assert "未指定座位号" in plain and "不会补位" in plain, (
        f"提示要写明现象与后果，实得：{plain}")
    assert "seatMissing" not in good, (
        f"已指定座位号的卡片不该有这个提示（否则它就成了背景噪音）：{good}")

    rows = _toc_rows(out["toc"])
    assert len(rows) == 1, rows
    assert "1" in rows[0]["text"] and "未指定座位号" in rows[0]["text"], (
        f"目录该段要计入这类提示（共 1 条），实得：{rows[0]['text']!r}")


def test_seat_missing_flag_is_red(tmp_path):
    """提示必须是红色的告警样式——用户扫读整页时靠颜色定位，做成灰色小字
    等于没做。颜色之外另有文字（「未指定座位号」），不只靠颜色。"""
    html, _sids = _seat_missing_page(tmp_path)
    decls = _css_decls(html, ".seatMissing")
    assert re.search(r"#(c0392b|c62828|b3261e|a33|d32f2f)", decls, re.I), (
        f".seatMissing 必须用告警红（与 .multiHint 同一套配色），实得：{decls}")


def test_seat_missing_flag_clears_when_seat_chosen(tmp_path):
    """补上座位号之后提示必须立刻消失（走 render()，不必刷新页面）。

    只在首屏判断一次的实现在这里会留着一条已经不成立的红字——用户会以为
    自己没改成功，反复点。
    """
    html, sids = _seat_missing_page(tmp_path)
    ck = f"6::{sids[0]}"
    out = _run_page(html, {ck: {"role": "创业者", "seat": ""}},
                    after=f"save({ck!r},'seat','1')")
    assert "seatMissing" not in out["main"], (
        f"座位号补上后提示应消失，实得：{out['main'][:600]}")


# ===========================================================================
# Task 53 · 需求三：删掉「本组该位置恒为工作人员/专家」
# 用户 2026-08-23 原话：「这个选项实际没用，删掉吧」。
# ===========================================================================

def test_position_exclusion_控件_removed(tmp_path):
    """页面上不得再有位置排除的任何痕迹：控件、样式、导出字段、容差常量。

    逐项断言而不是只看一个关键词——上一版这个功能横跨 CSS、卡片 HTML、
    一个统计函数、一个注入常量和导出 payload 五处，删漏任何一处都会留下
    一个点了没反应的控件（或一个仍然写进文件、下游却已不读的字段）。
    """
    html = _filled_page_with_candidates(tmp_path)
    for token in ("posExcl", "pos_exclude", "POS_EXCL_TOL",
                  "countPositionMatches", "本组该位置恒为",
                  "position_exclusions", "__POS_EXCL_TOL__"):
        assert token not in html, f"页面里不该再出现 {token!r}"

    card = _run_js(html, "candidateCard(SEGMENTS[0], SEGMENTS[0].candidates[0])")
    assert "恒为工作人员" not in card, card

    # 导出 payload 里也不许再有这个键（哪怕 store 里还留着上一版的残留值）
    seg = _segments_blob_from_html(html)[0]
    ck = f"{seg['order']}::{seg['candidates'][0]['stable_id']}"
    payload = _run_js_with_store(
        html, {ck: {"role": "创业者", "seat": "2",
                    "pos_exclude": True, "pos_exclude_role": "专家"}},
        "buildLabelsPayload()")
    assert "position_exclusions" not in payload, (
        f"localStorage 里的旧残留也不许再写进导出文件，实得 {sorted(payload)}")


# ===========================================================================
# Task 53 · 需求四：左侧目录跟随滚动
#
# 目录是 `position:fixed` + `overflow-y:auto`。段数多时（用户要标 60+ 组），
# 当前高亮的那一行经常在目录自己的可视区之外——高亮了等于没高亮。
# ===========================================================================

def _toc_scroll_page(tmp_path, n=6):
    specs = [(k, 2, f"企业{k}", 1) for k in range(1, n + 1)]
    return _render_toc_page(tmp_path, specs, seats_by_order={})


def _run_toc_scroll(html: str, hit_order: int) -> dict:
    """在 node 里跑真实页面 JS，喂一个 IntersectionObserver 回调，返回
    「哪一行被高亮了、哪一行被滚进了可视区」。

    shim 提供 `scrollIntoView`（真浏览器有）与一个可手工触发的
    IntersectionObserver（node 没有），其余一字不改。
    """
    script = _extract_script(html)
    harness = """
const SAVED = '{}';
const SCROLLED = [], HILITE = [];
function mkRow(seg){
  return {
    _seg: seg,
    classList: {add(c){ if(c==='toc-current') HILITE.push(seg); }, remove(){}},
    getAttribute(n){ return n === 'data-seg' ? String(seg) : null; },
    scrollIntoView(opt){ SCROLLED.push([seg, opt && opt.block]); },
  };
}
const ROWS = [1,2,3,4,5,6].map(mkRow);
const HEADS = [1,2,3,4,5,6].map(s=>({id:'seg-'+s}));
let IO_CB = null;
class IntersectionObserver {
  constructor(cb){ IO_CB = cb; }
  observe(){}
}
const els = {};
const el = id => (els[id] = els[id] || {innerHTML:'', textContent:'',
                   classList:{add(){}, remove(){}}});
const localStorage = { getItem: () => SAVED, setItem: () => {} };
const document = { getElementById: el,
  querySelectorAll: sel => (sel.indexOf('toc-row') >= 0 ? ROWS : HEADS) };
""" + script + """
IO_CB([{isIntersecting: true, target: {id: 'seg-__HIT__'}}]);
console.log(JSON.stringify({scrolled: SCROLLED, hilite: HILITE}));
"""
    harness = harness.replace("__HIT__", str(hit_order))
    r = subprocess.run(["node", "-e", harness], capture_output=True, text=True,
                       timeout=20)
    assert r.returncode == 0, f"Node 执行失败：{r.stderr}"
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_toc_scrolls_current_into_view(tmp_path):
    """高亮当前段的**同时**把那一行滚进目录的可视区。

    断言打在「真的调了滚动、而且是滚的那一行」上，不是打在源码文本上——
    源码里出现 `scrollIntoView` 但从没被调用过，看起来一样绿。
    """
    html = _toc_scroll_page(tmp_path)
    out = _run_toc_scroll(html, hit_order=5)
    assert out["hilite"] == [5], f"应高亮第 5 段，实得 {out['hilite']}"
    assert out["scrolled"] == [[5, "nearest"]], (
        f"被高亮的那一行必须同时滚进目录可视区，且用 block:'nearest'"
        f"（'start'/'center' 会连带把整页位置也拽走），实得 {out['scrolled']}")


def test_toc_scroll_degrades_without_scroll_into_view(tmp_path):
    """老浏览器没有 `scrollIntoView`（或不认对象参数）时静默降级为不滚动，
    不许抛错——一个 TypeError 会打断 IntersectionObserver 回调，连高亮
    一起坏掉，而页面照常渲染、没有任何报错可见。
    """
    html = _toc_scroll_page(tmp_path)
    script = _extract_script(html)
    harness = """
const SAVED = '{}';
const HILITE = [];
function mkRow(seg){
  return { classList: {add(c){ if(c==='toc-current') HILITE.push(seg); }, remove(){}},
           getAttribute(n){ return n === 'data-seg' ? String(seg) : null; } };
}
const ROWS = [1,2,3].map(mkRow);
const HEADS = [1,2,3].map(s=>({id:'seg-'+s}));
let IO_CB = null;
class IntersectionObserver { constructor(cb){ IO_CB = cb; } observe(){} }
const els = {};
const el = id => (els[id] = els[id] || {innerHTML:'', textContent:'',
                   classList:{add(){}, remove(){}}});
const localStorage = { getItem: () => SAVED, setItem: () => {} };
const document = { getElementById: el,
  querySelectorAll: sel => (sel.indexOf('toc-row') >= 0 ? ROWS : HEADS) };
""" + script + """
IO_CB([{isIntersecting: true, target: {id: 'seg-2'}}]);
console.log(JSON.stringify(HILITE));
"""
    r = subprocess.run(["node", "-e", harness], capture_output=True, text=True,
                       timeout=20)
    assert r.returncode == 0, f"没有 scrollIntoView 时不该抛错：{r.stderr}"
    assert json.loads(r.stdout.strip().splitlines()[-1]) == [2], (
        "降级后高亮仍要正常工作（只是不滚动）")


# ===========================================================================
# Task 53 · 需求五：导出后给出明确提示
#
# 用户 2026-08-23 原话：「再次导出的时候，没有任何提示，应该弹出提示」。
#
# 导出有**三条路径**（`run_all._build_export_upgrade` 的升级版接管前两条，
# 页面自带的 `exportLabels` 是第三条）：
#   ① 记住的句柄直写   ② 选择器重新选   ③ `<a download>` 回退
# 三条落地的位置完全不同（上次那个文件 / 刚选的那个文件 / 浏览器下载目录），
# 用户需要知道文件到底落在哪——所以提示文案必须能区分是哪一条。
#
# 不用 `alert()`：它阻塞主线程、样式无法控制、连续导出会叠一串模态框，
# 而且在部分浏览器里被当成弹窗拦掉——那就又回到「没有任何提示」。
# ===========================================================================

def _toast_msg(html: str, where: str, name: str, store: dict | None = None) -> str:
    """页面 JS 真实算出的提示文案（`showExportToast` 的返回值）。"""
    expr = (f"showExportToast({json.dumps(where)}, {json.dumps(name)},"
            " buildLabelsPayload())")
    return _run_js_with_store(html, store or {}, expr)


def test_export_shows_toast_with_destination(tmp_path):
    """三条路径都要有提示，且**能区分是哪一条**、写明落在哪、导了多少条。"""
    html = _filled_page_with_candidates(tmp_path)
    seg = _segments_blob_from_html(html)[0]
    cks = [f"{seg['order']}::{c['stable_id']}" for c in seg["candidates"]]
    store = {cks[0]: {"role": "创业者", "seat": "2"},
             cks[1]: {"role": "工作人员", "seat": ""}}

    msgs = {w: _toast_msg(html, w, n, store) for w, n in (
        ("handle", "上次的.json"), ("picker", "刚选的.json"),
        ("download", "entry_labels_x.json"))}

    for where, msg in msgs.items():
        assert "2" in msg, (
            f"{where}：必须写明导出了多少条标注（本例 2 条），实得 {msg!r}")
    assert "上次的.json" in msgs["handle"], msgs["handle"]
    assert "刚选的.json" in msgs["picker"], msgs["picker"]
    assert "entry_labels_x.json" in msgs["download"], msgs["download"]
    assert len(set(msgs.values())) == 3, (
        f"三条路径的文案必须互不相同（用户要据此知道文件落在哪），实得 {msgs}")
    assert "下载" in msgs["download"], (
        f"回退那条要说明是浏览器下载目录，实得 {msgs['download']!r}")


def test_export_toast_counts_only_real_labels(tmp_path):
    """条数取自**真正写进文件的**那些，不是候选总数、也不是 store 的键数。

    夹具里 store 有 3 个键，但只有 1 条是带角色的候选标注（另一条 role 为
    空、一条是座位否决）——报「3 条」会让用户以为标注比实际多。
    """
    html = _filled_page_with_candidates(tmp_path)
    seg = _segments_blob_from_html(html)[0]
    cks = [f"{seg['order']}::{c['stable_id']}" for c in seg["candidates"]]
    store = {cks[0]: {"role": "创业者", "seat": "2"},
             cks[1]: {"role": "", "seat": ""},
             "seatovr::5::1": {"rejected": True, "note": ""}}
    msg = _toast_msg(html, "download", "x.json", store)
    n = int(re.search(r"(\d+)\s*条标注", msg).group(1))
    assert n == 1, f"只有 1 条真标注（另一条角色为空不写进文件），实得 {msg!r}"
    assert "1 条座位否决" in msg or "座位否决 1" in msg, (
        f"座位否决另计并写出来（它也是要被下游读的输入），实得 {msg!r}")


def test_toast_is_not_alert(tmp_path):
    """不许用 `alert()`：阻塞主线程、样式不可控、连续导出叠一串模态框、
    还可能被弹窗拦截器吃掉——那就又回到「没有任何提示」。"""
    from pipeline.run_all import _build_export_upgrade

    html = _filled_page_with_candidates(tmp_path)
    _banner, injected = _build_export_upgrade(Path("/tmp/work/entry_labels_x.json"))
    for tag, text in (("页面", html), ("注入的导出升级", injected)):
        assert not re.search(r"\balert\s*\(", text), (
            f"{tag}脚本里不许出现 alert(")
        assert not re.search(r"\bconfirm\s*\(", text), (
            f"{tag}脚本里不许出现 confirm(")

    # 提示条必须是页面内的浮动元素，并且会自己淡出（不需要用户去点掉）
    decls = _css_decls(html, ".toast")
    assert "position:fixed" in decls.replace(" ", ""), (
        f".toast 应是页面内的浮动提示条，实得：{decls}")
    assert "transition" in decls or "opacity" in decls, (
        f".toast 要能淡出，实得：{decls}")
    script = _extract_script(html)
    assert re.search(r"setTimeout\([^)]*TOAST", script) or "TOAST_MS" in script, (
        "淡出时长要来自一个具名常量（config.ENTRY_REVIEW_TOAST_MS），不写死")


def test_injected_export_paths_all_notify(tmp_path):
    """注入的升级版导出（run_all）三条路径都要真的调到页面的提示函数。

    断言打在「真跑一遍这段 JS，提示被调了几次、参数是什么」上，不是打在
    源码文本上——`showExportToast` 出现在源码里但从没被调用，看起来一样绿。
    """
    from pipeline.run_all import _build_export_upgrade

    _banner, script_html = _build_export_upgrade(
        Path("/tmp/work/entry_labels_x.json"))
    m = re.search(r"<script>(.*)</script>", script_html, re.DOTALL)
    assert m, "导出升级必须是一个 <script> 块"
    script = m.group(1)

    prelude = """
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
function makeIdb(opts){
  const store = new Map();
  if (opts && opts.seed) store.set('entry_labels_x.json', opts.seed);
  return { open(){ const req = {};
    setTimeout(() => { req.result = { transaction(){ return { objectStore(){ return {
      get(k){ const r={}; setTimeout(()=>{ r.result=store.get(k); r.onsuccess&&r.onsuccess(); },0); return r; },
      put(v,k){ store.set(k,v); const r={}; setTimeout(()=>{ r.onsuccess&&r.onsuccess(); },0); return r; },
      delete(k){ store.delete(k); const r={}; setTimeout(()=>{ r.onsuccess&&r.onsuccess(); },0); return r; },
    }; } }; } }; req.onsuccess && req.onsuccess(); }, 0);
    return req; } };
}
function makeHandle(name, perm){
  return { name: name, _perm: perm,
    queryPermission(){ return Promise.resolve(this._perm); },
    requestPermission(){ this._perm='granted'; return Promise.resolve('granted'); },
    createWritable(){ return Promise.resolve({
      write(d){ LOG.push(['write', name]); return Promise.resolve(); },
      close(){ return Promise.resolve(); } }); } };
}
function toast(w, n){ LOG.push(['toast', w, n]); }
"""

    def run(scenario: str) -> list:
        harness = prelude + scenario + "\n" + script + """
(async function(){
  await window.exportLabels();
  await new Promise(r => setTimeout(r, 20));
  console.log(JSON.stringify(LOG));
})();
"""
        r = subprocess.run(["node", "-e", harness], capture_output=True,
                           text=True, timeout=30)
        assert r.returncode == 0, f"Node 执行失败：{r.stderr}"
        return json.loads(r.stdout.strip().splitlines()[-1])

    saved = run("""
const indexedDB = makeIdb({seed: makeHandle('记住的.json','granted')});
const window = { indexedDB: indexedDB, showExportToast: toast,
  showSaveFilePicker(){ return Promise.reject(new Error('不该被调用')); } };
""")
    assert ["toast", "handle", "记住的.json"] in saved, (
        f"① 直写记住的位置也要提示（用户原话正是「再次导出没有任何提示」），"
        f"实得 {saved}")

    picked = run("""
const indexedDB = makeIdb({});
const picked = makeHandle('新选的.json','granted');
const window = { indexedDB: indexedDB, showExportToast: toast,
  showSaveFilePicker(){ return Promise.resolve(picked); } };
""")
    assert ["toast", "picker", "新选的.json"] in picked, (
        f"② 走选择器那条也要提示，实得 {picked}")

    fallback = run("""
const indexedDB = null;
const window = { showExportToast: toast };
""")
    assert ["anchor"] in fallback, fallback
    assert any(e[0] == "toast" and e[1] == "download" for e in fallback), (
        f"③ <a download> 回退同样要提示（这条最需要——文件落在浏览器下载"
        f"目录，不说用户根本找不到），实得 {fallback}")


def test_injected_export_survives_missing_toast(tmp_path):
    """页面没有 `showExportToast`（旧页面 + 新 run_all）时导出照常成功。

    提示条是锦上添花，一个 TypeError 把导出整条打断，就是 Task 45 那次
    「改了但没生效且不报错」的重演。
    """
    from pipeline.run_all import _build_export_upgrade

    _banner, script_html = _build_export_upgrade(
        Path("/tmp/work/entry_labels_x.json"))
    script = re.search(r"<script>(.*)</script>", script_html, re.DOTALL).group(1)
    harness = """
const EXPORT_FILENAME='entry_labels_x.json';
function buildLabelsPayload(){ return {ok:1}; }
const LOG=[];
const document = { getElementById: () => ({textContent:'', style:{}, onclick:null}),
  createElement: () => ({set href(v){}, set download(v){}, click(){ LOG.push(['anchor']); }}) };
const URL={createObjectURL:()=> 'blob:x'};
function Blob(p){ this.parts=p; }
const indexedDB=null;
const window={};
""" + script + """
(async function(){ await window.exportLabels();
  await new Promise(r=>setTimeout(r,20));
  console.log(JSON.stringify(LOG)); })();
"""
    r = subprocess.run(["node", "-e", harness], capture_output=True, text=True,
                       timeout=30)
    assert r.returncode == 0, f"没有提示函数时导出不该抛错：{r.stderr}"
    assert ["anchor"] in json.loads(r.stdout.strip().splitlines()[-1]), (
        "导出本身必须照常完成")


def test_page_export_actually_calls_the_toast(tmp_path):
    """页面自带的 `exportLabels()`（第 ③ 条路径）必须**真的**调到提示条。

    只测文案函数不够：`showExportToast` 写得再对，`exportLabels` 里不调它
    就等于没做，而两条断言看起来都是绿的——这正是本项目反复栽的那类坑。
    这里真跑一遍 `exportLabels()`，断言打在「提示条元素上出现了什么字」。
    """
    html = _filled_page_with_candidates(tmp_path)
    seg = _segments_blob_from_html(html)[0]
    ck = f"{seg['order']}::{seg['candidates'][0]['stable_id']}"
    script = _extract_script(html)
    harness = (
        "const SAVED = " + json.dumps(json.dumps(
            {ck: {"role": "创业者", "seat": "2"}}, ensure_ascii=False)) + ";\n"
        """
const els = {};
const CREATED = [];
const el = id => (els[id] = els[id] || {innerHTML:'', textContent:'',
                   classList:{add(){}, remove(){}}});
const localStorage = { getItem: () => SAVED, setItem: () => {} };
let CLICKED = 0;
const document = {
  getElementById: id => (id === 'export-toast' ? (els[id] || null) : el(id)),
  querySelectorAll: () => [],
  createElement: tag => { const o = {tag: tag, className:'', textContent:'',
    set id(v){ this._id = v; els[v] = o; }, get id(){ return this._id; },
    set href(v){}, set download(v){}, click(){ CLICKED++; }};
    CREATED.push(o); return o; },
  body: { appendChild(){} },
};
const URL = { createObjectURL: () => 'blob:x' };
function Blob(p){ this.parts = p; }
""" + script + """
exportLabels();
const toast = els['export-toast'];
console.log(JSON.stringify({clicked: CLICKED,
  text: toast ? toast.textContent : null,
  cls: toast ? toast.className : null}));
"""
    )
    r = subprocess.run(["node", "-e", harness], capture_output=True, text=True,
                       timeout=20)
    assert r.returncode == 0, f"Node 执行失败：{r.stderr}"
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["clicked"] == 1, f"导出本身要真的发生，实得 {out}"
    assert out["text"] and "1 条标注" in out["text"], (
        f"导出后提示条上必须写出条数，实得 {out['text']!r}")
    assert "entry_labels_" in out["text"], (
        f"提示条要写明文件名（用户得知道文件叫什么），实得 {out['text']!r}")
    assert "toast-show" in (out["cls"] or ""), (
        f"提示条要真的显现出来（加上 .toast-show），实得 {out['cls']!r}")


# ===========================================================================
# Task 54 · 需求一：格位分配改成结构化选择题（页面这一侧）
#
# 用户原话：「能放弃解析，转换为选择吗，这样就能精准实时匹配到座位人是否
# 识别，比如选人数为 2 时，就给出这两人身份和座位号的选择题，且可留空，根据
# 选出的空，精准识别，不需要解析了」。
#
# 解析在 Python 侧时页面无法实时反馈：城市2组 order1 三个人早就写进备注了，
# 目录仍显示红色未完成——要等下一次跑流水线才知道解析结果。**实时性是这组
# 改动的核心价值**，下面 test_cells_count_toward_green_immediately 专门验它。
# ===========================================================================

def _cell_page(tmp_path, presenter_num=3, seats_by_order=None, labels=None,
               order=5, n_tracks=1):
    """一段、`n_tracks` 个候选（每个都有 4 张证据帧，四个格位都可用）。"""
    frame = _write_frame(tmp_path)
    seg = _seg_with_genders(order, presenter_num, [0] * presenter_num)
    dets = []
    for i in range(n_tracks):
        dets += track(7 + i, RIGHT_CX + 150 + 260 * i, STANDING_H, 6, frame,
                      order=order, kind="entry" if i % 2 == 0 else "exit")
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order=seats_by_order or {},
                        detections_fp="FP-54", entry_labels=labels or {})
    html = out.read_text(encoding="utf-8")
    blob = _segments_blob_from_html(html)[0]
    assert blob["candidates"], "夹具前提：必须至少有一个候选"
    assert blob["candidates"][0]["n_cells"] == 4, (
        f"夹具前提：候选要有 4 张证据帧（四个格位都可选），实得 "
        f"{blob['candidates'][0]['n_cells']}")
    return html, blob


def _seat_wrap_of(card_html):
    """卡片里「座位号」那一栏（<div class="seatWrap" …>…</div>）。"""
    m = re.search(r'<div class="seatWrap"[^>]*>.*?</div>', card_html, re.DOTALL)
    assert m, f"卡片上没有座位号栏：{card_html[:600]}"
    return m.group(0)


def _cand_card(html, store, seg_idx=0, cand_idx=0):
    return _run_js_with_store(
        html, store,
        f"candidateCard(SEGMENTS[{seg_idx}], SEGMENTS[{seg_idx}].candidates[{cand_idx}])")


def _selects(card_html):
    """把卡片里每个 <select> 拆成 (标签文字, [(value, 文本, 是否selected)])。"""
    out = []
    for m in re.finditer(r"<label>([^<]*)<select([^>]*)>(.*?)</select>",
                         card_html, re.DOTALL):
        opts = [(o.group(1), re.sub(r"<[^>]+>", "", o.group(3)).strip(),
                 "selected" in o.group(2))
                for o in re.finditer(r'<option value="([^"]*)"([^>]*)>(.*?)</option>',
                                     m.group(3), re.DOTALL)]
        out.append({"label": m.group(1).strip(), "attrs": m.group(2), "options": opts})
    return out


def _cell_selects(card_html):
    """格位分配区块里的下拉，按行分组：[[格位, 身份, 座位号], ...]。"""
    box = re.search(r'<div class="cellsBox">(.*?)</div>\s*(?:<div class="noteLegacy"|'
                    r'<textarea)', card_html, re.DOTALL)
    rows = re.findall(r'<div class="cellRow"[^>]*>(.*?)</div>\s*(?=<div class="cellRow"|$)',
                      card_html, re.DOTALL)
    assert rows, f"卡片上没有格位分配的行：{card_html[:800]}"
    return [_selects(r) for r in rows]


def test_cell_index_defaults_to_sequence(tmp_path):
    """人数 ≥2 → 展开**证据图格数**那么多行，格位号预填 1..N。

    行数跟的是格数不是人数（用户 2026-08-24：「2*2 里有 2 个人，我希望将这
    四个格位分给 2 个人」）——四格都是画面，每格都要说清是谁，多出来的行
    留空即可。留空不是「已判定为空」：身份与座位号一个都不预选。
    """
    html, blob = _cell_page(tmp_path)
    ck = f"{blob['order']}::{blob['candidates'][0]['stable_id']}"
    n_cells = blob["candidates"][0]["n_cells"]
    card = _cand_card(html, {ck: {"n_people": "3"}})
    rows = _cell_selects(card)
    assert len(rows) == n_cells, (
        f"证据图有 {n_cells} 格就该有 {n_cells} 行（与人数 3 无关），实得 {len(rows)}")
    for i, row in enumerate(rows):
        assert len(row) == 3, f"每行三个控件（格位/身份/座位号），实得 {row}"
        assert row[0]["label"].startswith("格位")
        picked = [v for v, _t, sel in row[0]["options"] if sel]
        assert picked == [str(i + 1)], (
            f"第 {i + 1} 行的格位号应默认预填 {i + 1}，实得 {picked}")
        assert [v for v, _t, sel in row[1]["options"] if sel] == [""], (
            f"身份必须默认留空（不许预填），实得 {row[1]['options']}")
        assert [v for v, _t, sel in row[2]["options"] if sel] == [""], (
            f"座位号必须默认留空，实得 {row[2]['options']}")


def test_cell_index_dropdown_excludes_cells_taken_by_other_rows(tmp_path):
    """格位下拉不列**别的行已占**的格位号。

    行数等于格数之后，每格恰好一行、四个格位号全都有主。若下拉仍随便让改，
    用户把第 1 行从格位1 改成格位3，就会出现两行都写着格位3、而格位1 没有
    任何一行认领——两行指着同一张图，另一格的人完全没被判定，页面上却看不
    出任何异常。
    """
    html, blob = _cell_page(tmp_path, presenter_num=3)
    ck = f"{blob['order']}::{blob['candidates'][0]['stable_id']}"
    rows = _cell_selects(_cand_card(html, {ck: {"n_people": "2"}}))
    got = [[v for v, _t, _s in row[0]["options"]] for row in rows]
    assert got == [["1"], ["2"], ["3"], ["4"]], (
        f"四行各自占住一个格位号，下拉里不该出现别人的格位，实得 {got}")


def test_cell_seat_dropdown_lists_only_vacant(tmp_path):
    """格位那一行的座位号下拉只列**尚空**的座位（沿用 Task 48 的
    effectiveVacant）+「不确定」。

    列出已被占用的座位，用户标了 entryexit_fill_plan 也会跳过（不覆盖已占
    座位）——标了却不生效，且没有任何提示。
    """
    # 需 3 人，就座已定座位 1 → 只剩 2、3 可选
    html, blob = _cell_page(tmp_path, presenter_num=3, seats_by_order={5: {99: 1}})
    assert blob["vacant_seats"] == [2, 3], f"夹具前提，实得 {blob['vacant_seats']}"
    ck = f"{blob['order']}::{blob['candidates'][0]['stable_id']}"
    card = _cand_card(html, {ck: {"n_people": "2"}})
    seat_opts = [v for v, _t, _s in _cell_selects(card)[0][2]["options"]]
    assert seat_opts == ["", "2", "3"], (
        f"座位号下拉只该有「不确定」+ 尚空的 2、3，实得 {seat_opts}")


def test_cells_count_toward_green_immediately(tmp_path):
    """**本需求的核心价值**：格位选择题一填，`determinedCount` 立刻增加、
    目录当场变绿，不必等跑流水线。

    在 node 里跑页面真实的 JS：先渲染一遍（红），再调 `saveCell(...)` 把
    两个格位填成两个座位的创业者，读**同一次运行**里重算出来的目录。
    """
    html, blob = _cell_page(tmp_path, presenter_num=3, seats_by_order={5: {99: 1}})
    sid = blob["candidates"][0]["stable_id"]
    ck = f"{blob['order']}::{sid}"

    before = _toc_rows(_run_page(html, {ck: {"n_people": "2"}})["toc"])[0]
    assert "toc-done-all" not in before["cls"], (
        f"夹具前提：还缺 2 人，初始不该是绿的，实得 {before['cls']!r}")

    after = _run_page(
        html, {ck: {"n_people": "2"}},
        after=(f"saveCell({blob['order']},'{sid}',0,'role','创业者');"
               f"saveCell({blob['order']},'{sid}',0,'seat','2');"
               f"saveCell({blob['order']},'{sid}',1,'role','创业者');"
               f"saveCell({blob['order']},'{sid}',1,'seat','3');"))
    row = _toc_rows(after["toc"])[0]
    assert "toc-done-all" in row["cls"], (
        f"两个格位填完（座位 2、3）+ 就座已定的座位 1 = 3 人，应立刻变绿，"
        f"实得 {row['cls']!r}")
    cells = after["store"][ck]["cells"]
    filled = [c for c in cells if c.get("role") or c.get("same_as")]
    assert filled == [{"cell": 1, "role": "创业者", "seat": 2},
                      {"cell": 2, "role": "创业者", "seat": 3}], (
        f"store 里存的必须是结构化的 cells，实得 {cells}")
    # 其余格位留空存着（行数 = 格数），留空 = 未判定，不参与补位
    assert [c["cell"] for c in cells] == [1, 2, 3, 4], (
        f"四格都该有一行，实得 {cells}")
    assert after["written"][ck]["cells"] == cells, (
        "内存里改了、写回 localStorage 的还是旧的 → 刷新即丢，同样是静默失效")


def test_note_prefilled_into_cells_on_page(tmp_path):
    """只有旧备注、没有 cells 的候选：解析结果**预填进控件**并提示
    「已由备注自动填充，请核对」——让用户一次性迁移过来。

    解析只在 Python 侧那一份（`entryexit.note_to_cells`），页面不另写一份
    JS 解析器：两处各写一份必然走样，而走样的表现是页面显示的分配与流水线
    实际补的人对不上。
    """
    html, blob = _cell_page(tmp_path, presenter_num=3)
    sid = blob["candidates"][0]["stable_id"]
    note = "1/3是创业者3，2/4是创业者2"          # 用户真实写法（order 9 华韵传媒）
    labels = {str(blob["order"]): {sid: {"role": "创业者", "n_people": 2,
                                          "note": note}}}
    html, blob = _cell_page(tmp_path, presenter_num=3, labels=labels)
    sid = blob["candidates"][0]["stable_id"]
    ck = f"{blob['order']}::{sid}"
    card = _cand_card(html, {ck: {"role": "创业者", "n_people": "2", "note": note}})

    assert "已由备注自动填充，请核对" in card, f"必须提示是自动填的：{card[:800]}"
    rows = _cell_selects(card)
    picked = [[v for v, _t, s in sel["options"] if s] for row in rows for sel in row]
    assert picked == [["1"], ["创业者"], ["3"],
                      ["2"], ["创业者"], ["2"],
                      ["3"], ["same:1"], [],
                      ["4"], ["same:2"], []], (
        f"「1/3是创业者3，2/4是创业者2」应填成 1→创业者座位3、2→创业者座位2、"
        f"3→同格位1、4→同格位2；重复的目标折成引用（不复制一份值），"
        f"实得 {picked}")
    assert "同格位1" in card, "重复的目标应折成「同格位X」引用，而不是复制一份值"


def test_unparsable_note_shown_readonly(tmp_path):
    """解析不出的备注：原文照旧显示在只读的备注栏里供参考，并说明它没能
    自动填充——不显示的话，用户永远不知道自己写的那句话去哪了。"""
    note = "另两个是混入人员，看不清谁是谁"
    html, blob = _cell_page(tmp_path, presenter_num=3)
    sid = blob["candidates"][0]["stable_id"]
    labels = {str(blob["order"]): {sid: {"role": "创业者", "n_people": 2,
                                          "note": note}}}
    html, blob = _cell_page(tmp_path, presenter_num=3, labels=labels)
    sid = blob["candidates"][0]["stable_id"]
    ck = f"{blob['order']}::{sid}"
    card = _cand_card(html, {ck: {"role": "创业者", "n_people": "2", "note": note}})

    m = re.search(r'<div class="noteLegacy">(.*?)</div>\s*(?:<textarea|$)',
                  card, re.DOTALL)
    assert m, f"备注原文必须只读展示出来：{card[:800]}"
    plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m.group(1)))
    assert note in plain, f"要显示原文，实得 {plain!r}"
    assert "不再参与补位判定" in plain, f"要写明它不再参与判定，实得 {plain!r}"
    assert "没能自动填成" in plain, f"要说明它没能自动填充，实得 {plain!r}"
    assert "已由备注自动填充" not in card, "解析不出就不该声称自动填过"


def test_labels_file_next_to_page_is_used_for_prefill(tmp_path):
    """不显式传标注时，按约定去 `entry_review.html` 同目录找
    `entry_labels_<SHEET>.json`——正式流水线里两者本就在同一个 work 目录。

    这条守住「代码改了但产物没变化」那类缺陷：run_all 只调
    `render_entry_review(...)`，不传标注，若约定这条路没接上，页面上一条
    预填都不会有，而且不会报错。
    """
    frame = _write_frame(tmp_path)
    seg = _seg_with_genders(5, 3, [0, 0, 0])
    dets = track(7, RIGHT_CX + 150, STANDING_H, 6, frame, order=5)
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, detections_fp="FP-54", entry_labels={})
    sid = _segments_blob_from_html(out.read_text(encoding="utf-8"))[0][
        "candidates"][0]["stable_id"]

    safe = config.sanitize_filename_component(config.SHEET)
    (tmp_path / f"entry_labels_{safe}.json").write_text(json.dumps({
        "_meta": {"detections_fingerprint": "FP-54"},
        "5": {sid: {"role": "创业者", "n_people": 2, "note": "1是创业者2"}},
    }, ensure_ascii=False), encoding="utf-8")

    render_entry_review([seg], dets, out, detections_fp="FP-54")   # 不传标注
    html = out.read_text(encoding="utf-8")
    card = _cand_card(html, {f"5::{sid}": {"role": "创业者", "n_people": "2",
                                            "note": "1是创业者2"}})
    assert "已由备注自动填充" in card, (
        f"同目录的标注文件应被读到并预填，实得：{card[:600]}")


# --- 「同格位X」（用户追加需求）------------------------------------------

def test_same_as_only_references_earlier_cells(tmp_path):
    """身份下拉里的「同格位X」**只列比自己小的格位号**。

    只能向前引用 → 每跳一次格位号严格减小 → 环在结构上不可能出现，不必
    事后检测环（`entryexit._resolve_same_as` 的同一条不变式）。
    """
    html, blob = _cell_page(tmp_path, presenter_num=3)
    ck = f"{blob['order']}::{blob['candidates'][0]['stable_id']}"
    rows = _cell_selects(_cand_card(html, {ck: {"n_people": "3"}}))
    refs = [[v for v, _t, _s in row[1]["options"] if v.startswith("same:")]
            for row in rows]
    assert refs == [[], ["same:1"], ["same:1", "same:2"],
                    ["same:1", "same:2", "same:3"]], (
        f"格位1 无可引用；格位2 只能引用 1；格位3 只能引用 1、2（**没有**"
        f"「同格位3」这种指向更大格位的选项），实得 {refs}")


def test_same_as_stores_reference_not_copy(tmp_path):
    """选「同格位1」后，store 里存的是 `{"cell":2,"same_as":1}` —— 引用，
    不是把格位1 的身份与座位号复制一份。

    复制会在用户回头改格位1 时留下一份不跟着变的旧副本，而且不报错——本
    项目在「同一份信息存两处」上已栽过多次。
    """
    html, blob = _cell_page(tmp_path, presenter_num=3)
    sid = blob["candidates"][0]["stable_id"]
    ck = f"{blob['order']}::{sid}"
    got = _run_page(html, {ck: {"n_people": "2"}},
                    after=(f"saveCell({blob['order']},'{sid}',0,'role','创业者');"
                           f"saveCell({blob['order']},'{sid}',0,'seat','2');"
                           f"saveCell({blob['order']},'{sid}',1,'role','same:1');"))
    assert got["store"][ck]["cells"][:2] == [
        {"cell": 1, "role": "创业者", "seat": 2}, {"cell": 2, "same_as": 1}], (
        f"第二行只该存引用，实得 {got['store'][ck]['cells']}")
    card = _cand_card(html, {ck: got["store"][ck]})
    assert "继承自格位 1" in card, f"页面要写明继承自哪一格：{card[:800]}"
    assert re.search(r"<select disabled[^>]*>\s*<option>2</option>", card), (
        f"座位号下拉应禁用并灰字显示继承来的 2：{card[:800]}")


def test_changing_source_cell_updates_dependents(tmp_path):
    """改了被引用的格位，引用它的格位**跟着变**——这正是「不复制」的价值。

    把格位1 从座位2 改成座位3 之后，选了「同格位1」的格位2 必须也变成
    座位3；若当初复制了一份，它会停在座位2 且不报错。
    """
    html, blob = _cell_page(tmp_path, presenter_num=3)
    sid = blob["candidates"][0]["stable_id"]
    ck = f"{blob['order']}::{sid}"
    store = {ck: {"n_people": "2",
                  "cells": [{"cell": 1, "role": "创业者", "seat": 2},
                            {"cell": 2, "same_as": 1}]}}
    got = _run_page(html, store,
                    after=f"saveCell({blob['order']},'{sid}',0,'seat','3');")
    cells = got["store"][ck]["cells"]
    assert cells[:2] == [{"cell": 1, "role": "创业者", "seat": 3},
                         {"cell": 2, "same_as": 1}], (
        f"改源格不该在引用行里留下任何副本，实得 {cells}")
    card = _cand_card(html, {ck: {"n_people": "2", "cells": cells}})
    assert re.search(r"<select disabled[^>]*>\s*<option>3</option>", card), (
        f"引用行显示的座位号应跟着变成 3，实得：{card[:800]}")


def test_changing_people_count_keeps_filled_cells(tmp_path):
    """改人数**不许删掉已填好的格位**。

    旧实现按人数截断（`cells.slice(0, n)`）：人数从 3 拨回 2，格位 3/4 上
    已经判定好的人被静默删掉，页面上只是那两行变空，没有任何提示。行数现在
    由格数决定，人数只管这块控件展不展开，因此一格也不该丢。
    """
    html, blob = _cell_page(tmp_path, presenter_num=3)
    sid = blob["candidates"][0]["stable_id"]
    ck = f"{blob['order']}::{sid}"
    store = {ck: {"n_people": "3",
                  "cells": [{"cell": 1, "role": "创业者", "seat": 2},
                            {"cell": 2, "role": "创业者", "seat": 3},
                            {"cell": 3, "same_as": 1},
                            {"cell": 4, "same_as": 2}]}}
    got = _run_page(html, store,
                    after=f"setPeople({blob['order']},'{sid}','2');")
    assert got["store"][ck]["cells"] == store[ck]["cells"], (
        f"人数改小不该动已填的格位，实得 {got['store'][ck]['cells']}")


def test_forward_reference_is_cleared_on_save(tmp_path):
    """指向**更大**格位的引用一律清掉——只能向前引用，环在结构上不可能出现。

    这类记录会从旧标注文件里读回来（早先按人数铺行时格位号可以乱序）。留着
    它，页面上写着「继承自格位3」而解引用一路解不开，那一格既不报错也补不
    进人。
    """
    html, blob = _cell_page(tmp_path, presenter_num=3)
    sid = blob["candidates"][0]["stable_id"]
    ck = f"{blob['order']}::{sid}"
    store = {ck: {"n_people": "2",
                  "cells": [{"cell": 1, "same_as": 3},
                            {"cell": 3, "role": "创业者", "seat": 2}]}}
    got = _run_page(html, store,
                    after=f"saveCell({blob['order']},'{sid}',1,'seat','3');")
    by_cell = {c["cell"]: c for c in got["store"][ck]["cells"]}
    assert "same_as" not in by_cell[1], (
        f"格位1 指向更大的格位3，这条引用必须清掉，实得 {by_cell[1]}")
    assert "toc-done-all" not in _toc_rows(got["toc"])[0]["cls"], (
        "清掉引用后只剩一个座位，需 3 人，不该算已确定")


def test_same_person_two_cells_counts_once(tmp_path):
    """格位1 与格位3 同为一人（「同格位1」）时只算**一个**座位。

    Task 50 的 `determinedCount` 本就按座位号去重，但「同格位」让这种情形
    从偶发变成常态，必须钉住：算两次的话，一个人会把两个座位都「填满」，
    页面提前变绿而实际还缺人。
    """
    html, blob = _cell_page(tmp_path, presenter_num=3, seats_by_order={5: {99: 1}})
    sid = blob["candidates"][0]["stable_id"]
    ck = f"{blob['order']}::{sid}"
    store = {ck: {"n_people": "2",
                  "cells": [{"cell": 1, "role": "创业者", "seat": 2},
                            {"cell": 3, "same_as": 1}]}}
    row = _toc_rows(_run_page(html, store)["toc"])[0]
    assert "toc-done-all" not in row["cls"], (
        f"座位1 已定 + 两格同一人（都指座位2）= 2 个座位，需 3 人，不该变绿"
        f"（算两次的话恰好凑够 3、会提前变绿而实际还缺人）：{row['cls']!r}")


def test_cells_are_exported(tmp_path):
    """导出的 JSON 带上 `cells`；**没填过格位的候选形状一字不变**
    （Task 46 起既有的四个键），否则下游与既有测试全要跟着动。"""
    html, blob = _cell_page(tmp_path, presenter_num=3, n_tracks=2)
    sids = [c["stable_id"] for c in blob["candidates"]]
    store = {
        f"{blob['order']}::{sids[0]}": {
            "role": "创业者", "seat": "2", "n_people": "1"},
        f"{blob['order']}::{sids[1]}": {
            "n_people": "2",
            "cells": [{"cell": 1, "role": "创业者", "seat": 3},
                      {"cell": 2, "same_as": 1}]},
    }
    payload = _run_js_with_store(html, store, "buildLabelsPayload()")
    body = payload[str(blob["order"])]
    assert "cells" not in body[sids[0]], (
        f"没填过格位的候选不许多出一个键，实得 {body[sids[0]]}")
    assert body[sids[1]]["cells"] == [{"cell": 1, "role": "创业者", "seat": 3},
                                       {"cell": 2, "same_as": 1}], body[sids[1]]
    assert body[sids[1]]["role"] == "", (
        "一个 2×2 里三个人分属不同座位时「这个候选的角色」没有单一答案——"
        "没选角色也必须能导出（只按 role 过滤会把逐格选好的东西整条丢掉）")


def test_cells_survive_a_reopen_when_file_has_them(tmp_path):
    """导出过 cells 之后重开页面，格位选择必须还在。

    run_all 注入的 PRELOAD 按**整条记录**覆盖 store，而它拼出来的记录里没有
    `cells`（`run_all._entry_labels_to_store` 只搬 role/seat/n_people/note）。
    不从标注文件把 cells 补回来，用户的劳动成果重开页面就没了，且不报错。
    """
    seats = {5: {99: 1}}                     # 就座阶段已定座位 1
    html, blob = _cell_page(tmp_path, presenter_num=3, seats_by_order=seats)
    sid = blob["candidates"][0]["stable_id"]
    labels = {str(blob["order"]): {sid: {
        "role": "", "n_people": 2, "note": "",
        "cells": [{"cell": 1, "role": "创业者", "seat": 2},
                  {"cell": 2, "role": "创业者", "seat": 3}]}}}
    html, blob = _cell_page(tmp_path, presenter_num=3, seats_by_order=seats,
                            labels=labels)
    sid = blob["candidates"][0]["stable_id"]
    ck = f"{blob['order']}::{sid}"
    # PRELOAD 那一侧能给的就这些（没有 cells）
    got = _run_page(html, {ck: {"role": "", "seat": "", "n_people": "2", "note": ""}})
    assert got["store"][ck]["cells"] == labels[str(blob["order"])][sid]["cells"], (
        f"文件里的 cells 必须被补回页面，实得 {got['store'][ck]}")
    assert "toc-done-all" in _toc_rows(got["toc"])[0]["cls"], (
        "补回来之后：就座已定座位1 + cells 的座位2、3 = 3 人，该段应为绿；"
        "不补回来就只剩座位1，用户会以为自己上次白标了")


# ===========================================================================
# Task 54 · 端到端（只读用户的真实标注）
# ===========================================================================

def test_real_notes_migrate_to_cells_and_turn_a_segment_green(tmp_path):
    """用户**手写的中文备注**迁成 cells 之后，能把一整段判为已完成。

    这条链是本需求的由来：城市2组 order1 的三条备注早就把三个人都写清楚了，
    页面却一直红着——解析在 Python 侧，页面无从得知结果。迁成结构化的 cells
    之后，选择题一填目录当场变绿。

    **段号不写死。** 原先这条测试钉的是 order1 恰好有三条备注、恰好覆盖三个
    座位——那是用户某一版标注的样子。他 2026-08-24 21:40 重标了城市2，order1
    的备注变了，测试当场变红而代码一字未动（同类坑本会话第三次：先是
    `test_real_labels_produce_expected_splits` 钉死 `[2, 3]`，再是
    `test_real_labels_fill_every_labelled_seat` 钉死 34/34）。

    改为**自己去找**：扫真实标注，挑出「备注解析出的座位 + 就座已定的座位
    正好铺满该段编制、且备注确实贡献了至少一个」的那一段来验。找不到就
    skip 并说清原因——绝不静静变成「什么都没验证」还照样绿。
    """
    import pytest
    from pipeline import entryexit as ee

    from pipeline import paths as _pp
    root = Path("work/城市2-4.26")
    # 位置由 pipeline.paths 解析（Task 56G：交付物归 work/Output/<组>/）。
    # 自己拼一遍拼错了，这条会静静地 skip——而 skip 就是「什么都没验证」。
    labels_file = _pp.labels_path(root, "城市2-4.26")
    segs_file, seats_file = root / "segments.json", root / "seats.json"
    for p in (labels_file, segs_file, seats_file):
        if not p.exists():
            pytest.skip(f"缺少用户的真实标注/产物 {p}（该组的 work 目录不在本机）")

    real = json.loads(labels_file.read_text(encoding="utf-8"))
    all_segs = {s["order"]: s
                for s in json.loads(segs_file.read_text(encoding="utf-8"))["segments"]}
    seats_json = json.loads(seats_file.read_text(encoding="utf-8"))

    pick = None
    for key, recs in real.items():
        if key in ee.LABEL_RESERVED_KEYS or not isinstance(recs, dict):
            continue
        order = int(key)
        seg_meta = all_segs.get(order)
        if not seg_meta:
            continue
        need = set(range(1, int(seg_meta["presenter_num"]) + 1))
        seated = {int(v) for v in (seats_json.get(str(order)) or {}).values()}
        notes, from_notes = [], set()
        for rec in recs.values():
            if not isinstance(rec, dict) or not rec.get("note"):
                continue
            parsed = ee.parse_cell_note_report(rec["note"])["parsed"]
            hit = {int(v["seat"]) for v in parsed.values()
                   if v.get("role") == "创业者" and v.get("seat")}
            if hit:
                notes.append(rec["note"])
                from_notes |= hit
        # 备注必须**确实贡献**了座位，否则这条测试验的只是就座结果
        if notes and from_notes and (seated | from_notes) >= need:
            pick = (order, seg_meta, seated, notes)
            break

    if pick is None:
        pytest.skip(
            "当前这一版真实标注里，没有哪一段是「备注解析出的座位 + 就座已定"
            "的座位」正好铺满编制的——这条端到端无从验证。不是失败，但也**没有"
            "验证过任何东西**，不要把它算作通过。")

    order, seg_meta, seated, notes = pick
    frame = _write_frame(tmp_path)
    n = int(seg_meta["presenter_num"])
    seg = _seg_with_genders(order, n, [1] * n, company=seg_meta["company"])
    dets = []
    for i in range(len(notes)):
        dets += track(20 + i, RIGHT_CX + 120 + 250 * i, STANDING_H, 6, frame,
                      order=order, kind="exit")
    # 就座已定的座位用假 track 号占住，与真实产物的座位号一致
    seats_by_order = {order: {900 + j: st for j, st in enumerate(sorted(seated))}}

    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order=seats_by_order,
                        detections_fp="FP-CITY", entry_labels={})
    sids = [c["stable_id"]
            for c in _segments_blob_from_html(out.read_text(encoding="utf-8"))[0]
            ["candidates"]]
    assert len(sids) == len(notes), (
        f"夹具前提：应有 {len(notes)} 个候选各承载一条真实备注，实得 {sids}")

    labels = {str(order): {sid: {"role": "", "seat": None,
                                 "n_people": 2, "note": note}
                           for sid, note in zip(sids, notes)}}
    render_entry_review([seg], dets, out, seats_by_order=seats_by_order,
                        detections_fp="FP-CITY", entry_labels=labels)
    html = out.read_text(encoding="utf-8")
    store = {f"{order}::{sid}": {"role": "", "seat": "", "n_people": "2",
                                 "note": note}
             for sid, note in zip(sids, notes)}

    got = _run_page(html, store)
    row = _toc_rows(got["toc"])[0]
    assert "toc-done-all" in row["cls"], (
        f"第 {order} 段的真实备注写清了全部创业者，迁成 cells 后应判为已完成，"
        f"实得 {row['cls']!r} · {row['text']!r}（备注：{notes}）")

    # 导出的 JSON 里要真的带上迁移后的 cells（不然下次重开又回到备注）
    payload = _run_js_with_store(html, store, "buildLabelsPayload()")
    exported = payload[str(order)]
    assert all("cells" in rec for rec in exported.values()), (
        f"每条备注都该迁成 cells，实得 {exported}")
    seats_from_cells = {c["seat"] for rec in exported.values()
                        for c in rec["cells"] if c.get("role") == "创业者"}
    need = set(range(1, int(seg_meta["presenter_num"]) + 1))
    assert seats_from_cells | set(seated) >= need, (
        f"迁移后的 cells 加上就座已定的应铺满第 {order} 段的编制 {sorted(need)}，"
        f"实得 cells={sorted(seats_from_cells)} 就座={sorted(seated)}")


# --- 格位「同格位X」选项不该依赖行是否存在（用户 2026-08-24 实测）---

def test_same_as_options_do_not_depend_on_row_existence(tmp_path):
    """格位 K 的身份下拉里「同格位1..K-1」恒可选，与行是否已生成无关。

    用户实测「无法设置格位3 格位4同上」。根因：`cellRoleOptions` 从
    `rows`（当前已生成的行）里取可引用的格位号，而 `cellRowsOf` 只按
    「人数」生成那么多行——人数选 2 时只有 2 行，用户把某行的格位号改到
    3 或 4 后，那一行的下拉里仍只有「同格位1」（来自另一行），拿不到
    「同格位2/3」。

    正确行为：可引用的格位号由**本格的格位号**决定（1..K-1），不看别的行
    存不存在。格位 1 无更小的格位，不给该选项。
    """
    frame = _write_frame(tmp_path)
    seg = mkseg(order=5, presenter_num=3)
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=5)
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={5: {}})
    html = out.read_text(encoding="utf-8")

    import re
    m = re.search(r"function cellRoleOptions\((.*?)\n\}", html, re.S)
    assert m, "未找到 cellRoleOptions"
    body = m.group(1)
    # 选项来源必须是格位号本身，不是遍历 rows
    assert "rows.forEach" not in body, (
        f"可引用格位不应从 rows 遍历取得（行不存在时就拿不到）：{body[:300]}")
    assert re.search(r"for\s*\(\s*let\s+\w+\s*=\s*1", body), (
        f"应从 1 循环到 here-1 生成选项：{body[:300]}")


def test_cell1_has_no_same_as_option(tmp_path):
    """格位 1 没有「同格位」——没有比它更小的格位可引用。"""
    frame = _write_frame(tmp_path)
    seg = mkseg(order=5, presenter_num=2)
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=5)
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={5: {}})
    html = out.read_text(encoding="utf-8")
    import re
    m = re.search(r"function cellRoleOptions\((.*?)\n\}", html, re.S)
    body = m.group(1)
    # 循环上界必须是 here（不含），格位 1 时循环体一次都不执行
    assert re.search(r"<\s*here", body), (
        f"循环上界应为 here（不含），使格位 1 无可引用项：{body[:300]}")


def test_filled_seat_carries_evidence_sheet(tmp_path):
    """已定座位要带上它的 2×2 证据图，供发现混入别人的格。

    用户实测：文化4组 order8 的 2×2 里有一格不是目标人（跟踪串号——四格
    来自同一条 track 的四帧，某一帧的框实际框到了别人）。四格红框标的都是
    同一个 track，模型无从分辨哪格才是目标，会把两个人的衣着混起来判断，
    且不报错。所以必须让用户在标注页就看到这张图。

    范围是**每个已定座位**（就座/递补/切分），不只多人框——串号可能发生在
    任何一条 track 上。
    """
    frame = _write_frame(tmp_path)
    seg = mkseg(order=5, presenter_num=2)
    dets = track(11, RIGHT_CX + 100, STANDING_H, 8, frame, order=5, kind="seated")
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={5: {11: 1}})
    blob = _segments_blob_from_html(out.read_text(encoding="utf-8"))[0]
    f0 = blob["filled_seats"][0]
    assert f0.get("sheet"), "已定座位应带 2×2 证据图"
    assert f0.get("n_cells"), "应给出实际格数，供渲染排除复选框"


def test_exclusion_checkboxes_rendered_per_cell(tmp_path):
    """每格一个「排除」复选框，勾掉的格不进 VLM 输入。"""
    frame = _write_frame(tmp_path)
    seg = mkseg(order=5, presenter_num=2)
    dets = track(11, RIGHT_CX + 100, STANDING_H, 8, frame, order=5, kind="seated")
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={5: {11: 1}})
    main = _run_page(out.read_text(encoding="utf-8"))["main"]
    assert "排除格" in main, f"应渲染逐格排除复选框，实得片段：{main[:400]}"


# --- 2×2 四格要能分给少于四个人（用户 2026-08-24 实测）---

def test_all_cells_get_rows_when_people_fewer_than_cells(tmp_path):
    """证据图有几格就给几行，不是有几个人给几行。

    用户实测：「一个图 2*2 里有 2 个人，还是只能标 2 个格位，而我希望将
    这四个格位分给 2 个人」。根因在 `cellRowsOf`：
        want = Math.min(peopleOf(rec), max)
    人数选 2 时只铺 2 行，格位 3/4 根本没有行可填——而 2×2 的四格全都是
    画面，每一格都得说清是谁。「同格位X」正是为「多格同一人」准备的，
    却因为行不存在而够不着。

    正确行为：行数 = 证据图的实际格数（留空的行不参与判定，不会误判成
    已确定），人数只决定「要不要展开格位分配」这一件事。
    """
    frame = _write_frame(tmp_path)
    seg = mkseg(order=5, presenter_num=3)
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=5)
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={5: {}})
    html = out.read_text(encoding="utf-8")

    cand = _segments_blob_from_html(html)[0]["candidates"][0]
    n_cells = cand["n_cells"]
    assert n_cells == 4, f"这条 track 应给出满四格的证据图，实得 {n_cells}"

    ck = f"5::{cand['stable_id']}"
    got = _run_js_with_store(
        html, {ck: {"role": "创业者", "seat": "1", "n_people": "2"}},
        "cellRowsOf(SEGMENTS[0], SEGMENTS[0].candidates[0], "
        "store[candKey(SEGMENTS[0], SEGMENTS[0].candidates[0])])")
    cells = [r["cell"] for r in got["rows"]]
    assert cells == [1, 2, 3, 4], (
        f"2×2 四格都该有行可填（人数选 2 也一样），实得 {cells}")


# ===========================================================================
# Task 56C · 一：座位「无可用画面」可标记（可取消），目录分三态
#
# 用户 2026-08-24：「如果某个座位的人没有合适的候选图，能标记下来，再抽帧
# 二次有更多的备选吗」。
#
# 此前某座位在所有候选里都找不到合适画面时，用户没有任何办法把这件事记下来
# ——它只会一直显示为「未确定」，与「还没标」混作一谈，下次打开页面无从
# 分辨哪些是真找不到、哪些只是没标完。而目录颜色正是用户决定「还要标哪些」
# 的唯一依据。
# ===========================================================================

def _nc_boxes(main_html: str) -> dict:
    """「本座位无可用画面」勾选项：{座位号: 是否勾上}。

    与 `_toc_rows` 同一套做法——先按 `data-seat` 切出每个勾选项，再只在该
    片段内找属性位上的 `checked`。整段裸着找 "checked" 会被
    `onchange="…this.checked)"` 里的那个词误判（Task 49 起卡片里到处都是
    这种事件代码），那是一眼看不出原因的测试失效。
    """
    out = {}
    for m in re.finditer(r'<label class="ncSeat[^"]*" data-seat="(\d+)">(.*?)</label>',
                         main_html, re.DOTALL):
        seat, inner = int(m.group(1)), m.group(2)
        tag = re.search(r"<input[^>]*>", inner)
        assert tag is not None, f"座位 {seat} 的勾选项里没有 <input>：{inner}"
        bare = re.sub(r'="[^"]*"', '=""', tag.group(0))
        out[seat] = "checked" in bare
    return out


def _nc_page(tmp_path, presenter_num=3, seats_by_order=None, entry_labels=None):
    """一段（order 5，需 presenter_num 人）+ 一个候选。默认座位 1 由就座
    阶段判定，座位 2/3 未定——正是「找不到合适候选」的现场。"""
    frame = _write_frame(tmp_path)
    seg = _seg_with_genders(5, presenter_num, [0, 1, 0][:presenter_num])
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=5)
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out,
                        seats_by_order=(seats_by_order if seats_by_order is not None
                                        else {5: {11: 1}}),
                        detections_fp="FP-56C", entry_labels=entry_labels or {})
    return out.read_text(encoding="utf-8")


def test_undetermined_seat_offers_no_candidate_checkbox(tmp_path):
    """每个**尚未确定**的座位都要有一个「本座位无可用画面」的勾选项；
    已确定的座位不给（给了就是邀请用户把已经定好的人标成没画面）。"""
    html = _nc_page(tmp_path)
    boxes = _nc_boxes(_run_page(html)["main"])
    assert sorted(boxes) == [2, 3], (
        f"座位 1 已由就座阶段定下，只有 2、3 该有勾选项，实得 {sorted(boxes)}")
    assert boxes == {2: False, 3: False}, (
        "没标过的座位一律不得预勾——预填等于把「没判定」伪装成「已判定」")


def test_no_candidate_mark_is_reversible(tmp_path):
    """勾错必须能勾回来。

    用户此前明确提过「标了创业者之后目前无法取消标注」这类坑：只写不删的
    标记等于一个单向阀门，勾错一次这一格就永远脏着。
    """
    html = _nc_page(tmp_path)
    on = _run_page(html, {}, after="toggleNoCandidate(5,2,true);")
    assert _nc_boxes(on["main"])[2] is True, "勾上之后重算，该项要显示为已勾"
    assert on["written"]["nocand::5::2"]["no_candidate"] is True, (
        f"勾选必须写回 localStorage，实得 {on['written']}")

    off = _run_page(html, {"nocand::5::2": {"no_candidate": True}},
                    after="toggleNoCandidate(5,2,false);")
    assert _nc_boxes(off["main"])[2] is False, "取消勾选后重算，该项必须变回未勾"
    assert off["written"]["nocand::5::2"]["no_candidate"] is False, (
        "取消必须**显式写 false** 落盘，不能只从内存里删——文件里那份还在，"
        f"下次打开又勾上了，实得 {off['written']}")


def test_no_candidate_exported_as_side_channel(tmp_path):
    """导出的 JSON 里走顶层旁路通道 `no_candidate_seats`，键是 (order, seat)。"""
    html = _nc_page(tmp_path)
    payload = _run_js_with_store(
        html, {"nocand::5::2": {"no_candidate": True},
               "nocand::5::3": {"no_candidate": False}},
        "buildLabelsPayload()")
    assert payload.get("no_candidate_seats") == {"5": [2]}, (
        f"只导出勾上的那些；取消掉的不写（写了下游会以为它还标着），"
        f"实得 {payload.get('no_candidate_seats')!r}")


def test_no_candidate_not_exported_when_nothing_marked(tmp_path):
    """一个都没勾时**不写这个键**——空字典会让 run_all 的
    `if entry_labels:` 把「一条标注都没有」的文件当成真值。"""
    html = _nc_page(tmp_path)
    payload = _run_js_with_store(html, {}, "buildLabelsPayload()")
    assert "no_candidate_seats" not in payload, payload


def test_determined_seat_is_not_exported_as_no_candidate(tmp_path):
    """座位后来定下来了，那条陈旧的标记不再导出——自清理。

    否则用户先标「没画面」、后来又找到人补上，文件里两种说法并存，
    加密抽帧还会去重抽一段其实已经不缺人的窗口。
    """
    html = _nc_page(tmp_path)
    blob = _segments_blob_from_html(html)[0]
    ck = f"{blob['order']}::{blob['candidates'][0]['stable_id']}"
    payload = _run_js_with_store(
        html, {"nocand::5::2": {"no_candidate": True},
               ck: {"role": "创业者", "seat": "2"}},
        "buildLabelsPayload()")
    assert "no_candidate_seats" not in payload, (
        f"座位 2 已被标成创业者，「无可用画面」这条应自动作废，实得 {payload}")


def test_no_candidate_does_not_count_as_determined(tmp_path):
    """「无可用画面」**不算已定**：目录不许因此变绿。

    变绿意味着「这段标完了、人都齐了」，而这里恰恰是一个人没找着。
    """
    html = _nc_page(tmp_path, presenter_num=2, seats_by_order={5: {11: 1}})
    rows = _toc_rows(_run_page(html, {"nocand::5::2": {"no_candidate": True}})["toc"])
    assert "toc-done-all" not in rows[0]["cls"], (
        f"标为无可用画面不等于座位已确定，实得 {rows[0]['cls']!r}")


def test_toc_distinguishes_three_states(tmp_path):
    """目录要能分辨三种状态：已定 / 未定 / 已标为无可用画面。

    用户靠目录颜色决定还要标哪些。「无可用画面」若与「还没标」同色，
    他每翻一组都得重新看一遍正文才知道这段到底还有没有事可做。
    """
    html = _nc_page(tmp_path, presenter_num=2, seats_by_order={5: {11: 1}})
    todo = _toc_rows(_run_page(html, {})["toc"])[0]
    marked = _toc_rows(_run_page(html, {"nocand::5::2": {"no_candidate": True}})["toc"])[0]
    html_done = _nc_page(tmp_path, presenter_num=2, seats_by_order={5: {11: 1, 12: 2}})
    done = _toc_rows(_run_page(html_done, {})["toc"])[0]

    classes = {todo["cls"], marked["cls"], done["cls"]}
    assert len(classes) == 3, (
        f"三种状态的目录行必须互不相同，实得 {classes}")
    assert "toc-nocand" in marked["cls"], marked["cls"]
    assert marked["mark"] not in (todo["mark"], done["mark"]) and marked["mark"], (
        "颜色之外必须另有非视觉标记（色觉障碍、截图、打印后颜色都失效），"
        f"实得 {marked['mark']!r} / 未定 {todo['mark']!r} / 已定 {done['mark']!r}")
    assert "无可用画面" in marked["text"], (
        f"目录行要写清有几个座位标了无可用画面，实得 {marked['text']!r}")


def test_partially_marked_segment_still_reads_as_todo(tmp_path):
    """三个座位缺两个、只标了一个「无可用画面」时，这一段仍是「还有事做」。

    只要还剩一个没看过的座位，就不能显示成「这段处理完了」。
    """
    html = _nc_page(tmp_path, presenter_num=3, seats_by_order={5: {11: 1}})
    row = _toc_rows(_run_page(html, {"nocand::5::2": {"no_candidate": True}})["toc"])[0]
    assert "toc-nocand" not in row["cls"], (
        f"还剩座位 3 没标，不该显示成已全部处理，实得 {row['cls']!r}")


def test_no_candidate_prefilled_from_labels_file(tmp_path):
    """上次导出的文件里的标记要回填到页面上。

    不回填的后果实测过（`position_exclusions` 那次）：换台电脑/换个浏览器
    打开，勾选框看起来全是空的，用户会重标一遍，或者以为标记丢了。
    """
    html = _nc_page(tmp_path, entry_labels={"no_candidate_seats": {"5": [3]}})
    boxes = _nc_boxes(_run_page(html)["main"])
    assert boxes == {2: False, 3: True}, (
        f"文件里标了座位 3，页面应回显为已勾，实得 {boxes}")


def test_local_uncheck_beats_labels_file(tmp_path):
    """页面上取消掉的，压过文件里那份旧的「已标记」——否则「能取消」
    这件事在下次打开页面时被文件悄悄撤销。"""
    html = _nc_page(tmp_path, entry_labels={"no_candidate_seats": {"5": [3]}})
    boxes = _nc_boxes(_run_page(html, {"nocand::5::3": {"no_candidate": False}})["main"])
    assert boxes[3] is False, f"页面上的取消必须压过文件，实得 {boxes}"


def test_export_count_ignores_side_channels(tmp_path):
    """导出提示条里的「已导出 N 条标注」只数真正的标注。

    顶层旁路通道（`cell_exclusions`、`no_candidate_seats`）与段号平级，
    按段展开去数会把它们算进标注条数——`no_candidate_seats` 的值还是数组，
    `Object.keys` 数的是下标，用户会看到一个凭空多出来的数字。
    """
    html = _nc_page(tmp_path)
    blob = _segments_blob_from_html(html)[0]
    ck = f"{blob['order']}::{blob['candidates'][0]['stable_id']}"
    n = _run_js_with_store(
        html,
        {ck: {"role": "创业者", "seat": "2"},
         "nocand::5::3": {"no_candidate": True}},
        "countExportedLabels(buildLabelsPayload())")
    assert n["labels"] == 1, f"只有 1 条真正的候选标注，实得 {n}"
    assert n.get("no_candidate") == 1, (
        f"「无可用画面」要单独报一个数（它不是标注、也不是座位否决），实得 {n}")


# --- 「上/下一组创业者」：整张卡片流转到相邻段去标（用户 2026-08-24 实测）---
#
# 用户原话两条，合起来是一个需求：
#   「选择下一组创业者后，目前还是无法指定是下一组创业者的几号位，并且无法
#     实时显示在下一组的候选框中」
#   「一旦选择『下一组创业者』该组图片可以立马流转到下一组图，在下一组进行标注」
#
# 所以座位号不在源段选——在**目标段**的候选区里选。那边才看得到该段已定的
# 是谁、还缺哪几个、各座位应为男应为女。源段留一张「已流转」的小卡片，能改
# 回来即可。


def _two_seg_page(tmp_path, tid=77, src_order=1, tgt_order=2,
                  tgt_presenter_num=3):
    """源段一个进出场候选 + 相邻段（座位 1 已由就座定下，2、3 空着）。"""
    frame = _write_frame(tmp_path)
    src = mkseg(order=src_order, presenter_num=1, company="源企业")
    tgt = mkseg(order=tgt_order, presenter_num=tgt_presenter_num, company="目标企业")
    dets = track(tid, RIGHT_CX + 200, STANDING_H, 6, frame, order=src_order)
    out = tmp_path / "e.html"
    render_entry_review([src, tgt], dets, out,
                        seats_by_order={tgt_order: {99: 1}})
    html = out.read_text(encoding="utf-8")
    blob = _segments_blob_from_html(html)
    src_idx = 0 if blob[0]["order"] == src_order else 1
    cand = blob[src_idx]["candidates"][0]
    return html, cand, src_idx


def test_next_group_candidate_shows_up_in_the_next_segments_list(tmp_path):
    """选了「下一组创业者」，这张候选**当场**出现在下一段的候选区里。

    此前它只是在源段被标了个角色就没了下文：源段的座位号栏写死
    `role==='创业者'` 才显示，别的角色整块藏起来；目标段更是完全不知道有这
    么个人。用户既指不了座位号，也看不到它——而 Python 侧恰恰要靠座位号才能
    把人流转过去，等于标了白标。
    """
    html, cand, _si = _two_seg_page(tmp_path)
    sid = cand["stable_id"]
    ck = f"1::{sid}"
    main = _run_page(html, {ck: {"role": "下一组创业者"}})["main"]

    # 目标段那一节里要出现这张候选（按 store 键认，键始终挂在**源段**上）
    tgt_part = main[main.index('id="seg-2"'):]
    assert ck in tgt_part, (
        f"标成「下一组创业者」之后，这张候选应出现在第 2 段的候选区里："
        f"{tgt_part[:600]}")
    assert "第 1 段" in tgt_part or "源企业" in tgt_part, (
        f"借来的卡片要写明它来自哪一段，否则用户不知道自己在标谁：{tgt_part[:600]}")


def test_borrowed_candidate_offers_the_target_segments_seats(tmp_path):
    """借来的卡片上，标「下一组创业者」的那一格列的是**目标段**的空位。

    列源段的空位就是把人指到了错的企业编制上；Python 侧按目标段的
    `presenter_num` 校验后跳过，只打印一行未落位，用户很难注意到。

    这条走**渲染出来的卡片**（端到端），池子本身由
    test_cell_seat_pool_follows_the_row_role 单独钉。
    """
    html, cand, _si = _two_seg_page(tmp_path, tgt_presenter_num=3)
    ck = f"1::{cand['stable_id']}"
    store = {ck: {"role": "下一组创业者", "n_people": "2",
                  "cells": [{"cell": 1, "role": "下一组创业者"}]}}
    card = _run_js_with_store(
        html, store,
        "borrowedCard(findSegByOrder(2), findSegByOrder(1), "
        "findSegByOrder(1).candidates[0])")
    rows = _cell_selects(card)
    seat_opts = [v for v, _t, _s in rows[0][2]["options"] if v]
    assert seat_opts == ["2", "3"], (
        f"第 2 段座位 1 已由就座定下，可选的是 2、3，实得 {seat_opts}")


def test_previous_group_flows_backwards(tmp_path):
    """「上一组创业者」流向 order-1。方向写死成 +1 是最容易犯的错。"""
    html, cand, _si = _two_seg_page(tmp_path, src_order=2, tgt_order=1,
                                    tgt_presenter_num=3)
    ck = f"2::{cand['stable_id']}"
    main = _run_page(html, {ck: {"role": "上一组创业者"}})["main"]
    tgt_part = main[main.index('id="seg-1"'):main.index('id="seg-2"')]
    assert ck in tgt_part, f"应流向 order-1 那一段：{tgt_part[:600]}"


def test_source_segment_says_the_card_flowed_out(tmp_path):
    """源段不再摆一张要人标的完整卡片，而是说清「已流转到第 N 段」。

    留着完整卡片会让用户在源段又标一遍座位号——那个座位号属于源段的编制，
    与目标段的座位号是两码事，标了不生效且不报错。
    """
    html, cand, si = _two_seg_page(tmp_path)
    ck = f"1::{cand['stable_id']}"
    card = _cand_card(html, {ck: {"role": "下一组创业者"}}, seg_idx=si)
    assert "已流转" in card, f"源段要说明它去哪了：{card[:600]}"
    assert "第 2 段" in card or "目标企业" in card, (
        f"要点名目标段，否则用户不知道去哪找：{card[:600]}")


def test_flowed_out_card_can_be_taken_back(tmp_path):
    """流转是可撤销的：角色单选仍在，改回「创业者」它就回到本段。

    用户此前明确提过「标了创业者之后目前无法取消标注」这类坑。一个只能送出
    去、送错了拿不回来的操作，等于逼他重新导出一份标注文件来改。
    """
    html, cand, si = _two_seg_page(tmp_path)
    ck = f"1::{cand['stable_id']}"
    card = _cand_card(html, {ck: {"role": "下一组创业者"}}, seg_idx=si)
    assert 'value="创业者"' in card, f"角色单选必须还在，能改回来：{card[:600]}"


def test_neighbour_role_at_the_edge_says_there_is_no_such_segment(tmp_path):
    """最后一段还标「下一组创业者」：没有下一段可指，说破。

    默不作声地当作已处理，用户会以为流转成功了——Python 侧会记一条未落位并
    打印，但那要等跑完整条流水线才看得到。
    """
    frame = _write_frame(tmp_path)
    only = mkseg(order=1, presenter_num=2, company="独苗企业")
    dets = track(78, RIGHT_CX + 200, STANDING_H, 6, frame, order=1)
    out = tmp_path / "e.html"
    render_entry_review([only], dets, out, seats_by_order={})
    html = out.read_text(encoding="utf-8")
    cand = _segments_blob_from_html(html)[0]["candidates"][0]
    ck = f"1::{cand['stable_id']}"
    card = _cand_card(html, {ck: {"role": "下一组创业者"}})
    assert "没有第 2 段" in card or "没有下一段" in card, (
        f"越界要说破，不能装作流转成功：{card[:600]}")


def test_cross_segment_assignment_counts_toward_the_target_segment(tmp_path):
    """在借来的卡片上选了座位 2，**当场**算进目标段的进度。

    不算进去的话，目标段的目录一直是红的、「还缺」也照旧列着座位 2，用户会
    跑回去把同一个人重复标一遍。
    """
    html, cand, _si = _two_seg_page(tmp_path)
    ck = f"1::{cand['stable_id']}"
    got = _run_js_with_store(
        html, {ck: {"role": "下一组创业者", "seat": "2"}},
        "determinedSeats(findSegByOrder(2))")
    assert got == [1, 2], (
        f"目标段应有就座定下的 1 + 流转来的 2，实得 {got}")


def test_cross_segment_assignment_does_not_count_toward_its_own_segment(tmp_path):
    """标的是隔壁的人，**不算**本段的进度——算进去本段会提前变绿，
    而它自己那个座位其实还空着。"""
    html, cand, _si = _two_seg_page(tmp_path)
    ck = f"1::{cand['stable_id']}"
    got = _run_js_with_store(
        html, {ck: {"role": "下一组创业者", "seat": "2"}},
        "determinedSeats(findSegByOrder(1))")
    assert got == [], f"源段不该因为标了隔壁的人而算作已定，实得 {got}"


# --- 已定座位的替换控件：改名 + 去掉备注框（用户 2026-08-24）---

def test_seat_override_checkbox_is_labelled_replace(tmp_path):
    """勾选项文字用「是否替换」，不再是「这个座位判错了」。

    用户 2026-08-24：「自动匹配的『这个座位判错了』，提示词改为『是否替换』
    即可」。这一栏的实际用途多数不是「机器判错了」，而是「我找到了更清楚的
    一张」——写成「判错了」会让人以为必须先认定它错才能换。
    """
    frame = _write_frame(tmp_path)
    seg = mkseg(order=5, presenter_num=2)
    dets = track(11, RIGHT_CX + 100, STANDING_H, 8, frame, order=5, kind="seated")
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={5: {11: 1}})
    html = out.read_text(encoding="utf-8")
    box = _run_js_with_store(html, {}, "seatOverrideControls(SEGMENTS[0], 1)")
    assert "是否替换" in box, f"勾选项应写「是否替换」，实得：{box}"
    assert "判错" not in box, f"不该再出现「判错」字样，实得：{box}"


def test_seat_override_has_no_free_text_note(tmp_path):
    """替换那一栏不再有备注填空框。

    用户 2026-08-24：「如果后面人为发现更清楚的进行替换，不需要备注的填空框
    了」。它本来也没人读——`seat_overrides` 的消费方只看 `rejected` 与
    `replacement_stable_id`，那个框写什么都不影响任何结果，纯粹占地方。
    """
    frame = _write_frame(tmp_path)
    seg = mkseg(order=5, presenter_num=2)
    dets = track(11, RIGHT_CX + 100, STANDING_H, 8, frame, order=5, kind="seated")
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={5: {11: 1}})
    html = out.read_text(encoding="utf-8")
    box = _run_js_with_store(html, {}, "seatOverrideControls(SEGMENTS[0], 1)")
    assert 'type="text"' not in box, f"备注填空框应已去掉，实得：{box}"
    assert "备注" not in box, f"「备注」标签也该一并去掉，实得：{box}"


# --- 用户 2026-08-24 三条界面/顺序调整 ---

def test_seat_override_has_no_replacement_dropdown(tmp_path):
    """替换那一栏只留「是否替换」，不再有「改用我指定的候选」下拉。

    用户原话：「『改用我指定的候选』哪个选项也是冗余的，删掉即可，根据我
    标注的进出场等进行选择即可」。确实冗余：勾了「是否替换」这个座位就被
    视为空缺，谁来补由用户在下面进出场候选里标的那份决定——同一件事让他在
    两个地方各说一遍，两处说法不一致时还得再定一套优先级。
    """
    frame = _write_frame(tmp_path)
    seg = mkseg(order=5, presenter_num=2)
    dets = track(11, RIGHT_CX + 100, STANDING_H, 8, frame, order=5, kind="seated")
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={5: {11: 1}})
    html = out.read_text(encoding="utf-8")
    box = _run_js_with_store(html, {}, "seatOverrideControls(SEGMENTS[0], 1)")
    assert "改用我指定的候选" not in box, f"该下拉应已删除，实得：{box}"
    assert "replacement_stable_id" not in box, (
        f"控件删了就不该还往 replacement_stable_id 里写，实得：{box}")
    assert "是否替换" in box, f"「是否替换」要留着：{box}"


def test_candidate_headings_are_big_and_red(tmp_path):
    """「进场候选」「退场候选」要显眼：字大、红色。

    用户原话：「进场候选 退场候选 几个字再大点，用红色，进行提醒」。这两行
    是分隔两个时间窗口的唯一标志，混过去就会把退场的人当成进场的人标。
    """
    html = _render_one(tmp_path)
    assert '<h3 class="kindTitle">进场候选</h3>' in html, (
        "两个标题要带上可加样式的类名，否则没法单独放大变红")
    decls = _css_decls(html, ".kindTitle")
    assert "red" in decls or "#c" in decls.lower() or "#e" in decls.lower(), (
        f".kindTitle 要是红色，实得 {decls!r}")
    m = re.search(r"font-size:\s*([\d.]+)(rem|em|px)", decls)
    assert m, f".kindTitle 要显式放大字号，实得 {decls!r}"
    size, unit = float(m.group(1)), m.group(2)
    big = (size >= 1.3) if unit in ("rem", "em") else (size >= 20)
    assert big, f"字号要明显大于正文，实得 {m.group(0)}"


def test_candidates_are_shown_in_time_order(tmp_path):
    """同一个窗口里的候选按**时间从前到后**排，不是按框高。

    用户原话：「所有待标注的图片呈现的顺序是从前到后吗…按照从前到后的时间
    顺序，方便标注」。进场窗口起自本段开始前 25 秒，因此它里面**先**是上一
    家还没走完的人、**后**才是本段进来的人；按框高排会把两拨人打乱交错，
    人工一路标下来根本对不上现场发生的顺序。

    截断仍按框高（挑哪几个给人看是另一回事，见 config.ENTRYEXIT_TOP_N），
    只是**展示顺序**改成时间。
    """
    from pipeline.entryexit import entryexit_candidates

    frame = _write_frame(tmp_path)
    seg = mkseg(order=2, presenter_num=3)
    # 故意让「时间靠后的人框更大」：按框高排会把它排到最前，按时间排则在后。
    dets = (track(31, RIGHT_CX + 100, STANDING_H * 0.6, 4, frame, order=2)
            + track(32, RIGHT_CX + 400, STANDING_H, 4, frame, order=2))
    for d in dets:
        if d.track_id == 32:
            object.__setattr__(d, "t", d.t + 100.0)

    cands = [c for c in entryexit_candidates(dets, seg, frame_w=W)
             if c["kind"] == "entry"]
    assert len(cands) == 2, f"夹具前提：应有两个进场候选，实得 {len(cands)}"
    assert cands[0]["t_mid"] < cands[1]["t_mid"], (
        f"应按时间从前到后，实得 t_mid={[c['t_mid'] for c in cands]}、"
        f"框高={[c['median_h'] for c in cands]}")


def test_export_bar_says_where_to_save_the_json(tmp_path):
    """导出按钮旁要写明**存到哪个目录**。

    浏览器只能弹「另存为」，选不了默认目录；用户不知道该存哪，多半会落进
    「下载」文件夹，而流水线在 `work/Output/<组>/` 那里找（Task 56G）。找不到
    的表现是「一条标注都没有」——流水线照常跑完、结果全退回就座基线，一处不
    报错。这一句提示是这条链上唯一能拦住它的地方。
    """
    html = _render_one(tmp_path)
    # 先剥掉 <!-- --> 注释再断言：解释这段代码为什么存在的注释里同样写着
    # 「work/Output/<组>/」，不剥就会匹配到注释、把「提示删了」测成通过
    # （变异测试实测存活；`_css_decls` 当年也栽在同一件事上）。
    visible = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    bar = re.search(r'<div class="exportbar">(.*?)</div>', visible, re.DOTALL)
    assert bar, "未找到导出悬浮条"
    assert "Output" in bar.group(1), (
        f"要写明存到 work/Output/<组>/ 下，实得：{bar.group(1)}")


# --- 借来的卡片要能逐格指定身份与座位（用户 2026-08-25 实测新能源2 order6）---

def test_borrowed_card_offers_roles_and_cells(tmp_path):
    """从相邻段流转过来的卡片，也要有角色单选和格位分配。

    用户实测（新能源2组 order 6）：那张 2×2 里挤了不止一个人，而借来的卡片
    只给了一个「本段座位号」下拉——一张图只能指一个人，另外几个无从表达。

    借来的卡片就是完整的候选卡片，只是显示在**目标段**里、并带一条来源徽标。
    """
    html, cand, _si = _two_seg_page(tmp_path)
    ck = f"1::{cand['stable_id']}"
    card = _run_js_with_store(
        html, {ck: {"role": "下一组创业者", "n_people": "2"}},
        "borrowedCard(findSegByOrder(2), findSegByOrder(1), "
        "findSegByOrder(1).candidates[0])")
    assert "来自第 1 段" in card, f"来源徽标要留着：{card[:300]}"
    assert 'value="创业者"' in card, f"角色单选要有：{card[:600]}"
    assert "格位" in card, f"人数≥2 时格位分配要展开：{card[:800]}"


def test_cell_seat_pool_follows_the_row_role(tmp_path):
    """格位那一行的座位号，池子跟着**这一行的身份**走。

    一行标「创业者」指的是**本段**的座位；标「上/下一组创业者」指的是相邻段
    的座位。此前一律取本段的空位池——把人指到了错的企业编制上，Python 侧按
    目标段的 presenter_num 校验后跳过，只打印一行，用户很难注意到。
    """
    html, cand, _si = _two_seg_page(tmp_path, tgt_presenter_num=3)
    ck = f"1::{cand['stable_id']}"
    store = {ck: {"n_people": "2", "cells": [
        {"cell": 1, "role": "创业者", "seat": None},
        {"cell": 2, "role": "下一组创业者", "seat": None},
    ]}}
    got = _run_js_with_store(
        html, store,
        "[seatPoolFor(findSegByOrder(1),'创业者'),"
        " seatPoolFor(findSegByOrder(1),'下一组创业者'),"
        " seatPoolFor(findSegByOrder(1),'上一组创业者')]")
    own, nxt, prv = got
    assert own == [1], f"第 1 段自己只有 1 个座位且没人占，实得 {own}"
    assert nxt == [2, 3], f"第 2 段座位 1 已由就座定下，可选 2、3，实得 {nxt}"
    assert prv == [], f"没有第 0 段，应给空池，实得 {prv}"


# --- 折叠区的候选也能标（用户 2026-08-25：order4 那个人就在里面）---

def _filtered_page(tmp_path):
    """一个展示候选 + 一个被高度规则挡进折叠区的候选。"""
    frame = _write_frame(tmp_path)
    seg = mkseg(order=3, presenter_num=2)
    dets = (track(60, RIGHT_CX + 200, STANDING_H, 6, frame, order=3, kind="exit")
            + track(61, RIGHT_CX + 500, STANDING_H * 0.12, 8, frame, order=3,
                    kind="exit", width=12.0))
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={3: {}})
    html = out.read_text(encoding="utf-8")
    blob = _segments_blob_from_html(html)[0]
    assert blob["filtered"], "夹具前提：要有被挡进折叠区的候选"
    return html, blob


def test_filtered_candidate_carries_a_stable_id(tmp_path):
    """折叠区的候选要带稳定标识——没有它就无从存标注、重检测后更无从对齐。"""
    _html, blob = _filtered_page(tmp_path)
    c = blob["filtered"][0]
    for k in ("stable_id", "median_cx", "median_y2", "n_cells"):
        assert k in c, f"折叠区候选缺字段 {k}：{sorted(c)}"
    assert "@" in c["stable_id"] and "#" in c["stable_id"], c["stable_id"]


def test_filtered_candidate_is_labelable(tmp_path):
    """折叠区的卡片要有角色单选、座位号、人数/格位——和正常候选一样。

    用户 2026-08-25 实测城市2 order 4：密抽之后那个人确实在候选里，只是框高
    占比 0.096（阈值 0.25）被挡了。原设计是「只读，看一眼确认没误挡」，前提
    是被挡的都是手或太小认不出；实际恰恰相反，真人就在里面而他没法标。
    """
    html, blob = _filtered_page(tmp_path)
    card = _run_js_with_store(html, {},
                              "filteredCard(SEGMENTS[0], SEGMENTS[0].filtered[0])")
    assert 'value="创业者"' in card, f"要有角色单选：{card[:600]}"
    assert "座位号" in card, f"要有座位号：{card[:600]}"
    assert "人数" in card, f"要有人数（多人框同样可能被挡）：{card[:600]}"


def test_filtered_card_still_shows_why_it_was_filtered(tmp_path):
    """能标了，但「为什么被挡」不能丢——用户要据此判断自己是不是在救一个
    本就该挡掉的东西（比如只拍到一只手）。"""
    html, blob = _filtered_page(tmp_path)
    card = _run_js_with_store(html, {},
                              "filteredCard(SEGMENTS[0], SEGMENTS[0].filtered[0])")
    assert "被挡原因" in card, f"原因必须留着：{card[:600]}"


def test_labelling_a_filtered_candidate_counts_and_exports(tmp_path):
    """在折叠区标的创业者，当场算进本段进度，并且能导出。

    只在页面上能点、导出时丢掉，等于白标——而丢掉是不报错的。
    """
    html, blob = _filtered_page(tmp_path)
    ck = f"3::{blob['filtered'][0]['stable_id']}"
    got = _run_js_with_store(html, {ck: {"role": "创业者", "seat": "1"}},
                             "determinedSeats(SEGMENTS[0])")
    assert got == [1], f"折叠区标的人要算进已定座位，实得 {got}"
    payload = _run_js_with_store(html, {ck: {"role": "创业者", "seat": "1"}},
                                 "buildLabelsPayload()")
    assert blob["filtered"][0]["stable_id"] in payload["3"], (
        f"导出里必须带上这条，实得 {list(payload.get('3', {}))}")


# --- 勾了「排除格N」的已定座位要能被指过去补格（用户 2026-08-25）---

def _seat_with_exclusion_page(tmp_path, order=8, seat=1, presenter_num=3):
    frame = _write_frame(tmp_path)
    seg = mkseg(order=order, presenter_num=presenter_num)
    dets = (track(11, RIGHT_CX + 100, STANDING_H, 8, frame, order=order,
                  kind="seated")
            + track(60, RIGHT_CX + 300, STANDING_H, 6, frame, order=order,
                    kind="exit"))
    out = tmp_path / "e.html"
    render_entry_review([seg], dets, out, seats_by_order={order: {11: seat}})
    return out.read_text(encoding="utf-8")


def test_seat_with_excluded_cells_becomes_selectable(tmp_path):
    """勾了「排除格N」的已定座位，要重新出现在座位号下拉里。

    用户实测两处（文化4 order 8 座位1、城市2 order 16 座位1）：剔掉某一格之后
    想再指一张好图补上，下拉里那个座位却是灰的「1（已有人，不会覆盖）」。

    这条提示在**没有排除**时是对的——自动递补确实不该顶掉就座结果。但一旦
    用户勾了排除，情况就变了：座位上的人没换，只是他那张 2×2 少了一格，需要
    **同一个人**的另一张画面补上。补格位走的是 `labelled_seat_candidates`
    （它有意绕开占用判断），所以指过去是真的会生效的——下拉却不让选。
    """
    html = _seat_with_exclusion_page(tmp_path)
    # 没勾排除时：座位 1 已有人，不可选（保持原行为）
    before = _run_js_with_store(html, {}, "effectiveVacant(SEGMENTS[0])")
    assert 1 not in before, f"没勾排除时座位 1 不该可选，实得 {before}"
    # 勾掉一格之后：座位 1 可选（指过去＝补那一格）
    after = _run_js_with_store(
        html, {"cellex::8::1": {"cells": [2]}}, "effectiveVacant(SEGMENTS[0])")
    assert 1 in after, f"勾了排除之后座位 1 应可选，实得 {after}"


def test_seat_option_says_it_will_top_up_not_overwrite(tmp_path):
    """那个座位在下拉里要写明「补格位」，不能只是一个光秃秃的号。

    用户看到的必须与实际发生的一致：指过去**不会顶掉**座位上的人，只是把
    被勾掉的那一格补回来。不写清楚，他会以为自己在替换整个人。
    """
    html = _seat_with_exclusion_page(tmp_path)
    blob = _segments_blob_from_html(html)
    cand = [c for c in blob[0]["candidates"] if c["kind"] == "exit"][0]
    card = _cand_card(html, {"cellex::8::1": {"cells": [2]},
                             f"8::{cand['stable_id']}": {"role": "创业者"}})
    assert "补格位" in card, f"要写明指过去是补格位，实得：{card[:900]}"


def test_filtered_section_stays_open_once_something_inside_is_labelled(tmp_path):
    """折叠区里只要有标过的候选，这一区就默认**展开**。

    用户实测城市2 order 4：在折叠区里认出人、标成创业者之后，页面一重绘那一区
    又收起来了——他看不到自己刚标的东西还在不在，只能每次手动展开去找。
    `<details>` 没有 open 属性时浏览器默认收起，而 render() 是整段重绘的。
    """
    html, blob = _filtered_page(tmp_path)
    sid = blob["filtered"][0]["stable_id"]
    closed = _run_js_with_store(html, {}, "filteredSection(SEGMENTS[0])")
    assert "<details" in closed and "open" not in closed.split(">")[0], (
        f"没标过时应保持收起（不干扰正常标注流程）：{closed[:200]}")
    opened = _run_js_with_store(
        html, {f"3::{sid}": {"role": "创业者", "seat": "1"}},
        "filteredSection(SEGMENTS[0])")
    assert "open" in opened.split(">")[0], (
        f"里面有标过的候选就该默认展开：{opened[:200]}")


def test_filtered_section_summary_says_how_many_were_labelled(tmp_path):
    """折叠标题上写明「其中 N 个已标」——收起状态下也看得到自己救回过几个。"""
    html, blob = _filtered_page(tmp_path)
    sid = blob["filtered"][0]["stable_id"]
    got = _run_js_with_store(
        html, {f"3::{sid}": {"role": "创业者", "seat": "1"}},
        "filteredSection(SEGMENTS[0])")
    assert "已标" in got, f"标题要写明已标几个：{got[:400]}"


# --- 借来的卡片上要能直接指「本段创业者」（用户 2026-08-25 新能源2 order2）---

def test_borrowed_cell_offers_this_segments_presenter(tmp_path):
    """借来的卡片上，格位身份要能直接选「本段创业者」。

    用户实测（新能源2 order 2）：卡片流转过来了，提示写着「请到那一段的候选区
    里指定座位号」，可那一段每个格位**选不了创业者身份和座位号**。

    根因是语义：这张卡片属于**源段**，所以选项里的「创业者」指的是源段的座位，
    座位池也是源段的；要把人放进**当前这一段**，得选「上/下一组创业者」——而
    卡片就摆在当前段里，用户没有理由想到还要选一个带方向的词。

    所以借来的卡片上把方向那一项改写成「本段创业者（第N段）」，存进去的仍是
    Python 认得的那个带方向的角色，座位池取本段的空位。
    """
    html, cand, _si = _two_seg_page(tmp_path, tgt_presenter_num=3)
    ck = f"1::{cand['stable_id']}"
    card = _run_js_with_store(
        html, {ck: {"role": "下一组创业者", "n_people": "2"}},
        "borrowedCard(findSegByOrder(2), findSegByOrder(1), "
        "findSegByOrder(1).candidates[0])")
    rows = _cell_selects(card)
    opts = [(v, txt) for v, txt, _s in rows[0][1]["options"]]
    # 必须钉住**文字挂在哪个 value 上**。只查「有没有『本段创业者』这几个字」
    # 是不够的：方向算反时那几个字会跑到「上一组创业者」头上，照样命中
    # （变异测试实测存活）——而那会把人指到相反的那一段去。
    labelled = [v for v, txt in opts if "本段创业者" in txt]
    assert labelled == ["下一组创业者"], (
        f"卡片来自第 1 段、显示在第 2 段，「本段创业者」必须挂在"
        f"「下一组创业者」上，实得 {opts}")
    assert any(v == "创业者" and "第 1 段" in txt for v, txt in opts), (
        f"原本那个「创业者」要写明指的是源段，免得两个选项看着一样：{opts}")


def test_borrowed_cell_seat_pool_is_the_host_segment(tmp_path):
    """选了「本段创业者」之后，座位号列的是**本段**（目标段）的空位。"""
    html, cand, _si = _two_seg_page(tmp_path, tgt_presenter_num=3)
    ck = f"1::{cand['stable_id']}"
    card = _run_js_with_store(
        html, {ck: {"role": "下一组创业者", "n_people": "2",
                    "cells": [{"cell": 1, "role": "下一组创业者"}]}},
        "borrowedCard(findSegByOrder(2), findSegByOrder(1), "
        "findSegByOrder(1).candidates[0])")
    rows = _cell_selects(card)
    seats = [v for v, _t, _s in rows[0][2]["options"] if v]
    assert seats == ["2", "3"], (
        f"第 2 段座位 1 已由就座定下，可选 2、3，实得 {seats}")


def test_people_options_go_up_to_the_cell_count(tmp_path):
    """人数最多可选 4——2×2 就是四格，四个人各占一格是完全可能的。

    用户 2026-08-25：「每个 2*2 图片，人数的选择应该是最多可以选择 4 人吧，
    目前最多只能选 3 人」。上限写死成 3 的后果：四个人的框标不全，第四个人
    没有格位行可填，只能眼看着他没人认领。上限跟着 CELL_COUNT 走，不另写死。
    """
    html = _render_one(tmp_path)
    opts = _run_js_with_store(html, {}, "peopleOptions('1')")
    import re as _re
    vals = _re.findall(r'<option value="(\d+)"', opts)
    assert vals == ["1", "2", "3", "4"], f"人数应可选 1-4，实得 {vals}"


def test_four_people_get_four_cell_rows(tmp_path):
    """人数选 4 时四行格位都在（一格一人）。"""
    html, blob = _cell_page(tmp_path, presenter_num=4)
    ck = f"{blob['order']}::{blob['candidates'][0]['stable_id']}"
    rows = _cell_selects(_cand_card(html, {ck: {"n_people": "4"}}))
    assert len(rows) == 4, f"四个人四行，实得 {len(rows)}"


# --- 借来的卡片只有 1 人时也要能指座位号（用户 2026-08-25 新能源2 order2）---

def test_borrowed_single_person_can_get_a_seat(tmp_path):
    """流转过来的卡片只有 1 个人时，也要能直接指定座位号。

    用户实测（新能源2 order 2）：「当上一组流转过来的照片只有 1 人时，无法单独
    指定座位号，如果是多人，可以根据格位指定座位号」。

    根因：候选层的座位号那一栏写死了 `role==='创业者'` 才显示，而流转过来的
    卡片角色恰恰是「上/下一组创业者」——整栏被藏起来；人数为 1 时又不展开格位
    分配，于是一个能填座位号的地方都没有。
    """
    html, cand, _si = _two_seg_page(tmp_path, tgt_presenter_num=3)
    ck = f"1::{cand['stable_id']}"
    card = _run_js_with_store(
        html, {ck: {"role": "下一组创业者", "n_people": "1"}},
        "borrowedCard(findSegByOrder(2), findSegByOrder(1), "
        "findSegByOrder(1).candidates[0])")
    assert "display:none" not in _seat_wrap_of(card), (
        f"流转过来的卡片必须能填座位号：{card[:800]}")
    opts = [v for v, _t, _s in _selects(_seat_wrap_of(card))[0]["options"] if v]
    assert opts == ["2", "3"], (
        f"座位池要取**本段**（第 2 段，座位 1 已由就座定下），实得 {opts}")


def test_seat_wrap_still_hidden_for_non_presenter_roles(tmp_path):
    """工作人员/专家仍不显示座位号——他们本就不占座位。

    **在借来的语境里也要测**：那一栏的显示条件是「本段创业者 或（借来的卡片
    且角色带方向）」，只在普通卡片上测的话，把条件放宽成「借来的卡片一律显示」
    照样能过（变异测试实测存活）。
    """
    html, cand, _si = _two_seg_page(tmp_path)
    ck = f"1::{cand['stable_id']}"
    plain = _cand_card(html, {ck: {"role": "工作人员"}}, seg_idx=0)
    assert "display:none" in _seat_wrap_of(plain), (
        f"普通卡片上工作人员不该出现座位号栏：{plain[:600]}")
    borrowed = _run_js_with_store(
        html, {ck: {"role": "工作人员"}},
        "borrowedCard(findSegByOrder(2), findSegByOrder(1), "
        "findSegByOrder(1).candidates[0])")
    assert "display:none" in _seat_wrap_of(borrowed), (
        f"借来的卡片上，角色若是工作人员同样不该出现座位号栏：{borrowed[:700]}")


def test_borrowed_role_label_names_the_direction_and_the_segment(tmp_path):
    """借来的卡片上那一项要写清**两件事**：属于本段、图来自哪一边。

    用户 2026-08-25：「新能源2 order2 有流转来自上一企业的截图，在这里提示词
    应写『本组创业者（来自上一段）』；order 6 有个『本组创业者（第6段）』，
    这个数字是自动的吗，要是在其他 order 也能显示正确吧」。

    数字确实是自动的（取 `hostSeg.order`），这条测试连**方向**一起钉住：图从
    后一段流过来时写「来自下一段」，从前一段流过来时写「来自上一段」。方向写
    反就是把人指到相反的那一段去。
    """
    # 源段 1 → 目标段 2：站在第 2 段看，图来自**上一段**
    html, cand, _si = _two_seg_page(tmp_path, src_order=1, tgt_order=2,
                                    tgt_presenter_num=3)
    ck = f"1::{cand['stable_id']}"
    card = _run_js_with_store(
        html, {ck: {"role": "下一组创业者", "n_people": "2"}},
        "borrowedCard(findSegByOrder(2), findSegByOrder(1), "
        "findSegByOrder(1).candidates[0])")
    opts = [(v, t) for v, t, _s in _cell_selects(card)[0][1]["options"]]
    hit = [t for v, t in opts if v == "下一组创业者"]
    assert hit and "第 2 段" in hit[0] and "来自上一段" in hit[0], (
        f"应写明「本段（第 2 段）· 图来自上一段」，实得 {hit}")

    # 源段 2 → 目标段 1：站在第 1 段看，图来自**下一段**
    html2, cand2, _si2 = _two_seg_page(tmp_path, src_order=2, tgt_order=1,
                                       tgt_presenter_num=3)
    ck2 = f"2::{cand2['stable_id']}"
    card2 = _run_js_with_store(
        html2, {ck2: {"role": "上一组创业者", "n_people": "2"}},
        "borrowedCard(findSegByOrder(1), findSegByOrder(2), "
        "findSegByOrder(2).candidates[0])")
    opts2 = [(v, t) for v, t, _s in _cell_selects(card2)[0][1]["options"]]
    hit2 = [t for v, t in opts2 if v == "上一组创业者"]
    assert hit2 and "第 1 段" in hit2[0] and "来自下一段" in hit2[0], (
        f"应写明「本段（第 1 段）· 图来自下一段」，实得 {hit2}")


def test_export_keeps_the_seat_for_neighbour_roles(tmp_path):
    """「上/下一组创业者」的座位号必须导出——它正是流转落位的唯一依据。

    用户 2026-08-25 实测新能源2：在流转过来的卡片上选了座位号，导出的文件里
    却是 `seat: null`，那条流转因此永远落不了地（`entryexit_fill_plan` 的跨段
    分支要求 `lab.get("seat")`），而页面上看着好好的、一处不报错。

    根因是导出那一句写死了 `rec.role==='创业者'` 才带座位号。带方向的角色同样
    占一个座位——只是占的是**相邻段**的座位。
    """
    html, cand, _si = _two_seg_page(tmp_path, tgt_presenter_num=3)
    ck = f"1::{cand['stable_id']}"
    payload = _run_js_with_store(
        html, {ck: {"role": "下一组创业者", "seat": "2", "n_people": "1"}},
        "buildLabelsPayload()")
    rec = payload["1"][cand["stable_id"]]
    assert rec["role"] == "下一组创业者", rec
    assert rec["seat"] == 2, (
        f"流转卡片上选的座位号必须导出，否则那条流转永远落不了地，实得 {rec}")


def test_export_still_drops_the_seat_for_staff_roles(tmp_path):
    """工作人员/不确定仍不带座位号——他们本就不占座位。

    残留一个座位号会让下游以为「这个座位有人了」，而那个人根本不是创业者。

    **「专家」2026-09-19 从这份名单里移出去了**，不是因为上面那句话不成立，
    而是因为专家的号已经换了意思：它不再是「第几个座位」，而是「第几位专家」
    （评审席 1..5），由 `judge_donations_from_candidates` 消费。担心的那种
    误认不会发生——`entryexit.py` 里每个读 `seat` 的分支都先卡 role
    （`role != "创业者": continue`），专家带着号也走不进座位分配。
    专家那一侧由 `test_a_judge_number_survives_export` 反过来钉住。
    """
    html, cand, _si = _two_seg_page(tmp_path)
    ck = f"1::{cand['stable_id']}"
    for role in ("工作人员", "不确定"):
        payload = _run_js_with_store(
            html, {ck: {"role": role, "seat": "2", "n_people": "1"}},
            "buildLabelsPayload()")
        rec = payload["1"][cand["stable_id"]]
        assert rec["seat"] is None, f"{role} 不该带座位号，实得 {rec}"


def test_borrowed_card_role_radios_say_this_segment(tmp_path):
    """借来的卡片上，**角色单选**也要写成「本段创业者」。

    用户 2026-08-25：「流转卡片的角色是应该改为『本组创业者』吧」。上一轮只改了
    格位下拉，候选层的角色单选还是原样——单人卡片根本不展开格位，用户看到的就
    是这排单选，上面写着「下一组创业者」，而卡片就摆在目标段里。

    存进去的值一字不变（Python 只认「上/下一组创业者」），只是显示成用户此刻
    脑子里的说法。
    """
    html, cand, _si = _two_seg_page(tmp_path, tgt_presenter_num=3)
    ck = f"1::{cand['stable_id']}"
    card = _run_js_with_store(
        html, {ck: {"role": "下一组创业者", "n_people": "1"}},
        "borrowedCard(findSegByOrder(2), findSegByOrder(1), "
        "findSegByOrder(1).candidates[0])")
    import re as _re
    radios = _re.findall(r'<input type="radio"[^>]*value="([^"]*)"[^>]*>\s*([^<]+)',
                         card)
    by_val = {v: t.strip() for v, t in radios}
    assert "下一组创业者" in by_val, f"存的值必须还是带方向的角色：{by_val}"
    assert "本段创业者" in by_val["下一组创业者"], (
        f"显示应写成「本段创业者」，实得 {by_val['下一组创业者']!r}")
    assert "第 2 段" in by_val["下一组创业者"], f"要带上本段段号：{by_val}"
    assert "第 1 段" in by_val.get("创业者", ""), (
        f"原来那个「创业者」要写明指的是源段，免得两项看着一样：{by_val}")


def test_plain_card_role_radios_are_unchanged(tmp_path):
    """普通卡片（不是借来的）上，角色单选一字不改。"""
    html = _render_one(tmp_path)
    card = _cand_card(html, {})
    import re as _re
    radios = _re.findall(r'<input type="radio"[^>]*value="([^"]*)"[^>]*>\s*([^<]+)',
                         card)
    for v, t in radios:
        assert t.strip() == v, f"普通卡片上不该改写，实得 {v!r} → {t.strip()!r}"


def test_neighbour_role_is_hidden_when_that_segment_does_not_exist(tmp_path):
    """指向不存在那一段的方向角色，不显示。

    用户 2026-08-25：「新能源2 order2 流转过来的图片，已经有了『第1段的创业者』，
    不需要『上一组创业者』这个选项了吧」。

    这张卡片属于第 1 段，「上一组创业者」指的是第 0 段——根本不存在，选了也只会
    在跑流水线时记一条「没有第 0 段」的未落位。同一条规则顺带管住首末两段：第一
    段不该有「上一组创业者」，最后一段不该有「下一组创业者」。

    选不中的选项就不该摆在那里——它只会让人以为自己漏了一种情况。
    """
    html, cand, _si = _two_seg_page(tmp_path, tgt_presenter_num=3)
    ck = f"1::{cand['stable_id']}"
    card = _run_js_with_store(
        html, {ck: {"role": "下一组创业者", "n_people": "1"}},
        "borrowedCard(findSegByOrder(2), findSegByOrder(1), "
        "findSegByOrder(1).candidates[0])")
    import re as _re
    vals = _re.findall(r'<input type="radio"[^>]*value="([^"]*)"', card)
    assert "上一组创业者" not in vals, (
        f"这张卡属于第 1 段，「上一组创业者」指向第 0 段，不该出现：{vals}")
    assert "下一组创业者" in vals and "创业者" in vals, vals


def test_a_selected_but_impossible_role_is_still_shown(tmp_path):
    """已经选中的那个角色即使指向不存在的段，也要显示出来。

    藏起来的话，下拉/单选里一个都不选中，用户看不出自己此前标过什么，更没法
    改回去——存着的值还在文件里，页面却装作没有。
    """
    frame = _write_frame(tmp_path)
    only = mkseg(order=1, presenter_num=2, company="独苗企业")
    dets = track(78, RIGHT_CX + 200, STANDING_H, 6, frame, order=1)
    out = tmp_path / "e.html"
    render_entry_review([only], dets, out, seats_by_order={})
    html = out.read_text(encoding="utf-8")
    cand = _segments_blob_from_html(html)[0]["candidates"][0]
    ck = f"1::{cand['stable_id']}"
    card = _cand_card(html, {ck: {"role": "下一组创业者"}})
    import re as _re
    vals = _re.findall(r'<input type="radio"[^>]*value="([^"]*)"', card)
    assert "下一组创业者" in vals, (
        f"选中的值必须还在，否则用户看不出自己标过什么、也改不回去：{vals}")
    assert "上一组创业者" not in vals, f"没选中且指向不存在的段，藏起来：{vals}"


def test_cell_role_dropdown_hides_impossible_neighbour_roles(tmp_path):
    """格位下拉同一套规则——两处显示不一致会让人以为其中一处坏了。"""
    html, cand, _si = _two_seg_page(tmp_path, tgt_presenter_num=3)
    ck = f"1::{cand['stable_id']}"
    card = _run_js_with_store(
        html, {ck: {"role": "下一组创业者", "n_people": "2"}},
        "borrowedCard(findSegByOrder(2), findSegByOrder(1), "
        "findSegByOrder(1).candidates[0])")
    vals = [v for v, _t, _s in _cell_selects(card)[0][1]["options"]]
    assert "上一组创业者" not in vals, f"指向第 0 段，不该出现：{vals}"
    assert "下一组创业者" in vals, vals


def test_borrowed_card_hides_the_opposite_direction_role(tmp_path):
    """借来的卡片上只留两种身份：本段的、源段的。反方向那个一律藏。

    用户 2026-08-26 实测城市2 order 16：流转过来的卡片上还有「上一组创业者」。

    上一版规则「目标段不存在就藏」只是**碰巧**修好了新能源2——那张卡属于第 1
    段，反方向指向第 0 段（不存在）。城市2 这张属于第 15 段，反方向指向第 14
    段，**存在**，于是照旧显示。

    正确的口径与「不存在」无关：这张卡摆在第 16 段里，它要回答的只有一个问题
    ——图里的人是**本段（16）**的，还是**源段（15）**的。指向第 14 段的选项与
    这张卡无关，选了只会把人流到一个跟这张图毫不相干的企业名下。
    """
    frame = _write_frame(tmp_path)
    segs = [mkseg(order=o, presenter_num=2, company=f"企业{o}")
            for o in (14, 15, 16)]
    dets = track(90, RIGHT_CX + 200, STANDING_H, 6, frame, order=15, kind="exit")
    out = tmp_path / "e.html"
    render_entry_review(segs, dets, out, seats_by_order={16: {99: 1}})
    html = out.read_text(encoding="utf-8")
    cand = next(c for s in _segments_blob_from_html(html) if s["order"] == 15
                for c in s["candidates"])
    ck = f"15::{cand['stable_id']}"

    # 角色**留空**：存着「下一组创业者」的话，「已选中的永远留着」那条会把它
    # 兜住，于是「连指向本段的那个也藏了」这种错照样能过（变异测试实测存活）。
    card = _run_js_with_store(
        html, {ck: {"role": "", "n_people": "1"}},
        "borrowedCard(findSegByOrder(16), findSegByOrder(15), "
        "findSegByOrder(15).candidates[0])")
    import re as _re
    vals = _re.findall(r'<input type="radio"[^>]*value="([^"]*)"', card)
    assert "上一组创业者" not in vals, (
        f"这张卡显示在第 16 段里，「上一组创业者」指向第 14 段——与本卡无关，"
        f"不该出现（第 14 段确实存在，所以「不存在就藏」那条规则管不住它）：{vals}")
    assert "下一组创业者" in vals, f"指向本段（16）的那个必须在：{vals}"
    assert "创业者" in vals, f"源段（15）的那个也要在：{vals}"

    cell_vals = [v for v, _t, _s in _cell_selects(_run_js_with_store(
        html, {ck: {"role": "", "n_people": "2"}},
        "borrowedCard(findSegByOrder(16), findSegByOrder(15), "
        "findSegByOrder(15).candidates[0])"))[0][1]["options"]]
    assert "上一组创业者" not in cell_vals, f"格位下拉同一套口径：{cell_vals}"


def test_plain_card_keeps_both_directions_when_both_exist(tmp_path):
    """普通卡片（不是借来的）上，前后两段都在时两个方向角色都要留着。

    「只留指向 host 的那个」只适用于**借来的**卡片——普通卡片没有 host，
    两个方向都是合法判断。
    """
    frame = _write_frame(tmp_path)
    segs = [mkseg(order=o, presenter_num=2, company=f"企业{o}")
            for o in (14, 15, 16)]
    dets = track(91, RIGHT_CX + 200, STANDING_H, 6, frame, order=15, kind="exit")
    out = tmp_path / "e.html"
    render_entry_review(segs, dets, out, seats_by_order={})
    html = out.read_text(encoding="utf-8")
    blob = _segments_blob_from_html(html)
    idx = [i for i, s in enumerate(blob) if s["order"] == 15][0]
    card = _cand_card(html, {}, seg_idx=idx)
    import re as _re
    vals = _re.findall(r'<input type="radio"[^>]*value="([^"]*)"', card)
    assert "上一组创业者" in vals and "下一组创业者" in vals, (
        f"普通卡片上前后两段都在，两个方向都该留着：{vals}")


# --- Task 56V：已定座位上的「这张图其实是座位 N」--------------------------

def test_filled_seat_offers_a_seat_reassign_dropdown(tmp_path):
    """已定座位卡片上要能就地改座位号。

    用户 2026-08-29 实测城市2 order 6：「自动标的 seat1 实际为 seat2，这样
    自动标错的如何调整，否则标错了，无法将该图用于其他座位」。

    此前这一路只能整体否决，改号得绕一圈：否决座位 1 → 去下面的进出场候选
    里找到同一个人 → 指给座位 2。那个人恰好没有好用的进出场候选时，这张本来
    很好的就座图就此浪费。
    """
    html = _filled_page_with_candidates(tmp_path, seats_by_order={5: {99: 1}})
    assert "seatReassign" in html, "缺「这张图其实是座位 N」这一项"
    assert 'onchange="saveReassign(' in html, (
        "改号下拉没接到 saveReassign——已定座位卡片是运行时由 JS 渲染的，"
        "这里断言的是那段源码")
    # 键的形状：与 seatOvrKey / cellExKey 同源，(段号, **原**座位号)。
    # track_id 作键会让改号在下次重检测后静默失效（Task 46）。
    m = re.search(r"function reassignKey\(seg, seat\)\{[^}]*\}", html)
    assert m, "缺 reassignKey"
    assert "'reassign::'+seg.order+'::'+seat" in m.group(0), (
        f"键必须由 (段号, 原座位号) 拼出，实得 {m.group(0)}")


def test_seat_reassign_dropdown_skips_the_seat_itself(tmp_path):
    """下拉里列本段其它座位号，**不列自己**。

    「改成自己」是没有意义的一项；留着它等于给用户一个什么都不做的选择，
    而下游还得专门把这种痕迹滤掉（`seat_reassign_of` 确实滤了，但那是兜底，
    不是让页面可以随便吐脏值的理由）。

    选项是运行时按 `presenter_num` 生成的，静态 HTML 里没有——这里钉的是
    那个循环里确实有跳过自己的那一句，以及上限取 `presenter_num`。
    """
    html = _filled_page_with_candidates(tmp_path, seats_by_order={5: {99: 1}})
    m = re.search(r"function seatReassignControl\(seg, seat\)\{(.*?)\n\}",
                  html, re.S)
    assert m, "找不到 seatReassignControl"
    body = m.group(1)
    assert "n <= seg.presenter_num" in body, (
        f"可选座位号的上限该是本段标注人数，实得 {body}")
    assert re.search(r"if \(n === seat\) continue;", body), (
        f"必须跳过自己这一项，实得 {body}")
    assert 'value=""' in body, "要有一项「不改」，否则用户选错了退不回去"


def test_export_writes_seat_reassign_as_a_top_level_key(tmp_path):
    """导出的 JSON 里 `seat_reassign` 与段号平级，形状是 {段: {原座位: 新座位}}。

    与 `seat_overrides` / `cell_exclusions` 同一层，`LABEL_RESERVED_KEYS`
    才认得它；写进某个段号底下会被当成一个候选的标注，静默失效。
    """
    html = _filled_page_with_candidates(tmp_path, seats_by_order={5: {99: 1}})
    assert "payload.seat_reassign" in html, (
        "导出函数里没有把改号写进 payload")
    # 空的不写：与另外三条通道逐字同一条规矩——挂一个空对象会让「一条标注都
    # 没有」的文件变成真值，下游打印「已启用」而实际什么都不做。
    m = re.search(r"if \(Object\.keys\(reassign\)\.length\) "
                  r"payload\.seat_reassign = reassign;", html)
    assert m, "空的改号表不许写进文件"


def test_setting_people_on_a_filtered_candidate_opens_the_cell_rows(tmp_path):
    """折叠区那张卡片把「人数」拨到 2，格位分配必须当场展开。

    用户 2026-09-08 实测城市1 order 2：折叠区里 entry track 20 那张 2×2 挤了
    不止一个人，人数拨到 2 之后**一行格位都没出来**，也没有任何提示。

    根因：`setPeople` / `saveCell` 拿 `(段号, stable_id)` 反查候选，而
    `findCandBySid` 只在 `seg.candidates` 里找——折叠区的候选在 `seg.filtered`
    里，反查落空，两个函数都**静默 return**。角色单选与座位号走的是
    `save(ck, ...)`（直接拿键、不反查），所以那两个控件一直是好的，坏的只有
    人数与格位这一半——于是折叠区的卡片看起来「能标」，实际标不了多人。
    """
    html, blob = _filtered_page(tmp_path)
    sid = blob["filtered"][0]["stable_id"]
    ck = "3::" + sid
    got = _run_page(html, {}, after=f"setPeople(3, {json.dumps(sid)}, '2')")
    assert got["store"].get(ck, {}).get("n_people") == "2", (
        f"人数没存进 store，`setPeople` 多半在反查候选那一步就 return 了："
        f"{got['store']}")
    parts = got["main"].split('<div class="filteredCand"')
    assert len(parts) == 2, "夹具前提：main 里恰有一张折叠区卡片"
    assert "cellsBox" in parts[1], (
        f"折叠区卡片上没有格位分配框：{parts[1][:800]}")
    n_rows = parts[1].count('class="cellRow"')
    assert n_rows == blob["filtered"][0]["n_cells"], (
        f"格位行数应等于证据图的格数 {blob['filtered'][0]['n_cells']}，实得 {n_rows}")


def test_choosing_a_role_for_one_cell_of_a_filtered_candidate_persists(tmp_path):
    """展开之后逐格选身份也得存得住——`saveCell` 与 `setPeople` 是同一条
    反查路径，只修一个等于人数能拨、选了不算数，而且照样不报错。"""
    html, blob = _filtered_page(tmp_path)
    sid = blob["filtered"][0]["stable_id"]
    ck = "3::" + sid
    got = _run_page(html, {}, after=(
        f"setPeople(3, {json.dumps(sid)}, '2');"
        f"saveCell(3, {json.dumps(sid)}, 1, 'role', '创业者');"
        f"saveCell(3, {json.dumps(sid)}, 1, 'seat', '1')"))
    cells = got["store"].get(ck, {}).get("cells") or []
    hit = [c for c in cells if c.get("cell") == 2]
    assert hit and hit[0].get("role") == "创业者" and hit[0].get("seat") == 1, (
        f"折叠区候选的逐格判定没存住：{cells}")
    assert got["written"] is not None and ck in got["written"], (
        "内存里存了但没写回 localStorage，重开页面就没了")


def _two_seg_filtered_page(tmp_path, src_order=1, tgt_order=2):
    """源段有一个展示候选 + 一个被挡进折叠区的候选，另有一个相邻段。"""
    frame = _write_frame(tmp_path)
    src = mkseg(order=src_order, presenter_num=1, company="源企业")
    tgt = mkseg(order=tgt_order, presenter_num=3, company="目标企业")
    dets = (track(77, RIGHT_CX + 200, STANDING_H, 6, frame, order=src_order,
                  kind="exit")
            + track(78, RIGHT_CX + 500, STANDING_H * 0.12, 8, frame,
                    order=src_order, kind="exit", width=12.0))
    out = tmp_path / "e.html"
    render_entry_review([src, tgt], dets, out, seats_by_order={tgt_order: {99: 1}})
    html = out.read_text(encoding="utf-8")
    blob = _segments_blob_from_html(html)
    src_idx = 0 if blob[0]["order"] == src_order else 1
    assert blob[src_idx]["filtered"], "夹具前提：源段要有被挡进折叠区的候选"
    return html, blob[src_idx]["filtered"][0], src_idx


def test_a_filtered_candidate_marked_next_group_reaches_the_next_segment(tmp_path):
    """折叠区里的候选标成「下一组创业者」，同样要流转到下一段去。

    与 `setPeople` 那条同一个根因家族：`borrowedCandidates` 只遍历
    `s.candidates`，折叠区的候选流转不过去。表现是源段写着「已流转到第 2 段，
    请到那一段指定座位号」，而第 2 段里根本没有这张卡片——用户按提示过去，
    找不到，也没有任何报错。座位号指不了，Python 侧就补不进去这个人。
    """
    html, cand, _si = _two_seg_filtered_page(tmp_path)
    ck = "1::" + cand["stable_id"]
    main = _run_page(html, {ck: {"role": "下一组创业者"}})["main"]
    tgt_part = main[main.index('id="seg-2"'):]
    assert ck in tgt_part, (
        f"折叠区标成「下一组创业者」的候选没有出现在第 2 段里："
        f"{tgt_part[:600]}")


def test_seat_missing_warning_counts_filtered_candidates_too(tmp_path):
    """折叠区里标了创业者却没选座位号，同样要计进「缺座位号」的提醒。

    凡有问题必须可见：这个数字是目录上那条「N 处标了创业者没选座位号」的
    来源。漏掉折叠区的话，用户把目录扫成全绿，而那个人一个也补不进去。
    """
    html, cand, _si = _two_seg_filtered_page(tmp_path)
    ck = "1::" + cand["stable_id"]
    got = _run_js_with_store(html, {ck: {"role": "创业者"}},
                             "segSeatMissingCount(findSegByOrder(1))")
    assert got == 1, f"折叠区那条没被计进缺座位号提醒，实得 {got}"


def test_filtered_section_stays_open_after_only_setting_the_people_count(tmp_path):
    """折叠区里刚把「人数」拨到 2，这一区必须**保持展开**。

    用户 2026-09-08 实测：「选择人数为 2 后，立马又折叠了，而不是让我继续勾选
    每个格位的人分别是谁」。`setPeople` 会触发整段重绘，而
    `filteredSection` 判断要不要带 `open` 时只看 `role` 与 `cells` ——此刻
    这条记录里只有 `n_people`，判为「没标过」，`<details>` 不带 open 就收起，
    刚展开的四行格位连同那张图一起被卷了回去。
    """
    html, blob = _filtered_page(tmp_path)
    sid = blob["filtered"][0]["stable_id"]
    opened = _run_js_with_store(html, {f"3::{sid}": {"n_people": "2"}},
                                "filteredSection(SEGMENTS[0])")
    assert "open" in opened.split(">")[0], (
        f"拨了人数就等于用户正在这一区里干活，不许收起：{opened[:200]}")


# --- Task 56V 续：改号之后，页面自己的座位账也要跟着改（用户 2026-09-08）---
#
#   「城市1 order4 一旦勾选『这张图其实是座位2』后，应当将该座位释放出来以供
#     选择，目前无法重新为座位1指定」
#
# 改号通道此前只做了两件事：存进 store、导出进 `seat_reassign`。页面自己的
# 「哪些座位还空着 / 哪些已经定了」一概不看它，于是原座位既选不了、又照旧
# 算作已定，目录跟着一起报错的数字。Python 侧（`registry` 的就座分支）的语义
# 是：改号后**原座位空出来**，**目标座位由这张图占着**，被顶掉的那位退回待标。
# 页面必须与它逐条一致，否则页面绿了而流水线补不上人。


def _reassigned(tmp_path, expr, src=1, dst=2):
    html = _filled_page_with_candidates(tmp_path)
    return _run_js_with_store(html, {f"reassign::5::{src}": {"seat": dst}}, expr)


def test_reassigning_a_filled_seat_frees_the_original_seat(tmp_path):
    """改号之后原座位要回到可选池里——那张图已经挪到别的号上了，这个号空了。"""
    got = _reassigned(tmp_path, "effectiveVacant(findSegByOrder(5))")
    assert 1 in got, f"原座位 1 没被释放出来，用户无法为它另指一个人：{got}"


def test_reassigning_a_filled_seat_moves_which_seat_counts_as_determined(tmp_path):
    """已定的是**目标**座位，不再是原座位。

    只放开可选池不改这一处的话，目录会一边说「座位 1 已定」、一边把座位 1
    列进可选，还继续把座位 2 算作缺人——三处口径互相打架。
    """
    got = _reassigned(tmp_path, "determinedSeats(findSegByOrder(5))")
    assert got == [2], f"改号后应是座位 2 已定、座位 1 待指认，实得 {got}"
    todo = _reassigned(tmp_path, "undeterminedSeats(findSegByOrder(5))")
    assert todo == [1], f"还缺的应是座位 1，实得 {todo}"


def test_reassign_target_stays_selectable_as_a_merge_target(tmp_path):
    """目标座位仍可选，且文案写明「不换人」。

    改号过来的那张图与用户在目标座位上标的候选是**合并**关系（Task 56W：
    「自动标注的座位1改为座位2……应与后边人工标注的座位2进行合并」）。
    直接把目标座位从池子里拿掉，就把那条路堵死了。

    夹具**两个座位都已定**（vacant_seats 为空）：目标座位本来就空着的话，
    它在池子里与这条改动无关，这条断言就是白写的——变异测试实测：去掉
    「把目标座位放进池子」那一句，用空座位当目标时一个测试都不红。
    """
    html = _filled_page_with_candidates(tmp_path, seats_by_order={5: {99: 1, 98: 2}})
    store = {"reassign::5::1": {"seat": 2}}
    blob = _segments_blob_from_html(html)[0]
    assert blob["vacant_seats"] == [], (
        f"夹具前提：两个座位都已定，池子本该是空的，实得 {blob['vacant_seats']}")
    vac = _run_js_with_store(html, store, "effectiveVacant(findSegByOrder(5))")
    assert 2 in vac, f"目标座位要留在池子里供合并，实得 {vac}"
    assert 1 in vac, f"原座位同样要释放，实得 {vac}"
    tops = _run_js_with_store(html, store, "topUpSeats(findSegByOrder(5))")
    assert 2 in tops, (
        f"目标座位上已经有改号过来的图，文案必须说明指过去不换人，实得 {tops}")
    done = _run_js_with_store(html, store, "determinedSeats(findSegByOrder(5))")
    assert done == [2], (
        f"被顶掉的那位退回待标，本段只剩座位 2 有主，实得 {done}")


def test_clearing_the_reassignment_puts_the_seat_back(tmp_path):
    """撤回改号（选「不改」＝删掉整条记录），账要原样退回去。"""
    html = _filled_page_with_candidates(tmp_path)
    got = _run_js_with_store(html, {}, "determinedSeats(findSegByOrder(5))")
    assert got == [1], f"没改号时就座定的是座位 1，实得 {got}"
    vac = _run_js_with_store(html, {}, "effectiveVacant(findSegByOrder(5))")
    assert 1 not in vac, f"没改号时座位 1 有人，不该在可选池里：{vac}"


def test_a_reassignment_pointing_at_itself_is_ignored(tmp_path):
    """「改成自己」不作数——那是用户点开下拉又选回原值的痕迹。

    判据必须与 Python 的 `entryexit.seat_reassign_of` **逐条一致**（跳过脏值、
    跳过指向自己、不另设上界）。两边不一致的表现是页面算出一套座位账、
    流水线算出另一套，而没有任何一处报错。
    """
    got = _reassigned(tmp_path, "determinedSeats(findSegByOrder(5))", dst=1)
    assert got == [1], f"改成自己应当忽略、维持原状，实得 {got}"
    # 「已定」那一项两边算出来一样（挪走再放回同一个号），分不出真假——
    # 分得出的是这两项：座位 1 上的人没动，不该被释放进可选池；导出里也
    # 不该留下这条什么都不做的痕迹（下游会以为这个座位被人工处理过）。
    vac = _reassigned(tmp_path, "effectiveVacant(findSegByOrder(5))", dst=1)
    assert 1 not in vac, f"座位 1 上的人没动，不该被释放：{vac}"
    payload = _reassigned(tmp_path, "buildLabelsPayload()", dst=1)
    assert "seat_reassign" not in payload, (
        f"「改成自己」不该写进导出：{payload.get('seat_reassign')}")


def test_export_actually_carries_the_reassignment(tmp_path):
    """导出里真的带着改号，形状 {段: {原座位: 新座位}}。

    此前这一路只有「源码里有没有那几行」的断言——改号通道整个走样（页面的
    座位账一概不看它）而全绿。行为断言才拦得住。
    """
    html = _filled_page_with_candidates(tmp_path)
    payload = _run_js_with_store(html, {"reassign::5::1": {"seat": 2}},
                                 "buildLabelsPayload()")
    assert payload.get("seat_reassign") == {"5": {"1": 2}}, (
        f"导出里没带上改号：{payload.get('seat_reassign')}")
    none = _run_js_with_store(html, {}, "buildLabelsPayload()")
    assert "seat_reassign" not in none, (
        "一条改号都没有时不许写这个键——空对象会让下游以为人工处理过")


# ---------------------------------------------------------------------------
# 候选图的画质：默认不压（用户 2026-09-08：「进入候选的图像是蒙了一层纱，
# 是有什么处理吗，用原始的抽帧图」）。
#
# 查下来「纱」本身是**源帧的运动模糊**（人贴着镜头快走、室内光暗），不是处理
# 加上去的——`build_evidence_sheet` 只裁剪加红框，不做白平衡、不做任何滤镜。
# 但顺着查出来一件真事：城市1 那一版页面里 183 张候选图**有 162 张被悄悄降到
# 700px 并重编码成 JPEG q85**，因为整页超了 12MB 的体积预算，而
# `_fit_images_to_budget` 的返回值被丢掉了，页面上、命令行里都没有一个字。
#
# 这不只是好看不好看：VLM 拿到的是 PNG 原图，人工在页面上核对的是压过的那张
# ——「人核对的图与模型看到的图不是同一张」正是本项目最忌讳的那一类。
# ---------------------------------------------------------------------------

def test_candidate_images_are_never_recompressed(tmp_path):
    """候选图保持 `build_evidence_sheet` 产出的 PNG 原样——**没有例外**。

    原先这条是「预算够用时不压」，另有一条测「超了预算要在页面上说出来」。
    图改成写文件之后（用户 2026-09-18）体积预算整个取消了：没有上限，也就
    没有「悄悄压一张」这件事可言。降级路径消失比「降级时要声明」更强。
    """
    html = _filled_page_with_candidates(tmp_path)
    blob = _segments_blob_from_html(html)[0]
    srcs = [c["img"] for c in blob["candidates"] if c.get("img")]
    assert srcs, "候选一张图都没有"
    for src in srcs:
        assert src.endswith(".png"), f"候选图被重编码成别的格式了：{src}"
        assert (tmp_path / src).read_bytes().startswith(b"\x89PNG"), src


def test_the_page_says_it_is_no_longer_self_contained(tmp_path):
    """页面必须写明图在同目录的文件夹里、转交要整个打包。

    凡有代价必须可见：不说的话，用户按老习惯只把那个 html 发出去，对方打开
    满屏是叉——而这一步在他那边什么都不报。
    """
    html = _filled_page_with_candidates(tmp_path)
    assert "review_img/" in html
    assert "打包" in html, "页面上没说转交要整个目录一起打包"
    assert "data:image" not in html


def test_labels_are_matched_against_the_folded_candidates_too(tmp_path):
    """重检测后按内容重新对齐时，配对池必须是**展示 ∪ 折叠区**。

    用户 2026-09-08 实测城市1：同一次运行里，挑选页算出「50 条只对上 27 条
    （54%），已作废」，而登记表算出「对上 36 条（72%），保留」——两处用了
    不同的池子。`registry` 与 `run_all` 都用 `candidates_for_labels`（展示 ∪
    折叠），只有这里传的是展示候选（还套着 ENTRYEXIT_TOP_N 的展示上限）。

    后果不是「少匹配几条」这么轻：整份被判作废之后 `FILE_LABELS` 变空，
    页面上 `hydrateCellsFromFile` 就恢复不了用户逐格选过的 `cells`——那份
    工作只剩在文件里，用户一旦从页面重新导出就没了，而且一处不报错。
    """
    html, blob = _filtered_page(tmp_path)          # 一个展示候选 + 一个折叠区候选
    sid = blob["filtered"][0]["stable_id"]
    labels = {"_meta": {"detections_fingerprint": "别的指纹", "sheet": "s",
                        "n_candidates": 2},
              "3": {sid: {"role": "创业者", "seat": 1,
                          "cells": [{"cell": 1, "role": "创业者", "seat": 1}]}}}
    lp = tmp_path / "entry_labels_x.json"
    lp.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")

    frame = _write_frame(tmp_path)
    seg = mkseg(order=3, presenter_num=2)
    dets = (track(60, RIGHT_CX + 200, STANDING_H, 6, frame, order=3, kind="exit")
            + track(61, RIGHT_CX + 500, STANDING_H * 0.12, 8, frame, order=3,
                    kind="exit", width=12.0))
    out = tmp_path / "e2.html"
    render_entry_review([seg], dets, out, seats_by_order={3: {}},
                        detections_fp="当前指纹", labels_path=lp)
    page = out.read_text(encoding="utf-8")
    m = re.search(r"const FILE_LABELS\s*=\s*(\{.*?\});\n", page, re.DOTALL)
    assert m, "页面里找不到 FILE_LABELS"
    got = json.loads(m.group(1))
    assert got, ("折叠区那条标注被判为配不上、整份作废了——配对池漏了折叠区")
    assert any(k.endswith(sid) for k in got), f"实得 {list(got)}"


def test_the_legacy_note_syntax_hint_is_gone(tmp_path):
    """备注框旁那段「旧的备注写法 / ✗ 不解析」的长提示已删（用户 2026-09-08：
    「这个提示可以删了」）。

    Task 54 起格位分配是结构化下拉，备注不再参与补位判定；那段提示教的是
    2026-08 之前的写法，留着只是占地方、还把人往旧路上引。

    **解析器本身不删**：用户已按备注标完两组，作废等于让他们重标；文件里
    带 note 的旧标注照旧能被解析成格位预填（见 `note_to_cells`），
    只是页面不再教人怎么写。
    """
    html = _filled_page_with_candidates(tmp_path)
    assert "noteHint" not in html, "那段提示还在"
    assert "旧的备注写法" not in html
    # 断言那一段独有的句子，不要断言「不解析」三个字——源码注释里也会出现
    # 它（`saveNoCandidateAt` 就写着「页面不解析」），那样断的是注释不是页面。
    assert "这类相对指代" not in html
    # 解析器与预填这条路仍在
    assert "note_cells" in html, "备注→格位的自动预填不该跟着删掉"


def test_a_seat_whose_card_moved_away_is_not_labelled_as_a_top_up(tmp_path):
    """改号挪走的座位不该再写「补格位，不换人」——那个座位已经空了。

    用户 2026-09-08 实测城市1 order 4：他先在座位 1 的 2×2 上勾了「排除格 3、
    格 4」，又勾了「这张图其实是座位 2」把整张挪走；之后在退场候选 track 29
    上标「创业者 · 座位 1」，下拉里却写着「1（补格位，不换人）」——而这条
    恰恰是座位 1 唯一的标注，不是给谁补格。

    `topUpSeats` 只看「这个已定座位有没有勾过逐格排除」，没看它是不是已经被
    改号挪走 / 被人工否决。文案与实际不符会让人以为自己指过去不生效。
    """
    html = _filled_page_with_candidates(tmp_path, seats_by_order={5: {99: 1}})
    store = {"cellex::5::1": {"cells": [3, 4]}}
    tops = _run_js_with_store(html, store, "topUpSeats(findSegByOrder(5))")
    assert tops == [1], f"只勾了排除、没改号时，仍是「补格位，不换人」：{tops}"

    store["reassign::5::1"] = {"seat": 2}
    tops = _run_js_with_store(html, store, "topUpSeats(findSegByOrder(5))")
    assert tops == [2], (
        f"座位 1 的图已挪到座位 2：座位 1 空了不该带这个文案，"
        f"座位 2 上有那张挪过来的图才该带，实得 {tops}")
    vac = _run_js_with_store(html, store, "effectiveVacant(findSegByOrder(5))")
    assert 1 in vac, f"座位 1 仍要可选（它空了），实得 {vac}"


def test_a_rejected_seat_is_not_labelled_as_a_top_up(tmp_path):
    """人工勾了「是否替换」的座位同理：勾了就等于空缺，指过去是换人不是补格。"""
    html = _filled_page_with_candidates(tmp_path, seats_by_order={5: {99: 1}})
    store = {"cellex::5::1": {"cells": [3]},
             "seatovr::5::1": {"rejected": True}}
    tops = _run_js_with_store(html, store, "topUpSeats(findSegByOrder(5))")
    assert tops == [], f"被否决的座位不该写「不换人」，实得 {tops}"


def test_no_candidate_seat_offers_a_timestamp_box(tmp_path):
    """勾了「本座位无可用画面」的座位旁边要有一个填重抽时刻的框。

    用户 2026-09-09：「标记为『座位无可用画面』时，给出一个 note 窗口，填写
    该座位重抽的时间戳，供重抽参考」。他往往**眼睛已经看到那个人在第几分钟
    出现了**，只是那一带没被抽到；记下来加密就能直接对着它抽。
    """
    html = _filled_page_with_candidates(tmp_path, seats_by_order={5: {}})
    sec = _run_js_with_store(html, {"nocand::5::1": {"no_candidate": True}},
                             "noCandidateSection(findSegByOrder(5))")
    assert "saveNoCandidateAt(" in sec, f"缺填时刻的输入框：{sec[:800]}"
    assert "20:34" in sec, "占位符要给出可接受的写法示例"
    # 没勾的座位不显示这个框——没勾就没有「重抽」这回事
    sec2 = _run_js_with_store(html, {}, "noCandidateSection(findSegByOrder(5))")
    assert "saveNoCandidateAt(" not in sec2, f"没勾的座位不该出现时刻框：{sec2[:600]}"


def test_the_timestamp_is_exported_under_its_own_channel(tmp_path):
    """导出成与段号平级的 `no_candidate_notes`，形状 {段: {座位: {at, text}}}。

    **不塞进 `no_candidate_seats`**：那一条的值是数组，另外三个组的既有文件
    里都是这个形状，改形状会让它们读不进来。
    """
    html = _filled_page_with_candidates(tmp_path, seats_by_order={5: {}})
    store = {"nocand::5::1": {"no_candidate": True, "at": "20:34"}}
    payload = _run_js_with_store(html, store, "buildLabelsPayload()")
    assert payload.get("no_candidate_seats") == {"5": [1, 2]} or \
           1 in payload.get("no_candidate_seats", {}).get("5", []), payload
    assert payload.get("no_candidate_notes") == {"5": {"1": {"at": "20:34", "text": ""}}}, (
        f"实得 {payload.get('no_candidate_notes')}")
    # 没填时刻就不写这条——空对象会让下游以为人工填过
    payload2 = _run_js_with_store(html, {"nocand::5::1": {"no_candidate": True}},
                                  "buildLabelsPayload()")
    assert "no_candidate_notes" not in payload2, payload2


def test_setting_people_back_to_one_hides_the_cell_rows(tmp_path):
    """人数改回 1，格位行要收起来（**但不删** store 里的 cells）。

    用户 2026-09-09 实测城市1 order 16 exit track 72：「选的是人数为 1，却还是
    出现了 4 个格位标注」。`cellRowsOf` 里人数只管「要不要**铺新行**」，已经
    存着的 cells 照旧渲染出来。

    不删是刻意的（`setPeople` 的注释记着这条教训）：用户可能只是拨错了又拨
    回去，删掉等于把他的工作悄悄毁掉。所以是**藏起来**，拨回 2 就全在。
    Python 侧 `cell_map_of_label` 同一条口径——两边不一致的话，页面藏起来了
    而流水线照旧按 cells 切分，一处不报错。
    """
    html = _filled_page_with_candidates(tmp_path)
    blob = _segments_blob_from_html(html)[0]
    sid = blob["candidates"][0]["stable_id"]
    ck = f"5::{sid}"
    cells = [{"cell": 1, "role": "创业者", "seat": 1}, {"cell": 2, "same_as": 1}]
    n2 = _run_js_with_store(html, {ck: {"n_people": "2", "cells": cells}},
                            "cellRowsOf(SEGMENTS[0], SEGMENTS[0].candidates[0],"
                            " store['" + ck + "']).rows.length")
    assert n2 >= 2, f"人数 2 时该显示格位行，实得 {n2}"
    n1 = _run_js_with_store(html, {ck: {"n_people": "1", "cells": cells}},
                            "cellRowsOf(SEGMENTS[0], SEGMENTS[0].candidates[0],"
                            " store['" + ck + "']).rows.length")
    assert n1 == 0, f"人数 1 时不该再显示格位行，实得 {n1}"
    kept = _run_js_with_store(html, {ck: {"n_people": "1", "cells": cells}},
                              "(setPeople(5,'" + sid + "','2'), "
                              "cellRowsOf(SEGMENTS[0], SEGMENTS[0].candidates[0],"
                              " store['" + ck + "']).rows.length)")
    assert kept >= 2, f"拨回 2 时原来的格位判定要全在，实得 {kept}"


def test_clearing_a_label_survives_a_page_reload(tmp_path):
    """「清除本条标注」必须扛得住重开页面。

    用户 2026-09-09：「清除了标注也没生效」。`clearLabel` 只是删掉 localStorage
    里的键，而页面每次打开都会把文件里那份 PRELOAD 盖回来——他的删除被悄悄
    撤销，而页面上看不出发生过什么。这与「取消勾选无可用画面写显式的 false、
    不删键」是同一条教训，只是当时没推广到这里。

    做法：清除时留一个墓碑键，PRELOAD 合并时跳过它；用户重新标这条候选时
    墓碑自动消失。
    """
    html = _filled_page_with_candidates(tmp_path)
    blob = _segments_blob_from_html(html)[0]
    sid = blob["candidates"][0]["stable_id"]
    ck = f"5::{sid}"
    got = _run_page(html, {ck: {"role": "创业者", "seat": "1"}},
                    after=f"clearLabel('{ck}');"
                          f"Object.assign(store, mergePreload({{'{ck}':"
                          f"{{role:'创业者',seat:'1'}}}}));")
    assert not got["store"].get(ck), (
        f"文件里的那份把清除盖回来了：{got['store'].get(ck)}")
    # 重新标一次，墓碑要让路
    got2 = _run_page(html, {}, after=f"clearLabel('{ck}');"
                                     f"save('{ck}','role','工作人员');")
    assert got2["store"].get(ck, {}).get("role") == "工作人员", got2["store"].get(ck)


# --- 评审席区块（用户 2026-09-17）----------------------------------------

def _judges_page(tmp_path, n_seg=2, entry_labels=None):
    """左侧三位评审人 + 右侧一个进出场候选的挑选页。

    `entry_labels`：上一轮导出的标注（页面据此回填），默认没有。
    """
    frame = _write_frame(tmp_path)
    segs = [mkseg(order=o, presenter_num=1) for o in range(1, n_seg + 1)]
    dets = []
    for o in range(1, n_seg + 1):
        for i, (cx, y2) in enumerate([(200.0, 900.0), (320.0, 780.0), (450.0, 660.0)]):
            dets += track(20 + i, cx, 0.35 * 1080, 6, frame, order=o,
                          kind="seated", bottom=y2)
        dets += track(60, RIGHT_CX + 200, STANDING_H, 5, frame, order=o, kind="exit")
    out = tmp_path / "e.html"
    render_entry_review(segs, dets, out, detections_fp="FP-J",
                        entry_labels=entry_labels)
    return out.read_text(encoding="utf-8")


def test_page_has_a_group_level_judges_section(tmp_path, monkeypatch):
    """挑选页顶部要有一块「本组评审人」，恒 5 个号、空缺可见。

    用户 2026-09-17 要在 entry 与 vlm 两个产出里都看到专家 1-5 的 2×2。
    评审人是**组级**的：挂在每一段底下会让同一张脸要认 N 遍。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    assert "本组评审人" in html
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    assert m, "页面里找不到 JUDGES 数据"
    data = json.loads(m.group(1))
    assert data["n_expected"] == 5
    assert len(data["candidates"]) == 3
    assert [c["judge"] for c in data["candidates"]] == [1, 2, 3]
    assert all(c["img"] for c in data["candidates"]), "每位都要带 2×2 图"


def test_judges_section_shows_the_empty_slots(tmp_path, monkeypatch):
    """只检出 3 位时，4、5 号必须作为**缺口**显示出来。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    sec = _run_js_with_store(html, {}, "judgesSection()")
    assert "专家 4" in sec and "专家 5" in sec, sec[:600]
    assert "尚无候选" in sec, sec[:600]


def test_judge_numbering_can_be_changed_and_exported(tmp_path, monkeypatch):
    """每位候选都能改号 / 标「不是评审人」，并导出到 `judges` 通道。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    sid = json.loads(m.group(1))["candidates"][0]["stable_id"]
    sec = _run_js_with_store(html, {}, "judgesSection()")
    assert "saveJudge(" in sec, sec[:600]
    payload = _run_js_with_store(
        html, {f"judge::{sid}": {"judge": "4"}}, "buildLabelsPayload()")
    assert payload.get("judges") == {sid: {"judge": 4, "role": ""}}, payload.get("judges")
    payload2 = _run_js_with_store(
        html, {f"judge::{sid}": {"role": "不是评审人"}}, "buildLabelsPayload()")
    assert payload2["judges"][sid]["role"] == "不是评审人"
    none = _run_js_with_store(html, {}, "buildLabelsPayload()")
    assert "judges" not in none, "一条都没标时不许写这个键"


def test_judge_cards_are_one_per_row_at_candidate_size(tmp_path, monkeypatch):
    """评审人卡片一位一行，图的大小与创业者候选图**同一条规则**。

    用户 2026-09-18：「现在本组评审人5个人的图排成了一行，太小不容易分辨，
    一个专家一列，大小参考展示创业者的2*2图」。原先是 5 张 220px 定宽卡片挤在
    一行里，2×2 缩到那个尺寸根本看不清衣服。

    做法是**复用创业者卡片的类**（`cand` + `imgwrap`），而不是另抄一份尺寸：
    另抄一份必然日后走样，而走样的表现正是这次——两边看着像是一回事，实际
    一个 220px 一个 62%。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    sec = _run_js_with_store(html, {}, "judgesSection()")
    assert 'class="cand judgeCard' in sec, sec[:400]
    assert 'class="imgwrap' in sec, "图要放进与创业者同一个 imgwrap 列里"
    # 定宽会把它锁回小图——这一行就是这次要去掉的东西
    css = _css_decls(html, ".judgeCard")
    assert "width:" not in css.replace("max-width:", ""), css
    # 尺寸只此一处来源：备选那几张 2×2 也挂 imgwrap，不另抄一份百分比
    assert sec.count('class="imgwrap') >= 1, sec[:400]
    row = _css_decls(html, ".judgeRow")
    assert "flex-wrap" not in row, f"一位一行，不再横排换行：{row}"


def test_judge_card_offers_a_checkbox_per_cell(tmp_path, monkeypatch):
    """每位评审人的 2×2，四格各有一个「这一格不是本人」的勾选（用户 2026-09-18）。

    「专家的 2×2 需要每个格子进行人工核实，因为有的格子混进了其他专家」。
    跨段归并按位置容差并人，相邻两位评审人本来就坐得近，某一段跟丢重识别后
    隔壁那位很容易被并进同一个候选。四格红框标的都是同一个人，**模型无从
    分辨哪格才是本人**，会把两个人的衣着混起来判断——不报错，数字全错。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    sec = _run_js_with_store(html, {}, "judgesSection()")
    assert "saveJudgeCell(" in sec, sec[:800]
    for k in (1, 2, 3, 4):
        assert f"格{k}" in sec, f"缺第 {k} 格的勾选：{sec[:800]}"


def test_judge_cell_choices_are_exported(tmp_path, monkeypatch):
    """标过的格写进 `judges[stable_id].cells`，键是格位号。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    sid = json.loads(m.group(1))["candidates"][0]["stable_id"]
    payload = _run_js_with_store(
        html, {f"judge::{sid}": {"cells": {"4": "x", "2": "x"}}},
        "buildLabelsPayload()")
    assert payload["judges"][sid]["cells"] == {"2": "x", "4": "x"}, payload["judges"]


def test_a_judge_marked_only_by_cell_is_still_written_out(tmp_path, monkeypatch):
    """只勾了格、没改号也没改身份的那一条，同样要导出。

    导出只写「人工动过的」记录。若判断「动过」只看号与身份，逐格核实的结果
    就整份丢掉——用户勾完、导出、重开页面，格子又全回来了，而一处不报错。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    sid = json.loads(m.group(1))["candidates"][0]["stable_id"]
    payload = _run_js_with_store(
        html, {f"judge::{sid}": {"cells": {"3": "x"}}}, "buildLabelsPayload()")
    assert sid in (payload.get("judges") or {}), payload.get("judges")
    assert payload["judges"][sid]["cells"] == {"3": "x"}


def test_unticked_judge_cells_write_nothing(tmp_path, monkeypatch):
    """勾上又取消的格位不留痕——空列表也不写，否则「没标过」与「标过又撤回」
    在文件里长得一样，人工核对无从分辨。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    sid = json.loads(m.group(1))["candidates"][0]["stable_id"]
    payload = _run_js_with_store(
        html, {f"judge::{sid}": {"cells": {"2": False}}}, "buildLabelsPayload()")
    assert "judges" not in payload, payload.get("judges")
    # 旧版存的布尔 false 也一样：没标过就是没标过
    p2 = _run_js_with_store(
        html, {f"judge::{sid}": {"cells": {"2": ""}}}, "buildLabelsPayload()")
    assert "judges" not in p2, p2.get("judges")


def test_judge_cells_beyond_the_real_frame_count_are_not_offered(tmp_path,
                                                                 monkeypatch):
    """证据图不足 4 张时，多出来的格不给勾——那几格是灰块，勾了也没有帧可丢，
    人却会以为自己已经核实过。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    cands = json.loads(m.group(1))["candidates"]
    assert all(c.get("n_cells") for c in cands), "候选要带上真实格数"


# ---------------------------------------------------------------------------
# 挑选页引用图片文件，不再把图嵌进 HTML（用户 2026-09-18）
#
# 此前一整组的 264 张图以 base64 嵌进页面，80 MB 压到 32 MB 才装得下——而
# 「人核对的图比模型看到的糊」正是本项目最忌讳的那一类。改成写文件之后压缩
# 这一步整个消失：人看到的与送模型的是**同一批字节**。
#
# 顺带的两笔：base64 比原始字节胀 33%，浏览器要一口气解析 33 MB 的文档；
# 改后页面只剩 1 MB 上下，图交给浏览器按需解码。实测每张图写文件 0.08 ms，
# 而原先光是降采样重编码一轮就要 9.23 ms。
# ---------------------------------------------------------------------------

def test_page_references_image_files_instead_of_embedding_them(tmp_path):
    """页面里一个 `data:` 内嵌图都不许有，图全部落成文件。"""
    out = tmp_path / "entry_review.html"
    frame = _write_frame(tmp_path)
    seg = mkseg(order=2, presenter_num=2)
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    render_entry_review([seg], dets, out, detections_fp="FP-IMG")
    html = out.read_text(encoding="utf-8")
    assert "data:image" not in html, "还有图被嵌进了 HTML"
    blob = _segments_blob_from_html(html)[0]
    srcs = [c["img"] for c in blob["candidates"] if c.get("img")]
    assert srcs, "候选一张图都没有"
    for src in srcs:
        assert not src.startswith("data:"), src
        assert (out.parent / src).is_file(), f"引用了不存在的文件：{src}"


def test_referenced_files_are_the_uncompressed_originals(tmp_path):
    """盘上那张就是 `build_evidence_sheet` 的原样输出，一个字节不差。

    这是整件事的要点：核对页与送模型的那张同源，挑选页此前却是压过的。
    """
    from pipeline.annotate import build_evidence_sheet
    out = tmp_path / "entry_review.html"
    frame = _write_frame(tmp_path)
    seg = mkseg(order=2, presenter_num=2)
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    render_entry_review([seg], dets, out, detections_fp="FP-IMG")
    blob = _segments_blob_from_html(out.read_text(encoding="utf-8"))[0]
    cand = blob["candidates"][0]
    got = (out.parent / cand["img"]).read_bytes()
    assert got.startswith(b"\x89PNG"), "候选图被重编码成别的格式了"
    from pipeline.entryexit import candidates_for_labels
    picked = candidates_for_labels(dets, seg, 1440, 1080)
    match = [c for c in picked if c["stable_id"] == cand["stable_id"]]
    assert match, cand["stable_id"]
    want = build_evidence_sheet([d.frame_path for d in match[0]["best_dets"]],
                                [(d.x1, d.y1, d.x2, d.y2) for d in match[0]["best_dets"]],
                                label_cells=True)
    assert got == want, "盘上的图与 build_evidence_sheet 的输出不一致"


def test_images_go_beside_the_page_in_one_folder(tmp_path):
    """图落在页面旁边**同一个**目录里——整组打包发出去时不会漏。

    页面不再自包含了：交给别人标必须整个目录打包（`work/操作说明.md` 流程四
    本来就是这么写的）。图散落在几处的话，漏掉哪一处都是「打开页面一半是叉」。
    """
    out = tmp_path / "entry_review.html"
    frame = _write_frame(tmp_path)
    seg = mkseg(order=2, presenter_num=2)
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    render_entry_review([seg], dets, out, detections_fp="FP-IMG")
    blob = _segments_blob_from_html(out.read_text(encoding="utf-8"))[0]
    dirs = {str(Path(c["img"]).parent) for c in blob["candidates"] if c.get("img")}
    assert dirs == {"review_img"}, dirs
    assert (out.parent / "review_img").is_dir()


def test_stale_images_from_an_earlier_run_are_cleared(tmp_path):
    """上一轮留下、这一轮不再引用的图要清掉。

    候选数量随重抽帧变化，留着的孤儿图既占盘又让人以为页面还在用它。清理
    **只限**这个由本函数独占的目录，不碰别处任何东西。
    """
    out = tmp_path / "entry_review.html"
    frame = _write_frame(tmp_path)
    seg = mkseg(order=2, presenter_num=2)
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    render_entry_review([seg], dets, out, detections_fp="FP-IMG")
    orphan = out.parent / "review_img" / "seg99_cand42.png"
    orphan.write_bytes(b"stale")
    render_entry_review([seg], dets, out, detections_fp="FP-IMG")
    assert not orphan.exists(), "上一轮的孤儿图还在"
    blob = _segments_blob_from_html(out.read_text(encoding="utf-8"))[0]
    assert all((out.parent / c["img"]).is_file()
               for c in blob["candidates"] if c.get("img")), "清理误伤了在用的图"


def test_the_judges_and_filled_seats_use_the_same_image_folder(tmp_path):
    """评审人图、已定座位图、证据图**同一套命名、同一个目录**。

    分两套路径必然日后走样，而走样的表现是某一类图静默变成叉。
    """
    from pipeline import config as _c
    out = tmp_path / "entry_review.html"
    frame = _write_frame(tmp_path)
    segs = [mkseg(order=o, presenter_num=1) for o in (1, 2)]
    dets = []
    for o in (1, 2):
        for i, (cx, y2) in enumerate([(200.0, 900.0), (320.0, 780.0)]):
            dets += track(20 + i, cx, 0.35 * 1080, 6, frame, order=o,
                          kind="seated", bottom=y2)
        dets += track(60, RIGHT_CX + 200, STANDING_H, 5, frame, order=o, kind="exit")
    import pytest as _pt
    with _pt.MonkeyPatch.context() as mp:
        mp.setattr(_c, "JUDGE_MIN_FRAMES", 5)
        render_entry_review(segs, dets, out, detections_fp="FP-IMG")
    html = out.read_text(encoding="utf-8")
    assert "data:image" not in html
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    for c in json.loads(m.group(1))["candidates"]:
        assert c["img"].startswith("review_img/"), c["img"]
        assert (out.parent / c["img"]).is_file(), c["img"]


def test_each_image_gets_its_own_file(tmp_path):
    """每张图一个文件名，互不覆盖。

    共用一个名字的话后写的会盖掉先写的，页面上每张卡片显示的都是**同一个
    人**——而文件都在、路径都对、一处不报错。人会照着那张图把整段的座位全
    标错，且没有任何迹象提示他看的不是这个候选。
    """
    out = tmp_path / "entry_review.html"
    frame = _write_frame(tmp_path)
    segs = [mkseg(order=o, presenter_num=2) for o in (1, 2)]
    dets = []
    for o in (1, 2):
        dets += track(70 + o, RIGHT_CX + 120, STANDING_H, 6, frame, order=o)
        dets += track(80 + o, RIGHT_CX + 420, STANDING_H * 0.9, 6, frame, order=o)
    render_entry_review(segs, dets, out, detections_fp="FP-IMG")
    blob = _segments_blob_from_html(out.read_text(encoding="utf-8"))
    srcs = [c["img"] for seg in blob
            for c in (seg["candidates"] + seg["filtered"]) if c.get("img")]
    assert len(srcs) >= 3, f"夹具前提：要有好几张图才测得出覆盖，实得 {srcs}"
    assert len(set(srcs)) == len(srcs), f"有图共用了文件名：{srcs}"
    blobs = [(out.parent / s).read_bytes() for s in srcs]
    assert len(set(blobs)) > 1, "所有图的内容都一样——多半是互相盖掉了"


def test_judge_cell_offers_who_it_really_is(tmp_path, monkeypatch):
    """每一格给的是「这一格是谁」的下拉，不只是勾掉（用户 2026-09-18）。

    只能勾掉是不够的：城市1 实测四位专家共勾掉 7 格，专家1 与专家3 各只剩
    1 格——模型拿一格判整场衣着，比拿四格混着两个人还糟。那些格没有坏，只是
    归错了人；说清是谁，帧就换到那一位名下。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    sec = _run_js_with_store(html, {}, "judgesSection()")
    assert "saveJudgeCell(" in sec, sec[:800]
    for k in (1, 2, 3, 4):
        assert f"格{k}" in sec, f"缺第 {k} 格：{sec[:800]}"
    assert "本人" in sec, "没有「本人」这个默认项"
    assert "其他人" in sec, "没有「其他人（丢掉）」这一项"
    assert sec.count("<select") >= 5, "每格该有一个下拉（外加编号那个）"


def test_judge_cell_choice_is_exported_as_who(tmp_path, monkeypatch):
    """导出成 `judges[sid].cells = {格位号: 专家号 | "x"}`。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    sid = json.loads(m.group(1))["candidates"][0]["stable_id"]
    payload = _run_js_with_store(
        html, {f"judge::{sid}": {"cells": {"2": "3", "4": "x"}}},
        "buildLabelsPayload()")
    assert payload["judges"][sid]["cells"] == {"2": 3, "4": "x"}, payload["judges"]


def test_a_cell_left_as_the_person_writes_nothing(tmp_path, monkeypatch):
    """选回「本人」的格不留痕——空着与「选过又改回来」在文件里长得一样才对。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    sid = json.loads(m.group(1))["candidates"][0]["stable_id"]
    payload = _run_js_with_store(
        html, {f"judge::{sid}": {"cells": {"2": ""}}}, "buildLabelsPayload()")
    assert "judges" not in payload, payload.get("judges")


def test_the_old_checkbox_state_still_counts_as_other(tmp_path, monkeypatch):
    """浏览器里存着的旧勾选（布尔 true）照旧算「其他人」。

    用户 2026-09-18 11:30 已经用勾选版标完城市1 导出了，localStorage 里存的
    是 `{cells:{"2":true}}`。换成下拉后读不出它，就等于把他那半天的工作静默
    清空——而页面上一个字都不会说。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    sid = json.loads(m.group(1))["candidates"][0]["stable_id"]
    payload = _run_js_with_store(
        html, {f"judge::{sid}": {"cells": {"2": True}}}, "buildLabelsPayload()")
    assert payload["judges"][sid]["cells"] == {"2": "x"}, payload["judges"]


def test_judge_card_shows_the_spare_pool(tmp_path, monkeypatch):
    """每位专家除当前 2×2 外，还要给出**备选**的几张 2×2（用户 2026-09-18）。

    「针对 2×2 不足的专家，重新给出新的挑选图，以实现 2×2 四张图」。城市1 标完
    五位只剩 1/3/3/2/3 格；补上来的备选可能还是同一个错的人，所以备选也必须
    看得见、标得了——只自动顶上不给看，等于把一张没人验过的脸送进模型。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    c = json.loads(m.group(1))["candidates"][0]
    assert len(c["sheets"]) >= 2, c["sheets"]
    assert c["sheets"][0]["cells"] == [1, 2, 3, 4], c["sheets"][0]
    assert c["sheets"][1]["cells"][0] == 5, c["sheets"][1]
    for sh in c["sheets"]:
        assert sh["img"].startswith("review_img/"), sh
        assert (tmp_path / sh["img"]).is_file(), sh


def test_the_spare_cells_can_be_marked_on_the_page(tmp_path, monkeypatch):
    """备选那几格同样有「这一格是谁」的下拉，格位号接着往下排。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    sec = _run_js_with_store(html, {}, "judgesSection()")
    assert "格5" in sec, "备选的第 5 格没有下拉"
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    sid = json.loads(m.group(1))["candidates"][0]["stable_id"]
    payload = _run_js_with_store(
        html, {f"judge::{sid}": {"cells": {"6": "2"}}}, "buildLabelsPayload()")
    assert payload["judges"][sid]["cells"] == {"6": 2}, payload["judges"]


def test_the_card_says_which_cells_the_model_will_see(tmp_path, monkeypatch):
    """卡片要写明「现在送模型的是哪四格」。

    备选自动顶上之后，人看着一排 2×2 分不清哪四张真的会进模型——而这正是
    他要核的东西。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    sid = json.loads(m.group(1))["candidates"][0]["stable_id"]
    sec = _run_js_with_store(
        html, {f"judge::{sid}": {"cells": {"2": "x"}}}, "judgesSection()")
    assert "送模型" in sec, sec[:900]
    # 空出来的位置不自动顶上（用户 2026-09-18），所以这里是三格不是四格
    assert "格 1、3、4（" in sec, sec[sec.find("送模型") - 200:sec.find("送模型") + 200]


def test_the_spare_sheets_are_numbered_by_their_pool_position(tmp_path, monkeypatch):
    """备选那张 2×2 上画的是 5/6/7/8，不是 1/2/3/4。

    图上还画 1-4 的话，人看着第二张图里的「2」去标，指的却是池里的第 6 格
    ——他标的号与流水线用的号从此各说各话，而一处不报错。
    """
    from pipeline import config as _c
    from pipeline.annotate import build_evidence_sheet
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    out = tmp_path / "e.html"
    frame = _write_frame(tmp_path)
    segs = [mkseg(order=o, presenter_num=1) for o in range(1, 7)]
    dets = []
    for o in range(1, 7):
        for i, (cx, y2) in enumerate([(200.0, 900.0), (320.0, 780.0)]):
            dets += track(20 + i, cx, 0.35 * 1080, 6, frame, order=o,
                          kind="seated", bottom=y2)
    render_entry_review(segs, dets, out, detections_fp="FP-P")
    html = out.read_text(encoding="utf-8")
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    c = json.loads(m.group(1))["candidates"][0]
    pool = judge_candidates(dets, 1440, 1080)[0]["pool_dets"]
    second = c["sheets"][1]
    assert second["cells"] == [5, 6, 7, 8], second["cells"]
    chunk = pool[4:8]
    want = build_evidence_sheet([d.frame_path for d in chunk],
                                [(d.x1, d.y1, d.x2, d.y2) for d in chunk],
                                label_cells=True, cell_labels=[5, 6, 7, 8])
    assert (out.parent / second["img"]).read_bytes() == want, (
        "备选那张图上的编号与它的池位置对不上")


def test_spare_cells_can_be_topped_up_into_the_two_by_two(tmp_path, monkeypatch):
    """备选那几格有「补选进 2×2」的勾选；默认那四格不给勾。

    用户 2026-09-18：「不需要总是勾选 4 个，默认选的被标记为其他人后，再从
    后面标的选中进 2×2 递补」。给默认四格也配勾选框的话，人会以为不勾就不
    算数，反倒把好端端的几格漏掉。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    sec = _run_js_with_store(html, {}, "judgesSection()")
    assert "togglePick(" in sec, sec[:900]
    assert "补选进 2×2" in sec, sec[:900]
    # 默认那四格：不给勾，写明它们本来就在里面
    assert "默认在 2×2 里" in sec, sec[:1200]
    i = sec.find('class="jcell">格1')
    assert "togglePick(" not in sec[i:sec.find('class="jcell">格2')], sec[i:i + 400]


def test_picks_are_exported(tmp_path, monkeypatch):
    """挑中的格写进 `judges[sid].picks`，按挑的次序。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    sid = json.loads(m.group(1))["candidates"][0]["stable_id"]
    payload = _run_js_with_store(
        html, {f"judge::{sid}": {"picks": [9, 2, 5]}}, "buildLabelsPayload()")
    assert payload["judges"][sid]["picks"] == [9, 2, 5], payload["judges"]
    none = _run_js_with_store(html, {}, "buildLabelsPayload()")
    assert "judges" not in none, "一格都没挑时不许写这个键"


def test_the_page_stars_the_default_cells_plus_the_top_ups(tmp_path, monkeypatch):
    """★ = 默认那四格里还算本人的 + 补选的，与 Python 侧同一条口径。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    sid = json.loads(m.group(1))["candidates"][0]["stable_id"]
    sec = _run_js_with_store(
        html, {f"judge::{sid}": {"cells": {"1": "x"}, "picks": [6]}},
        "judgesSection()")
    assert "格 2、3、4、6" in sec, sec[sec.find("送模型") - 100:sec.find("送模型") + 250]


def test_the_page_says_how_many_are_still_needed(tmp_path, monkeypatch):
    """还差几格必须写出来——挑了三格就导出，2×2 里会留个灰块，而人以为挑完了。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    sid = json.loads(m.group(1))["candidates"][0]["stable_id"]
    sec = _run_js_with_store(
        html, {f"judge::{sid}": {"cells": {"1": "x", "2": "x", "3": "x"}},
               "picks": [9]}, "judgesSection()")
    assert "还差 3 格" in sec, sec[sec.find("送模型") - 100:sec.find("送模型") + 300]


def test_a_cell_that_is_not_the_person_cannot_be_picked(tmp_path, monkeypatch):
    """标成「其他人」的格不给挑——挑了也不作数，让它可勾就是留个假象。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    sid = json.loads(m.group(1))["candidates"][0]["stable_id"]
    sec = _run_js_with_store(
        html, {f"judge::{sid}": {"cells": {"2": "x"}}}, "judgesSection()")
    i = sec.find("togglePick('" + sid + "',2")
    assert i > 0, sec[:600]
    assert "disabled" in sec[max(0, i - 220):i + 60], sec[max(0, i - 220):i + 60]


def test_the_page_realigns_judge_labels_after_a_re_extraction(tmp_path, monkeypatch,
                                                              capsys):
    """挑选页也要把挪了位的评审人标注对回来。

    只在登记表那一侧对齐是不够的：人打开页面看到的是「没人标过」，会重标一遍，
    而登记表里其实还留着旧的——两边各说各话，且一处不报错。
    """
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    out = tmp_path / "e.html"
    frame = _write_frame(tmp_path)
    segs = [mkseg(order=o, presenter_num=1) for o in range(1, 7)]
    dets = []
    for o in range(1, 7):
        for i, (cx, y2) in enumerate([(200.0, 900.0), (320.0, 780.0)]):
            dets += track(20 + i, cx, 0.35 * 1080, 6, frame, order=o,
                          kind="seated", bottom=y2)
    sid = judge_candidates(dets, 1440, 1080)[0]["stable_id"]
    cx, y2 = sid.split("#")[1].split(",")
    stale = f"judge#{int(cx) - 1},{int(y2) + 1}"
    render_entry_review(segs, dets, out, detections_fp="FP-R",
                        entry_labels={"judges": {stale: {"judge": 4}}})
    html = out.read_text(encoding="utf-8")
    blob = json.loads(re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL).group(1))
    got = [c for c in blob["candidates"] if c["stable_id"] == sid]
    assert got and got[0]["judge"] == 4, [(c["stable_id"], c["judge"])
                                          for c in blob["candidates"]]
    assert "评审席标注" in capsys.readouterr().out


def test_the_toc_lists_each_judge_with_their_position(tmp_path, monkeypatch):
    """左侧目录里要有评审席一栏，逐位列出**画面里的位置**（用户 2026-09-18）。

    评审席是全组共用的一块，挂在页首；企业目录有十几行，人一滚下去就再也看不到
    它。目录里给一行入口，随时点得回来；位置（cx,y2）写出来，是因为席位号来自
    几何基线，人得能对着画面判断「这一位是不是真坐在第 3 号」——城市2 就是
    `judge#35,668`（cx=35，紧贴画面左边缘）占了 3 号位，而真正那位在 cx=565。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    toc = _run_js_with_store(html, {}, "judgeTocHtml()")
    assert "评审席" in toc, toc[:400]
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    for c in json.loads(m.group(1))["candidates"]:
        assert str(int(c["median_cx"])) in toc, (c["stable_id"], toc[:600])


def test_the_toc_judge_rows_link_to_their_cards(tmp_path, monkeypatch):
    """目录里每位评审人都要能点回他自己那张卡片。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    toc = _run_js_with_store(html, {}, "judgeTocHtml()")
    sec = _run_js_with_store(html, {}, "judgesSection()")
    ids = set(re.findall(r'id="(judge-\d+)"', sec))
    assert ids, sec[:400]
    for jid in ids:
        assert f'href="#{jid}"' in toc, (jid, toc[:600])


def test_the_toc_shows_how_many_cells_each_judge_still_needs(tmp_path, monkeypatch):
    """目录里要显示每位还差几格——缺格的那几位才是要回去处理的。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    sid = json.loads(m.group(1))["candidates"][0]["stable_id"]
    toc = _run_js_with_store(
        html, {f"judge::{sid}": {"cells": {"1": "x", "2": "x"}}}, "judgeTocHtml()")
    assert "差 2" in toc, toc[:800]


def test_the_judge_toc_comes_before_the_company_toc(tmp_path, monkeypatch):
    """评审席那一栏排在企业目录**之前**（用户 2026-09-18）。

    评审席是全组共用的一块、挂在页首；企业目录有十几行，排在它后面就要滚到底
    才看得见，而评审席恰恰是最先要核的东西。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    i_j = html.index('id="toc-judges"')
    i_c = html.index('id="toc-list"')
    assert i_j < i_c, "评审席那一栏排在了企业目录后面"


def test_the_page_counts_cells_handed_over_by_others(tmp_path, monkeypatch):
    """页面上的「还差几格」要把**别人还回来的格**算进去（用户 2026-09-18）。

    登记表一直是算的（`_kept = 本人的 + 收进的`），页面只算本人的。城市2 实测
    专家3 自己四格全标掉、收了 `judge#207,696` 的两格，登记表说「2 格、还差 2」
    而挑选页说「四格齐」——同一件事两个说法，人照着页面就以为标完了。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    cands = json.loads(m.group(1))["candidates"]
    a, b = cands[0]["stable_id"], cands[1]["stable_id"]
    store = {
        # 专家1 自己四格全标掉
        f"judge::{a}": {"cells": {"1": "x", "2": "x", "3": "x", "4": "x"}},
        # 专家2 把自己的格1、格2 指给专家1
        f"judge::{b}": {"cells": {"1": 1, "2": 1}},
    }
    sec = _run_js_with_store(html, store, "judgesSection()")
    # 按卡片边界切，不能按「专家 2」切——那几个字也出现在专家1 自己的编号下拉
    # 里（`<option value="2">专家 2</option>`），切出来的 head 连提示行都不含，
    # 断言就成了对着下拉比对（第一版正是这样，必红）。
    cards = sec.split('class="cand judgeCard')
    head = cards[1]
    assert "还差 2 格" in head, head[:900]
    assert "收进 2 格" in head, head[:900]


def test_the_toc_counts_cells_handed_over_by_others(tmp_path, monkeypatch):
    """目录里的「差 N 格」同样要算上收进的格——两处口径必须一致。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    cands = json.loads(m.group(1))["candidates"]
    a, b = cands[0]["stable_id"], cands[1]["stable_id"]
    store = {
        f"judge::{a}": {"cells": {"1": "x", "2": "x", "3": "x", "4": "x"}},
        f"judge::{b}": {"cells": {"1": 1, "2": 1}},
    }
    toc = _run_js_with_store(html, store, "judgeTocHtml()")
    row = toc[toc.find("专家 1"):toc.find("专家 2")]
    assert "差 2 格" in row, row


def test_a_judge_fully_supplied_by_others_reads_as_complete(tmp_path, monkeypatch):
    """四格全靠别人还回来时，页面要说「齐」而不是「差 4」。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    m = re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
    cands = json.loads(m.group(1))["candidates"]
    a, b = cands[0]["stable_id"], cands[1]["stable_id"]
    store = {
        f"judge::{a}": {"cells": {"1": "x", "2": "x", "3": "x", "4": "x"}},
        f"judge::{b}": {"cells": {"1": 1, "2": 1, "3": 1, "4": 1}},
    }
    toc = _run_js_with_store(html, store, "judgeTocHtml()")
    row = toc[toc.find("专家 1"):toc.find("专家 2")]
    assert "四格齐" in row, row


def test_the_toc_hides_extras_once_every_seat_is_filled(tmp_path, monkeypatch):
    """席位坐满时，目录里不再列「未入席 N 位」（用户 2026-09-18）。

    那一行是「从这里把人提上来补位」的入口，坐满之后只是噪声。评审席区块里的
    候选卡片照旧列出来——哪天发现坐错了号（城市2 的 `judge#35,668` 就是贴着
    画面边缘挤进 3 号位的），还得从那里把人提回席位。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    # 席位压到 2 而场上有 3 位：坐满且确实有人没入席。三配三的话 extra 本来
    # 就是空的，断言恒真。
    monkeypatch.setattr(_c, "JUDGE_COUNT", 2)
    html = _judges_page(tmp_path, n_seg=6)
    toc = _run_js_with_store(html, {}, "judgeTocHtml()")
    assert "未入席" not in toc, toc[:600]
    sec = _run_js_with_store(html, {}, "judgesSection()")
    assert "未入席的候选" in sec, "区块里也不见了，就再也提不回席位"


def test_the_toc_lists_extras_while_a_seat_is_empty(tmp_path, monkeypatch):
    """还有空号时那一行必须在。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)      # 5 个席位、3 位候选
    toc = _run_js_with_store(html, {}, "judgeTocHtml()")
    assert "尚无候选" in toc, toc[:600]


def test_a_cell_marked_as_a_judge_offers_judge_numbers(tmp_path, monkeypatch):
    """格位身份选「专家」时，座位号下拉变成 1..N 的评审人号（用户 2026-09-19）。

    文化4 的专家5 坐在右侧靠投影仪那头，`judge_candidates` 只看左区，他一个候选
    都出不来。右区的人都在创业者候选里，所以指认他的唯一出口就是这一栏——而
    此前这里给的是本段空着的**创业者座位号**，根本指不到评审席上。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    pool = _run_js_with_store(html, {}, "seatPoolFor(SEGMENTS[0],'专家')")
    assert pool == list(range(1, _c.JUDGE_COUNT + 1)), pool


def test_other_cell_roles_keep_the_entrepreneur_seat_pool(tmp_path, monkeypatch):
    """其余身份照旧用本段空着的创业者座位号——这一改不能溢到别的身份上。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    for role in ("创业者", "工作人员", "不确定"):
        pool = _run_js_with_store(html, {}, f"seatPoolFor(SEGMENTS[0],'{role}')")
        assert pool != list(range(1, _c.JUDGE_COUNT + 1)) or len(pool) != 5, role
        assert pool == _run_js_with_store(
            html, {}, "effectiveVacant(SEGMENTS[0])"), role


def test_a_one_person_candidate_marked_as_a_judge_can_pick_a_number(tmp_path,
                                                                    monkeypatch):
    """人数为 1 的候选选「专家」后，必须出现 1..N 的评审人号（用户 2026-09-19）。

    实测文化4 的 entry/exit track 2：选了专家却没有任何地方能指第几位。这一栏
    原先写死只给「创业者」看——人数为 1 时又不展开格位分配，于是整张卡片没有
    一个能填号的地方，右侧那位评审人就此指认不了。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    blob = _segments_blob_from_html(html)[0]
    cand = blob["candidates"][0]
    ck = f"{blob['order']}::{cand['stable_id']}"   # candKey 用的是双冒号
    main = _run_js_with_store(
        html, {ck: {"role": "专家"}},
        "candidateCard(SEGMENTS[0], SEGMENTS[0].candidates[0], null)")
    # 按标签定位，不能按「座位号」三个字——那几个字在下方「清除本条标注」的
    # 说明里也出现（"座位号、人数、备注一并清掉"），找过去就落在了说明文字上。
    assert "第几位专家" in main, main[:900]
    i = main.find("第几位专家")
    wrap = main[max(0, i - 200):i]
    assert "display:none" not in wrap, wrap
    opts = main[i:main.index("</select>", i)]
    for n in range(1, _c.JUDGE_COUNT + 1):
        assert f'value="{n}"' in opts, (n, opts)


def test_a_one_person_entrepreneur_still_sees_the_entrepreneur_seats(tmp_path,
                                                                     monkeypatch):
    """选「创业者」时照旧是本段空着的创业者座位号——这一改不能溢到别的身份。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    blob = _segments_blob_from_html(html)[0]
    cand = blob["candidates"][0]
    ck = f"{blob['order']}::{cand['stable_id']}"   # candKey 用的是双冒号
    main = _run_js_with_store(
        html, {ck: {"role": "创业者"}},
        "candidateCard(SEGMENTS[0], SEGMENTS[0].candidates[0], null)")
    vacant = _run_js_with_store(html, {}, "effectiveVacant(SEGMENTS[0])")
    assert "第几位专家" not in main, "创业者不该看到评审席的号"
    i = main.find("座位号")
    opts = main[i:main.index("</select>", i)]
    for n in vacant:
        assert f'value="{n}"' in opts, (n, vacant, opts)


def test_filled_seat_images_are_candidate_sized(tmp_path, monkeypatch):
    """已定座位的图与创业者候选图**同一条尺寸规则**（用户 2026-09-19）。

    「entry 界面自动标好的图不够大，导致无法辨别，要求自动识别好的大小和其他
    候选图大小一致」。原先是 180px 定宽、横排换行——2×2 缩到那个尺寸根本看不清
    衣服，而这一栏恰恰是要人核对「机器定的这个座位对不对」。

    同一条要求 2026-09-18 已经对评审人卡片提过一次（`2871f3a`），当时只改了被
    点名的那一处。做法与那次相同：**复用创业者卡片的类**（`cand` + `imgwrap`），
    不另抄一份尺寸——另抄一份必然日后走样，而走样的表现正是这次。
    """
    frame = _write_frame(tmp_path)
    seg = mkseg(order=6, presenter_num=2, company="乙企业")
    dets = track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
    kf = tmp_path / "keyframes"
    _write_jpg(_seat_dir(kf, seg.order, seg.company, 1) / "crop_seated_01_t1.0.jpg", 30)
    html = _render_filled_page(tmp_path, seg, dets, {seg.order: {11: 1}}, kf)

    # 定宽是这次要去掉的东西
    css = _css_decls(html, ".filledSeat")
    assert "width:" not in css.replace("max-width:", ""), css
    row = _css_decls(html, ".fsRow")
    assert "flex-wrap" not in row, f"一座一行，不再横排换行：{row}"
    # 图必须走与创业者同一个 imgwrap 列
    cards = _filled_cards(html)
    assert cards, "夹具前提：要有已定座位卡片"
    assert 'class="cand filledSeat' in html, "未复用创业者卡片的类"
    assert 'class="imgwrap' in html.split('class="cand filledSeat')[1][:600], \
        "图没有放进 imgwrap 列"


def test_the_page_renders_despite_a_duplicate_judge_number(tmp_path, monkeypatch):
    """两位被指到同一号时，挑选页仍要出得来（用户 2026-09-19，电子27）。

    `judge_slots` 拒绝猜是对的，但抛异常会**挡住修复的唯一工具**：页面出不来
    → 改不了号 → 页面永远出不来。页面要降级——冲突的那几位当作未指号——并把
    冲突写在页面上。
    """
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    out = tmp_path / "e.html"
    frame = _write_frame(tmp_path)
    segs = [mkseg(order=o, presenter_num=1) for o in range(1, 7)]
    dets = []
    for o in range(1, 7):
        for i, (cx, y2) in enumerate([(200.0, 900.0), (320.0, 780.0), (450.0, 660.0)]):
            dets += track(20 + i, cx, 0.35 * 1080, 6, frame, order=o,
                          kind="seated", bottom=y2)
    cands = judge_candidates(dets, 1440, 1080)
    labels = {"judges": {cands[0]["stable_id"]: {"judge": 4},
                         cands[1]["stable_id"]: {"judge": 4},
                         cands[2]["stable_id"]: {"judge": 2}}}
    render_entry_review(segs, dets, out, detections_fp="FP-C",
                        entry_labels=labels)      # 不抛
    html = out.read_text(encoding="utf-8")
    assert "专家 4" in html
    assert "同时被指为" in html, "页面上没有把冲突说出来"
    blob = json.loads(re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL).group(1))
    got = {c["stable_id"]: c["judge"] for c in blob["candidates"]}
    # 冲突的两位降级为「未指号」（回到几何基线），第三位的 2 号保留
    assert got[cands[2]["stable_id"]] == 2, got


def test_the_conflict_note_names_what_is_visible_on_the_page(tmp_path, monkeypatch):
    """重号提示要给**页面上看得到的东西**（用户 2026-09-19）。

    原先只印内部标识 `judge#271,557`，用户的原话是「人工不知道是哪两个 track
    冲突了」——而评审人候选是跨段按位置归并的，每一位由十来个 (段, track) 拼成，
    根本没有「那个 track」。页面上写的是「几何基线第 N 位 · 位置 cx,y2 · N 帧」，
    提示就该说这些。
    """
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    out = tmp_path / "e.html"
    frame = _write_frame(tmp_path)
    segs = [mkseg(order=o, presenter_num=1) for o in range(1, 7)]
    dets = []
    for o in range(1, 7):
        for i, (cx, y2) in enumerate([(200.0, 900.0), (320.0, 780.0), (450.0, 660.0)]):
            dets += track(20 + i, cx, 0.35 * 1080, 6, frame, order=o,
                          kind="seated", bottom=y2)
    cands = judge_candidates(dets, 1440, 1080)
    labels = {"judges": {cands[0]["stable_id"]: {"judge": 4},
                         cands[2]["stable_id"]: {"judge": 4}}}
    render_entry_review(segs, dets, out, detections_fp="FP-C", entry_labels=labels)
    html = out.read_text(encoding="utf-8")
    # 按结构边界切，不要「往前数 200 个字符」——那会落进上一段，而 `</p>`
    # 先撞上的是上一段的收尾，切出来的片段里根本没有提示正文。
    note = html[html.index('<p class="judgeConflict">'):]
    note = note[:note.index("</p>")]
    assert "几何基线第 1 位" in note and "几何基线第 3 位" in note, note
    for c in (cands[0], cands[2]):
        assert f"{c['median_cx']:.0f},{c['median_y2']:.0f}" in note, (c, note)
        assert str(c["n_frames"]) in note, note


def test_a_judge_number_survives_export(tmp_path, monkeypatch):
    """整张标成「专家 N」时，那个 N 必须导出（用户 2026-09-19，电子27）。

    实测：用户在多处把候选标成专家 5，文件里却全是 `seat: null`，于是
    `judge_donations_from_candidates` 一条都收不进去——他标了等于没标，而
    一处不报错。

    根因是导出那一段的白名单写死了「创业者 + 上/下一组创业者」。这一行上方的
    注释记着同一个 bug 之前为相邻角色发生过一次；这次是「专家」又漏了一遍。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    blob = _segments_blob_from_html(html)[0]
    cand = blob["candidates"][0]
    ck = f"{blob['order']}::{cand['stable_id']}"
    payload = _run_js_with_store(
        html, {ck: {"role": "专家", "seat": "5"}}, "buildLabelsPayload()")
    rec = payload[str(blob["order"])][cand["stable_id"]]
    assert rec["role"] == "专家"
    assert rec["seat"] == 5, rec


def test_roles_without_a_seat_still_export_null(tmp_path, monkeypatch):
    """工作人员/不确定仍然不带座位号——他们本来就不占座位。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    blob = _segments_blob_from_html(html)[0]
    cand = blob["candidates"][0]
    ck = f"{blob['order']}::{cand['stable_id']}"
    for role in ("工作人员", "不确定"):
        payload = _run_js_with_store(
            html, {ck: {"role": role, "seat": "2"}}, "buildLabelsPayload()")
        rec = payload[str(blob["order"])][cand["stable_id"]]
        assert rec["seat"] is None, (role, rec)


def test_every_role_that_shows_a_seat_control_exports_its_seat(tmp_path, monkeypatch):
    """凡是页面上显示座位号控件的身份，导出时都必须带上那个号。

    这条把「显示」与「导出」两份名单绑在一起。它们分开维护已经出过两次事：
    先是相邻角色（上/下一组创业者），再是「专家」——两次都是页面上选得了、
    文件里是 null，而一处不报错。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path, n_seg=6)
    blob = _segments_blob_from_html(html)[0]
    cand = blob["candidates"][0]
    ck = f"{blob['order']}::{cand['stable_id']}"
    roles = _run_js_with_store(html, {}, "ROLES")
    for role in roles:
        shows = _run_js_with_store(
            html, {ck: {"role": role}},
            f"candidateCard(SEGMENTS[0],SEGMENTS[0].candidates[0],null)"
            f".indexOf('display:none') < 0")
        payload = _run_js_with_store(
            html, {ck: {"role": role, "seat": "1"}}, "buildLabelsPayload()")
        rec = payload[str(blob["order"])][cand["stable_id"]]
        if shows:
            assert rec["seat"] == 1, f"{role}：页面上能选号，导出却丢了"


def test_export_carries_the_page_code_fingerprint(tmp_path):
    """导出要带上**导出它的那份页面代码**的指纹。

    用户 2026-09-19 实测电子27：修好「专家的号被导出过滤掉」之后重出了页面，
    他重新导出，文件里那五条专家的 seat **还是 null**。我在命令行上拿那份
    真页面跑 `buildLabelsPayload()`，导出的是 `seat: 5`——页面是对的，文件
    是错的，两件事都成立，而我没有任何办法判断这份文件到底是哪份页面导出的
    （浏览器很可能还开着改版之前的那个标签页）。

    `detections_fingerprint` 帮不上忙：它标的是**检测结果**，重出页面时检测
    没重跑，指纹一字不变。所以另立一个标页面代码的指纹。
    """
    html = _render_one(tmp_path)
    sid = _segments_blob_from_html(html)[0]["candidates"][0]["stable_id"]
    payload = _run_js_with_store(
        html, {f"2::{sid}": {"role": "创业者", "seat": "1"}}, "buildLabelsPayload()")
    fp = payload["_meta"].get("page_code_fp")
    assert fp, f"_meta 里没有 page_code_fp：{payload['_meta']}"
    assert fp == entryreview.page_code_fp(), (
        f"导出带的指纹要等于当前页面代码的指纹，实得 {fp!r} "
        f"≠ {entryreview.page_code_fp()!r}")
    assert payload["_meta"].get("page_built_at"), "还要带一个人能读的生成时刻"


def test_page_code_fingerprint_tracks_the_page_source():
    """指纹必须**跟着页面代码变**，否则它标不出「这是旧页面导出的」。

    写死一个常量、或者拿版本号当指纹，都能让上一条测试通过而这条红——
    那样的指纹永远相等，正好在需要它报警的时候一声不吭。
    """
    before = entryreview.page_code_fp()
    orig = entryreview._TEMPLATE
    try:
        entryreview._TEMPLATE = orig + "\n<!-- 改一个字 -->\n"
        assert entryreview._TEMPLATE != orig, "改写模板失败，这条测试没测到东西"
        assert entryreview.page_code_fp() != before, (
            "改了页面代码，指纹却没变——它标不出新旧")
    finally:
        entryreview._TEMPLATE = orig
    assert entryreview.page_code_fp() == before, "恢复后指纹要回到原值"


def test_filled_seat_card_shows_only_the_evidence_sheet(tmp_path):
    """已定座位卡片上**只有 2×2**，不再有那张就座单图。

    用户 2026-09-20 实测电子28：「本段已定座位会出现一个空白框是为何」——
    39 个已定座位全是灰方块。单图取自 `keyframes/`，而 `export_keyframes`
    只在**阶段 8（生成画廊）**里调用，人工核实发生在阶段 5 之后：
    **人需要它的时候它必然不存在**，等它出现时人早标完了。

    更要紧的是它与「人核对的图 = 送模型的图」相冲突：送进 VLM 的是
    `build_evidence_sheet` 拼的 2×2，而单图是另一张就座裁剪。把一张模型
    看不到的图摆在卡片最显眼处，正是本项目反复出事的那一类。

    这里**故意把裁剪图准备齐**：只测「没有图时不显示」的话，把单图改回来
    也照样通过——要钉住的是「有也不显示」。
    """
    frame = _write_frame(tmp_path)
    seg = mkseg(order=6, presenter_num=2, company="乙企业")
    dets = (track(7, RIGHT_CX + 200, STANDING_H, 6, frame, order=seg.order)
            # 座位 1 那一位的就座帧：2×2 就是从这些帧拼出来的
            + track(11, RIGHT_CX, STANDING_H, 6, frame, kind="seated",
                    order=seg.order))
    kf = tmp_path / "keyframes"
    _write_jpg(_seat_dir(kf, seg.order, seg.company, 1) / "crop_seated_01_t1.0.jpg", 30)

    out = tmp_path / "entry_review.html"
    render_entry_review([seg], dets, out, seats_by_order={seg.order: {11: 1}},
                        detections_fp="FP-48")
    html = out.read_text(encoding="utf-8")
    row = _segments_blob_from_html(html)[0]["filled_seats"][0]
    assert "img" not in row and "img_source" not in row, (
        f"已定座位记录里不该再有单图字段：{row}")
    assert row.get("sheet"), f"2×2 必须还在：{row}"

    main = _run_page(html)["main"]
    cards = _filled_cards(main)
    assert len(cards) == 1, f"应有一张已定座位卡片，实得 {len(cards)}"
    assert "filledsheet" in cards[0]["img"], (
        f"卡片上显示的那张图应当就是 2×2，实得 {cards[0]['img']!r}")
    assert "无图" not in cards[0]["text"], f"不该再有「无图」字样：{cards[0]['text'][:200]}"
    assert 'class="noimg"' not in main, "灰色占位方块不该再出现在页面上"


def test_render_entry_review_no_longer_takes_keyframes_dir(tmp_path):
    """`keyframes_dir` 这个参数只服务于那张单图，一并去掉。

    留着一个没人读的参数，下一个人会以为「传了就有图」——两个生产调用点
    （run_all 与本模块的 __main__）本来就没传过它，它从来只走默认值。
    """
    import inspect
    sig = inspect.signature(render_entry_review)
    assert "keyframes_dir" not in sig.parameters, (
        f"参数应已删除，实得 {list(sig.parameters)}")


def test_a_candidate_marked_not_a_judge_stays_visible_and_revocable(tmp_path, monkeypatch):
    """标了「不是评审人」之后，这条候选**仍要看得见**，且勾能取消。

    用户 2026-09-20 实测电子28：「专家2 3 点了不是评审人后，就无法撤销显示
    那些帧图了，还需要核实修改」。`judgeSlots()` 里那句
    `if (judgeIsExcluded(c)) return;` 把它从席位和「未入席」两边同时剔除，
    整张卡片连同复选框一起消失——标记留在 localStorage 里，页面上再也没有
    一个能取消它的控件，那几帧也没法复核。

    这正是本项目「删除是一种陈述，不是缺席」那条规矩：人工的否定要**显式
    可见**、可撤回，不能表现为「东西不见了」。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    sid = json.loads(re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
                     .group(1))["candidates"][0]["stable_id"]
    store = {f"judge::{sid}": {"role": "不是评审人"}}
    sec = _run_js_with_store(html, store, "judgesSection()")

    assert sid in sec, f"被标掉的候选仍要在页面上列出来，实得：{sec[:400]}"
    assert "不是评审人" in sec
    # 那张卡片里必须有一个**已勾选**的复选框，取消它即可撤回
    card = sec[sec.index(sid):]
    card = card[:card.index('class="cand judgeCard') ] if 'class="cand judgeCard' in card[1:] else card
    assert "checked" in card, f"复选框要是勾上的状态（否则无从取消）：{card[:400]}"
    assert "saveJudge(" in card, "撤回用的还是同一个 saveJudge"
    assert "jsheetImg" in card or "review_img" in card, (
        f"那几帧要能看到，供人工复核：{card[:400]}")
    assert "未入席的候选" not in card, (
        f"标题要说它是被标掉的，写成「未入席」会让人以为它还能排号：{card[:400]}")


def test_a_candidate_marked_not_a_judge_takes_no_seat(tmp_path, monkeypatch):
    """但它**不占席位**——这正是标它的目的。

    只把「看得见」修好而让它继续占号，等于把用户的否定当没听见。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    cands = json.loads(re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
                       .group(1))["candidates"]
    sid = cands[0]["stable_id"]
    slots = _run_js_with_store(html, {f"judge::{sid}": {"role": "不是评审人"}},
                               "judgeSlots()")
    placed = [c and c["stable_id"] for c in slots["slots"].values()]
    assert sid not in placed, f"被标掉的不该占席位，实得 {placed}"
    assert sid not in [c["stable_id"] for c in slots["extra"]], (
        "也不该混在「未入席的候选」里——那一栏是还能排号的人")
    assert sid in [c["stable_id"] for c in slots.get("excluded", [])], (
        f"要单独归到一处，页面才好把它列出来：{slots.get('excluded')}")


def test_no_excluded_row_when_nobody_is_marked(tmp_path, monkeypatch):
    """没人被标掉时不出现这一栏——常驻的空栏是纯噪声。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    sec = _run_js_with_store(html, {}, "judgesSection()")
    assert "标为「不是评审人」" not in sec, sec[:300]


def test_judge_marks_come_back_from_the_label_file(tmp_path, monkeypatch):
    """标注文件里的评审席标记要能回填进页面——localStorage 不是唯一来源。

    用户 2026-09-21：「已标为『不是评审人』刷新了怎么没找到」。上一条改动让
    被标掉的候选重新显示出来，但 `judgeIsExcluded` 只读 localStorage：换台
    电脑、换个浏览器、清过站点数据，或者像这次一样标记不知何时丢了，页面就
    当他从没标过——那一位照常回到席位上，**页面上没有任何痕迹说他曾被否决**。

    文件那一侧本来是通的：导出会写 `judges[sid].role`，Python 的
    `judge_slots` 也认它。缺的正是回来的那一段——与 `FILE_NO_CANDIDATE`
    同一条规矩：**localStorage 里没有这个键时才回填**，人当下的取舍压过文件。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    sid = json.loads(re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
                     .group(1))["candidates"][0]["stable_id"]

    # 文件里说这一位不是评审人，localStorage 空空如也
    html2 = _judges_page(tmp_path, entry_labels={
        "judges": {sid: {"judge": None, "role": "不是评审人"}}})
    sec = _run_js_with_store(html2, {}, "judgesSection()")
    assert "已标为「不是评审人」" in sec, (
        f"文件里标过的，重开页面必须还看得见：{sec[:400]}")
    slots = _run_js_with_store(html2, {}, "judgeSlots()")
    assert sid in [c["stable_id"] for c in slots.get("excluded", [])], slots.get("excluded")


def test_the_page_beats_the_file_for_judge_marks(tmp_path, monkeypatch):
    """人当下在页面上取消的，压过文件里那条旧的。

    反过来做（文件压过页面）就是「取消了刷新一下又回来」——本项目已经在
    「清除标注」上栽过一次，同一条坑不踩第二遍。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    sid = json.loads(re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
                     .group(1))["candidates"][0]["stable_id"]
    html2 = _judges_page(tmp_path, entry_labels={
        "judges": {sid: {"judge": None, "role": "不是评审人"}}})
    # 页面上已把它改回「是评审人」（显式空串）
    slots = _run_js_with_store(html2, {f"judge::{sid}": {"role": ""}}, "judgeSlots()")
    assert sid not in [c["stable_id"] for c in slots.get("excluded", [])], (
        "页面上撤回之后，文件里那条旧标记不许把它再拉回去")


def test_an_excluded_judge_is_announced_where_the_click_happened(tmp_path, monkeypatch):
    """标掉一位之后，**在他原来的位置上**就要看到怎么撤回。

    用户 2026-09-21：「现在专家1 误触了『不是评审人』，找不到恢复的界面」。
    卡片确实还在（上一条改动），但它落到评审席区块最下方的一栏里，而人的
    视线停在刚才那张卡片原来的地方——那儿现在只剩一句「尚无候选——这一位
    可能没到场，也可能被漏检了」，读起来像是检测漏了，完全看不出是自己刚点的。

    误触的撤销必须在**出事的地方**能看见，不能要求人先知道页面底下另有一栏。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    cands = json.loads(re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
                       .group(1))["candidates"]
    sid = [c["stable_id"] for c in cands if c.get("judge") == 1][0]
    sec = _run_js_with_store(html, {f"judge::{sid}": {"role": "不是评审人"}},
                             "judgesSection()")
    head = sec[:sec.index("已标为「不是评审人」")]
    assert "不是评审人" in head, (
        f"空出来的那一格要说明是被人工标掉的，而不是「尚无候选」：{head[-400:]}")
    assert "#judge-dropped" in head, (
        f"并且要能一键跳到可以取消的地方：{head[-400:]}")


def test_the_dropped_row_is_in_the_table_of_contents(tmp_path, monkeypatch):
    """目录里也要有这一栏——评审席区块在页首，人一滚下去就看不见了。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    html = _judges_page(tmp_path)
    sid = json.loads(re.search(r"const JUDGES=(\{.*?\});\n", html, re.DOTALL)
                     .group(1))["candidates"][0]["stable_id"]
    toc = _run_js_with_store(html, {f"judge::{sid}": {"role": "不是评审人"}},
                             "judgeTocHtml()")
    assert "不是评审人" in toc, f"目录里要有这一行：{toc}"
    assert "#judge-dropped" in toc, f"要能点过去：{toc}"
    clean = _run_js_with_store(html, {}, "judgeTocHtml()")
    assert "不是评审人" not in clean, f"没人被标掉时不出现这一行：{clean}"
