import base64
import csv
import io
import json
from pathlib import Path

from pipeline.annotate import PersonAnnotation
from pipeline.validate import Violation
from pipeline.gallery import render_gallery, annotation_to_row
from pipeline import config


def mkann(pid="seg02_seat1", staff=0):
    return PersonAnnotation(
        person_id=pid, seg_order=2, company="北斗羲和", seat_index=1, presenter_num=3,
        gender_female=0, speaks_pre=1, speaks_qa=0, seg_start=960, seg_end=1836,
        vlm={"formality_1_5": 5, "outer_hue_class": "蓝", "gender_judged": "男",
             "outer_color_name": "藏青", "has_tie": "不可见", "evidence_quality": "good",
             "confidence": 0.9},
        pixel={"hue_class_pixel": "蓝", "hue_class_agree": 1, "outer_hue_corr": 220.0,
               "color_granularity_achieved": "11类", "hue_class_best": "蓝"},
        best_frame_seated="work/frames/02/seated/000100.000.jpg",
        suspect_staff=staff,
    )


def test_row_is_flat_and_has_key_columns():
    row = annotation_to_row(mkann())
    assert row["company"] == "北斗羲和"
    assert row["seat_index"] == 1
    assert row["formality_1_5"] == 5
    assert row["hue_class_best"] == "蓝"
    assert row["color_granularity_achieved"] == "11类"
    assert all(not isinstance(v, (dict, list)) for v in row.values()), "行必须是扁平的"


def test_gallery_html_is_written_and_self_contained(tmp_path):
    out = tmp_path / "gallery.html"
    render_gallery([mkann()], [], out)
    html = out.read_text(encoding="utf-8")
    assert "北斗羲和" in html
    assert "localStorage" in html
    assert "导出人工校对版 CSV" in html, (
        "Task 33：按钮文案改为「导出人工校对版 CSV」，与新增的 result.csv 并存说明呼应"
    )
    assert "http://" not in html and "https://" not in html, "画廊必须自包含，不依赖外部资源"


def test_violations_are_rendered(tmp_path):
    out = tmp_path / "gallery.html"
    v = [Violation("人数闸门", 2, "", "标注 3 人，检出 2 人")]
    render_gallery([mkann()], v, out)
    assert "人数闸门" in out.read_text(encoding="utf-8")


def test_suspect_staff_section_present(tmp_path):
    out = tmp_path / "gallery.html"
    render_gallery([mkann(staff=1)], [], out)
    html = out.read_text(encoding="utf-8")
    assert "疑似工作人员" in html
    assert "恢复" in html, "被剔除者必须可一键恢复"


def test_annotation_to_row_drops_binary_values():
    """bytes（如证据图 PNG 字节）不能漏进展平后的行——原黑名单只挡
    dict/list，bytes 会漏过去，污染 CSV 单元格并让 json.dumps 崩溃。"""
    a = mkann()
    a.vlm["evidence_png"] = b"\x89PNG\r\n\x1a\n"
    a.vlm["ok_str"] = "still here"
    row = annotation_to_row(a)
    assert "evidence_png" not in row, "bytes 值必须被过滤掉，不能进入展平后的行"
    assert row["ok_str"] == "still here", "正常字符串值不应被误伤"
    json.dumps(row, ensure_ascii=False)  # 不应抛 TypeError


def test_storage_key_is_namespaced_per_video():
    """localStorage 键必须按视频（config.SHEET）命名空间隔离，否则跨视频
    打开两份画廊会互相覆盖人工校对结果——person_id 形如 seg02_seat1，
    跨视频必然碰撞。"""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "gallery.html"
        render_gallery([mkann()], [], out)
        html = out.read_text(encoding="utf-8")

    assert f"dress_review_v1::{config.SHEET}" in html
    assert "KEY='dress_review_v1'" not in html, "不能残留未命名空间化的裸赋值"
    assert 'KEY="dress_review_v1"' not in html, "不能残留未命名空间化的裸赋值"


def test_company_name_is_escaped_in_group_header():
    """企业名来自 Book1.xlsx，含 & < " 的名称若不转义会破坏分组标题所在
    的 HTML 结构（XSS / 结构破坏）。

    转义发生在浏览器执行 JS 模板字面量时（`` `<h2>${esc(c)}</h2>` ``），
    不发生在 Python 生成 HTML 的阶段——所以静态文件里永远不会出现
    "&amp;"/"&lt;" 字面量，无论修没修都不会出现。能在 Python 层面真正
    锁住这个缺陷的办法，是断言分组标题这一插值点的 JS 源码确实调用了
    esc()，而不是直接插值裸的 `c`。企业名原文（含 & < "）会以 JSON 字符
    串字面量的形式出现在 __ROWS__ 数据里，这是预期且安全的（位于
    <script> 标签内部的字符串字面量中，不会被 HTML 解析器当作标签），
    不代表分组标题本身没有转义，所以不能用"页面里还有没有 <script> 子串"
    这种粗糙的断言，那种断言修没修都会通过。
    """
    import tempfile
    from pathlib import Path

    a = mkann()
    a.company = 'A & B <script>"x"'
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "gallery.html"
        render_gallery([a], [], out)
        html = out.read_text(encoding="utf-8")

    assert "`<h2>${esc(c)}</h2>`" in html, "分组标题必须对企业名调用 esc() 转义"
    assert "`<h2>${c}</h2>`" not in html, "不能存在未转义的裸插值点"


# ---------------------------------------------------------------------------
# Task 25 · 缺陷一 —— 复核图几乎不含创业者：best_frame_seated 是 1440x1080
# 整帧缩成 520x390，人只占最右一小条。build_evidence_sheet 生成的 2x2 红框
# 证据图才是给人工复核用的正确素材，应优先内联进画廊缩略图。
# ---------------------------------------------------------------------------

def _tiny_png_bytes(color=(10, 20, 30)) -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), color).save(buf, format="PNG")
    return buf.getvalue()


def test_gallery_thumbnail_uses_evidence_sheet(tmp_path):
    """现行实现下必红：gallery.py:81 无条件用 best_frame_seated 整帧。

    img 的 src 是浏览器执行 JS 模板字面量时才拼出来的（`` `data:${r._thumb_type
    ||'image/jpeg'};base64,${r._thumb}` ``），不会以成品字符串的形式出现在这份
    静态文件里——所以既要检查喂给 JS 的 ROWS 数据本身（_thumb/_thumb_type 字段
    确实是证据图而非整帧），也要检查 JS 源码里 media type 确实绑定到
    `r._thumb_type`（而非写死 jpeg，那是控制器变异清单第 (c) 条）。
    """
    a = mkann()
    a.pixel["_evidence_png"] = _tiny_png_bytes((250, 30, 30))
    out = tmp_path / "gallery.html"
    render_gallery([a], [], out)
    html = out.read_text(encoding="utf-8")

    import base64
    expected_b64 = base64.b64encode(a.pixel["_evidence_png"]).decode()

    rows = _rows_blob_from_html(html)
    row = next(r for r in rows if r["person_id"] == a.person_id)
    assert row["_thumb"] == expected_b64, "应内联证据图本身的字节，而不是整帧缩略图"
    assert row["_thumb_type"] == "image/png", "证据图是 PNG，media type 不能写成 jpeg"

    assert "data:${r._thumb_type" in html, (
        "JS 渲染 <img> 时必须把 media type 绑定到 r._thumb_type，不能写死 jpeg"
    )


def test_gallery_falls_back_when_no_evidence_sheet(tmp_path):
    """没有证据图时（例如缺座位的占位行）仍要能正常出图、不崩。"""
    a = mkann()
    assert "_evidence_png" not in a.pixel
    out = tmp_path / "gallery.html"
    render_gallery([a], [], out)      # 不应抛异常
    html = out.read_text(encoding="utf-8")
    assert "北斗羲和" in html

    # 连 best_frame_seated 也没有的极端情况（真实的缺座位占位行）
    a2 = mkann(pid="seg02_seat2")
    a2.best_frame_seated = ""
    a2.pixel.pop("_evidence_png", None)
    out2 = tmp_path / "gallery2.html"
    render_gallery([a2], [], out2)    # 不应抛异常
    assert "北斗羲和" in out2.read_text(encoding="utf-8")


def test_evidence_png_key_dropped_from_row_by_annotation_to_row():
    """单元级护栏：pixel 里的 _evidence_png 必须被 annotation_to_row 的
    标量白名单过滤掉，不能进入展平后的行（进而不能进 CSV/JSON）。"""
    import json
    a = mkann()
    a.pixel["_evidence_png"] = _tiny_png_bytes()
    row = annotation_to_row(a)
    assert "_evidence_png" not in row
    json.dumps(row, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Task 25 · 缺陷二（改稿）—— 用户改主意：has_jacket 直接废弃（无下游依赖），
# 改为让 formality_1_5 的每一档含义在复核界面里自解释。量表文案唯一定义在
# pipeline.annotate.FORMALITY_LABELS，PROMPT/画廊/CSV 三处从它派生，不允许
# 各抄一份（Ruling R39 的教训）。
# ---------------------------------------------------------------------------

def _fields_blob_from_html(html: str) -> list:
    """从生成的静态 HTML 里把 __FIELDS__ 替换后的真实 JSON 数据抠出来解析，
    而不是用正则去匹配 JS 源码文本——这样能拿到 Python 对象，直接断言
    formality_1_5 选项的 value 部分究竟是不是纯数字。"""
    import re
    m = re.search(r"FIELDS=(.*?);\nconst KEY", html, re.DOTALL)
    assert m is not None, "未在 HTML 里找到 FIELDS 赋值"
    return json.loads(m.group(1))


def _formality_options_from_html(html: str) -> list:
    fields = _fields_blob_from_html(html)
    formality = next(f for f in fields if f[0] == "formality_1_5")
    # json.loads 把元组还原成 list；统一转回 (value, label) 元组，
    # 方便测试代码按元组比较。
    return [tuple(o) for o in formality[1]]


def _rows_blob_from_html(html: str) -> list:
    """从生成的静态 HTML 里把 __ROWS__ 替换后的真实 JSON 数据抠出来解析。"""
    import re
    m = re.search(r"const ROWS=(.*?), VIOL=(.*?), FIELDS=(.*?);\nconst KEY", html, re.DOTALL)
    assert m is not None, "未在 HTML 里找到 ROWS 赋值"
    return json.loads(m.group(1))


def test_gallery_formality_options_show_meanings(tmp_path):
    out = tmp_path / "gallery.html"
    render_gallery([mkann()], [], out)
    html = out.read_text(encoding="utf-8")
    opts = _formality_options_from_html(html)
    labels = [ol for _ov, ol in opts]
    assert any("成套西装" in l for l in labels)
    assert any("翻领短袖衫" in l for l in labels)  # Task 43：Polo 已改为中文
    assert len(opts) == 5


def test_gallery_formality_option_values_stay_numeric(tmp_path):
    """最重要的护栏：<select> 的 value 属性必须是纯数字 "1".."5"，不能是
    带含义的中文标签全文——否则会静默把脏值污染进导出的 result.csv。"""
    out = tmp_path / "gallery.html"
    render_gallery([mkann()], [], out)
    html = out.read_text(encoding="utf-8")

    # (1) 数据层：FIELDS 里 formality_1_5 每个选项的 value 部分必须是纯数字。
    opts = _formality_options_from_html(html)
    values = [ov for ov, _ol in opts]
    assert values == ["1", "2", "3", "4", "5"]
    for ov, ol in opts:
        assert ol != ov, "标签不能退化成裸值，说明含义没有渲染出来"

    # (2) 渲染层：JS 源码必须把 <option> 的 value 属性显式绑定到 ov（选项的
    # 值部分），而不是绑定到 ol（标签）或干脆不写 value 属性——后者会让
    # 浏览器把 <option> 的可见文本当成 value，导出 CSV 时污染数据列。
    assert 'value="${esc(ov)}"' in html, (
        "JS 渲染 <option> 时必须显式写出 value=\"${esc(ov)}\"，绑定到选项值而非标签"
    )


def test_gallery_shows_formality_legend(tmp_path):
    """画廊顶部要有静态可见（不依赖 JS 执行）的量表说明区，复核时随时对照。"""
    out = tmp_path / "gallery.html"
    render_gallery([mkann()], [], out)
    html = out.read_text(encoding="utf-8")
    # 只看 <script> 之前的静态部分——证明这是页面本身的可见内容，
    # 不是只存在于喂给 JS 的 JSON 数据里。
    static_part = html.split("<script>")[0]
    assert "正式程度量表" in static_part
    from pipeline.annotate import FORMALITY_LABELS
    for n, label in FORMALITY_LABELS.items():
        assert f"{n} = {label}" in static_part, f"档位 {n} 未出现在静态说明区"


def test_gallery_formality_labels_come_from_shared_source(tmp_path, monkeypatch):
    """控制器变异清单第 (g) 条：若 gallery.py 把量表文案另抄一份字面量，
    而不是从 pipeline.annotate.FORMALITY_LABELS 现取，这条必须能抓到——
    通过 monkeypatch 该字典并确认画廊输出（下拉框选项与静态说明区）
    都跟着变化来验证。"""
    import pipeline.annotate as annotate_mod

    monkeypatch.setattr(annotate_mod, "FORMALITY_LABELS", {1: "仅测试用占位标签"})
    out = tmp_path / "gallery.html"
    render_gallery([mkann()], [], out)
    html = out.read_text(encoding="utf-8")

    opts = _formality_options_from_html(html)
    assert opts == [("1", "1 = 仅测试用占位标签")], (
        "下拉框选项未跟随 FORMALITY_LABELS 变化，说明 gallery.py 另抄了一份"
    )
    static_part = html.split("<script>")[0]
    assert "仅测试用占位标签" in static_part, (
        "静态说明区未跟随 FORMALITY_LABELS 变化，说明 gallery.py 另抄了一份"
    )


def test_gallery_dropdown_follows_formality_labels(tmp_path):
    """Task 43：不依赖 monkeypatch，直接用当前真实的 FORMALITY_LABELS（用户
    这次给的五档新文案）逐档、逐字比对画廊下拉框选项——比单纯"跟着变化"更
    严格，能同时抓住"忘了更新其中一档"这类局部漂移（变异清单(b)）。"""
    from pipeline.annotate import FORMALITY_LABELS
    out = tmp_path / "gallery.html"
    render_gallery([mkann()], [], out)
    html = out.read_text(encoding="utf-8")
    opts = _formality_options_from_html(html)
    expected = [(str(n), f"{n} = {label}") for n, label in FORMALITY_LABELS.items()]
    assert opts == expected


def test_result_row_has_formality_label():
    from pipeline.annotate import FORMALITY_LABELS
    row = annotation_to_row(mkann())
    assert row["formality_label"] == f"5 = {FORMALITY_LABELS[5]}"


def test_result_row_formality_label_empty_when_no_formality():
    a = mkann()
    a.vlm.pop("formality_1_5", None)
    row = annotation_to_row(a)
    assert row.get("formality_1_5") in (None, "", 0, "0") or "formality_1_5" not in row
    assert row["formality_label"] == ""


# ---------------------------------------------------------------------------
# Task 32 · 要求三（真 bug，优先级最高）—— HTML <select> 在字段值为空字符串
# 时没有任何 option 匹配，浏览器默认选中第一个 option，未识别的人会显示成
# 「西装外套 / 有 / 红 / …」，看起来像识别成功了；用户一旦改动该卡片其他
# 字段，这个假值就会被 save() 存进 store 并在导出 CSV 时写进 result.csv。
#
# 这些测试直接在 Node 里执行画廊真实产出的 card() 函数（纯字符串拼接，不
# 依赖 DOM），拿到浏览器会渲染出的那段 HTML 字符串本身来断言——而不是只
# 匹配 Python 模板源码里的字面量，这样才能真正锁住“空值时选中谁”这个
# 运行期行为。
# ---------------------------------------------------------------------------

import re
import subprocess


def _extract_script(html: str) -> str:
    m = re.search(r"<script>(.*)</script>", html, re.DOTALL)
    assert m is not None, "未在 HTML 里找到 <script> 区块"
    return m.group(1)


def _run_card_js(html: str, person_id: str) -> str:
    """在 Node 里跑一遍生成 HTML 里真实的 <script> 源码（用最小 shim 顶掉
    浏览器全局对象 document/localStorage，其余代码一字不改），然后调用
    card() 拿到该 person_id 对应卡片实际会渲染出的 HTML 字符串。"""
    script = _extract_script(html)
    harness = (
        "const localStorage = { getItem: () => null, setItem: () => {} };\n"
        "const document = { getElementById: () => ({ innerHTML: '' }) };\n"
        + script
        + f"\nconsole.log(card(ROWS.find(r=>r.person_id==={json.dumps(person_id)})));\n"
    )
    result = subprocess.run(
        ["node", "-e", harness], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, f"Node 执行画廊 JS 失败：{result.stderr}"
    return result.stdout


def _select_html_for_field(card_html: str, field: str) -> str:
    """从 card() 产出的 HTML 里，抠出某个字段对应 <select>...</select> 的
    完整片段（label 紧跟在 select 之前，用它定位是哪个字段）。"""
    m = re.search(
        rf"<label>{re.escape(field)}</label>\s*<select[^>]*>(.*?)</select>",
        card_html, re.DOTALL,
    )
    assert m is not None, f"未在卡片 HTML 里找到字段 {field} 对应的 <select>"
    return m.group(1)


def _first_option(select_inner_html: str) -> tuple[str, bool]:
    """返回 select 内第一个 <option> 的 (value, 是否带 selected)。"""
    m = re.search(r'<option\s+value="([^"]*)"\s*(selected)?\s*>', select_inner_html)
    assert m is not None, "未找到任何 <option>"
    return m.group(1), bool(m.group(2))


def test_empty_field_renders_empty_placeholder_option(tmp_path):
    """字段值为空的标注，该 select 的第一个 option 必须是 value="" 且被
    selected——现行实现（无占位项）下必红：第一个 option 会是 FIELDS 里
    真实选项列表的第一项，且大概率不会带 selected（除非该项的 value 恰好
    等于空字符串这个巧合，正常情况下不会）。"""
    a = mkann()
    a.vlm.pop("formality_1_5", None)  # formality_1_5 留空 = 未识别
    out = tmp_path / "gallery.html"
    render_gallery([a], [], out)
    html = out.read_text(encoding="utf-8")

    card_html = _run_card_js(html, a.person_id)
    select_inner = _select_html_for_field(card_html, "formality_1_5")
    value, selected = _first_option(select_inner)
    assert value == "", "空值字段的 select 第一个 option 的 value 必须是空字符串"
    assert selected, "空值字段的 select 第一个 option（占位项）必须被 selected"


def test_placeholder_option_value_is_empty_not_label(tmp_path):
    """护栏：占位项的 value 必须是空字符串，不能是「（未识别）」之类的
    文案本身——否则用户改动同一卡片的其他字段触发 save() 整行落库后，
    这个占位文案会被写进导出 CSV 的该字段列，而不是留空。"""
    a = mkann()
    a.vlm.pop("formality_1_5", None)
    out = tmp_path / "gallery.html"
    render_gallery([a], [], out)
    html = out.read_text(encoding="utf-8")

    card_html = _run_card_js(html, a.person_id)
    select_inner = _select_html_for_field(card_html, "formality_1_5")
    value, _selected = _first_option(select_inner)
    assert value == "", f"占位项 value 必须是空字符串，实际是 {value!r}"
    assert "未识别" not in value


def test_identified_field_still_selects_its_value(tmp_path):
    """有值的字段仍要正确选中该值本身，不能被新加的占位项抢走 selected。"""
    a = mkann()  # formality_1_5 = 5（有值）
    out = tmp_path / "gallery.html"
    render_gallery([a], [], out)
    html = out.read_text(encoding="utf-8")

    card_html = _run_card_js(html, a.person_id)
    select_inner = _select_html_for_field(card_html, "formality_1_5")
    value, selected = _first_option(select_inner)
    assert value == "", "第一个 option 始终应该是占位项本身"
    assert not selected, "有值时占位项不能被 selected"

    m = re.search(r'<option value="5"\s*(selected)?\s*>', select_inner)
    assert m is not None and m.group(1), "formality_1_5=5 时，值为 5 的 option 必须被 selected"


def test_unidentified_row_is_visually_marked(tmp_path):
    """整行都没识别到人（formality_1_5 与 outer_garment_type 均为空）的
    卡片必须带有区分类名（.unidentified）与醒目提示文字——不能只靠下拉框
    恰好为空让用户自己发现。"""
    a = mkann()
    a.vlm.pop("formality_1_5", None)
    assert "outer_garment_type" not in a.vlm  # mkann() 本就没给这个字段
    out = tmp_path / "gallery.html"
    render_gallery([a], [], out)
    html = out.read_text(encoding="utf-8")

    card_html = _run_card_js(html, a.person_id)
    assert "unidentified" in card_html, "未识别行的卡片必须带 unidentified 类名"
    assert "未识别到人" in card_html, "未识别行必须有醒目的中文提示文字"

    # 对照组：有值的字段不应触发这个标记。
    b = mkann(pid="seg02_seat3")
    b.vlm["outer_garment_type"] = "西装外套"
    out2 = tmp_path / "gallery2.html"
    render_gallery([b], [], out2)
    html2 = out2.read_text(encoding="utf-8")
    card_html2 = _run_card_js(html2, b.person_id)
    assert "unidentified" not in card_html2, "有识别结果的行不应带 unidentified 类名"


def test_gallery_marks_backfilled_rows(tmp_path):
    """Task 44：source='entryexit' 的行必须在卡片上显眼标出衣着来自进出场
    画面，seat_uncertain=1 时还要追加座位号未经确认的提示；source='seated'
    的普通行不应出现这个标记（对照组）。"""
    a = mkann(pid="seg02_seat1")
    a.source = "entryexit"
    a.seat_uncertain = 1
    out = tmp_path / "gallery.html"
    render_gallery([a], [], out)
    html = out.read_text(encoding="utf-8")

    card_html = _run_card_js(html, a.person_id)
    assert "衣着取自进出场画面" in card_html, "递补行必须标出衣着来自进出场画面"
    assert "座位号未经确认" in card_html, "seat_uncertain=1 时必须追加座位号未确认的提示"

    b = mkann(pid="seg02_seat2")
    assert b.source == "seated"
    out2 = tmp_path / "gallery2.html"
    render_gallery([b], [], out2)
    html2 = out2.read_text(encoding="utf-8")
    card_html2 = _run_card_js(html2, b.person_id)
    assert "衣着取自进出场画面" not in card_html2, "普通就座行不应出现递补标记"
    assert "座位号未经确认" not in card_html2


# ---------------------------------------------------------------------------
# Task 32 · 要求一 —— 图尽量大、默认不缩放；只有整份画廊预估体积超过
# config.GALLERY_MAX_MB 才回退到缩放。
# ---------------------------------------------------------------------------


def _noisy_square_png_bytes(size: int) -> bytes:
    """随机噪声像素，PNG 压缩不掉——纯色方块几乎不占字节，测不出「超预算」。"""
    import io as _io
    import os
    from PIL import Image
    im = Image.frombytes("RGB", (size, size), os.urandom(size * size * 3))
    buf = _io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def test_evidence_image_is_not_downscaled_by_default(tmp_path):
    """现行实现下必红：_evidence_thumb 超过 120000 字节的 b64 阈值就会缩到
    520x520。一张 800x800 的证据图理应超过该旧阈值，但只要没超过
    config.GALLERY_MAX_MB 总预算，就必须原样内联（体积虽大但远低于
    12MB 上限）。"""
    a = mkann()
    # 用随机噪声而非纯色方块：纯色 PNG 几乎压缩到接近 0 字节，无论新旧实现
    # 都不会触发缩放，测不出差异。噪声图 b64 后妥妥超过旧阈值 120000 字节，
    # 但仍远低于新预算 config.GALLERY_MAX_MB=12MB，能真正区分新旧行为。
    a.pixel["_evidence_png"] = _noisy_square_png_bytes(800)
    out = tmp_path / "gallery.html"
    render_gallery([a], [], out)
    html = out.read_text(encoding="utf-8")

    rows = _rows_blob_from_html(html)
    row = next(r for r in rows if r["person_id"] == a.person_id)
    assert row["_thumb_type"] == "image/png", "未超预算时不应把 PNG 转成 JPEG"

    import io as _io
    from PIL import Image
    decoded = Image.open(_io.BytesIO(base64.b64decode(row["_thumb"])))
    assert decoded.size == (800, 800), "未超预算时证据图必须按原尺寸内联，不缩放"


def test_images_are_downscaled_only_when_over_budget(tmp_path, monkeypatch):
    """把预算调到很小，逼总体积超限，断言确实发生了缩放，且页面上有
    体积超限提示文字（顶部 + stdout 两处，Task 32 要求）。"""
    monkeypatch.setattr(config, "GALLERY_MAX_MB", 0.01)  # 约 10KB，一张 800x800 噪声图就爆
    a = mkann()
    a.pixel["_evidence_png"] = _noisy_square_png_bytes(800)
    out = tmp_path / "gallery.html"
    render_gallery([a], [], out)
    html = out.read_text(encoding="utf-8")

    rows = _rows_blob_from_html(html)
    row = next(r for r in rows if r["person_id"] == a.person_id)

    import io as _io
    from PIL import Image
    decoded = Image.open(_io.BytesIO(base64.b64decode(row["_thumb"])))
    assert decoded.size != (800, 800), "超预算时必须发生缩放"
    assert max(decoded.size) < 800, "缩放后的图片应比原图小"

    static_part = html.split("<script>")[0]
    assert "为控制体积已缩放" in static_part, "超限缩放时页面顶部必须有提示"
    assert "keyframes/" in static_part


def test_downscale_notice_printed_to_stdout(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "GALLERY_MAX_MB", 0.01)
    a = mkann()
    a.pixel["_evidence_png"] = _noisy_square_png_bytes(800)
    out = tmp_path / "gallery.html"
    render_gallery([a], [], out)
    captured = capsys.readouterr()
    assert "为控制体积已缩放" in captured.out, "超限缩放必须在 stdout 明确提示"


# ---------------------------------------------------------------------------
# Task 32 · 要求二 —— 选项区改为横向流式排布，紧凑不喧宾夺主。
# ---------------------------------------------------------------------------


# Task 33 用户反馈 Task 32 的横向流式布局「选项都挤到一起了，图片右边剩了
# 很大的空白」——下面这条曾锁住的 flex-wrap 横排正是被投诉的对象，改为
# 竖向排列后随之更新（Task 33 要求新增测试之一）。


def test_fields_are_stacked_vertically(tmp_path):
    out = tmp_path / "gallery.html"
    render_gallery([mkann()], [], out)
    html = out.read_text(encoding="utf-8")
    static_part = html.split("<script>")[0]

    m = re.search(r"\.fields\{([^}]*)\}", static_part)
    assert m is not None, "未找到 .fields 的 CSS 规则"
    rule = m.group(1)
    assert "flex" in rule, ".fields 仍应是 flex 容器"
    assert "column" in rule, ".fields 必须竖向排列（flex-direction:column），不能横向挤在一起"
    assert "wrap" not in rule, "竖向排列不应再依赖横向换行（flex-wrap）"


def test_card_info_column_fills_remaining_width(tmp_path):
    """卡片布局「图片在左、字段区在右」，字段区所在的信息列必须占满剩余
    宽度（flex:1），不能让图片右边空出一大片——这正是用户投诉的原因。"""
    out = tmp_path / "gallery.html"
    render_gallery([mkann()], [], out)
    html = out.read_text(encoding="utf-8")
    static_part = html.split("<script>")[0]

    m = re.search(r"\.info\{([^}]*)\}", static_part)
    assert m is not None, "未找到卡片信息列（.info）的 CSS 规则"
    assert "flex:1" in m.group(1).replace(" ", ""), (
        ".info 必须 flex:1 占满剩余宽度，否则字段区旁边会留出空白"
    )


# ---------------------------------------------------------------------------
# Task 33 · 改动四 —— 人工校对版 CSV 另存，带组名，与 VLM 版并存。
#
# exportCsv() 的行为无法只从 Python 模板源码断言（文件名/human_edited_fields
# 列都是运行期算出来的），沿用 Task 32 已有的 Node 直跑 <script> 手法：
# 补一套比 _run_card_js 更完整的浏览器全局对象 shim（document.createElement /
# URL.createObjectURL / Blob），真正调用 exportCsv()，从它构造的 Blob 与
# <a download> 属性里读出实际产出的文件名与 CSV 内容。
# ---------------------------------------------------------------------------


def _run_export_js(html: str, store: dict) -> dict:
    """在 Node 里跑一遍画廊真实的 <script> 源码，预置 localStorage 里的
    `store`（模拟用户已经做过的人工校对），调用 exportCsv()，返回
    {"download": <a>.download 的值, "content": 传给 Blob 的完整字符串}。"""
    script = _extract_script(html)
    harness = (
        f"const __STORE__ = {json.dumps(store, ensure_ascii=False)};\n"
        "const localStorage = { getItem: () => JSON.stringify(__STORE__), setItem: () => {} };\n"
        "const document = { getElementById: () => ({ innerHTML: '' }),\n"
        "  createElement: () => { const el = { download: '', href: '', click(){} }; "
        "global.__LAST_A__ = el; return el; } };\n"
        "class Blob { constructor(parts, opts) { global.__LAST_BLOB__ = { parts, opts }; } }\n"
        "const URL = { createObjectURL: () => 'blob:mock' };\n"
        + script
        + "\nexportCsv();\n"
        "console.log(JSON.stringify({download: __LAST_A__.download, "
        "content: __LAST_BLOB__.parts.join('')}));\n"
    )
    result = subprocess.run(
        ["node", "-e", harness], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, f"Node 执行 exportCsv 失败：{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_export_filename_contains_sheet_name(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SHEET", "测试组4月1日视频")
    a = mkann()
    out = tmp_path / "gallery.html"
    render_gallery([a], [], out)
    html = out.read_text(encoding="utf-8")

    result = _run_export_js(html, {})
    assert config.SHEET in result["download"], (
        f"导出文件名应含组名，实得 {result['download']!r}"
    )
    assert result["download"].startswith("result_人工校对_"), (
        f"导出文件名应以 result_人工校对_ 开头，实得 {result['download']!r}"
    )
    assert result["download"] != "result.csv", "不能再是与组无关的旧文件名"


def test_export_includes_human_edited_fields_column(tmp_path):
    """用户改过 formality_1_5 一个字段后导出：CSV 必须新增
    human_edited_fields 列，且该行的值里出现 formality_1_5（真的记录了
    改动过什么，而不是恒为空/恒为固定占位）。未被人工碰过的行则应留空。"""
    a = mkann(pid="seg02_seat1")
    out = tmp_path / "gallery.html"
    render_gallery([a], [], out)
    html = out.read_text(encoding="utf-8")

    store = {a.person_id: {"formality_1_5": "3", "human_verified": 1}}
    result = _run_export_js(html, store)

    content = result["content"].lstrip("﻿")
    rows = list(csv.DictReader(io.StringIO(content)))
    assert "human_edited_fields" in rows[0], "导出 CSV 必须新增 human_edited_fields 列"
    row = next(r for r in rows if r["person_id"] == a.person_id)
    assert row["human_edited_fields"] == "formality_1_5", (
        f"应恰好记录被改动的字段名，实得 {row['human_edited_fields']!r}"
    )


def test_human_edited_fields_excludes_meta_flag_and_stays_empty_when_untouched():
    """护栏：
    1. human_verified 只是「这行被碰过」的元标志，不应作为字段名混进
       human_edited_fields 列，否则每一行只要存过任何字段就会多出一个
       无意义的 'human_verified' 噪声值。
    2. 完全没被人工校对过的行，human_edited_fields 必须是空字符串——不能
       恒为某个固定占位值（变异 (h)：不记录实际改动）。
    """
    a = mkann(pid="seg02_seat1")
    b = mkann(pid="seg02_seat2")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "gallery.html"
        render_gallery([a, b], [], out)
        html = out.read_text(encoding="utf-8")

    store = {a.person_id: {"has_tie": "有", "human_notes": "领带颜色特殊", "human_verified": 1}}
    result = _run_export_js(html, store)
    content = result["content"].lstrip("﻿")
    rows = list(csv.DictReader(io.StringIO(content)))

    edited_row = next(r for r in rows if r["person_id"] == a.person_id)
    edited_fields = set(edited_row["human_edited_fields"].split(","))
    assert edited_fields == {"has_tie", "human_notes"}, (
        f"应恰好是被改过的两个字段，且不含 human_verified，实得 {edited_fields}"
    )

    untouched_row = next(r for r in rows if r["person_id"] == b.person_id)
    assert untouched_row["human_edited_fields"] == "", (
        "未被人工校对过的行，human_edited_fields 必须留空"
    )


# ---------------------------------------------------------------------------
# Task 36 · 改动一 —— 卡片标题与顶部违规清单都要标出 order 序号，方便与
# 页面最上方那份「校验意见/违规清单」按 order 对照，不再只能靠企业名
# 人肉匹配。
# ---------------------------------------------------------------------------


def test_card_title_contains_order_number(tmp_path):
    """卡片标题里必须带 order 序号，且与企业名有明确、可编程区分的分隔
    （不能只是字符串里凑巧出现同样的数字——用专门的 <span class="order-num">
    包裹，断言这个包裹本身，而不是"标题文本里出现了这个数字"这种可能被
    企业名本身碰巧满足的弱断言）。"""
    a = mkann()  # seg_order = 2
    out = tmp_path / "gallery.html"
    render_gallery([a], [], out)
    html = out.read_text(encoding="utf-8")

    card_html = _run_card_js(html, a.person_id)
    m = re.search(r'<span class="order-num">#(\d+)</span>', card_html)
    assert m is not None, "卡片标题里必须有 <span class=\"order-num\"> 包裹的 order 序号"
    assert m.group(1) == str(a.seg_order), f"order 序号应为 {a.seg_order}，实得 {m.group(1)}"
    # order 序号必须出现在企业名之前（用户核对时先看序号再核对企业名）。
    order_pos = card_html.index(m.group(0))
    company_pos = card_html.index(a.company)
    assert order_pos < company_pos, "order 序号应出现在企业名之前"


def test_card_title_order_number_distinguishable_from_company(tmp_path):
    """护栏：假如企业名本身包含数字（例如"第2号企业"），"卡片文本里出现
    了 order 数字"这种粗糙断言会被企业名碰巧满足。用一个企业名里含
    order 数值本身的场景，确认 order-num span 里的数字确实来自 r.order
    字段，而不是从企业名文本里顺带带出来的。"""
    a = mkann()
    a.seg_order = 7
    a.company = "第2号创业企业"  # 企业名里混入了一个不同的数字（2），干扰粗糙断言
    out = tmp_path / "gallery.html"
    render_gallery([a], [], out)
    html = out.read_text(encoding="utf-8")

    card_html = _run_card_js(html, a.person_id)
    m = re.search(r'<span class="order-num">#(\d+)</span>', card_html)
    assert m is not None
    assert m.group(1) == "7", f"order-num 必须是 a.seg_order=7，不能被企业名里的 2 干扰，实得 {m.group(1)}"


def test_violation_list_shows_order(tmp_path):
    """顶部违规清单（#viol）里每条都必须带 order 序号，口径要与卡片标题
    一致——同一个 order 值必须能在两处对上号。"""
    a = mkann()
    v = [Violation("人数闸门", a.seg_order, a.person_id, "标注 3 人，检出 2 人")]
    out = tmp_path / "gallery.html"
    render_gallery([a], v, out)
    html = out.read_text(encoding="utf-8")

    script = _extract_script(html)
    # VIOL 渲染点必须显式引用 v.order（而不是干脆不显示，或者显示别的字段
    # 冒充 order）。
    assert re.search(r"esc\(v\.order\)", script), "违规清单必须显式渲染 v.order"

    # 用真实浏览器等价的方式跑一遍渲染逻辑，而不是只看模板源码字面量：
    # 直接执行 VIOL.map(...) 那一行，拿到实际产出的 HTML 字符串来断言。
    harness = (
        "const localStorage = { getItem: () => null, setItem: () => {} };\n"
        "const document = { getElementById: () => ({ innerHTML: '' }) };\n"
        + script
        + "\nconsole.log(document.getElementById('viol').innerHTML);\n"
    )
    # document.getElementById 在 harness 里是个假实现，直接读不到赋值结果；
    # 改为直接对 VIOL 手工跑一遍同款 map 表达式来验证渲染内容本身。
    harness2 = (
        "const esc = s => String(s??'').replace(/[&<>\"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[c]));\n"
        f"const VIOL = {json.dumps([{'rule': vv.rule, 'order': vv.seg_order, 'person_id': vv.person_id, 'detail': vv.detail} for vv in v], ensure_ascii=False)};\n"
        "console.log(VIOL.map(v=>`<div class=\"viol\"><b>${esc(v.rule)}</b> order ${esc(v.order)} ${esc(v.person_id)} \\u2014 ${esc(v.detail)}</div>`).join(''));\n"
    )
    result = subprocess.run(["node", "-e", harness2], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    viol_html = result.stdout
    assert f"order {a.seg_order}" in viol_html, "违规清单渲染出的 HTML 必须带 order 序号"


# ---------------------------------------------------------------------------
# Task 36 · 改动二 —— 图片按 config.GALLERY_IMG_SCALE 用 CSS 缩放显示，
# 不重新采样像素（重新采样会真的损失像素，不可逆）。
# ---------------------------------------------------------------------------


def _css_root_var(static_part: str, name: str) -> str:
    m = re.search(rf"--{re.escape(name)}\s*:\s*([0-9.]+)", static_part)
    assert m is not None, f"未在 CSS 里找到 --{name} 变量声明"
    return m.group(1)


def test_image_scale_is_css_not_resample(tmp_path):
    """本任务最重要的护栏：内联图片的解码后尺寸必须仍是原图尺寸（证明
    没有重新采样像素），同时 CSS 里必须有基于 config.GALLERY_IMG_SCALE
    的缩放声明（证明缩放是靠 CSS 显示层做的，不是靠改字节数据）。"""
    a = mkann()
    a.pixel["_evidence_png"] = _noisy_square_png_bytes(800)
    out = tmp_path / "gallery.html"
    render_gallery([a], [], out)
    html = out.read_text(encoding="utf-8")

    rows = _rows_blob_from_html(html)
    row = next(r for r in rows if r["person_id"] == a.person_id)

    import io as _io
    from PIL import Image
    decoded = Image.open(_io.BytesIO(base64.b64decode(row["_thumb"])))
    assert decoded.size == (800, 800), (
        "内联图片解码后尺寸必须仍是原图 800x800——若变小说明发生了重采样，"
        "而不是纯 CSS 显示缩放"
    )

    static_part = html.split("<script>")[0]
    scale_var = _css_root_var(static_part, "img-scale")
    assert float(scale_var) == config.GALLERY_IMG_SCALE

    m = re.search(r"\.card img\{([^}]*)\}", static_part)
    assert m is not None, "未找到 .card img 的 CSS 规则"
    rule = m.group(1)
    assert "var(--img-scale)" in rule, ".card img 的宽度必须通过 var(--img-scale) 引用缩放变量，不能写死百分比"


def test_img_scale_comes_from_config(tmp_path, monkeypatch):
    """改 config.GALLERY_IMG_SCALE 后，生成的 CSS 缩放变量要跟着变——
    不能写死 0.8（变异 (d)）。"""
    monkeypatch.setattr(config, "GALLERY_IMG_SCALE", 0.5)
    a = mkann()
    out = tmp_path / "gallery.html"
    render_gallery([a], [], out)
    html = out.read_text(encoding="utf-8")
    static_part = html.split("<script>")[0]
    scale_var = _css_root_var(static_part, "img-scale")
    assert scale_var == "0.5", f"CSS 缩放变量应跟随 config.GALLERY_IMG_SCALE=0.5，实得 {scale_var}"
    assert "0.8" not in static_part.split("--img-scale")[1][:20], (
        "不能残留写死的 0.8"
    )


# ---------------------------------------------------------------------------
# Task 36 · 改动三 —— 右侧选项文字调大到易读区间，同时保持竖排（不许
# 退回 Task 32 那版横向流式，那版被用户否掉过一次）。
# ---------------------------------------------------------------------------


def _css_rule_font_size_px(static_part: str, selector_pattern: str) -> float:
    m = re.search(selector_pattern, static_part)
    assert m is not None, f"未找到 CSS 规则：{selector_pattern}"
    rule = m.group(1)
    fm = re.search(r"font-size\s*:\s*(\d+(?:\.\d+)?)px", rule)
    assert fm is not None, f"规则里没有 font-size(px)：{rule}"
    return float(fm.group(1))


def test_field_font_sizes_are_readable(tmp_path):
    out = tmp_path / "gallery.html"
    render_gallery([mkann()], [], out)
    html = out.read_text(encoding="utf-8")
    static_part = html.split("<script>")[0]

    label_size = _css_rule_font_size_px(static_part, r"\.fields \.field label\{([^}]*)\}")
    select_size = _css_rule_font_size_px(static_part, r"\.fields select\{([^}]*)\}")
    assert label_size >= 14, f"label font-size 应 >=14px，实得 {label_size}px"
    assert select_size >= 14, f"select font-size 应 >=14px，实得 {select_size}px"


def test_fields_remain_vertically_stacked(tmp_path):
    """字号调大之后，字段区仍必须是竖排（flex-direction:column），不能
    为了塞更多字段又退回 Task 32 那版横向流式 flex-wrap 布局——那版已被
    用户明确否掉过一次（Task 33）。"""
    out = tmp_path / "gallery.html"
    render_gallery([mkann()], [], out)
    html = out.read_text(encoding="utf-8")
    static_part = html.split("<script>")[0]

    m = re.search(r"\.fields\{([^}]*)\}", static_part)
    assert m is not None, "未找到 .fields 的 CSS 规则"
    rule = m.group(1)
    assert "column" in rule, ".fields 必须保持竖向排列（flex-direction:column）"
    assert "wrap" not in rule, "不能退回横向流式（flex-wrap）"

    # 字段行内部（label + select 同一行）仍应保持横向对齐，这与"整个
    # 字段区是竖排列表"不矛盾——每一行内部横排，行与行之间竖排堆叠。
    m2 = re.search(r"\.fields \.field\{([^}]*)\}", static_part)
    assert m2 is not None, "未找到 .fields .field 的 CSS 规则"
    assert "flex" in m2.group(1)


# --- 画廊解耦阶段 5（用户 2026-08-17 要求）---

def test_evidence_png_is_read_from_disk_when_absent_in_memory(tmp_path, monkeypatch):
    """内存里没有 _evidence_png 时，应从 keyframes/ 读回那张图。

    此前画廊只认 `a.pixel["_evidence_png"]`，而这个键在写 annotations.json
    时被刻意剥掉（它是二进制）。于是想单独重跑阶段 6 看新版页面，就必须连
    阶段 5 的 VLM 一起重跑——几十秒的事被拖成半小时，还白烧一遍 VLM 调用。

    但图其实一直在盘上（export_keyframes 存了 evidence_sent_to_model.png，
    实测三组 27/27/30 张俱在）。改为内存优先、磁盘兜底。
    """
    from pipeline import gallery
    from pipeline.annotate import PersonAnnotation
    from PIL import Image

    kf = tmp_path / "keyframes" / "07_某公司" / "seat2"
    kf.mkdir(parents=True)
    Image.new("RGB", (800, 800), (10, 20, 30)).save(kf / "evidence_sent_to_model.png")

    a = PersonAnnotation(person_id="seg07_seat2", seg_order=7, company="某公司",
                         seat_index=2, presenter_num=3, gender_female=0,
                         speaks_pre=0, speaks_qa=0, seg_start=0, seg_end=100,
                         vlm={"formality_1_5": 3, "confidence": 0.7})
    assert "_evidence_png" not in a.pixel, "前提：内存里没有图"

    out = tmp_path / "g.html"
    gallery.render_gallery([a], [], out, keyframes_dir=tmp_path / "keyframes")
    html = out.read_text(encoding="utf-8")
    # img 的 src 是 JS 模板字面量，运行时才拼成 data:...；源码里只能查载荷本身
    expect = base64.b64encode((kf / "evidence_sent_to_model.png").read_bytes()).decode()
    assert expect in html, "应从磁盘读到证据图并内联"
    assert '"_thumb_type": "image/png"' in html


def test_memory_evidence_takes_priority_over_disk(tmp_path):
    """内存里有图时优先用内存的——刚跑完阶段 5 的那次不该多一次磁盘往返。"""
    from pipeline import gallery
    from pipeline.annotate import PersonAnnotation
    from PIL import Image
    import io

    kf = tmp_path / "keyframes" / "07_某公司" / "seat2"
    kf.mkdir(parents=True)
    Image.new("RGB", (800, 800), (255, 0, 0)).save(kf / "evidence_sent_to_model.png")

    buf = io.BytesIO()
    Image.new("RGB", (800, 800), (0, 255, 0)).save(buf, format="PNG")
    mem = buf.getvalue()

    a = PersonAnnotation(person_id="seg07_seat2", seg_order=7, company="某公司",
                         seat_index=2, presenter_num=3, gender_female=0,
                         speaks_pre=0, speaks_qa=0, seg_start=0, seg_end=100,
                         vlm={"formality_1_5": 3}, pixel={"_evidence_png": mem})

    out = tmp_path / "g.html"
    gallery.render_gallery([a], [], out, keyframes_dir=tmp_path / "keyframes")
    html = out.read_text(encoding="utf-8")
    assert base64.b64encode(mem).decode() in html, "内存里的图应优先"


def test_missing_on_both_sides_falls_back_to_frame_not_crash(tmp_path):
    """内存与磁盘都没有时，回退到整帧路径，不得抛异常。"""
    from pipeline import gallery
    from pipeline.annotate import PersonAnnotation

    a = PersonAnnotation(person_id="seg09_seat1", seg_order=9, company="无图公司",
                         seat_index=1, presenter_num=1, gender_female=0,
                         speaks_pre=1, speaks_qa=0, seg_start=0, seg_end=100)
    out = tmp_path / "g.html"
    gallery.render_gallery([a], [], out, keyframes_dir=tmp_path / "keyframes")
    assert out.exists() and "无图公司" in out.read_text(encoding="utf-8")
