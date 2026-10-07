"""Task 56F · 交付包：把「人工核实完、可以直接拿去打标」的东西单独拢一份。

用户 2026-08-24：「将人工标记核实后用于 VLM 的文件单独放在
`work/Output/` 各个组名下，这样方便由其他人标注，标注完成后将产出文件发我，
我可以依赖最终产物进行 VLM 判定」。

这份包要**离开这台机器**——别人拿到手里没有 `work/<组>/frames/` 那几万张
帧，而登记表里存的恰恰是「某帧某个框」。所以包必须自包含：证据图要落成真
正的 PNG 文件，不能只留一条指向本机的路径。
"""
import json
from pathlib import Path

import pytest

from pipeline import bundle, registry
from pipeline.annotate import build_evidence_sheet
from pipeline.detect import Detection
from pipeline.schedule import Segment

W = 1440
H = int(W * 0.75)


def _frame(tmp_path, name="f.jpg"):
    """一张**内容不均匀**的帧。

    纯色帧会让这一整类测试失去意义：任意两个框裁出来的像素完全相同，「把第
    一格的框套到所有帧上」这种错位在证据图上一个字节都不差（变异测试实测
    存活）。用固定种子的噪声，保证不同框 → 不同像素，同时可复现。
    """
    from PIL import Image
    import numpy as np
    p = tmp_path / name
    rng = np.random.default_rng(20260824)
    Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(p)
    return p


def _mkreg(tmp_path, sheet="测试组", n_seats=2, with_frames=True):
    """一份最小但形状真实的登记表：一段、n_seats 个座位，各带 2 帧。"""
    fp = _frame(tmp_path)
    seats = {}
    for k in range(1, n_seats + 1):
        # 同一座位的两帧要用**不同的框**：都一样的话，「把第一格的框套到所有
        # 帧上」这种错位在证据图上看不出差别，逐字节比对也照样通过（变异测试
        # 实测存活）。
        frames = ([{"frame_path": str(fp), "t": float(i),
                    "box": [100.0 + 60 * k + 40 * i, 200.0 + 30 * i,
                            220.0 + 60 * k + 40 * i, 700.0 + 30 * i],
                    "conf": 0.9} for i in range(2)] if with_frames else [])
        seats[str(k)] = {"seat": k, "source": "cell_split", "confirmed": True,
                         "track_id": 60 + k, "kind": "exit",
                         "stable_id": f"exit@1#{k}", "cell_indices": [1, 2],
                         "note": "", "seat_uncertain": 0, "frames": frames}
    return {
        "_meta": {"schema": registry.REGISTRY_SCHEMA, "sheet": sheet,
                  "detections_fingerprint": "FP-TEST",
                  "built_at": "2026-08-24T12:00:00+00:00",
                  "labels_file": f"entry_labels_{sheet}.json",
                  "labels_digest": "abc123def456",
                  "labels_mtime": "2026-08-24T11:00:00+00:00",
                  "n_segments": 1, "n_seats_expected": n_seats,
                  "n_seats_resolved": n_seats,
                  "counts_by_source": {"seated": 0, "entryexit": 0,
                                       "cell_split": n_seats}},
        "segments": {"1": {"company": "测试企业", "presenter_num": n_seats,
                           "rejected_seats": [], "unresolved_seats": [],
                           "seats": seats}},
    }


def test_bundle_dir_is_per_group(tmp_path):
    """每组一个子目录，别人拿走的是一整个文件夹，不是散落的几个文件。"""
    got = bundle.bundle_dir(tmp_path, "城市2-4.26")
    assert got == tmp_path / "Output" / "城市2-4.26", got


def test_bundle_writes_one_evidence_png_per_seat(tmp_path):
    """每个座位落一张 PNG，文件名带段号与座位号，肉眼就能对上。"""
    reg = _mkreg(tmp_path)
    out = bundle.write_bundle(reg, tmp_path / "Output" / "测试组", sheet="测试组")
    pngs = sorted((tmp_path / "Output" / "测试组" / "evidence").glob("*.png"))
    assert [p.name for p in pngs] == ["seg01_seat1.png", "seg01_seat2.png"], (
        f"每座位一张、名字带段号座位号，实得 {[p.name for p in pngs]}")
    assert out["n_evidence"] == 2, out


def test_bundle_evidence_is_byte_identical_to_what_the_vlm_gets(tmp_path):
    """落盘的 PNG 必须与打标时送进模型的那张**逐字节相同**。

    本项目栽过一次「人核对了一张模型没看过的图」：两处各裁一份必然走样。
    所以这里不许另裁——一律走 `annotate.build_evidence_sheet`，用登记表里
    存的那几帧那几个框。
    """
    reg = _mkreg(tmp_path, n_seats=1)
    bundle.write_bundle(reg, tmp_path / "Output" / "测试组", sheet="测试组")
    rec = reg["segments"]["1"]["seats"]["1"]
    paths, boxes = registry._preview_frames(rec)
    expect = build_evidence_sheet(paths, boxes)
    got = (tmp_path / "Output" / "测试组" / "evidence" / "seg01_seat1.png").read_bytes()
    assert got == expect, "落盘的证据图与打标时送进模型的那张不是同一张"


def test_bundle_manifest_carries_the_labels_digest(tmp_path):
    """清单里带上这包产自哪一份人工标注（摘要）。

    包会离开这台机器再传回来，没有摘要就无从判断手上这份是不是最新的——
    「产物没重新生成、代码却改了」这类静默陈旧本项目已栽过五次以上。
    """
    reg = _mkreg(tmp_path)
    bundle.write_bundle(reg, tmp_path / "Output" / "测试组", sheet="测试组")
    man = json.loads((tmp_path / "Output" / "测试组" / "manifest.json")
                     .read_text(encoding="utf-8"))
    assert man["labels_digest"] == "abc123def456", man
    assert man["schema"] == registry.REGISTRY_SCHEMA, man
    assert man["sheet"] == "测试组", man


def test_bundle_lists_seats_without_evidence_instead_of_dropping_them(tmp_path):
    """没有可用证据图的座位要**列出来**，不能悄悄少一张 PNG。

    静默少一张，收件人只会以为这个座位本来就不存在；而它其实是「这个人还
    没定下来」或「帧文件不在了」，两种都需要有人去处理。
    """
    reg = _mkreg(tmp_path, n_seats=2, with_frames=False)
    out = bundle.write_bundle(reg, tmp_path / "Output" / "测试组", sheet="测试组")
    assert out["n_evidence"] == 0, out
    assert sorted(out["missing_evidence"]) == ["seg01_seat1", "seg01_seat2"], out
    man = json.loads((tmp_path / "Output" / "测试组" / "manifest.json")
                     .read_text(encoding="utf-8"))
    assert man["missing_evidence"], "清单里也要写明缺哪些，别人才看得到"


def test_bundle_copies_the_registry_itself(tmp_path):
    """登记表本身也进包——它是「谁坐哪个座位」的唯一权威，打标要读它。"""
    reg = _mkreg(tmp_path)
    bundle.write_bundle(reg, tmp_path / "Output" / "测试组", sheet="测试组")
    p = tmp_path / "Output" / "测试组" / "seat_registry_测试组.json"
    assert p.exists(), "登记表要进包"
    assert json.loads(p.read_text(encoding="utf-8"))["_meta"]["labels_digest"] \
        == "abc123def456"


def test_bundle_moves_legacy_files_out_of_the_group_dir(tmp_path):
    """组目录里还留着的交付物**移动**进包，不是复制。

    用户 2026-08-24：「Output 已经有的文件在原组文件夹下就不用有了…不用重复
    存储」。复制正好造出他要消灭的那份重复；两处迟早不一致，而不一致时哪
    一份作数没人说得清，且不报错。
    """
    src = tmp_path / "work" / "测试组"
    src.mkdir(parents=True)
    (src / "entry_review.html").write_text("<html>页</html>", encoding="utf-8")
    lp = src / "entry_labels_测试组.json"
    lp.write_text("{}", encoding="utf-8")
    reg = _mkreg(tmp_path)
    reg["_meta"]["labels_digest"] = registry.labels_digest(lp)
    out = bundle.write_bundle(reg, bundle.bundle_dir(src.parent, "测试组"),
                              sheet="测试组", work=src)
    d = bundle.bundle_dir(src.parent, "测试组")
    assert (d / "entry_review.html").exists(), "标注页要进包"
    assert (d / "entry_labels_测试组.json").exists(), "人工标注要进包"
    assert not (src / "entry_review.html").exists(), "旧位置必须清掉，别重复存"
    assert not (src / "entry_labels_测试组.json").exists()
    assert sorted(out["migrated_from_group_dir"]) == [
        "entry_labels_测试组.json", "entry_review.html"], out
    assert "entry_review.html" in out["present"], out


def test_bundle_survives_missing_side_files(tmp_path):
    """标注页/标注文件哪儿都没有时照样出包，清单里如实记下没有。"""
    reg = _mkreg(tmp_path)
    out = bundle.write_bundle(reg, tmp_path / "Output" / "测试组",
                              sheet="测试组", work=tmp_path / "不存在")
    assert out["migrated_from_group_dir"] == [], out
    assert out["present"] == [f"seat_registry_测试组.json"], out
    assert (tmp_path / "Output" / "测试组" / "manifest.json").exists()


def test_bundle_readme_says_what_to_send_back(tmp_path):
    """包里要有一份说明：怎么标、标完把**哪个文件**发回来。

    收件人不是本项目的人，不写清楚他会把整个文件夹压缩回来，或者只回一张
    截图。要送回的就是 entry_labels_<组>.json 那一个文件。
    """
    reg = _mkreg(tmp_path)
    bundle.write_bundle(reg, tmp_path / "Output" / "测试组", sheet="测试组")
    txt = (tmp_path / "Output" / "测试组" / "README.md").read_text(encoding="utf-8")
    assert "entry_labels_测试组.json" in txt, f"要点名送回哪个文件：{txt}"
    assert "entry_review.html" in txt, f"要说清在哪标：{txt}"


def test_bundle_readme_names_the_file_to_send_back_with_the_page_too(tmp_path):
    """带上标注页的那条 README 分支同样要点名送回哪个文件。

    README 有两条分支（附了 entry_review.html / 没附），只测其中一条时，另
    一条改坏了一个测试都不红（变异测试实测存活）。而**实际发出去的包永远是
    带页的那条**——恰恰是没被测到的那条。
    """
    src = tmp_path / "work" / "测试组"
    src.mkdir(parents=True)
    (src / "entry_review.html").write_text("<html>页</html>", encoding="utf-8")
    lp = src / "entry_labels_测试组.json"
    lp.write_text("{}", encoding="utf-8")
    reg = _mkreg(tmp_path)
    reg["_meta"]["labels_digest"] = registry.labels_digest(lp)
    d = bundle.bundle_dir(src.parent, "测试组")
    bundle.write_bundle(reg, d, sheet="测试组", work=src)
    txt = (d / "README.md").read_text(encoding="utf-8")
    assert "双击" in txt, f"带页时要告诉他怎么打开：{txt}"
    assert txt.count("entry_labels_测试组.json") >= 1, (
        f"要点名送回哪个文件：{txt}")


def test_bundle_refuses_a_stale_registry(tmp_path):
    """登记表落后于人工标注时**不出包**。

    出了包，别人拿到的是旧图、按旧图标注，回来的结果与当前标注对不上，而
    一路无人报错。宁可在这里停下。
    """
    work = tmp_path / "work"
    work.mkdir()
    (work / "entry_labels_测试组.json").write_text(
        json.dumps({"1": {"x": {"seat": 1}}}), encoding="utf-8")
    reg = _mkreg(tmp_path)          # labels_digest 是编的，与上面那份对不上
    with pytest.raises(registry.RegistryStaleError):
        bundle.write_bundle(reg, tmp_path / "Output" / "测试组",
                            sheet="测试组", work=work)


def test_missing_evidence_wording_distinguishes_judges(tmp_path, monkeypatch, capsys):
    """缺图的提示要分清「座位」和「评审席」——评审人不是创业者座位。

    用户 2026-09-17 加进来的评审席恒有 5 个号，只到 4 位时第 5 号必然缺图；
    照旧印成「1 个座位没有证据图」会让人以为少了一个创业者。
    """
    from pipeline.bundle import _missing_evidence_text
    assert "评审席" not in _missing_evidence_text(["seg01_seat2"])
    t = _missing_evidence_text(["judge5"])
    assert "座位" not in t and "评审席" in t, t
    both = _missing_evidence_text(["seg01_seat2", "judge5"])
    assert "座位" in both and "评审席" in both, both


def test_readme_warns_the_page_needs_its_image_folder(tmp_path):
    """README 要提醒：标注页的图在 `review_img/`，别把 html 单独拆出来发。

    图改成文件引用之后（用户 2026-09-18），只转发那个 html 的话对方打开满屏
    是叉——而他那边什么都不报，只会以为「这页坏了」。
    """
    from pipeline.bundle import _write_readme
    out = tmp_path / "README.md"
    _write_readme(out, "测试组", 3, 3, [], True)
    txt = out.read_text(encoding="utf-8")
    assert "review_img" in txt, txt
    assert "自包含" not in txt, "README 还在说标注页是自包含的"
