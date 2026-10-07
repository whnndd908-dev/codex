"""跨组汇总：把各组视频各自的产物拼成一份可直接分析的总表。

为什么要做成正式步骤而不是临时脚本
----------------------------------
第一次手写这段拼接就出了错：`result_*` 的通配把 `result_wide_*` 一并匹配
了进去，长表实际拿到的是宽表，行数从 103 悄悄变成 45。两张表都有
`company`、`order` 这些列，扫一眼根本看不出来。产物文件名带组名（Task 33）
之后，同一目录下 `result_X.csv` / `result_wide_X.csv` / `result_人工校对_X.csv`
三者前缀互相包含，这个坑只会更容易踩。

于是把「挑哪个文件」这件事收进 `find_group_csv` 并用测试锁死，而不是每次
都靠人记得写排除条件。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from pipeline import config

# 汇总要合并的三张表：长表（每人一行）、宽表（每企业一行）、项目排名。
KINDS = ("result", "result_wide", "project_ranking")

_OUT_NAME = {
    "result": "all_groups_long.csv",
    "result_wide": "all_groups_wide.csv",
    "project_ranking": "all_groups_ranking.csv",
}


@dataclass(frozen=True)
class GroupSpec:
    """一组视频的汇总所需信息。

    只含汇总用得到的三项，不复制 config 里那一整套抽帧/检测参数——那些
    是跑流水线时才需要的，混进来只会让人以为改这里就能换组。
    """
    key: str        # 组标识，进 group_key 列
    panel: str      # 评审组全称，进 panel 列
    work: Path      # 该组产物目录


# 三组视频的注册表。跑流水线时 config.WORK 等常量指向其中一组（由重跑脚本
# 改写），但汇总必须同时看到全部三组，因此单独列在这里。
GROUPS = [
    GroupSpec("城市2-4.26", "城市精细化管理与应急保障2组",
              config.ROOT / "work" / "城市2-4.26"),
    GroupSpec("文化4-4.28", "文化科技融合4组",
              config.ROOT / "work" / "文化4-4.28"),
    GroupSpec("新能源2-4.27", "新能源与高效节能2组",
              config.ROOT / "work" / "新能源2-4.27"),
]


def find_group_csv(work: Path, kind: str) -> Path | None:
    """在一组的产物目录里挑出指定种类的 CSV。

    三个前缀互相包含，必须显式排除而不能只靠 startswith：
        result_X.csv            ← kind="result" 要的
        result_wide_X.csv       ← kind="result_wide" 要的，但也 startswith("result_")
        result_人工校对_X.csv    ← 人工校对版，任何自动汇总都不该吃进去

    找不到时返回 None 而非抛异常：某组可能还没跑完，该由调用方决定是跳过
    还是报错，不能在这里把整份汇总带崩。
    """
    if not work.is_dir():
        return None

    prefix = kind + "_"
    # 排除「名字更长的兄弟 kind」——不手写具体字符串。
    #
    # 手写排除踩过两次坑，都是同一类错误的不同表现：
    #   第一次：忘了排除，kind="result" 吃进 result_wide_X.csv
    #   第二次：排除写成 rest.startswith("wide_")，结果漏掉了改名前的旧文件
    #           result_wide.csv（rest 是 "wide.csv"，不带下划线）
    # 说明按字符串前缀逐条手写排除本身就不牢靠。改为从 KINDS 推导：凡是以
    # 本 kind 前缀开头的其它 kind，其全部文件一律排除。
    siblings = [k for k in KINDS if k != kind and k.startswith(prefix)]

    cands = []
    for p in sorted(work.glob("*.csv")):
        name = p.name
        if not name.startswith(prefix):
            continue
        if any(name.startswith(s) for s in siblings):
            continue
        # 人工校对版永远排除：它含人工改动，混进 VLM 版汇总会让两者无法比较
        if name[len(prefix):].startswith("人工校对_"):
            continue
        cands.append(p)
    return cands[0] if cands else None


def combine_rows(specs: list[GroupSpec], kind: str) -> tuple[list[dict], list[str]]:
    """把各组同一种类的 CSV 纵向拼起来。

    - `panel` / `group_key` 固定为前两列，保证堆起来之后还能分辨行的来源。
    - 各组列不一致时取并集（某组可能跑于旧版代码），缺的列留空字符串——
      不能填 0，0 在本项目多处是合法值。
    - 缺产物的组打印一行并跳过，绝不静默少一整组。
    """
    rows: list[dict] = []
    cols: list[str] = ["panel", "group_key"]

    for spec in specs:
        src = find_group_csv(spec.work, kind)
        if src is None:
            print(f"   ⚠ {spec.key} 缺 {kind} 产物，已跳过（汇总将不含该组）")
            continue
        with src.open(encoding="utf-8-sig") as fh:
            n = 0
            for r in csv.DictReader(fh):
                # 组标识写在 **r 之后：源文件若自带 panel/group_key 同名列
                # （例如误把上一版汇总表放回了产物目录），必须由注册表说了算。
                # 反过来写会让"这行来自哪一组"被文件内容改写，而汇总表看上去
                # 完全正常——发现于变异测试。
                merged = {**r, "panel": spec.panel, "group_key": spec.key}
                rows.append(merged)
                n += 1
                for k in merged:
                    if k not in cols:
                        cols.append(k)
        print(f"   {spec.key:14s} {src.name[:44]:46s} {n:>4} 行")

    # 列并集下，早出现的行缺后来才见到的列——统一补空串，避免 DictWriter
    # 写出 None 或整行错位
    for r in rows:
        for c in cols:
            r.setdefault(c, "")
    return rows, cols


def write_combined(specs: list[GroupSpec] | None = None,
                   out_dir: Path | None = None) -> list[Path]:
    """产出三张汇总表，返回实际写出的文件路径。"""
    specs = specs if specs is not None else GROUPS
    out_dir = out_dir if out_dir is not None else (config.ROOT / "work")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for kind in KINDS:
        rows, cols = combine_rows(specs, kind)
        if not rows:
            print(f"   ⚠ {kind}: 三组都没有产物，未生成汇总表")
            continue
        p = out_dir / _OUT_NAME[kind]
        with p.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f">> {p.name}: {len(rows)} 行 × {len(cols)} 列")
        written.append(p)
    return written


if __name__ == "__main__":
    print("⑧ 跨组汇总")
    write_combined()
