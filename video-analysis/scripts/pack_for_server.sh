#!/usr/bin/env bash
# 打包上传给服务器的「代码包」——不含视频、不含 .env、不含中间产物。
#
# 用法：  bash scripts/pack_for_server.sh [输出目录]
# 产出：  dress_server_<日期>.tar.gz  （约 116 MB，主要是权重）
#
# 视频**不在包里**：它们 9-22 GB，单独用 rsync/scp 传，断线能续。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT}"
STAMP="$(date +%Y%m%d)"
TARBALL="$OUT_DIR/dress_server_${STAMP}.tar.gz"

cd "$ROOT"

# 一律显式列出要打包的东西，不用 `--exclude` 排除法：漏排一项就会把 .env
# 或 1.3 GB 的 frames/ 带上去，而 tar 不会有任何提示。
ITEMS=(
  pipeline
  tests
  pyproject.toml
  Book1.xlsx
  "2017创新video.dta"
  yolo26x.pt
  work/服务器部署清单.md
  work/操作说明.md
)

for f in "${ITEMS[@]}"; do
  [ -e "$f" ] || { echo "✘ 缺少 $f，中止" >&2; exit 1; }
done

# 硬护栏：密钥绝不进包。
if printf '%s\n' "${ITEMS[@]}" | grep -qx '.env'; then
  echo "✘ .env 出现在打包清单里，中止" >&2; exit 1
fi

# 排掉 __pycache__：无用，且带过去的是**本机**编译的字节码——服务器 Python
# 版本或路径一旦不同，读到过期的 .pyc 是最难查的一类问题。
tar --exclude='__pycache__' --exclude='*.pyc' -czf "$TARBALL" "${ITEMS[@]}"

echo "已打包：$TARBALL"
ls -la "$TARBALL" | awk '{printf "  %.1f MB\n", $5/1048576}'
echo
echo "包内不含（按设计）："
echo "  · .env           —— 密钥用服务器自己的环境变量注入"
echo "  · *.mp4          —— 单独 rsync，见下"
echo "  · work/<组>/     —— frames/detections 等中间产物由服务器自己生成"
echo
echo "核验包里确实没有密钥："
if tar -tzf "$TARBALL" | grep -q '\.env'; then
  echo "  ✘ 包里发现 .env！不要上传这个包" >&2; exit 1
else
  echo "  ✔ 未发现 .env"
fi
echo
echo "下一步："
echo "  scp $TARBALL 服务器:~/"
echo "  rsync -avP --partial '城市精细化管理1组4月27日视频.mp4' 服务器:~/dress/"
