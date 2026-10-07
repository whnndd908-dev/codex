# 视频分析项目与云环境配置

本仓库保存本次任务已经完成的最终项目文件：视频人物分析流水线、测试、
人工核验方法文档、CUDA 环境安装与启动配置、依赖锁、原始 DTA 数据及已有结果。
本次 GitHub 同步只复制文件和提交，不重新运行项目、分析、安装或测试，
也不重新生成已有结果。

## 主要文件

| 路径 | 内容 |
| --- | --- |
| `video-analysis/pipeline/` | 排期、抽帧、YOLO 检测跟踪、人工核验、登记表、模型编码和输出代码 |
| `video-analysis/tests/` | 已有的 31 个测试模块，保留原文件 |
| `video-analysis/scripts/` | 原打包与辅助分析脚本 |
| `video-analysis/video-human-coding/` | 人工身份核验、证据图、修正、增量处理与验证方法文档 |
| `video-analysis/data/2017创新video.dta` | 用户上传数据的完整工作副本，457,967 字节，与原件一致 |
| `video-analysis/work/城市1-4.27/segments.json` | 已生成的排期结果，未重建 |
| `video-analysis/models/yolo26x.pt.sha256` | 本地 YOLO 权重校验值，权重本体不提交 |
| `video-analysis-cloud/requirements.in` | 云环境直接依赖清单 |
| `video-analysis-cloud/requirements.lock` | CUDA 依赖固定版本及包哈希 |
| `video-analysis-cloud/install.sh` | 已完成并保存的安装脚本 |
| `video-analysis-cloud/run.py` | Linux 路径、缓存和 GPU 检查启动器 |
| `video-analysis-cloud/fetch-segformer.py` | 按固定版本及可信哈希获取分割模型的辅助程序 |
| `video-analysis-cloud/README.md`、`start-skill.md` | 环境说明、启动与故障记录 |
| `video-analysis-cloud/segformer-source.json` | 分割模型来源、固定修订、文件大小和上游哈希 |
| `video-analysis-cloud/logs/` | 之前已经存在的验证记录，没有重新执行验证 |
| `video-analysis-cloud/smoke-output/` | 之前生成的示例视频、抽帧、检测、证据图和报告，原样保留 |
| `environment-config.json` | 已保存的安装、启动和网络配置草稿导出，不包含密钥值 |
| `source-sha256.original.json` | 原始附件解压文件校验记录，路径相对 `video-analysis/` |
| `docs/README-交付说明.md` | 先前完整 ZIP 交付说明，作为历史记录保留 |

## 大文件、来源与本地位置

GitHub 普通 Git 单文件上限为 100 MiB。大型权重、长视频和分发 ZIP 不提交到
本仓库，也没有强制加入 Git 或转换为 Git LFS。本地原完整 ZIP、分卷和全部
项目文件均保留。

| 文件 | 来源 | 本地放置位置与状态 |
| --- | --- | --- |
| `yolo26x.pt`，118,667,365 字节，约 113.17 MiB | [Ultralytics 官方 v8.4.0 发布](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt) | `/workspace/video-analysis/models/yolo26x.pt`；已存在，仅省略 Git 上传。仓库对应位置为 `video-analysis/models/yolo26x.pt` |
| `model.safetensors`，109,493,236 字节，约 104.42 MiB | [mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes/tree/584abc1e1d260e23c0fc627c5217a09b2b461046)，固定修订 `584abc1e1d260e23c0fc627c5217a09b2b461046` | `/workspace/video-analysis-cloud/huggingface/hub/models--mattmdjaga--segformer_b2_clothes/snapshots/584abc1e1d260e23c0fc627c5217a09b2b461046/model.safetensors`；此前下载受 CDN 403 阻塞，未取得完整文件。下载辅助程序及来源元数据已提交 |
| `城市精细化管理1组4月27日视频.mp4` | 用户的研究视频，尚未提供；不推测下载来源 | `/workspace/video-analysis/data/城市精细化管理1组4月27日视频.mp4`；尚不存在。也可用启动器 `--video` 指定真实文件 |
| `video-analysis-final.zip`，111,963,961 字节 | 本次任务已有完整交付包 | `/workspace/exports/video-analysis-final.zip`；完整原包保留，不提交分发副本 |

已有小型 DTA 数据正常提交；以后如使用更大的数据集，应保存在本地
`video-analysis/data/`，在文档记录文件名、来源与放置位置，检查大小后再决定
是否提交。不要将超过普通 Git 文件限制的文件加入提交。

YOLO 权重 SHA-256：

```text
9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92
```

## 环境与已有验证记录

此前准备了 Python 3.12、PyTorch 2.8.0+cu126、CUDA 12.6 运行库、
Torchvision 0.23.0+cu126、Ultralytics 8.4.174、Transformers 4.57.6 和相关依赖。
应用命令与安装脚本保留原云环境绝对路径：
`/workspace/video-analysis` 和 `/workspace/video-analysis-cloud`。
从 GitHub 获取后，如需运行，应将这两个目录放在对应位置，再按
[云环境说明](video-analysis-cloud/README.md) 和
[启动说明](video-analysis-cloud/start-skill.md) 操作；本次同步不执行这些步骤。

之前记录的完整测试结果为 1,250 通过、7 失败、11 跳过；6 个失败缺少真实
视频/抽帧，另 1 个默认 FFmpeg 路径检查按其预期前提单独重跑通过。
这些是历史验证结果，不能视为本次提交的重新测试结果。
当时 GPU 未挂载、分割模型 CDN 返回 403、真实研究视频尚未提供，完整 GPU
工作流尚未完成验证。本次同步不改变或重新检查这些运行状态。

`.gitignore` 排除虚拟环境、缓存、编译字节码、临时文件、凭据文件、
大型模型权重和分发归档；代码、配置、依赖、说明及既有结果已保留。
