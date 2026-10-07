# 最终项目交付包

本 ZIP 保留本次任务已完成的源代码、测试、配置、安装和启动辅助文件、依赖清单、
方法文档、原始 DTA 数据工作副本、YOLO 模型权重以及现有结果和验证记录。
本次整理仅进行文件打包与压缩包完整性检查，没有运行项目、安装依赖、重新测试
项目或重新生成已有结果。

## 目录

- video-analysis/：上传项目的原始代码、测试、scripts、方法文档、data、models 和 work。
- video-analysis-cloud/：最终 CUDA 安装脚本、启动器、requirements.in、requirements.lock、
  环境说明、模型来源元数据、现有验证日志和 smoke-output 示例结果。
- environment-config.json：已保存云环境草稿中安装、启动及网络配置的只读导出。
- source-sha256.original.json：原始附件解压文件的校验记录；路径相对 video-analysis/。
- SHA256SUMS.txt：包内文件的 SHA-256 校验清单（不包含清单自身）。

## 路径与数据

现有启动脚本保留原云环境路径，使用时将 video-analysis 和 video-analysis-cloud
两个目录放在 /workspace 下。具体用法见 video-analysis-cloud/README.md。
保留数据位于 video-analysis/data/2017创新video.dta，与上传原件逐字节一致。
YOLO 权重位于 video-analysis/models/yolo26x.pt。
已生成排期位于 video-analysis/work/城市1-4.27/segments.json。
现有示例视频、帧、检测和证据图位于 video-analysis-cloud/smoke-output/；
它们属于之前的示例验证，不能替代尚未提供的真实研究视频。

本包没有 .git、Python 虚拟环境、__pycache__、编译字节码、测试缓存、包下载缓存、
未完成的模型下载和 macOS 系统附带文件。临时 Hugging Face 缓存没有打包；
模型来源、固定版本、哈希及下载辅助程序已保留。

GPU 挂载、服装分割模型 CDN 访问及真实视频的待办状态保留在原环境说明中。
打包不改变这些状态，也不声称重新验证过 GPU 或完整视频流水线。
