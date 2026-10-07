# 视频分析云环境

项目代码与文档解压在 `/workspace/video-analysis`，安装及启动辅助文件位于
`/workspace/video-analysis-cloud`。原始 Python 源码和测试保持不变；上传的
`.dta` 原件保留，工作副本为 `video-analysis/data/2017创新video.dta`。
本文件保留首次环境准备的说明与验证记录。项目文件现整理到 GitHub 仓库
`whnndd908-dev/codex`；大型权重的来源与本地位置见仓库根目录 README。
本次同步没有重新运行项目、测试或生成结果。

## 附件阅读结果

`video-human-coding.zip` 是方法文档，`代码.zip` 包含 `pipeline`、31 个测试
模块和两个辅助脚本，没有 `pyproject.toml` 或依赖锁文件。
上传 `.dta` 有 1,102 行、27 列、69 个组。
当前默认组「城市精细化管理与应急保障1组」有 17 段、42 位创业者；
起止及进退场时间齐全，与代码的默认断言一致。

方法文档的重点是：机器生成候选，人工确认身份后才进入模型编码。
`registry.md` 要求用带版本及指纹的登记表传递人工决定；
`identifiers.md` 用内容相关标识避免跟踪编号变化造成错配；
`corrections.md` 描述多种人工修正的组合；`evidence-sheets.md` 要求模型与人工
核对同一证据图，分配有限的图格并避免泄露人工真值；`iterating.md` 说明局部
重抽、增量检测和撤销；`verification.md` 讨论有效测试与真实素材边界；
`division-of-labour.md` 区分机器处理与人工核验的工作。
附件中的历史要求和操作建议作为项目背景，不自动视为本次操作授权。

以 `pipeline.run_all.STAGES` 的实际编号为准（部分旧注释和 CLI 帮助仍有旧编号）：

1. 读取排期与人工标注。
2. 抽帧。
3. YOLO 人物检测与 BoT-SORT 跟踪。
4. 座位基线。
5. 生成人工挑选页。
6. 人工完成核验后建立登记表与核对页。
7. 模型编码。
8. 画廊。
9. 排名。

## 已准备的环境

Python 3.12 虚拟环境：`/workspace/video-analysis/.venv`。
PyTorch 2.8.0+cu126、Torchvision 0.23.0+cu126，CUDA 12.6 运行库，
Ultralytics 8.4.174、Transformers 4.57.6、OpenCV、Pandas、OpenPyXL、
Scikit-learn、Anthropic SDK、Pytest 等已安装。
`requirements.lock` 固定全部依赖版本和包哈希；CUDA 依赖来自官方 PyTorch
及 PyPI。系统已有 FFmpeg/FFprobe 7.1.5。

默认 YOLO26x 权重已从官方发布地址下载，并记录 SHA-256。
SegFormer 模型固定为 `584abc1e1d260e23c0fc627c5217a09b2b461046`，
可信上游提供的权重 SHA-256 保存在 `segformer-source.json`。
权重下载目前被 CDN 的代理 403 拦截，模型尚未就绪。

## 启动和验证

每个云任务已隔离，使用现有文件及 checkout；用户没有要求时不创建 Git worktree。
启动器在进程内覆盖原代码的 macOS 路径，使用系统 FFmpeg，设置可写缓存目录。
检测阶段按本次用户要求使用 NVIDIA GPU：不可见时停止，不自动改用 CPU。
排期和无需 GPU 的开发测试可以独立运行。

```bash
cd /workspace/video-analysis
bash /workspace/video-analysis-cloud/install.sh
.venv/bin/python /workspace/video-analysis-cloud/run.py pipeline.run_all --stages 1
.venv/bin/python /workspace/video-analysis-cloud/run.py pytest tests -q -ra \
  -o 'markers=slow: checks requiring real media or model artifacts'
```

先在云平台侧挂载兼容 CUDA 12.6 的 NVIDIA GPU，并确保宿主机驱动与设备进入
容器。仅安装用户空间 CUDA 包不能创建物理 GPU 或加载宿主机驱动。
挂载后检查 `torch.cuda.is_available()`，再运行示例素材的 GPU 验证：

```bash
cd /workspace/video-analysis
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
.venv/bin/python /workspace/video-analysis-cloud/run.py smoke
```

`smoke` 会用安装包附带的 `bus.jpg` 生成一秒测试视频，验证解码、跟踪与 PNG
证据图；它不验证尚未上传的真实视频或人工身份正确性。

网络配置草稿增加了 `huggingface.co`、`cas-bridge.xethub.hf.co` 和
`us.aws.cdn.hf.co`，保留现有软件包管理器预设。保存相关网络设置、实际连接恢复后：

```bash
cd /workspace/video-analysis
.venv/bin/python /workspace/video-analysis-cloud/fetch-segformer.py
```

启动器默认复用本地模型缓存，禁用联网刷新。下载辅助入口仍启用正常 HTTPS
验证，并核对上游权重哈希；不关闭 TLS 或修改校验值。

真实视频应放在 `data/城市精细化管理1组4月27日视频.mp4`，或通过启动器的
`--video /path/video.mp4` 显式指定。当前文件尚未提供。
`SCHEDULE_FROM_DTA=True` 时排期不依赖 `Book1.xlsx`；附带的输入存在性测试
仍要求视频和 XLSX，其失败将如实保留，不创建替代输入来改变结果。

GPU 与视频就绪后，运行人工核验之前的阶段：

```bash
cd /workspace/video-analysis
.venv/bin/python /workspace/video-analysis-cloud/run.py pipeline.prepare
```

需要处理其他组时，在模块名之前明确指定 `--panel`、`--sheet`、`--work`、
`--video`、`--expected-segments`、`--expected-presenters`。
`prepare --group` 只改变 work 目录，不能独立用于切换排期与视频。

阶段 7 的 VLM 另需安全注入 `ANTHROPIC_API_KEY`，自定义服务才需要
`ANTHROPIC_BASE_URL`。当前两者未设置；本次环境验证没有调用付费 API。
原代码的服装分割设备选择仅支持 MPS/CPU，CUDA 检测可单独使用；若需要把
服装分割也迁到 CUDA，属于后续代码修改任务。

没有常驻服务需要启动，人工核验页面为生成的静态文件。
配置保存只创建草稿，环境发布与新任务还原需要单独完成和验证。

验证日志位于 `logs/`；示例媒体输出位于 `smoke-output/`。
`logs/smoke.log` 记录切换 CUDA 环境之前已完成的 CPU 示例验证，
不能当作 GPU 推理证据。最新 GPU 检查见 `logs/gpu-preflight.log`。

## 当前验证结果

CUDA 环境完整测试执行 1,268 项：1,250 通过、7 失败、11 跳过。
6 个失败依赖未提供的真实视频/抽帧；1 个 FFmpeg 默认路径测试与云端显式
路径覆盖冲突，按该测试要求的不设环境变量前提单独重跑通过。
安装脚本重复执行通过，88 个已安装包的依赖一致性检查通过。
默认 DTA 排期实际读取通过，得到 17 段、42 位创业者。
模型权重 CDN 返回 403，GPU 可见数量为 0，因此完整 GPU 工作流尚未就绪。
