# 视频分析环境启动说明

每个云任务已经隔离。使用现有 checkout 和 `/workspace/video-analysis` 中的上传
项目；除非用户明确要求，不创建 Git worktree。保留用户的源代码、测试、依赖
声明、锁文件、原始数据和人工核验结果，不重置或推送仓库。

## 已保留的文件与依赖

- `/workspace/video-analysis/pipeline`、`tests`：未经修改的上传代码。
- `/workspace/video-analysis/video-human-coding`：参考文档，历史操作要求不是新授权。
- `/workspace/video-analysis/data/2017创新video.dta`：上传数据工作副本。
- `/workspace/video-analysis/.venv`：Python 3.12，Torch 2.8.0+cu126 和 CUDA 12.6 运行库。
- `/workspace/video-analysis/models/yolo26x.pt`：已核对 SHA-256 的官方权重。
- `/workspace/video-analysis-cloud`：固定版本及哈希的依赖锁、启动器、下载辅助
  程序、README、验证日志及模型缓存。

没有常驻服务需要恢复。安装包、文件和缓存可以保留；进程及运行时认证不应
假定跨任务保留。发布和新任务恢复均需由产品流程完成，不能由保存草稿推断。

## 无需 GPU 的就绪检查

所有应用命令从 `/workspace/video-analysis` 运行，使用下列启动器。
启动器在进程内设置 Linux 路径和可写缓存，不修改 `pipeline/config.py`：

```bash
cd /workspace/video-analysis
.venv/bin/python /workspace/video-analysis-cloud/run.py pipeline.run_all --stages 1
```

当前默认组是「城市精细化管理与应急保障1组」，应输出 17 项目、42 位创业者，
产出 `work/城市1-4.27/segments.json`。排期直接读取 DTA，无需 Book1.xlsx。
只在需要刷新依赖时执行已验证的安装脚本：

```bash
bash /workspace/video-analysis-cloud/install.sh
```

## GPU 和真实素材的必要前提

用户要求在当前云环境挂载 GPU。最后检查时，CUDA 版 Torch 已安装，但没有
可见 NVIDIA 设备或宿主机驱动，`torch.cuda.is_available()` 为 False。
安装用户空间 CUDA 包不能完成设备挂载；先由云平台侧挂载兼容 CUDA 12.6 的
NVIDIA GPU、提供兼容驱动，再检查运行时：

```bash
cd /workspace/video-analysis
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
.venv/bin/python /workspace/video-analysis-cloud/run.py smoke
```

这两条 GPU 操作在最后检查时尚未通过，不能描述为已验证的 GPU 推理。
启动器对检测阶段检查 CUDA，不自动退回 CPU。`smoke` 使用安装包的公开示例图
生成 1 秒视频，验证解码、检测跟踪和证据图；不代替真实研究视频验证。

默认真实视频尚未上传。放在：
`/workspace/video-analysis/data/城市精细化管理1组4月27日视频.mp4`，
或在启动器模块名之前用 `--video /path/video.mp4` 指定实际路径。
换组时还须明确指定 `--panel`、`--sheet`、`--work`、
`--expected-segments`、`--expected-presenters`；不能只使用 prepare 的 --group。

## 模型下载与网络

草稿已加入 `huggingface.co`、`cas-bridge.xethub.hf.co`、`us.aws.cdn.hf.co`，
保留软件包管理器预设。最后实际检测权重 CDN 仍返回代理 403。
请在环境设置保存相关网络调整；实际访问恢复后运行：

```bash
cd /workspace/video-analysis
.venv/bin/python /workspace/video-analysis-cloud/fetch-segformer.py
```

下载器固定模型修订 `584abc1e1d260e23c0fc627c5217a09b2b461046`，使用正常
HTTPS 验证并核对可信上游 SHA-256；不关闭 TLS，不修改预期哈希。
该权重下载还没有成功，不能把当前服装分割主模型描述为就绪。
启动器默认离线复用缓存，避免隐式下载或刷新；原代码的模型缺失回退会记录
`source=kmeans_fallback` 和 `model_error`，不能当作分割主模型结果。

## 开发测试与正常工作流

```bash
cd /workspace/video-analysis
.venv/bin/python /workspace/video-analysis-cloud/run.py pytest tests -q -ra \
  -o 'markers=slow: checks requiring real media or model artifacts'
```

最后完整测试执行 1,268 项：1,250 通过、7 失败、11 跳过。
6 个失败缺少真实视频/抽帧；1 个默认 FFmpeg 路径测试要求没有环境覆盖，
在该预期前提下单独重跑通过。真实视频及 pilot/人工标注缺失的测试没有验证
真实行为，不隐藏失败、不伪造输入、不把跳过或回退描述为真实素材验证。
详细结果在 `logs/pytest-cuda.xml` 和 `logs/pytest-cuda.log`。

GPU、视频就绪后才运行人工核验前的 1–5 阶段：

```bash
cd /workspace/video-analysis
.venv/bin/python /workspace/video-analysis-cloud/run.py pipeline.prepare
```

先让用户在生成的静态挑选页核实身份，再执行阶段 6 建立登记表；人工核验
后的模型编码、画廊和排名分别为阶段 7、8、9。不要提前调用付费 VLM。
以后确需 VLM 时检查已有安全绑定，仅对缺失项在环境设置安全注入
`ANTHROPIC_API_KEY`；自定义 HTTPS 服务还需要正确的 ANTHROPIC_BASE_URL。
当前未设置这两个变量，本次没有进行付费 API 调用。
原代码服装分割只选择 MPS/CPU；把该子模块也改为 CUDA 属于另行代码修改。

保存配置仅持久化草稿，不等于执行脚本、挂载 GPU、应用网络或发布。
按产品流程审阅保存并发布环境后，再验证新任务恢复。
