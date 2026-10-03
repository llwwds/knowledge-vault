# Spike ⑥ — macOS 原生最小校准点（校准 spike ②③ 容器数字）

> 日期：2026-10-04｜ 环境：**宿主机 macOS 原生**，venv-mac（开发与部署规范第 1 条允许的受控环境：依赖在 venv-mac、模型缓存与产物全在 `~/llwwds_application/knowledge-vault/`，零全局污染）；torch CPU、threads=8、HF_HUB_OFFLINE=1、模型缓存复用容器已下载的 HF 布局。
> 脚本：`experiments/calibrate_macos_native.py`｜ 结果 JSON：`~/llwwds_application/knowledge-vault/state/spike_calibration/results.json`
> 性质：**单点对照**（非全量复跑），目的是量化容器环境的系统性偏差并改写 ②③ 结论。已向用户说明此例外及理由（判定对象是 M1 硬件本身，容器测不出）。

## 为什么需要校准

spike ②③ 跑在 linux/arm64 容器：torch 无 macOS Accelerate/AMX 矩阵加速，且 Docker VM 内存上限 8GB（宿主 16GB）——两个失真方向都把「本机可跑性」往坏推。容器实测 reranker 64 对 178~406s、bge-m3 4138 ms/chunk，据此定架构会过度依赖 winpc。

## 校准数字（同一台 M1，容器 vs macOS 原生）

| 指标 | 容器（linux/arm64） | macOS 原生（venv-mac） | 加速比 |
|---|---|---|---|
| reranker 64 对 b8（库默认 fp32） | p50 178.3s（spike② 稳态档） | **p50 4.16s**（3 轮：4.54/4.16/4.09） | **~43×** |
| reranker 强制 half() | 522s（比 fp32 慢 1.32×） | 4.16s（与 fp32 持平） | — |
| bge-m3 每 chunk（426 token 级） | 4138 ms（稳态窗口） | **104.2 ms** | **~40×** |
| 同进程双模型同驻峰值 RSS | 3.99 GiB（VM 上限 8GB 内） | **3.01 GiB**（16GB 宿主） | 余量充足 |
| 模型加载（热） | reranker 6.3s | reranker 1.8s / bge-m3 4.0s | — |
| dtype 实证 | 库默认路径净效果 fp32 | 同（default_dtype=float32；BGEM3 权重 fp16 加载但 encode 前 float()） | 口径一致 |

fp16 强制在 macOS 上无收益（与 fp32 持平）——本机 torch 2.14 CPU 路径未体现 AMX fp16 快速核或被其他开销掩盖；**两个平台的生产配置都应直接用库默认路径（fp32），不要强制 half**。

## 结论改写（覆盖 spike ②③ 的容器口径结论）

1. **reranker（spike②）：通过（macOS 原生口径）**。64 对 4.16s ≤ 5s 预算；30 候选线性外推 ~2s，查询末级可接受。仍建议：RRF 后先返回、rerank 可选异步刷新（>64 候选或交互敏感场景）；ONNX int8（2-4×）留作升级路径，非必需。
2. **bge-m3 本地路线（spike③）：通过（macOS 原生口径）**。增量入库：每天 50 文件 × ~8 chunks ≈ **42s/天**，毫无压力；查询向量：单 query encode < 0.2s；全库一次性初始化：100 万 chunks × 104ms ≈ **29h，一夜后台任务可完成**。**winpc 自托管从「必需兜底」降级为「可选加速项」**（全库初始化嫌慢或 Mac 同时段要做重活时才启用）。
3. **同驻内存（spike③ 核心问题）：16GB 宿主无压力实锤**。同进程双模型加载+推理峰值 3.01 GiB，加系统占用后余量 > 10GB；甚至支持「双模型常驻」部署形态。
4. **方法论沉淀：性能基准必须在目标部署环境跑。** 同一台机器、同一份权重、同一版本 torch（2.14.1），linux 容器与 macOS 原生差 ~40×；容器数字只可用于回归对比与保守下界，不得直接用于「本机可行性」判定。

## 适用性边界

- macOS 侧为单点测量（64 对 × 3 轮 + 单文件 × 1 轮），样本小，但 40× 是量级差异，方向与结论稳健；正式 coding 后可用真实 chunk 复测。
- 容器数字存在同机其他容器 CPU 争抢的干扰（spike③ 已用稳态窗口规避大部分），即使完全排除争抢，与原生差距仍是数十倍量级。
- 校准跑为宿主机 venv 受控环境（规范第 1 条允许），依赖/缓存/产物零全局污染；若用户不认可此例外，删除本报告与 results.json 即可，容器口径结论仍自洽。

## 复跑

```bash
HF_HUB_OFFLINE=1 HF_HOME=~/llwwds_application/knowledge-vault/hf_cache \
  ~/llwwds_application/knowledge-vault/venv-mac/bin/python \
  experiments/calibrate_macos_native.py
```
