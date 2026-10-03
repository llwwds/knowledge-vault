# Spike ② — bge-reranker-v2-m3 CPU 延迟与内存（64 对 query-passage）

> 日期：2026-10-04（数据于 knowledge-vault-dev 容器内实测）｜ 环境：knowledge-vault-dev 专属容器（linux/arm64，Python 3.12.15，torch 2.14.1，FlagEmbedding 1.4.2，transformers 5.18.0，8 CPU，Docker VM 内存上限 8GB，cgroup 未单独限额），宿主 MacBook Air M1 16GB。
> **读数须知**：容器内 torch 的 aarch64 CPU GEMM 走 NEON 路径，未用到 macOS Accelerate/AMX；本报告全部数字为**保守上界**。macOS 原生 venv（Accelerate/AMX）预计有数倍提升，最终部署前应在宿主 venv 复测对照。但"该配置不可交互式使用"这一量级结论方向在宿主上不会反转。
> 脚本：`experiments/bench_reranker_cpu.py`（幂等，含 `--quick`；每配置独立子进程，慢轮自适应降轮次并记录 `rounds_slow_applied`）｜ 结果 JSON：容器内 `/root/llwwds_application/knowledge-vault/state/spike2_reranker/results.json`（同目录 `frag_*.json` 为各配置原始片段）。

## 方法

- 模型：`BAAI/bge-reranker-v2-m3`（XLMRobertaForSequenceClassification，24 层 / hidden 1024，参数量 567.8M）。`FlagReranker(..., use_fp16=True, devices="cpu")`，纯 CPU（容器内 cuda / mps 实测均不可用）。
- 工作负载：1 条固定中文技术 query + 64 条合成中文技术 passage（模板 + 固定词表 + 固定种子 seed=20261004，每条 189-400 字符；不含任何真实 vault 内容）。tokenize 后 pair 长度 p50=280 / mean=288 / max=361（≈真实笔记 chunk 的中等偏短档）。
- `compute_score` 全部使用库默认参数（`normalize=False`、`max_length=512`）；延迟 = 一次 `compute_score(64 对)` 调用的完整耗时。
- threads 对比：`torch.set_num_threads(4)` vs `8`，各 1 预热 + 2 计测轮，按 p50 选优；胜出档（8）再跑全量协议（2 预热 + 计测，请求 10 轮；单轮 > 20s 触发慢轮自适应，实际计测 5 轮）。
- 批大小对比：64 一批（单次前向）vs 8×8 分批（`batch_size=8`，同进程先后测，各自预热）。
- 精度口径：FlagEmbedding 1.4.2 的 `BaseReranker` 在 CPU 上**从不调用 `model.half()`**（`compute_score_single_gpu` 内有 `if device == "cpu": self.use_fp16 = False` 分支，init 也不 half），transformers 5.18 按 config（torch_dtype=float32）加载 → **库默认路径在 CPU 上实测为 fp32**（worker 记录 `model_dtype=torch.float32`）。因此补跑强制 `model.half()` 的真 fp16 对照（fp32→fp16 转换后推理）。
- 内存：`resource.ru_maxrss`（linux 单位 KB，进程级峰值）+ `/proc/self/status` VmRSS / VmHWM。

## 模型信息

| 指标 | 数值 |
|---|---|
| 参数量 | 567.8M |
| 磁盘缓存体积（HF cache 实测） | 2.29 GB（fp32 safetensors 单文件 2.27 GB） |
| 加载耗时 | 首次冷读 58.1s（virtiofs + 反序列化）；权重入页缓存后 6.3-7.1s |
| 内存中权重体积 | fp32 2.12 GiB；fp16 1.06 GiB |
| 加载后 RSS（fp32） | ~1.2 GiB（safetensors mmap 惰性驻留，权重页随推理逐步进入 RSS） |
| 推理峰值 RSS | fp32（64 一批）**3.46 GiB**；fp16 对照 4.49 GiB（含 fp32→fp16 转换双份驻留的伪峰，见下） |

## 延迟（64 对 query-passage，一次 compute_score）

| 配置 | 轮次（预热+计测） | p50 | p95 | max | 单对均摊（p50） |
|---|---|---|---|---|---|
| threads=4，fp32，64 一批 | 1+2 | 504.2 s | 519.0 s | 520.7 s | 7.88 s |
| **threads=8，fp32，64 一批（选优档）** | 2+5 | **395.7 s** | **405.6 s** | 407.3 s | 6.18 s |
| threads=8，fp32，**8×8 分批** | 1+5 | **178.3 s** | 191.4 s* | 193.2 s | 2.79 s |
| threads=8，强制 fp16，64 一批 | 1+2 | 522.2 s | 523.5 s | 523.7 s | 8.16 s |

\* 8×8 的 p95 由 5 轮原始数据推得（脚本只直接落了 b8 的 p50）。threads 对比用探测口径（4 档 504.2s vs 8 档 438.6s，t8 提速 ~13%）；主协议里 t8 进一步降到 395.7s（页缓存与 JIT 路径更热），排名不变，8 线程最终收益 ~1.27×。

关键观察：

1. **小批显著快于大批**：8×8 分批比 64 一批快 **2.22×**（178.3s vs 395.7s）。CPU 上大批前向的工作集（batch 64 × ~361 token 的注意力矩阵与中间激活）超出缓存/内存带宽友好区间；**生产在 CPU 上应固定小批（8）**。
2. **fp16 在该栈上更慢**：fp16 比 fp32 慢 1.32×（522.2s vs 395.7s）——aarch64 torch CPU 无快速 fp16 GEMM 路径，转换开销吃掉收益。**CPU 上不要用 `model.half()`**。
3. **数值一致性**：8×8 与 64 一批对同一输入的得分最大差 1e-5，分批无损。分数范围（normalize=False 原始 logit）[-8.34, -0.51]，fp16 对照一致，区分度正常。
4. fp16 对照的 4.49 GiB 峰值是"先载 fp32 再 half()"的转换伪峰（fp32 2.12 GiB + fp16 1.06 GiB 同时驻留）；若生产真要 fp16，应在 `from_pretrained` 时直接以 fp16 dtype 加载，峰值预期 ~2.5 GiB 以内。但鉴于第 2 条，CPU 上此举无意义。

## 内存与余量

- fp32 主路径推理峰值 RSS 3.46 GiB（64 一批时的峰值；8×8 相位更低，未单独计量）。
- Docker VM 上限 8 GB → 余量 ≈ 4.5 GB，**满足 ≥ 2GB 判据**；与 bge-m3 embedding（实验③）同驻内存可容纳。
- 内存不是瓶颈；瓶颈是延迟。

## 结论：调整

判据「64 对单批 p95 ≤ 5s 且内存余量 ≥ 2GB」：内存达标（余量 ≈ 4.5 GB），**延迟严重不达标**——64 对单批 p95 = 405.6s（fp32/threads=8/64 一批），即使采用更优的 8×8 分批（p95 ≈ 191.4s），仍超预算 **38-81×**；单对均摊 2.8-6.2s，意味着查询内 rerank 在该执行后端上不可交互式使用。且 fp16 强制对照证明该栈上无免费的精度提速。

**建议（按优先级）**：

1. **换执行后端，其次换模型规格**。要满足 5s/64 对需 ≥ 38× 提速，单靠一项优化不够：
   - ONNX Runtime int8（或 OpenVINO）：568M 交叉编码器在 aarch64 CPU 通常 3-8×；
   - 叠加更小规格（bge-reranker-base 278M ≈ 2×，或官方蒸馏版）；
   - 两者叠加约 6-16×，64 对 p95 可压到 ~12-30s——仍不满足 5s。**因此在线查询末级 rerank 在 8 核 ARM CPU 上应视为不可行，或在 ≤ 8-12 候选的小池上接受秒级延迟。**
2. **架构落位**：把 rerank 移出首响路径——先返回 RRF 融合结果，rerank 后异步刷新/二次排序；或仅在用户翻页/追问时触发。等价地，rerank 只服务"精排离线批处理"。
3. torch 路径若保留：threads 固定 8（与容器 CPU 数一致，收益 ~1.27×，勿与他进程争抢）；batch 固定 8（收益 2.22×）；normalize 维持默认 False，打分后自行 sigmoid/阈值化。
4. 候选数上限（fp32、8×8 分批、线性外推）：30 候选 ≈ 84s、64 ≈ 178s、96 ≈ 267s、128 ≈ 357s；若用 64 一批口径则再 ×2.2。**任何"查询内同步 rerank 数十候选"的方案在该后端上都应直接否决。**
5. **宿主复测**：上线前在 macOS 原生 venv（Accelerate/AMX）跑同一脚本 `--quick` 校准倍率；若原生路径有 ≥ 5× 提升，上述结论需按倍率重估（尤其"小池在线 rerank"是否可行）。

## 复跑

```bash
# 自检（真实模型，轮次少：协议 1 预热 + 3 计测；模型未缓存时先下载 ~2.3GB）
docker exec -w /repo knowledge-vault-dev /opt/venv/bin/python experiments/bench_reranker_cpu.py --quick
# 正式（threads 探测 + 全量协议 + 精度对照；容器内约 2.5-3.5 小时，单轮 ~6-9 分钟占大头）
docker exec -w /repo knowledge-vault-dev /opt/venv/bin/python experiments/bench_reranker_cpu.py
# 结果与片段
docker exec knowledge-vault-dev cat /root/llwwds_application/knowledge-vault/state/spike2_reranker/results.json
```
