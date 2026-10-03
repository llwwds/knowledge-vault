# Spike ③ — bge-m3 本地 CPU embedding 吞吐与同驻内存（10 文件 × ~512-token chunks）

> 日期：2026-10-04（数据于 knowledge-vault-dev 容器内实测）｜ 环境：knowledge-vault-dev 专属容器（linux/arm64，Python 3.12.15，torch 2.14.1，FlagEmbedding 1.4.2，transformers 5.18.0，huggingface_hub 1.33.0，8 CPU，Docker VM 内存上限 8GB（MemTotal 8,024,748 kB，cgroup 未单独限额），cgroup memory.max=max），宿主 MacBook Air M1 16GB。
> **读数须知**：与实验②相同，容器内 torch 走 NEON 路径，未用到 macOS Accelerate/AMX；本报告全部吞吐数字为**保守上界**（宿主原生 venv 预计有数倍提升，部署前需宿主复测校准）。内存结论受 Docker VM 8GB 上限约束：容器内不 OOM 只能证明峰值 ≤8GB，16GB 宿主判断见「同驻内存与 16GB 外推」一节。
> 脚本：`experiments/bench_bge_m3_local.py`（幂等，含 `--quick`；各阶段独立子进程，frag 文件落盘可断点核对）｜ 结果 JSON：容器内 `/root/llwwds_application/knowledge-vault/state/spike3_bge_m3/results.json`（同目录 `frag_*.json` 为各阶段原始片段）。

## 方法

- 模型：`BAAI/bge-m3`（XLMRoberta-large 骨干 + colbert/sparse 线性头，实测 568.8M 参数，稠密输出 1024 维，L2 归一化）。主测量显式 `BGEM3FlagModel(..., use_fp16=False, devices="cpu")`（fp32 口径，与实验②结论一致）；另设 dtype probe 子进程用**纯库默认参数**实例化，实证 use_fp16 在 CPU 上的真实行为（见下）。
- 工作负载：10 个模板 + 固定词表 + 固定种子（seed=20261004）合成的「典型笔记文件」（正文 4218-4321 字符，不含任何真实 vault 内容），按 XLM-R tokenizer 贪心句子打包切成 **7-8 个 ≤512 token 的 chunk，共 77 chunks**（token p50=460 / mean=426 / max=491）。
- 逐文件吞吐：每个文件一次 `encode(chunks, batch_size=8, max_length=1024, return_dense=True)`（含 tokenize/排序/填充/前向/池化全流程，生产口径），2 计测轮；预热为文件 0 前 4 chunk。
- 批大小敏感性：77 chunks 全集，一批 64 vs 8×8，各 1-2 计测轮（模型已充分预热；批 64 单轮 > 420s 触发慢轮自适应）。
- 同驻内存：同一进程先 load BGEM3FlagModel 推理一次（8 chunks），再 load `FlagReranker('BAAI/bge-reranker-v2-m3')`（库默认路径，复用实验②缓存）推理一次（64 对合成 pair，batch_size=8），记录 `ru_maxrss`（linux KB→GiB）。OOM 时脚本自动回退为「两进程分别 RSS 相加」估算（本次未触发）。
- `torch.set_num_threads(8)`（实验②选优档）；`TOKENIZERS_PARALLELISM=false`。

## 模型信息

| 指标 | 数值 |
|---|---|
| 磁盘缓存体积（(dev,ino) 去重实测） | 2.29 GB（2,295,435,830 bytes；pytorch_model.bin 单文件 2.27 GB） |
| 参数量 | 568.8M（骨干 567.8M + colbert 头 1.05M + sparse 头 1K） |
| 内存中权重体积 | fp32 2.12 GiB（dense 维 1024） |
| 加载耗时 | fp32 冷读 ~58s 量级（virtiofs，同实验②）；页缓存热后 **2.5-7.7s** |
| 加载后 RSS | ~1.21 GiB（safetensors mmap 惰性驻留）；预热一次后 2.43 GiB（权重全部进 RSS） |

**dtype 实测（use_fp16 在 CPU 上的真实行为，与实验②口径对照）**：FlagEmbedding 1.4.2 中两个基类的行为机制**不同**但净效果一致——

- `BGEM3FlagModel`（本实验，`M3Embedder`）：`use_fp16` **默认 True**，init 经 `get_model_torch_dtype()` 把 `torch_dtype=float16` 传给 `from_pretrained`（probe 实测加载后 dtype=torch.float16，RSS 1.99 GiB）；但 `encode_single_device` 开头有 `if device == "cpu": self.model.float()`，首次 encode 就地转回 fp32（probe 实测 encode 后 dtype=torch.float32，RSS 3.42 GiB，含转换期 fp16+fp32 双份驻留，probe 峰值 3.85 GiB）。即库默认路径在 CPU 上是「**fp16 加载 → fp32 推理**」，白白多一次加载与转换。
- `BaseReranker`（实验②）：init 不 half()，compute 分支在 CPU 上直接禁用 fp16，全程 fp32。
- **结论：CPU 上两者实际都是 fp32 推理；embedding 应显式传 `use_fp16=False`，省掉无意义的 fp16 加载与 cast（还省 ~1GB 转换伪峰）。** 强制 fp16 在该栈更慢的结论（实验②，1.32×）对 embedding 同样适用（同骨干同算子）。

**下载来源说明**：实验执行时 huggingface.co 与 hf-mirror.com 均不可达（TLS 被重置，VPN 未开），bge-m3 改从 ModelScope 镜像（BAAI 官方同步仓库）下载，逐文件 sha256 校验后按 HF 缓存规范手工构建 blobs/snapshots/refs（revision=modelscope-master），`HF_HUB_OFFLINE=1` 运行；该仓库无 safetensors，权重为 pytorch_model.bin。onnx/ 与 imgs/ 未拉取（若从 HF 全量 snapshot_download，缓存会额外包含 ~2.2GB onnx 副本）。实验②的 reranker 缓存为 HF 原源下载，直接复用，未重新下载。

## 吞吐

### 逐文件（batch_size=8，max_length=1024，每文件一次 encode）

| 文件 | chunks | tokens | 轮 1 ms | 轮 2 ms | p50 ms | 每 chunk ms |
|---|---|---|---|---|---|---|
| 例01 | 8 | 3268 | 135,908 | 61,092 | 98,500 | 12,312 |
| 例02 | 8 | 3280 | 136,369 | 70,760 | 103,564 | 12,946 |
| 例03 | 8 | 3327 | 80,404 | 65,600 | 73,002 | 9,125 |
| 例04-例10（7 汇总口径） | 7-8 | 3239-3335 | 52,518-71,871 | 52,519-63,000 | 54,898-66,197 | 7,843-8,275 |
| **汇总 p50** | 7.7 | 3269 | — | — | **64,694** | **8,087** |

**读数警告（重要）**：逐文件阶段的前 ~25 分钟恰逢宿主上其他容器/进程争抢 CPU（同机另有 5 个运行中容器），前两轮被拉长到 136s；同进程随后在**无争抢窗口**测得的全集批对比（下表 batch_8 = 4,138 ms/chunk）与 quick 自检（不同时段，4,231 ms/chunk）互相印证，**稳态真实值约 4.1-4.2 s/chunk（每 1000 token ≈ 9.7s）**，逐文件阶段的 8.1s/chunk 应视为受争抢污染的上界。交叉验证：同驻阶段 reranker 64 对 b8 实测 2.70s/pair，与实验②独立测得的 2.79s/pair 一致（±3%），证明争抢窗口之外 CPU 状态与实验②口径相同。折算稳态：**一个 7-8 chunk 的笔记文件 ≈ 32s**（4,138 ms/chunk × 7.7）。

### 批大小敏感性（77 chunks 全集，同进程同口径）

| 模式 | 计测轮 | 全集耗时 p50 | 每 chunk ms | 相对 |
|---|---|---|---|---|
| **batch_size=8（8×8 分批）** | 2 | **318.6 s** | **4,138** | 1.00× |
| batch_size=64（一批 64） | 1（慢轮自适应截断） | 579.8 s | 7,530 | 1.82× |

方向与实验②一致（大批劣化），幅度较小（pair 前向 2.22× → 单序列 1.82×）。**生产固定 batch_size=8**。数值一致性：两种批模式的 dense 向量最大余弦距离 3e-7，分批无损。

### 全库外推（依据：稳态 batch_size=8 = 4,138 ms/chunk，426 token/chunk）

| 场景 | 计算 | 结果 |
|---|---|---|
| 全库一次性入库（5 万文件 × 20 chunks = 100 万 chunks） | 1e6 × 4.138s | **1,149.5 h ≈ 48 天**（batch 64 口径 2,091.6 h） |
| 4h 预算内的全库规模上限 | 4h ÷ 4.138s/chunk | **≈ 3,479 chunks ≈ 173 个文件**（20 chunks/文件口径）；按本实验 7.7 chunks/文件口径 ≈ 452 个 |
| 增量入库：每天 10 个文件（200 chunks） | 200 × 4.138s | **≈ 13.8 min/天** |
| 增量入库：每天 50 个文件 | 1000 × 4.138s（20 chunks/文件口径） | ≈ 69 min/天 |
| 单 chunk 延迟（查询侧构造查询向量） | — | ≈ 4.1 s（426 token）；查询句更短，按 9.7ms/token 线性折算，64 token 查询 ≈ 0.6s |
| 宿主 M1 原生（AMX/Accelerate）外推 | 按 3-5× 提速（待宿主校准） | 全库 230-380 h；4h 预算内 ≈ 900-1500 chunks |

（若真实库 chunk 比 426 token 短——实验②实测真实笔记 chunk pair p50=280——按 9.7ms/token 线性缩放，全库入库仍在数百小时量级，结论不变。）

## 同驻内存与 16GB 外推

**容器实测（同进程顺序推理：先 embedding 8 chunks，再 reranker 64 对 b8，均 fp32）**：

| 相位 | RSS | 说明 |
|---|---|---|
| embedding 加载后 | 1.18 GiB | mmap 惰性 |
| embedding encode 后 | 2.44 GiB | 权重全部进 RSS |
| reranker 加载后 | 2.75 GiB | |
| reranker 推理中 | — | 两模型权重都在（4.24 GiB 名义和），部分干净文件页可被内核回收再缺页 |
| **进程峰值 ru_maxrss** | **3.991 GiB**（4,185,272 KB） | **8GB VM 内无 OOM**，未触发两进程相加回退 |
| 阶段 1 单独峰值（maxrss after phase1） | 2.56 GiB | embedding 单模型推理峰值 |

（dtype probe 单独测得 fp16→fp32 转换路径峰值 3.85 GiB——若误用库默认 use_fp16=True，同驻前就先吃掉这一伪峰，再叠加两模型常驻将逼近 6GB，8GB VM 内有险。显式 use_fp16=False 后无此伪峰。）

**16GB 宿主判断（两种口径，标注待宿主校准）**：

1. **同进程顺序推理**（每条查询：embedding 构造向量 → 召回 → rerank 精排，两模型同进程先后各推理一次）：容器实测峰值 3.99 GiB，且这是在 8GB VM（内存压力更大、页回收更激进）下的观测——**8GB VM 都绰绰有余，16GB 宿主更无问题**。判据余量：8GB VM 内余量 ≈ 4GB，满足实验②「≥2GB」标准。
2. **双模型常驻**（两模型永久驻留内存随时服务）：权重和 4.24 GiB（fp32）+ torch 运行时与激活峰值 ~1.3 GiB（b8 口径）+ Python 解释器 ~0.3 GiB ≈ **6 GiB 应用层**；macOS 系统与其他应用通常占 3-5 GB，16GB 宿主余量 ≈ 5-7 GB，**可行但不奢侈**。若需再压缩：两模型分时复用（顺序加载/卸载，或 embedding 用完释放）即为容器已验证的口径 1；或 reranker 按实验②建议改 ONNX int8（峰值同步大幅下降）。
3. 若未来上 fp16 权重直载（CPU 上不推荐，更慢）可省一半权重内存——但对该栈无性能意义。

## 结论：调整

判据逐条核对（判据：单文件 embedding 毫秒级~秒级、全库一次性入库 < 4h、同驻或分时复用内存可行）：

1. **内存：通过**。同进程顺序推理峰值 3.991 GiB（8GB VM 内），双模型常驻估算 ~6 GiB，16GB 宿主两种口径均可行。
2. **单文件 embedding：部分通过**。稳态 4.1 s/chunk（9.7 ms/token）、~32 s/文件（7.7 chunks）——单 chunk 达到「秒级」，但一个 20-chunk 文件 ≈ 80-90s，远超「毫秒级」；作为 watcher 增量入库（保存后约半分钟到一分半进索引）可接受，作为交互内嵌步骤偏慢。
3. **全库一次性入库 < 4h：严重不通过**。容器实测外推 1,149.5 h（48 天）@ 100 万 chunks；即使按宿主 AMX 3-5× 乐观提速仍 230-380 h；winpc x86 服务器按 2-4× 容器速率外推仍 300-600 h。**任何本地 CPU 方案都无法满足 4h 全库入库**；4h 预算内本地只能吃下 ~170-450 个文件。

**调整方向（数据依据）**：

1. **全库初始化不走本地 CPU**。触发条件：待入库库规模 > ~200 个文件（20 chunks/文件口径）即破 4h。选项按优先级：(a) 全库 embedding 移 winpc 自托管（需先实测 winpc CPU 速率，外推区间 300-600h，仍不达 4h——若 winpc 也只有 CPU，须换 GPU 或接受「后台慢速回填」）；(b) 改判据为「增量入库 + 后台慢速回填全库」，初始化期间新内容优先、旧内容按队列消化；(c) 缩小模型规格（bge-m3 568M → 更小 embedding 模型），吞吐近似线性提升，但会牺牲多语/长文档质量，需另立实验评估召回质量。
2. **增量与查询路径本地可跑**：每天几十文件的增量入库（13-69 min/天，可后台分片）、单查询向量构造（~0.6s，64 token 查询）在本容器 CPU 已可用；宿主 AMX 校准后预期再快数倍。
3. **与 reranker 同驻**：无内存障碍（3.99 GiB 峰值），且 rerank 本就应按实验②结论移出首响路径；两模型在 CPU 上的瓶颈都是延迟而非内存。生产固定：`use_fp16=False`、`batch_size=8`、`threads=8`。
4. **待宿主校准项**：macOS 原生 venv（Accelerate/AMX）跑同脚本 `--quick` 校准吞吐倍率；宿主 16GB 上重测同驻峰值（预期与容器观测一致或更低）。

> **【2026-10-04 校准更新，结论按本报告自设第 4 条规则重估】**：macOS 原生最小校准点（`docs/spike-6-macos-native-calibration.md`）实测 **104.2 ms/chunk**（本报告容器稳态 4138 ms，加速比 ~40×）、同进程双模型同驻峰值 **3.01 GiB**（16GB 宿主）。**重估结论：通过（macOS 原生口径）**——增量入库 ~42s/天（50 文件×8 chunks）、查询向量 <0.2s、全库 100 万 chunks 初始化 ≈ **29h（一夜后台任务）**；原「全库移 winpc」调整方向降级为**可选加速项**（嫌慢或 Mac 被占用时才启用）。同驻内存无压力实锤，支持双模型常驻形态。生产配置：库默认 fp32 路径（显式 `use_fp16=False` 省转换伪峰）、batch=8、threads=8。

**决策问题回答**：

- **bge-m3 本地可跑吗？** 能跑、数值正确（fp32 口径，稠密 1024 维 L2 归一化），内存无压力；吞吐为 9.7 ms/token（容器稳态），增量场景（每天几十文件）可用，全库一次性入库（10 万-100 万 chunks）在 CPU 上是千小时级，不可行——本地定位应为「增量 + 查询向量」，全库初始化移 winpc 或改后台回填。以上为容器数字与外推，待宿主校准（方向不会反转：量级差距 250× 不是 3-5 倍硬件提升能弥合的）。
- **与 reranker 同驻在 16GB 上吃紧吗？** 不吃紧。容器 8GB VM 内同进程顺序推理峰值 3.99 GiB（无 OOM）；16GB 宿主双模型常驻估算 ~6 GiB 应用层 + 系统 3-5 GB，余量 5-7 GB。同驻内存不是该架构的约束项，延迟才是（实验②）。

## 复跑

```bash
# 自检（3 文件；模型未缓存时需先解决下载，见下）
docker exec -e HF_HUB_OFFLINE=1 -w /repo knowledge-vault-dev /opt/venv/bin/python experiments/bench_bge_m3_local.py --quick
# 正式（10 文件 77 chunks；容器内约 55 分钟，bench 阶段 ~45 分钟占大头）
docker exec -e HF_HUB_OFFLINE=1 -w /repo knowledge-vault-dev /opt/venv/bin/python experiments/bench_bge_m3_local.py
# 结果与片段
docker exec knowledge-vault-dev cat /root/llwwds_application/knowledge-vault/state/spike3_bge_m3/results.json
```

模型缓存：bge-m3 已在 `/app_runtime/hf_cache/hub/models--BAAI--bge-m3`（ModelScope 镜像下载，sha256 校验，revision=modelscope-master）；huggingface.co 可达时也可直接 `snapshot_download('BAAI/bge-m3')` 由脚本自动下载（去掉 `HF_HUB_OFFLINE=1`）。bge-reranker-v2-m3 复用实验②缓存，脚本不会重新下载。
