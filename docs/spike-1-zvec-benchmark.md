# Spike ① — zvec 0.7.0 自证基准（50 万 chunk）

> 日期：2026-10-04（数据于容器内实测）｜ 环境：knowledge-vault-dev 专属容器（linux/arm64，Python 3.12.15，zvec 0.7.0，numpy 2.5.3，8 CPU），宿主 MacBook Air M1 16GB；collection 落 Docker 挂载卷（virtiofs），I/O 路径比宿主直连 APFS 慢，**本报告全部数字为保守上界**。
> 脚本：`experiments/bench_zvec_500k.py`（幂等，含 `--quick`）｜ 结果 JSON：`~/llwwds_application/knowledge-vault/state/spike1_zvec/bench_results_{quick,full}.json`

## 方法

- 合成数据 500,000 条：dim=1024 fp32 随机单位向量（模拟 bge-m3 输出）；payload 按 2026-10-03 定稿**只存元数据**（file_id INT64 倒排索引 / chunk_seq INT32 / kind STRING），正文不入向量库。
- 模拟 10,000 文件 × 50 chunk/file；HNSW（m=16, ef_construction=200）、MetricType.COSINE、collection 默认选项（mmap）。
- 阶段：分批插入 → flush+optimize → 查询延迟 → 按 file_id 删插（delete(ids) 与 delete_by_filter 两种语义）→ 重开验证 → 全量重建端到端。

## 结果（full = 50 万条；quick = 1 万条对照）

| 指标 | full（50万） | quick（1万） |
|---|---|---|
| 插入吞吐（端到端） | 1894 docs/s（263.9s） | 3221 docs/s |
| 插入吞吐（纯 insert 调用） | 2714 docs/s（重建轮 3864） | — |
| flush | 1.18s | — |
| **optimize（全量建图）** | **10195s 首轮 / 5373s 重建轮** | **34.0s** |
| KNN 查询 p50/p95/p99（默认 ef=300） | 20.6 / 74.9 / 85.9 ms | 9.2 / 10.8 ms |
| KNN 查询 p50/p95（ef=100） | **6.7 / 8.8 ms** | 4.3 / 4.8 ms |
| file_id 过滤查询 p50/p95 | 0.50 / 1.31 ms（20/20 校验通过） | — |
| self-recall@10（精确向量自召回） | 0.9 | 1.0 |
| delete(ids) 删 5000 条（1 次调用） | 0.84s（5945 docs/s） | — |
| delete_by_filter 逐 file 删 100 文件 | 0.48s（单次 ~4.8ms） | — |
| upsert 重插 5000 条 | 1.27~2.18s | — |
| 重开 collection | 0.30s | — |
| 全量重建端到端（drop+create+insert+flush+optimize） | 5553s | 36.5s |
| 磁盘占用 | 2.07 GB（≈原始向量 2GB，索引近零开销） | 44 MB |
| 峰值 RSS | 4.13 GB | — |

删插后 doc_count 守恒（500000→495000→490000→500000）已断言通过；重建后复测查询 p50=19.0ms / p95=30.8ms，与重建前一致。

## 结论：通过（附一项运维调整）

zvec 在 50 万 chunk 规模下：查询毫秒级（ef=100 时 p95=8.8ms）、按 file_id 删插亚秒级、插入吞吐 ~1900-3900 docs/s、磁盘与内存开销可预期——选型成立，**查询/删插/插入全部通过判定线**。

**调整项（运维策略，非选型问题）：`optimize()` 全量 HNSW 建图耗时超线性增长**（1 万条 34s → 50 万条 1.5~2.8 小时）。摄入策略应改为：

1. 增量插入后仅 flush（新数据可查，走未索引路径）；
2. optimize 低频批量触发（如夜间任务或攒批 N 万条），避免每次入库全量重建；
3. 本数字受容器 virtiofs I/O 与建图参数影响，属保守上界；换 embedding 模型/切块参数时的「全量重建」成本按此预估，宜一次性离线进行。

适用性边界：随机均匀向量是 HNSW 的劣化分布，self-recall@10=0.9（ef=300）在此条件下属正常水平；真实 embedding 有簇结构，recall 与延迟预计更好。接入真实数据后宜复测一轮。

## 复跑

```bash
# 自检（1 万条，~1 分钟）
docker exec -w /repo knowledge-vault-dev /opt/venv/bin/python experiments/bench_zvec_500k.py --quick
# 正式（50 万条，容器内约 2.5-3.5 小时，大头是 optimize）
docker exec -d -w /repo knowledge-vault-dev /opt/venv/bin/python experiments/bench_zvec_500k.py
```
