# Spike 4：递归 CTE 多跳图遍历基准（10 万边自证）

日期：2026-10-03
状态：已完成（全合成数据，无任何真实 vault 内容）

## 背景

图路选型已定稿：边（双链/引用）存 SQLite 邻接表，用递归 CTE 做 x 跳扩展。此前调研引用了外部参照「十万边 2 跳 ~305ms / 3 跳 ~706ms」（simple-graph 风格基准）。本实验在本地自证：真实边数量级（10 万边）下，递归 CTE 的 1/2/3 跳延迟是否可用。判定基准：个人库单用户，10 万边 3 跳 p95 ≤ 1s 为通过；秒级为调整。

## 环境

| 项 | 值 |
| --- | --- |
| 机器 | MacBook Air M1（Apple M1），16 GB，macOS 15.7.5 arm64，APFS SSD |
| Python | 3.12.13（`~/llwwds_application/knowledge-vault/venv-mac/`，仅标准库 sqlite3） |
| SQLite | 3.50.4 |
| 数据库 | WAL 模式，运行时产物在 `~/llwwds_application/knowledge-vault/state/spike4_cte/` |
| 脚本 | `experiments/bench_recursive_cte.py`（seed=20261003，幂等重建） |

## 方法

表结构（按定稿方案）：

```sql
CREATE TABLE edges(
    src INTEGER NOT NULL,
    dst INTEGER NOT NULL,
    PRIMARY KEY(src, dst)
) WITHOUT ROWID;
CREATE INDEX idx_edges_dst ON edges(dst);
```

递归 CTE 模板（有向 x 跳遍历，UNION 在 CTE 内按 (node, depth) 去重，外层对 node 去重计数）：

```sql
WITH RECURSIVE walk(node, depth) AS (
    SELECT :start, 0
    UNION
    SELECT e.dst, w.depth + 1
    FROM walk w JOIN edges e ON e.src = w.node
    WHERE w.depth < :max_depth
)
SELECT COUNT(DISTINCT node) FROM walk
```

- 合成随机有向图：主规模 10 万边 / 2 万节点；对照规模 5 万边 / 1 万节点；快检规模 1 万边 / 2 千节点。内存去重采样保证边数精确等于目标，单语句 `executemany` 落库。
- 每个 (规模, 跳数) 从 100 个固定 seed 随机起点各跑一次，记录单次毫秒延迟 p50/p95/max 与返回的 DISTINCT 节点数（含起点自身）。
- 附加对照（各取同 seed 前 50 起点，可与主矩阵直接对比）：UNION vs UNION ALL（去重 CTE 内 vs 外）、性能 PRAGMA（cache_size=-65536 / temp_store=MEMORY / mmap_size=256MB）开关。
- 单次计时只包住一条 `execute + fetchone`，整个遍历在 SQLite C 引擎内完成，无应用层逐行往返。

## 结果

### 主矩阵（基准模板：UNION 内去重，默认 PRAGMA，n=100）

| 规模 | 跳数 | p50 (ms) | p95 (ms) | max (ms) | DISTINCT 节点数均值 |
| --- | --- | --- | --- | --- | --- |
| 100k | 1 | 0.02 | 0.02 | 0.11 | 6.0 |
| 100k | 2 | 0.03 | 0.04 | 0.12 | 31.3 |
| 100k | 3 | 0.09 | **0.17** | 0.71 | 158.3 |
| 50k | 1 | 0.02 | 0.02 | 0.09 | 6.3 |
| 50k | 2 | 0.03 | 0.04 | 0.06 | 33.0 |
| 50k | 3 | 0.10 | 0.21 | 0.24 | 164.8 |

db 文件大小：100k 边 2.2 MB，50k 边 1.1 MB。扇出验证了图的分支结构：平均出度 5 的随机图，1/2/3 跳触达节点数 6 / 31 / 158 ≈ 1+5 / +25 / +125（去重后），CTE 语义正确、确实在逐层扩展。

### 附加对照（n=50）

| 规模 | 变体 | 3 跳 p50/p95/max (ms) | 说明 |
| --- | --- | --- | --- |
| 100k | base UNION（内去重） | 0.09 / 0.17 / 0.25 | 与 UNION ALL 用同 50 起点 |
| 100k | UNION ALL（外层 DISTINCT） | 0.06 / 0.12 / 0.18 | 略快 ~30% |
| 100k | UNION + 性能 PRAGMA | 0.08 / 0.17 / 0.25 | 与默认 PRAGMA 无可测差异 |
| 50k | base UNION（内去重） | 0.10 / 0.21 / 0.24 | |
| 50k | UNION ALL（外层 DISTINCT） | 0.06 / 0.14 / 0.18 | 略快 |
| 50k | UNION + 性能 PRAGMA | 0.09 / 0.21 / 0.32 | 无可测差异 |

- **UNION vs UNION ALL**：3 跳场景下 UNION ALL 略快。原因：3 跳内路径数与去重后节点数同阶（~150 行），UNION 逐行维护去重集合的开销大于它省下的重复扩展；此时把去重挪到外层一次性 DISTINCT 更划算。注意两点边界：(1) UNION 去重按 (node, depth) 二元组，同一节点在不同深度仍会重复扩展，并非严格 BFS 首达；(2) 在多路径重叠严重或跳数更深的图上，UNION ALL 的行数按路径数组合爆炸，外层去重反而会成为瓶颈——本文 3 跳 + 稀疏随机图恰好是它占优的区间，换图形态结论可能反转，不宜据此直接改模板。
- **PRAGMA（cache_size/temp_store/mmap_size）**：无可测差异。db 只有 2.2 MB，默认页缓存与 OS 页缓存已足够，符合预期；数据量到 GB 级才需要重开这组开关。

### --quick 快检对照（1 万边 / 2 千节点，n=100）

| 跳数 | p50 (ms) | p95 (ms) | max (ms) | 节点数均值 |
| --- | --- | --- | --- | --- |
| 1 | 0.01 | 0.01 | 0.25 | 6.1 |
| 2 | 0.02 | 0.03 | 0.04 | 31.2 |
| 3 | 0.08 | 0.16 | 0.20 | 150.6 |

## 与外部参照的差异解释

本机 10 万边 3 跳 p95 0.17ms，比调研参照（2 跳 ~305ms / 3 跳 ~706ms）快约三个数量级，**不是同条件下的"更差"，而是参照数字不可直接迁移**，原因按影响排序：

1. **执行层不同**：simple-graph 是应用层（Ruby）图库，每次扩展在解释器层逐行取数、逐节点再查询，305ms/706ms 大头是解释器往返开销；本实验整棵遍历是单条 SQL，全部在 SQLite C 引擎内完成，无逐行应用层往返。这正是"边存 SQLite + 递归 CTE"选型的价值所在。
2. **图形态不同**：均匀随机图（出度 5）3 跳仅触达 ~158 节点；参照基准与真实知识库图多为幂律形态，枢纽节点扇出大，扩展集可差几个数量级。本实验的均匀随机图**低估了真实图的枢纽扇出**，这是本结论最大的适用性边界。
3. 硬件（M1 + APFS）与 SQLite 版本（3.50.4）差异为次要因素：数据全在页缓存内时两者影响有限。

## 结论与建议

10 万边 3 跳 p95 = 0.17ms，远低于 1s 判定线，两数量级余量。

结论：通过

风险与后续建议：

- 本结论适用于「边量 10 万级、扩展由索引前缀扫描驱动」的机制验证；枢纽扇出是真实风险，接入真实 vault edges 后应在本框架上用真实数据复测（真实样本走 `testdata/`，不入仓库）。
- SQL 侧保险措施（按需启用）：对枢纽节点限制单跳扩展扇出（子查询 + LIMIT）、查询超时兜底；数据量到 GB 级再开 cache_size/temp_store/mmap 组。
- 升级路径保持预案：若真实图出现「枢纽节点 3 跳触达数万节点」且延迟进入数百 ms 区间，再评估 igraph 内存图方案；当前证据下不必提前引入。

## 复跑命令

```bash
# 完整基准（10 万边 + 5 万边，约 1s）
~/llwwds_application/knowledge-vault/venv-mac/bin/python experiments/bench_recursive_cte.py

# 快检版（1 万边）
~/llwwds_application/knowledge-vault/venv-mac/bin/python experiments/bench_recursive_cte.py --quick
```

脚本幂等：每次运行自动删除并重建 `~/llwwds_application/knowledge-vault/state/spike4_cte/` 下自己生成的 db 文件；固定 seed=20261003，数字可复现。
