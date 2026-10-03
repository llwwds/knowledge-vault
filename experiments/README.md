# experiments/

Spike 实验与性能基准脚本（开发与部署规范第 3 条：测试即开发，实验同为开发活动）。

- 脚本进本目录，随仓库版本管理；结论报告进 `../docs/`。
- 真实 vault 样本一律放设备本地 `~/llwwds_application/knowledge-vault/testdata/`，**永不进 git**（隐私红线，规范第 4 条）；仓库内只允许合成或脱敏样例。
- 实验运行时产物（数据库、zvec collection、模型缓存）放 `~/llwwds_application/knowledge-vault/` 下的 `state/`、`hf_cache/`。
- 运行方式：`~/llwwds_application/knowledge-vault/venv-mac/bin/python experiments/<脚本>.py`。

## 已有实验

| 脚本 | 对应 spike | 结论 |
|---|---|---|
| `bench_zvec_500k.py` | ① zvec 50 万 chunk 自证基准 | `../docs/spike-1-zvec-benchmark.md` |
| `bench_reranker_m1.py` | ② bge-reranker-v2-m3 M1 CPU fp16 | `../docs/spike-2-reranker-m1.md` |
| `bench_bge_m3_local.py` | ③ bge-m3 本地可跑性与同驻内存 | `../docs/spike-3-bge-m3-local.md` |
| `bench_recursive_cte.py` | ④ 递归 CTE 3 跳真实边数 | `../docs/spike-4-recursive-cte.md` |
| `jieba_vocab_audit.py` | ⑤ jieba 切分质量抽查与自定义词典 | `../docs/spike-5-jieba-vocab.md` |
