# knowledge-vault skill 使用参考（中文）

> 目标：agent 以最低阻力调用个人知识库自由检索。一条命令、结构化输出、零前置配置。

## 调用方式

```bash
python3 <skill目录>/scripts/kv.py search "查询词" [选项]
```

脚本自动完成：探测 `http://127.0.0.1:8770` 服务 → 离线则后台拉起 `kv-serve` → 仍失败则回退直连 `kv-search --no-rerank`。agent 无需关心服务生命周期。

## 子命令

| 命令 | 说明 |
|---|---|
| `search "查询词"` | 混合召回（FTS+jieba / 向量 / 图扩展 → RRF 融合）。默认 `--top-j 8`、不 rerank（亚秒级） |
| `search "查询词" --rerank` | 开启 bge-reranker 精排（64 对约 4s），重要查询用 |
| `search "..." --tag "02 项目" --tag "03 笔记"` | 按 context_tag（目录层级标签）过滤，可重复 |
| `search "..." --status library` | 按登记状态过滤（now=当前工作集 / library=馆藏） |
| `search "..." --json` | 完整 JSON 输出（含 file_id/chunk_id/file_path/text 全文） |
| `stats` | 库概况（documents/chunks/edges/版本） |
| `file <file_id>` | 单文件登记详情（路径/时间/状态/hash） |
| `serve-status` / `serve-start` | 服务状态 / 后台拉起 |

## 输出结构

默认格式（人类可读）：

```
0.0325 [fts,vector] #123 项目文档标题
    chunk 正文摘要……（160 字截断）
--- {'vector': 'present', ...}   ← meta（stderr）
```

`--json` 时每个结果含：`file_id`（登记主键）、`chunk_id`（f{file_id}-c{seq}）、`kind`（chunk/summary）、`score`（最终排序分，rerank 后为精排分）、`sources`（命中来源路）、`file_path`（相对真源区 `~/Documents/knowledge-vault_file/`）、`title`、`text`（索引层 chunk 原文，可直接作 context）。

## 检索质量要点

- 中文查询直接写自然语言；专有工具名/项目代号已入 jieba 用户词典（如 zvec、bge-m3、knowledge-vault）。
- 词法与向量双路独立召回后 RRF 融合——精确术语命中走 FTS 路，语义改写走向量路；结果 `sources` 字段标注命中来源。
- 04 信息（041/042 等敏感目录）与凭据文件按裁决排除出向量库，FTS 覆盖范围见入库配置；**检索无结果 ≠ 库里没有**。
- 大文件（>50MB）只登记不切块，仅有 summary 的文件以 `kind=summary` 记录参与召回。

## 写入边界（重要）

- 本 skill **只读**（外加 optimize 触发）。知识库内容的唯一合法写入路径：用户在 Obsidian 仓库编辑 → 入库落位同步到真源区 → watcher/管线自动索引。
- agent 发现「库里没有某知识」时，正确动作是建议用户写入 Obsidian，而不是尝试直写索引。

## 故障排查

- `服务离线且拉起失败`：检查 `KV_CLI` 路径是否存在（默认 `~/llwwds_application/knowledge-vault/venv-mac/bin/kv`）；手动 `KV_CLI.../kv kv-serve --port 8770` 前台看报错。
- 端口冲突：设 `KV_API_PORT` 换端口（脚本与 kv-serve 用同一变量）。
- 结果为空且无报错：确认查询词与库内容语言一致；试 `--json` 看 meta；用 `stats` 确认 documents/chunks 非零（刚重入库完成前为空属正常）。
