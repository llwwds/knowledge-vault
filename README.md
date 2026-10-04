# knowledge-vault

个人知识库的后端服务。它是一个无交互界面的应用（headless），对外只提供检索与写入 API，供人类前端（Obsidian）与 agent 调用。

## 定位

在 `code_file` 之外，个人知识内容的**真源**目前分散在 Obsidian vault、原始资产区（PDF / 图片 / Office / 音视频 / 超大文件）等位置。knowledge-vault 不取代这些真源，而是在它们旁边维护一套**可随时重建的派生索引层**，提供：

- 统一登记：为每个文件建立元数据（路径、hash、frontmatter、类型、时间戳）。
- 混合召回：关系型过滤 + 向量语义 + 关键词（词法）+ 图关系，多路融合后 rerank 出 top-k。
- 图导航：从 Obsidian 双链派生节点与边，支持多跳遍历与上下文扩展。

## 意义

知识库的内容会越来越大、格式越来越杂（md、PDF、图片、Office、数据文件、音视频），且要同时服务两个读者：

1. **人类** —— 需要自由地读、写、编辑，用 Obsidian 当界面。
2. **agent** —— 需要高性能、结构化、事务性的增删改查，以及准确且能解释的召回。

这两个读者对「存储」和「检索」的要求互相冲突：人类要的是「文件 + 文件夹 + 编辑器」，agent 要的是「数据库 + 索引 + 查询」。knowledge-vault 用「真源与索引分离」的方式同时满足两者：真源保持人类友好的形态，索引层提供 agent 友好的形态，两者由一条摄入管线持续同步。

一句话：**它是把散落的个人知识，变成一个可被准确、高性能地查询的「知识金库」的那层后端。**

## 代码结构

标准 Python 包（hatchling + src 布局），阶段1交付登记层 + 全文索引 + 图查询；zvec 向量库、摄入与召回管线由后续阶段交付。

```
src/knowledge_vault/
├── config.py       # KV_* 环境变量注入 → VaultConfig（state 目录 / vault 根 / userdict / file_id 种子）
├── schema.sql      # documents / chunks / edges / chunks_fts(FTS5) + 索引 DDL，schema_version 管理
├── store.py        # 连接管理（WAL/foreign_keys）、建库与迁移、Store 薄门面
├── file_id.py      # file_id 单调分配器（专用 counter 表，永不回收，KV_FILE_ID_SEED 可抬高起点）
├── registry.py     # 登记：register_file / soft_delete / move_file / get / iterate（软删不过滤）
├── textindex.py    # chunks 写入同步 FTS5（jieba 预分词 + userdict + span 预处理）、search / snippet
├── graph.py        # edges 写入与递归 CTE 多跳扩展（UNION 去重口径）
└── data/userdict.txt  # jieba 用户词典（随包打包，experiments/userdict_v0.txt 的拷贝）
tests/              # pytest 全套（合成数据 + tmp_path 隔离，不读真实 vault/快照）
```

运行时配置（环境变量，均可在启动时注入）：`KV_STATE_DIR`（默认 `~/llwwds_application/knowledge-vault/state`）、`KV_VAULT_ROOT`（默认 `~/Documents/knowledge-vault_file`）、`KV_USERDICT`（默认包内 `data/userdict.txt`，置空禁用）、`KV_FILE_ID_SEED`（默认 1）。

开发与测试在专属容器内进行（`bash dev/container/run.sh` 创建/重同步依赖），跑测试：

```bash
docker exec -w /repo knowledge-vault-dev /opt/venv/bin/python -m pytest
```
