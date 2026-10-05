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

## 版本语义（一级版本号）

- **v1.x —— 知识库就是我的 Obsidian 仓库**：知识以 md 文件形式散落在 Obsidian vault 中，检索靠 Obsidian 自身，没有索引层。
- **v2.x —— 知识库就是诸位眼前的这个系统（knowledge-vault）**：Obsidian 仓库回归「人类编辑界面」，knowledge-vault 在旁维护原文件真源区（`~/Documents/knowledge-vault_file/`，与开发/部署位置严格三分离）与可随时重建的派生索引层，对外提供 CLI / HTTP API / web 看板 / agent skill。

| 位置 | 角色 |
|---|---|
| `~/Documents/obsidian_file/` | 人类编辑区（Obsidian 仓库，内容源头） |
| `~/Documents/knowledge-vault_file/` | **原文件真源区**（入库落位，系统索引的对象） |
| `~/Documents/code_file/knowledge-vault/` | 开发仓库（代码/测试/实验，本 README 所在） |
| `~/llwwds_application/knowledge-vault/` | 部署实例（应用本体 + state/索引 + 模型缓存） |

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

## Web 看板

`kv-serve` 启动后，浏览器打开 `http://127.0.0.1:8770/ui` 即得单页看板（HTML/CSS/JS 全部内嵌在 `src/knowledge_vault/webui.py`，无第三方前端依赖、无 CDN 外链，离线可用）。看板提供：

- **概览卡**：documents（total/active/deleted）、chunks（按 kind）、edges、向量库状态、db 体积与 state 路径、包版本与 schema 版本；每 30s 轮询 `GET /stats` 自动刷新。
- **最近登记表**：file_id 倒序 20 条（title/status/path/mtime，软删除记录带标记），点击行或检索结果弹出 `GET /files/{id}` 详情。
- **检索框**：调 `POST /search`，可开关 rerank、调 top-j，结果带 score 与 sources 徽标。
- **optimize 按钮**：确认后触发 `POST /optimize`。

### 皮肤系统

皮肤 = 一组 CSS 变量（`--bg` / `--bg-elev` / `--fg` / `--fg-muted` / `--accent` / `--border` / `--danger` / `--radius` / `--font-sans` / `--font-mono`）+ 显示名，注册在后端 `THEMES` 注册表（`webui.py`）。顶栏切换器经 `GET /ui/theme.css?name=<name>` 拉取变量集即时换肤，选择写入 localStorage 记忆；`GET /ui?theme=<name>` 可在 URL 上指定皮肤，未知名字一律回退默认皮肤 `xai-dark`（xAI 审美：纯黑背景、白/浅灰文字、#222 极细边框、小圆角、大字距标题、等宽数字）。

**自定义皮肤只需注册一个 CSS 变量集**，前端与路由自动生效：

```python
from knowledge_vault.webui import THEMES, REQUIRED_THEME_VARS, Theme, register_theme

register_theme("paper-light", Theme(
    display_name="Paper Light",
    vars={
        "--bg": "#ffffff", "--bg-elev": "#f5f5f5",
        "--fg": "#111111", "--fg-muted": "#6f6f6f",
        "--accent": "#111111", "--accent-fg": "#ffffff",
        "--border": "#e0e0e0", "--danger": "#d4494f",
        "--radius": "4px",
        "--font-sans": "system-ui, sans-serif",
        "--font-mono": "ui-monospace, Menlo, monospace",
    },
))
```

开发与测试在专属容器内进行（`bash dev/container/run.sh` 创建/重同步依赖），跑测试：

```bash
docker exec -w /repo knowledge-vault-dev /opt/venv/bin/python -m pytest
```
