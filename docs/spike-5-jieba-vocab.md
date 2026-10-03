# Spike 5：jieba 切分质量抽查与自定义词典初版（真实 vault 抽样）

日期：2026-10-03
状态：已完成（真实样本与含真实内容的明细仅存设备本地 testdata，仓库内一律脱敏示例）

## 背景

全文路定案为 FTS5 + jieba 预分词。中文技术笔记里大量项目代号、工具名、中英混合词可能被 jieba 切碎，导致词法召回质量下降。本实验从真实 vault 抽 20 个 md 做切分质量审计：自动检测「技术词被切碎」的形态并人工复核，产出自定义词典初版（userdict_v0），量化 before/after 改善，并给出词典维护约定与剩余问题的处理建议。

## 环境

| 项 | 值 |
| --- | --- |
| 运行环境 | Docker 容器 `knowledge-vault-dev`（linux/arm64） |
| Python | 3.12.15（容器内 `/opt/venv/bin/python`） |
| jieba | 0.42.1（精确模式，HMM 默认开） |
| 挂载 | `/repo` 代码仓库（读写）；`/vault` Obsidian vault（**只读**）；`/app_runtime` → 设备本地运行时 |
| 脚本 | `experiments/jieba_vocab_audit.py`（seed=20261003，幂等） |
| 产物 | `/app_runtime/testdata/jieba_audit/`：`audit_detail.json`（含真实上下文，永不进 git）+ `samples/`（20 个源文件原样副本） |

## 抽样方法

- 递归收集 `/vault` 全部 `.md`，排除目录：`99 废纸篓`、`.obsidian`、`.trash`、`.git`、`obsidian仓库快照备份`、`.laneweave`、`.cairn`；vault 根部零散文件（2 个）与文件数 <5 的一级目录（如 assets）不参与分层。
- **隐私护栏**：文件名疑似凭据（key/token/密钥/密码等）的文件直接排除，共跳过 29 个，只计数不落路径。
- 分层配额：按一级目录，以 sqrt(文件数) 为理想配比、每层 [1, 5] 截断、最大余数法补齐到 20；层内优先取「大小中位 ±4 倍」带，固定 seed 随机抽取。
- 抽中文件**原样复制**到 testdata `samples/`（保留相对路径，sha256 校验一致），`/vault` 零写入。

| 一级目录 | 库内 md 数 | 抽样数 |
| --- | --- | --- |
| 01 资料 | 4927 | 5 |
| 02 项目 | 2035 | 5 |
| 03 笔记 | 1181 | 5 |
| 00 收集箱 | 462 | 3 |
| 04 信息 | 23 | 1 |
| 03 Subagent 任务与回执 | 7 | 1 |
| 合计 | 8635 | 20 |

20 篇剥离 frontmatter 与代码块后正文：291–8052 字符，中位 2553，合计 54547 字符；原文件 0.7–14.3 KB。

## 检测方法

对每篇正文（剥离 YAML frontmatter、围栏代码块，折叠链接目标后）用 `jieba.tokenize` 精确模式分词；把「偏移连续、字符全属 ASCII 技术集 `[A-Za-z0-9_+#./&%-]`」的相邻 token run 拼成 span，多 token 即疑似切碎，按形态分类：

| 类型 | 含义 | userdict 可修？ |
| --- | --- | --- |
| `SEP_FIXABLE` | `-` `.` `+` 等分隔被切（均在 jieba re_han 内） | 可 |
| `EXT_ONLY` | 点前缀扩展名/属性（`.md`、`.innerHTML`） | 多数可 |
| `SEP_REGEX_BOUNDARY` | `/` `_` 边界（不在 jieba re_han 内，DAG 无法跨块合并） | **不可** |
| `ALNUM_MIX` / `CASE_SPLIT` / `ALPHA_FRAG` | 字母数字混排 / 驼峰 / 连写英文词被拆 | 可 |

合理形态单列不计切错：数字序列（日期/版本/编号 + 有序列表标记）83 处、英文词 + 句末句点粘连 66 处、纯分隔符（注释符）16 处——共 165 处噪音已剔除。检测器边界：只覆盖 ASCII 技术 span 形态，纯中文专名切碎不在此射程（后续可用词典对照法补测）。

## 切错清单（脱敏示例，原句见 testdata/audit_detail.json）

基线（无 userdict）：核心切错 **608 处 span**，命中 **18/20** 文件，去重后 433 种模式。分类计数：

| 类型 | 数量 | 占比 | 典型形态（脱敏） |
| --- | --- | --- | --- |
| SEP_REGEX_BOUNDARY | 327 | 53.8% | `connect_mcp → connect / _ / mcp`；`XXX_YYY → XXX / _ / YYY`；个人绝对路径 `<个人绝对路径> → / / <目录> / …`；`mcp__server__tool → mcp / __ / server / __ / tool` |
| SEP_FIXABLE | 231 | 38.0% | `knowledge-vault → knowledge / - / vault`；`Few-Shot → Few / - / Shot`（3 种大小写变体并存）；`Node.js → Node / . / js`；`<模块>.json → <模块> / . / json`；内部编号 `Sxx-xx → Sxx / - / xx` |
| EXT_ONLY | 50 | 8.2% | `.md → . / md`（**全库性最高频单模式，32 次**）；`.canvas → . / canvas` |

人工复核结论：`Few-Shot`、`knowledge-vault`、`Node.js`、扩展名类为真切错（丢短语完整性、BM25 权重稀释）；snake_case 标识符与路径类真切但两侧切分一致、整词查询仍可召回，主要损失是噪音 token 与权重分散；列表标记/句点/注释符为检测噪音（已剔除）。

**关键机制发现**：jieba 默认 re_han 不含 `/` 与 `_`，这两类边界是正则级切分，**任何 userdict 都修不了**（占切错 53.8%）；而 `-` `.` 在 re_han 内，词典可合并。另经默认词典探针验证：核心模型名 `bge-m3 → bge / - / m3`、`bge-reranker-v2-m3 → bge / - / reranker / - / v2 / - / m3` 必被切碎（本 20 样本未覆盖到这两个词，见「维护约定」v1 全库扫描）。

## userdict_v0.txt

共 **24 条**，格式 `词 频次 词性`（jieba `load_userdict` 格式，频次统一 100000，词性 n）：

| 组 | 条目 | 说明 |
| --- | --- | --- |
| 核心栈/模型名（11） | `zvec` `bge-m3` `bge-reranker-v2-m3` `FlagEmbedding` `FTS5` `jieba` `HNSW` `rerank` `reranker` `Obsidian` `frontmatter` | 项目召回词表的核心，部分默认整存、部分必切碎，全部固定 |
| 复核确认切碎（8） | `knowledge-vault`、`Few-Shot`/`few-shot`/`Few-shot`、`Node.js`/`node.js`、`AGENTS.md`、`SKILL.md` | 20 样本内确认真切错的通用词（含大小写变体，jieba 大小写敏感） |
| 通用扩展名（5） | `.md` `.canvas` `.json` `.jsonl` `.html` | 消 `.md` 型最高频噪音 |

隐私边界：不含用户名、不含私人路径、不含真实笔记标题；个体项目代号/内部编号（如 `Sxx-xx`、`<工具>-mcp-node` 型）不进 v0，走增量机制。

**反例（重要）**：实测「默认词典已整存的通用词」**不能收**——加入 `API`/`MCP`/`DeepSeek` 后，jieba DAG 被这些词覆盖，HMM 不再兜底，把原本整存的驼峰复合词拆开（`APIRouter → API / Router`、`MCPClient → MCP / Client` 等，实测 8 处回归）。v0 已剔除该组。

## Before / After（同 20 文件、同 seed 复测）

| 指标 | before | after | 变化 |
| --- | --- | --- | --- |
| 核心切错 span 总数 | 608 | 557 | **-8.4%** |
| └ 词典可修类（SEP_FIXABLE + EXT_ONLY） | 281 | 230 | **-18.1%** |
| └ 通用技术词本体切碎 | 52 | **0** | **-100%**（另有 3 处路径包裹形态，词已整存，仅剩 `/` 边界） |
| └ SEP_REGEX_BOUNDARY（词典不可修） | 327 | 327 | 0（原理性上限，见下） |
| 命中文件 | 18/20 | — | 剩余命中均为个体标识符/路径 |

`Few-Shot`、`knowledge-vault`、`Node.js`、`.md` 等通用词全部整存；剩余切错主体为个人项目标识符、文件名、内部编号与绝对路径——属「两侧切分一致、召回不硬失败」类，由增量词典与管线预处理消化。

## 用法与加载时机建议

```python
import jieba
jieba.load_userdict("path/to/userdict.txt")   # 进程启动、首次分词前调用一次，幂等
```

- 正式管线中词典进包内 assets（或配置项指定），**索引侧与查询侧必须加载同一份文件**，否则两侧 token 形态不对称、召回失效。
- 加载在 `jieba.initialize()`/首次分词前完成即可；`load_userdict` 内部会触发初始化，进程内加载一次，勿在请求路径反复加载。

## 词典维护约定

1. **准入**：a) 默认词典会切碎或形态不稳的核心栈词/模型名；b) 通用约定文件名与扩展名；c) 全库统计高频（建议 ≥10 个文件出现）且默认切碎的专名——v1 用脚本全库扫描生成候选（本次 20 样本对 bge-m3 等核心词覆盖不足，即为此故）。
2. **禁入**：默认已整存的通用词（驼峰吸入回归，见反例）；用户名、私人路径、真实笔记标题、含密钥文件相关词。
3. **回归**：每次变更词典后跑复测命令，核对「通用技术词切碎 = 0」且 `ALPHA_FRAG`/`CASE_SPLIT` 无新增。
4. 剩余的 `/` `_` 边界类（53.8%）词典原理上修不了：建议摄入预处理（下划线归一化或技术 span 保护为原子 token）并在查询侧镜像同规则；若未来证实召回受损，再评估 wangfenjin/simple 等 FTS5 自定义分词器（未验证，不据此直接切换）。

## 结论与建议

默认 jieba 对本 vault 的专名切碎是系统性的（18/20 文件命中，608 处）；userdict 24 条把通用技术词切碎清零（52 → 0，含默认必切碎的 `bge-m3`/`bge-reranker-v2-m3`），词典可修类改善 -18.1%，方案成立；剩余主体是 `/` `_` 正则级边界类与个体标识符，交由增量词典 + 摄入预处理决策，不构成更换分词器的证据。

结论：通过

## 复跑命令

```bash
# 基线（无 userdict）
docker exec -w /repo knowledge-vault-dev /opt/venv/bin/python experiments/jieba_vocab_audit.py

# 复测（加载 userdict，输出 before/after 对比）
docker exec -w /repo knowledge-vault-dev /opt/venv/bin/python experiments/jieba_vocab_audit.py \
    --userdict experiments/userdict_v0.txt
```

脚本幂等：每次运行重建 testdata `samples/` 并重写 `audit_detail.json`；固定 seed=20261003，两跑同样本，数字可复现。`--total/--cap/--seed/--vault/--out` 可覆盖默认值。
