-- knowledge-vault 登记层 schema v1
-- 阶段1交付：documents / chunks / edges + chunks_fts（FTS5, external content）。
-- WAL 由 store.connect() 在每个连接上设置（journal_mode 持久化在库文件里，重复设置无害）。
PRAGMA journal_mode = WAL;

-- ---------------------------------------------------------------- documents
-- 15 字段登记记录。file_id 由 file_id.py 单调分配、永不回收；同一物理路径
-- 最多一条活跃记录（软删除不占用路径），同内容不同路径 = 不同 file_id。
CREATE TABLE IF NOT EXISTS documents (
    file_id       INTEGER PRIMARY KEY,              -- 程序生成，单调递增，永不回收
    file_path     TEXT NOT NULL,                    -- 物理文件路径（登记时原样保存）
    file_type     TEXT NOT NULL,                    -- 扩展名推断（无扩展名 = unknown）
    size_bytes    INTEGER NOT NULL,
    is_large      INTEGER NOT NULL DEFAULT 0,       -- 0/1；超大文件仅登记元数据
    title         TEXT NOT NULL,                    -- 默认取文件名去扩展名
    context_tag   TEXT NOT NULL DEFAULT '[]',       -- JSON 数组字符串
    summary       TEXT,                             -- 可空，禁止硬写占位内容
    status        TEXT NOT NULL DEFAULT 'library' CHECK (status IN ('now', 'library')),
    content_hash  TEXT NOT NULL,                    -- sha256 前 12 位
    created_at    TEXT NOT NULL,                    -- ISO8601，默认取 mtime
    mtime         TEXT NOT NULL,                    -- ISO8601
    registered_at TEXT NOT NULL,                    -- ISO8601
    updated_at    TEXT NOT NULL,                    -- ISO8601
    deleted_at    TEXT                              -- 软删除时间；NULL = 活跃
);

-- 活跃路径唯一：软删除记录不占用路径（move/重登记可复用路径、拿新 file_id）
CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_active_path
    ON documents(file_path) WHERE deleted_at IS NULL;

-- ------------------------------------------------------------------- chunks
-- 切块登记。chunk_id 形如 f{file_id}-c{seq}；kind='summary' 约定用 chunk_seq=0。
-- 注意：chunks.text 存原文；FTS 索引的写入/删除由 textindex.py 用分词后的文本同步。
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    file_id    INTEGER NOT NULL REFERENCES documents(file_id),
    chunk_seq  INTEGER NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN ('chunk', 'summary')),
    text       TEXT NOT NULL,
    char_count INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id, chunk_seq);

-- chunks_fts：external content 指向 chunks（按 rowid 对齐）。
-- 写入约定：写入方先用 jieba 分词（空格拼接）再 INSERT 到本表；查询侧镜像同分词。
-- 禁止对本表执行 'rebuild'——那会用 chunks 原文（未分词）重建索引。
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='rowid',
    tokenize='unicode61'
);

-- -------------------------------------------------------------------- edges
-- 图边（双链/引用）。PK 前缀支持 src 出边扫描；dst 索引支持递归 CTE 反向扩展。
CREATE TABLE IF NOT EXISTS edges (
    src       INTEGER NOT NULL,
    dst       INTEGER NOT NULL,
    edge_type TEXT NOT NULL,
    PRIMARY KEY (src, dst, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);

-- ------------------------------------------------------------- 内部支撑表
-- file_id 单调分配器计数器（永不回收；见 file_id.py）
CREATE TABLE IF NOT EXISTS id_counters (
    name  TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);

-- schema 版本（store.py 建库/迁移用）
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
