"""knowledge-vault：个人知识库后端（headless）。

阶段1范围：登记层（SQLite documents/chunks/edges + FTS5 全文）+ 图查询。
向量库（zvec）、摄入管线、召回管线与 API 由后续阶段交付。

快速上手::

    from knowledge_vault import Store

    with Store() as kv:                    # 配置来自 KV_* 环境变量（见 config.py）
        doc = kv.register_file("notes/a.md")
        kv.add_chunks(doc.file_id, ["第一段文本", "第二段文本"])
        hits = kv.search("文本")
        kv.add_edge(doc.file_id, other_id, "link")
        nbrs = kv.expand(doc.file_id, max_hops=2)
"""

from .config import VaultConfig, load_config, packaged_userdict_path
from .file_id import current_file_id, next_file_id
from .graph import ExpandResult, add_edge, expand, remove_edges_by_src
from .registry import (
    Document,
    DuplicateFileError,
    RegistryValueError,
    UnknownFileIdError,
    compute_content_hash,
    derive_title,
    get_file,
    infer_file_type,
    iterate_files,
    move_file,
    register_file,
    soft_delete,
    utc_iso,
)
from .store import SCHEMA_VERSION, SchemaVersionError, Store, connect, init_db
from .textindex import (
    ChunkHit,
    Tokenizer,
    add_chunk,
    add_chunks,
    add_summary_chunk,
    chunk_id_for,
    default_preprocess,
    delete_chunks_for_file,
    make_span_preprocess,
    search,
    search_tokens,
    snippet,
)

__version__ = "2.4.1.0"

__all__ = [
    # config
    "VaultConfig",
    "load_config",
    "packaged_userdict_path",
    # store
    "Store",
    "connect",
    "init_db",
    "SCHEMA_VERSION",
    "SchemaVersionError",
    # file_id
    "next_file_id",
    "current_file_id",
    # registry
    "Document",
    "DuplicateFileError",
    "UnknownFileIdError",
    "RegistryValueError",
    "register_file",
    "soft_delete",
    "move_file",
    "get_file",
    "iterate_files",
    "compute_content_hash",
    "infer_file_type",
    "derive_title",
    "utc_iso",
    # textindex
    "Tokenizer",
    "ChunkHit",
    "add_chunk",
    "add_chunks",
    "add_summary_chunk",
    "delete_chunks_for_file",
    "search",
    "search_tokens",
    "snippet",
    "chunk_id_for",
    "default_preprocess",
    "make_span_preprocess",
    # graph
    "ExpandResult",
    "add_edge",
    "remove_edges_by_src",
    "expand",
    "__version__",
]
