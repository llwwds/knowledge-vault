"""registry：登记语义 / hash / 软删除 / move / 读取侧。"""

from __future__ import annotations

import hashlib
from datetime import datetime

import pytest

from knowledge_vault import (
    DuplicateFileError,
    RegistryValueError,
    UnknownFileIdError,
    get_file,
    iterate_files,
    move_file,
    register_file,
    soft_delete,
)


def sha12(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def assert_iso8601(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert value.endswith("Z")
    return parsed


class TestRegister:
    def test_minimal_fields_and_rules(self, conn, make_file):
        f = make_file("notes/hello.md", "# 你好\n正文内容")
        doc = register_file(conn, f)

        assert doc.file_id == 1
        assert doc.file_path == str(f)
        assert doc.file_type == "md"
        assert doc.title == "hello"                      # 文件名去扩展名
        assert doc.size_bytes == f.stat().st_size
        assert doc.is_large is False
        assert doc.context_tag == []
        assert doc.summary is None                       # 可空，禁止硬写
        assert doc.status == "library"                   # 默认枚举值
        assert doc.content_hash == sha12(f)              # sha256 前 12 位
        assert len(doc.content_hash) == 12
        assert doc.deleted_at is None

        assert_iso8601(doc.mtime)
        assert doc.created_at == doc.mtime               # created_at 用 mtime
        assert doc.updated_at == doc.registered_at
        assert_iso8601(doc.registered_at)

    def test_type_inference_casefold_and_unknown(self, conn, make_file):
        assert register_file(conn, make_file("a/b/c.PDF", "x")).file_type == "pdf"
        assert register_file(conn, make_file("d.canvas", "x")).file_type == "canvas"
        assert register_file(conn, make_file("noext", "x")).file_type == "unknown"

    def test_explicit_fields(self, conn, make_file):
        f = make_file("proj.md", "x")
        doc = register_file(
            conn,
            f,
            status="now",
            context_tag=["项目", "召回"],
            summary="人工写的摘要",
            title="自定义标题",
        )
        assert doc.status == "now"
        assert doc.context_tag == ["项目", "召回"]
        assert doc.summary == "人工写的摘要"
        assert doc.title == "自定义标题"

    def test_is_large_threshold_and_override(self, conn, make_file):
        small = make_file("s1.bin", "x")            # 1 字节
        assert register_file(conn, small).is_large is False
        assert register_file(conn, make_file("s2.bin", "x"), is_large=True).is_large is True
        assert register_file(conn, make_file("s3.bin", "x"), is_large_threshold=1).is_large is True
        assert register_file(
            conn, make_file("s4.bin", "x"), is_large_threshold=1, is_large=False
        ).is_large is False

    def test_duplicate_active_path_rejected(self, conn, make_file):
        f = make_file("dup.md", "x")
        register_file(conn, f)
        with pytest.raises(DuplicateFileError):
            register_file(conn, f)

    def test_same_content_different_paths_two_ids(self, conn, make_file):
        a = make_file("one.md", "完全相同的内容")
        b = make_file("two.md", "完全相同的内容")
        d1, d2 = register_file(conn, a), register_file(conn, b)
        assert d1.file_id != d2.file_id
        assert d1.content_hash == d2.content_hash   # 与物理文件 1:1，不是与内容 1:1

    def test_missing_file_rejected(self, conn):
        with pytest.raises(FileNotFoundError):
            register_file(conn, "/nonexistent/dir/x.md")

    def test_missing_file_with_metadata_overrides(self, conn):
        doc = register_file(
            conn,
            "/nonexistent/dir/x.md",
            content_hash="abcdef123456",
            size_bytes=3,
            mtime="2026-01-01T00:00:00Z",
        )
        assert doc.file_type == "md"
        assert doc.title == "x"
        assert doc.content_hash == "abcdef123456"
        assert doc.created_at == doc.mtime == "2026-01-01T00:00:00Z"

    def test_invalid_status_rejected(self, conn, make_file):
        with pytest.raises(RegistryValueError):
            register_file(conn, make_file("x.md", "x"), status="archive")

    def test_context_tag_stored_as_json(self, conn, make_file):
        doc = register_file(conn, make_file("t.md", "x"), context_tag=["a", "b"])
        raw = conn.execute(
            "SELECT context_tag FROM documents WHERE file_id = ?", (doc.file_id,)
        ).fetchone()[0]
        assert raw == '["a", "b"]' or raw == '["a","b"]'


class TestSoftDelete:
    def test_sets_only_deleted_at(self, conn, make_file):
        f = make_file("sd.md", "x")
        before = register_file(conn, f)
        after = soft_delete(conn, before.file_id)

        assert after.deleted_at is not None
        assert_iso8601(after.deleted_at)
        for field, value in before.as_dict().items():
            if field == "deleted_at":
                continue
            assert after.as_dict()[field] == value, field

    def test_idempotent(self, conn, make_file):
        doc = register_file(conn, make_file("sd2.md", "x"))
        first = soft_delete(conn, doc.file_id)
        second = soft_delete(conn, doc.file_id)
        assert second.deleted_at == first.deleted_at

    def test_explicit_timestamp(self, conn, make_file):
        doc = register_file(conn, make_file("sd3.md", "x"))
        deleted = soft_delete(conn, doc.file_id, deleted_at="2026-09-01T00:00:00Z")
        assert deleted.deleted_at == "2026-09-01T00:00:00Z"

    def test_unknown_id(self, conn):
        with pytest.raises(UnknownFileIdError):
            soft_delete(conn, 999)


class TestReregisterAfterDelete:
    def test_path_released_and_layer_does_not_filter(self, conn, make_file):
        f = make_file("cycle.md", "x")
        d1 = register_file(conn, f)
        soft_delete(conn, d1.file_id)
        d2 = register_file(conn, f)

        assert d2.file_id == d1.file_id + 1
        assert d2.deleted_at is None

        all_rows = list(iterate_files(conn))            # 登记层默认不过滤
        assert [d.file_id for d in all_rows] == [d1.file_id, d2.file_id]
        active = list(iterate_files(conn, include_deleted=False))
        assert [d.file_id for d in active] == [d2.file_id]

    def test_status_filter(self, conn, make_file):
        register_file(conn, make_file("now1.md", "x"), status="now")
        register_file(conn, make_file("lib1.md", "x"))
        assert [d.status for d in iterate_files(conn, status="now")] == ["now"]


class TestMove:
    def test_move_creates_new_record_and_soft_deletes_old(self, conn, make_file, tmp_path):
        src = make_file("old/plan.md", "计划内容")
        doc = register_file(conn, src, context_tag=["项目"], summary="s", status="now")

        new_path = tmp_path / "new" / "renamed.canvas"
        moved = move_file(conn, doc.file_id, new_path)

        assert moved.file_id == doc.file_id + 1
        assert moved.file_path == str(new_path)
        assert moved.title == "renamed"                 # 按新路径重推
        assert moved.file_type == "canvas"              # 按新路径重推
        assert moved.content_hash == doc.content_hash   # 内容 hash 不变
        assert moved.size_bytes == doc.size_bytes
        assert moved.created_at == doc.created_at       # 时间侧字段继承
        assert moved.mtime == doc.mtime
        assert moved.context_tag == doc.context_tag
        assert moved.summary == doc.summary
        assert moved.status == doc.status
        assert moved.deleted_at is None

        old = get_file(conn, doc.file_id)
        assert old.deleted_at is not None               # 旧记录软删除
        assert old.file_path == doc.file_path           # 旧记录保留原路径

    def test_move_title_override(self, conn, make_file, tmp_path):
        doc = register_file(conn, make_file("a.md", "x"))
        moved = move_file(conn, doc.file_id, tmp_path / "b.md", title="手动标题")
        assert moved.title == "手动标题"

    def test_move_to_occupied_path_rejected_until_freed(self, conn, make_file):
        a = make_file("a.md", "x")
        b = make_file("b.md", "y")
        da, db = register_file(conn, a), register_file(conn, b)

        with pytest.raises(DuplicateFileError):
            move_file(conn, db.file_id, a)              # a 仍活跃

        soft_delete(conn, da.file_id)                   # 释放路径
        moved = move_file(conn, db.file_id, a)          # 可移入
        assert moved.file_path == str(a)

    def test_move_unknown_id(self, conn, tmp_path):
        with pytest.raises(UnknownFileIdError):
            move_file(conn, 999, tmp_path / "c.md")

    def test_move_soft_deleted_record_rejected(self, conn, make_file, tmp_path):
        doc = register_file(conn, make_file("gone.md", "x"))
        soft_delete(conn, doc.file_id)
        with pytest.raises(RegistryValueError):
            move_file(conn, doc.file_id, tmp_path / "elsewhere.md")


class TestReadSide:
    def test_get_file_missing_returns_none(self, conn):
        assert get_file(conn, 12345) is None

    def test_iterate_ordered_by_file_id(self, conn, make_file):
        for name in ("c.md", "a.md", "b.md"):
            register_file(conn, make_file(name, "x"))
        ids = [d.file_id for d in iterate_files(conn)]
        assert ids == sorted(ids) == [1, 2, 3]

    def test_document_as_dict_roundtrip(self, conn, make_file):
        doc = register_file(conn, make_file("d.md", "x"), context_tag=["t"])
        payload = doc.as_dict()
        assert set(payload) == {
            "file_id", "file_path", "file_type", "size_bytes", "is_large", "title",
            "context_tag", "summary", "status", "content_hash", "created_at",
            "mtime", "registered_at", "updated_at", "deleted_at",
        }
