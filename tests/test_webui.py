"""web 看板与皮肤系统测试：THEMES 注册 / CSS 渲染 / 新只读路由 / 旧路由回归。

全部走合成数据与线程内服务器（stdlib urllib），不触真实 vault/state。
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from knowledge_vault import api, webui
from knowledge_vault.webui import (
    DEFAULT_THEME,
    REQUIRED_THEME_VARS,
    THEMES,
    Theme,
    register_theme,
    render_index,
    render_theme_css,
    resolve_theme,
)

from _phase3_helpers import build_kb


# ------------------------------------------------------------------ HTTP 工具


def _get_raw(url: str) -> tuple[int, str, bytes]:
    """GET 请求，返回 (status, Content-Type, body)（非 2xx 也原样返回）。"""
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read()


def _get_json(url: str) -> tuple[int, dict]:
    status, _, raw = _get_raw(url)
    return status, json.loads(raw.decode("utf-8") or "{}")


def _post_json(url: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


@pytest.fixture()
def kb(store, make_file) -> dict:
    return build_kb(store, make_file)


@pytest.fixture()
def api_factory(config):
    """起线程的 API 服务器工厂；用完自动 shutdown（同 test_api.py 约定）。"""

    created: list = []

    def _factory(reranker=None) -> str:
        server = api.make_server(config, host="127.0.0.1", port=0, reranker=reranker)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        created.append((server, thread))
        host, port = server.server_address[:2]
        return f"http://{host}:{port}"

    yield _factory
    for server, thread in created:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _make_theme(**overrides) -> Theme:
    """变量齐全的测试皮肤工厂（默认全部变量同色，可单点覆盖）。"""
    vars_ = {var: "#123456" for var in REQUIRED_THEME_VARS}
    vars_.update(overrides)
    return Theme(display_name="Test Theme", vars=vars_)


# ------------------------------------------------------------------ 皮肤注册表


class TestThemes:
    def test_default_theme_registered(self):
        assert DEFAULT_THEME == "xai-dark"
        assert DEFAULT_THEME in THEMES
        assert THEMES[DEFAULT_THEME].display_name

    @pytest.mark.parametrize("name", sorted(THEMES))
    def test_every_theme_has_required_vars(self, name):
        missing = [v for v in REQUIRED_THEME_VARS if v not in THEMES[name].vars]
        assert not missing, f"皮肤 {name} 缺少变量: {missing}"

    def test_xai_dark_palette(self):
        vars_ = THEMES["xai-dark"].vars
        assert vars_["--bg"] == "#000000"  # 纯黑背景
        assert vars_["--bg-elev"] == "#0a0a0a"  # 近黑浮起表面
        assert vars_["--fg"] == "#ffffff"
        assert vars_["--border"] == "#222222"  # 极细边框
        assert vars_["--radius"] == "2px"  # 小圆角

    def test_render_theme_css_contains_all_vars(self):
        css = render_theme_css(DEFAULT_THEME)
        assert ":root" in css
        for var in REQUIRED_THEME_VARS:
            assert f"{var}:" in css
        assert "color-scheme: dark;" in css

    def test_render_theme_css_deterministic_sorted(self):
        css = render_theme_css(DEFAULT_THEME)
        assert css == render_theme_css(DEFAULT_THEME)
        var_lines = [ln for ln in css.splitlines() if ln.strip().startswith("--")]
        assert var_lines == sorted(var_lines, key=lambda ln: ln.split(":")[0].strip())

    def test_render_theme_css_unknown_and_none_fall_back(self):
        default = render_theme_css(DEFAULT_THEME)
        assert render_theme_css("no-such-theme") == default
        assert render_theme_css(None) == default

    def test_resolve_theme(self):
        assert resolve_theme("xai-dark") == "xai-dark"
        assert resolve_theme("nope") == DEFAULT_THEME
        assert resolve_theme(None) == DEFAULT_THEME

    def test_register_theme_success_and_served(self):
        theme = _make_theme(**{"--accent": "#ffb000"})
        register_theme("test-amber", theme)
        try:
            assert resolve_theme("test-amber") == "test-amber"
            assert "--accent: #ffb000" in render_theme_css("test-amber")
        finally:
            THEMES.pop("test-amber", None)  # 清理，不污染全局注册表

    def test_register_theme_rejects_missing_vars(self):
        incomplete = Theme(display_name="Broken", vars={"--bg": "#000000"})
        with pytest.raises(ValueError, match="缺少必需 CSS 变量"):
            register_theme("broken", incomplete)
        assert "broken" not in THEMES  # 校验失败不落注册表

    def test_register_theme_rejects_duplicate(self):
        with pytest.raises(ValueError, match="已注册"):
            register_theme(DEFAULT_THEME, _make_theme())

    def test_register_theme_overwrite(self, monkeypatch):
        # 快照现值，测试结束由 monkeypatch 还原，不污染全局注册表
        monkeypatch.setitem(THEMES, DEFAULT_THEME, THEMES[DEFAULT_THEME])
        register_theme(DEFAULT_THEME, _make_theme(**{"--accent": "#00ff00"}), overwrite=True)
        assert THEMES[DEFAULT_THEME].vars["--accent"] == "#00ff00"


# ------------------------------------------------------------------ 页面渲染


class TestRenderIndex:
    def test_structure_and_endpoints(self):
        html = render_index(DEFAULT_THEME)
        assert html.startswith("<!doctype html>")
        assert "__THEME_CSS__" not in html and "__BOOT_JSON__" not in html
        # 看板调用的全部路由都出现在内嵌 JS 里
        for endpoint in ("/stats", "/search", "/files", "/ui/theme.css", "/optimize"):
            assert endpoint in html
        assert "setInterval(refreshStats, 30000)" in html  # 30s 自动刷新
        assert "localStorage" in html  # 皮肤选择记忆
        assert '"xai-dark"' in html  # BOOT 启动数据带当前皮肤

    def test_embeds_theme_css_and_boot(self):
        html = render_index(DEFAULT_THEME)
        assert "--bg: #000000" in html  # 当前皮肤变量内嵌于 <style id="theme-style">
        assert "schemaVersion" in html and '"displayName": "xAI Dark"' in html

    def test_unknown_theme_falls_back(self):
        assert render_index("no-such-theme") == render_index(DEFAULT_THEME)


# ------------------------------------------------------------------ 新只读路由


class TestDashboardRoutes:
    def test_ui_returns_html(self, api_factory):
        base = api_factory()
        status, ctype, raw = _get_raw(f"{base}/ui")
        assert status == 200
        assert ctype.startswith("text/html")
        html = raw.decode("utf-8")
        assert html.startswith("<!doctype html>")
        assert "--bg: #000000" in html

    def test_ui_trailing_slash_ok(self, api_factory):
        base = api_factory()
        status, _, _ = _get_raw(f"{base}/ui/")
        assert status == 200

    def test_ui_theme_param_override_and_unknown_fallback(self, api_factory):
        base = api_factory()
        _, _, default = _get_raw(f"{base}/ui")
        _, _, named = _get_raw(f"{base}/ui?theme=xai-dark")
        _, _, unknown = _get_raw(f"{base}/ui?theme=does-not-exist")
        assert named == default  # 显式指定 == 默认
        assert unknown == default  # 未知皮肤回退默认

    def test_theme_css_route(self, api_factory):
        base = api_factory()
        status, ctype, raw = _get_raw(f"{base}/ui/theme.css?name=xai-dark")
        assert status == 200
        assert ctype.startswith("text/css")
        css = raw.decode("utf-8")
        for var in REQUIRED_THEME_VARS:
            assert f"{var}:" in css
        _, _, missing_name = _get_raw(f"{base}/ui/theme.css")
        _, _, unknown_name = _get_raw(f"{base}/ui/theme.css?name=nope")
        assert missing_name == raw  # 缺省 name → 默认皮肤
        assert unknown_name == raw  # 未知 name → 默认皮肤

    def test_custom_theme_served_end_to_end(self, api_factory, monkeypatch):
        """注册即生效：新增皮肤无需改前端，/ui 与 /ui/theme.css 都能出。"""
        monkeypatch.setitem(
            THEMES, "test-amber", _make_theme(**{"--accent": "#ffb000", "--bg": "#101010"})
        )
        base = api_factory()
        status, _, raw = _get_raw(f"{base}/ui/theme.css?name=test-amber")
        assert status == 200
        assert "--accent: #ffb000" in raw.decode("utf-8")
        status, _, raw = _get_raw(f"{base}/ui?theme=test-amber")
        assert status == 200
        html = raw.decode("utf-8")
        assert "--bg: #101010" in html
        assert '"test-amber"' in html  # 顶栏切换器的 BOOT 皮肤清单带上新皮肤

    def test_stats_json(self, api_factory, kb):
        base = api_factory()
        status, body = _get_json(f"{base}/stats")
        assert status == 200
        for key in ("version", "state_dir", "db_size_bytes", "documents", "chunks", "edges", "vector"):
            assert key in body
        assert body["documents"]["total"] == 5
        assert body["documents"]["active"] == 4
        assert body["documents"]["deleted"] == 1
        assert body["chunks"]["total"] == 7
        assert body["chunks"]["by_kind"] == {"chunk": 6, "summary": 1}
        assert body["edges"] == 3
        assert body["vector"]["present"] is False  # tmp state 无 zvec 目录

    def test_files_list_order_and_deleted_flag(self, api_factory, kb):
        base = api_factory()
        status, body = _get_json(f"{base}/files")
        assert status == 200
        assert body["count"] == 5
        ids = [f["file_id"] for f in body["files"]]
        assert ids == sorted(ids, reverse=True)  # file_id 倒序 = 最近登记在前
        assert body["files"][0]["file_id"] == kb["large"]
        by_id = {f["file_id"]: f for f in body["files"]}
        assert by_id[kb["d"]]["deleted_at"]  # 含软删除记录并带 deleted_at
        assert by_id[kb["a"]]["deleted_at"] is None
        for f in body["files"]:
            for key in ("file_id", "title", "file_path", "file_type", "size_bytes", "status", "mtime", "deleted_at"):
                assert key in f

    def test_files_list_limit(self, api_factory, kb):
        base = api_factory()
        status, body = _get_json(f"{base}/files?limit=2")
        assert status == 200
        assert body["count"] == 2
        assert len(body["files"]) == 2
        assert body["files"][0]["file_id"] == kb["large"]

    def test_files_list_invalid_limit_400(self, api_factory, kb):
        base = api_factory()
        for bad in ("abc", "0", "-1"):
            status, body = _get_json(f"{base}/files?limit={bad}")
            assert status == 400, bad
            assert "error" in body

    def test_new_routes_method_guard_405(self, api_factory):
        base = api_factory()
        for path in ("/ui", "/ui/theme.css", "/stats", "/files"):
            status, body = _post_json(f"{base}{path}", {})
            assert status == 405, path
            assert "error" in body

    def test_unknown_route_still_404(self, api_factory):
        base = api_factory()
        status, body = _get_json(f"{base}/nope")
        assert status == 404
        assert "error" in body


# ------------------------------------------------------------------ 旧路由回归


class TestLegacyRoutesRegression:
    def test_health_ok(self, api_factory):
        base = api_factory()
        status, body = _get_json(f"{base}/health")
        assert status == 200
        assert body["ok"] is True
        assert body["service"] == "knowledge-vault"

    def test_get_file_regression(self, api_factory, kb):
        base = api_factory()
        status, body = _get_json(f"{base}/files/{kb['a']}")
        assert status == 200
        assert body["title"] == "机器学习笔记"
        status, _ = _get_json(f"{base}/files/99999")
        assert status == 404
        status, _ = _get_json(f"{base}/files/abc")
        assert status == 400

    def test_search_regression(self, api_factory, kb):
        base = api_factory()
        status, body = _post_json(f"{base}/search", {"query": "机器学习"})
        assert status == 200
        assert body["items"]
        assert any("fts" in it["sources"] for it in body["items"])

    def test_optimize_regression(self, api_factory, kb):
        base = api_factory()
        status, body = _post_json(f"{base}/optimize", {})
        assert status == 200
        assert body["fts_optimized"] is True
