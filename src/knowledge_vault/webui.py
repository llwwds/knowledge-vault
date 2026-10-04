"""单页 web 看板与换皮肤系统（stdlib only，HTML/CSS/JS 全内嵌，离线可用）。

组成：

- :data:`THEMES` —— 皮肤注册表（``name → Theme``）。**新增皮肤 = 注册一个
  CSS 变量集**：构造 :class:`Theme` 后调用 :func:`register_theme`（或直接
  ``THEMES["name"] = Theme(...)``），前端顶栏切换器与 ``GET /ui/theme.css``
  自动生效，无需改任何前端代码。
- :func:`render_theme_css` —— 皮肤名 → ``:root { --var: value; ... }`` CSS 文本；
  未知皮肤回退 :data:`DEFAULT_THEME`。
- :func:`render_index` —— 单页看板 HTML（内嵌基础样式、当前皮肤 CSS 与启动
  数据）。基础样式只引用 CSS 变量，换肤 = 换变量集；前端切换器经
  ``GET /ui/theme.css?name=...`` 拉取新变量集替换 ``<style id="theme-style">``，
  并写入 localStorage 记忆选择（下次打开自动恢复）。

看板数据全部来自只读 HTTP 路由（``/stats`` / ``/search`` / ``/files`` /
``/files/{id}``），写入只暴露 optimize 触发（confirm 后 ``POST /optimize``）。
概览卡每 30s 轮询 ``/stats`` 自动刷新。无任何 CDN / 第三方前端依赖。

内置皮肤 ``xai-dark``：xAI 官网审美——纯黑背景、白/浅灰文字、1px 极细边框、
小圆角、大字距标题、等宽数字、无装饰。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import __version__
from .store import SCHEMA_VERSION

__all__ = [
    "Theme",
    "THEMES",
    "DEFAULT_THEME",
    "REQUIRED_THEME_VARS",
    "register_theme",
    "resolve_theme",
    "render_theme_css",
    "render_index",
]


# ------------------------------------------------------------------ 皮肤注册表


@dataclass(frozen=True)
class Theme:
    """一套皮肤 = CSS 变量集 + 显示名。

    ``vars`` 的键是含 ``--`` 前缀的 CSS 变量名；:data:`REQUIRED_THEME_VARS`
    中的变量必须全部提供（基础样式按这些名字取值），其余变量可选（如
    ``--accent-fg``：accent 底色上的前景色）。
    """

    display_name: str
    vars: dict[str, str]
    #: 喂给 CSS ``color-scheme`` 的值（影响滚动条/表单控件的原生配色）
    color_scheme: str = "dark"


#: 基础样式依赖的全部 CSS 变量；注册皮肤时缺一不可
REQUIRED_THEME_VARS = (
    "--bg",  # 页面底色
    "--bg-elev",  # 浮起表面（输入框/卡片悬浮/弹层）
    "--fg",  # 主文字
    "--fg-muted",  # 次级文字/标签
    "--accent",  # 强调（主按钮底色、焦点）
    "--border",  # 1px 边框
    "--danger",  # 危险/软删除标记
    "--radius",  # 圆角（xAI 风格取极小值）
    "--font-sans",  # 正文无衬线字体栈
    "--font-mono",  # 数字/路径/代码等宽字体栈
)

#: xai-dark：纯黑/近黑背景、白/浅灰文字、#222 极细边框、极简无装饰
_XAI_DARK_VARS: dict[str, str] = {
    "--bg": "#000000",
    "--bg-elev": "#0a0a0a",
    "--fg": "#ffffff",
    "--fg-muted": "#8a8a8a",
    "--accent": "#ffffff",
    "--accent-fg": "#000000",
    "--border": "#222222",
    "--danger": "#e5484d",
    "--radius": "2px",
    "--font-sans": (
        "-apple-system, 'Helvetica Neue', Helvetica, Arial,"
        " 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif"
    ),
    "--font-mono": (
        "'SF Mono', 'JetBrains Mono', Menlo, Consolas,"
        " 'Liberation Mono', monospace"
    ),
}

#: 皮肤注册表：name → Theme。新增皮肤 = ``register_theme(name, Theme(...))``
THEMES: dict[str, Theme] = {
    "xai-dark": Theme(display_name="xAI Dark", vars=dict(_XAI_DARK_VARS)),
}

#: 默认皮肤（未知名字一律回退到它）
DEFAULT_THEME = "xai-dark"


def register_theme(name: str, theme: Theme, *, overwrite: bool = False) -> None:
    """注册皮肤并校验变量齐全；重名默认报错（``overwrite=True`` 覆盖）。

    也可直接 ``THEMES[name] = theme``，但走本函数能在注册时发现缺变量，
    而不是等页面渲染出残缺样式。
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"皮肤名必须是非空字符串，收到 {name!r}")
    if not isinstance(theme, Theme):
        raise ValueError(f"theme 必须是 Theme 实例，收到 {type(theme)!r}")
    missing = [v for v in REQUIRED_THEME_VARS if v not in theme.vars]
    if missing:
        raise ValueError(f"皮肤 {name!r} 缺少必需 CSS 变量: {missing}")
    if name in THEMES and not overwrite:
        raise ValueError(f"皮肤 {name!r} 已注册；覆盖请传 overwrite=True")
    THEMES[name] = theme


def resolve_theme(name: str | None) -> str:
    """请求方给的皮肤名 → 注册表里的实际名字；未知/缺省回退默认皮肤。"""
    if isinstance(name, str) and name in THEMES:
        return name
    return DEFAULT_THEME


def render_theme_css(name: str | None) -> str:
    """渲染皮肤 CSS（``:root`` 变量块）；未知皮肤回退 :data:`DEFAULT_THEME`。

    输出是「已解析皮肤名」的确定性函数（未知名字回退后与默认皮肤字节级
    一致，利于测试与缓存）；变量按名字排序输出。
    """
    resolved = resolve_theme(name)
    theme = THEMES[resolved]
    lines = [f"/* knowledge-vault webui theme: {resolved} */", ":root {"]
    if theme.color_scheme:
        lines.append(f"  color-scheme: {theme.color_scheme};")
    for var in sorted(theme.vars):
        lines.append(f"  {var}: {theme.vars[var]};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _boot_payload(current_theme: str) -> dict:
    """注入页面的启动数据（版本、schema 版本、皮肤清单）。"""
    return {
        "version": __version__,
        "schemaVersion": SCHEMA_VERSION,
        "currentTheme": current_theme,
        "defaultTheme": DEFAULT_THEME,
        "themes": [
            {"name": name, "displayName": THEMES[name].display_name}
            for name in sorted(THEMES)
        ],
    }


# ------------------------------------------------------------------ 页面模板

#: 单页看板模板。占位符用 ``__XXX__`` 标记 + ``str.replace`` 注入（避免
#: ``str.format`` 与 CSS/JS 的花括号冲突）。
_INDEX_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>knowledge-vault</title>
<style id="theme-style">
__THEME_CSS__</style>
<style>
/* 基础样式：只引用皮肤 CSS 变量，换肤即换变量集，零第三方依赖 */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--fg);
  font-family: var(--font-sans); font-size: 14px; line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}
.mono { font-family: var(--font-mono); }
.hidden { display: none !important; }

/* ---- 顶栏 ---- */
header {
  display: flex; align-items: center; gap: 12px;
  padding: 14px 24px; border-bottom: 1px solid var(--border);
  position: sticky; top: 0; z-index: 20; background: var(--bg);
}
.brand { font-size: 12px; font-weight: 700; letter-spacing: 0.4em; text-transform: uppercase; }
.brand em { font-style: normal; color: var(--fg-muted); }
.badge {
  font-size: 11px; color: var(--fg-muted); letter-spacing: 0.08em;
  border: 1px solid var(--border); border-radius: var(--radius); padding: 1px 8px;
}
.spacer { flex: 1; }
.theme-label { font-size: 10px; letter-spacing: 0.25em; text-transform: uppercase; color: var(--fg-muted); }

input, select, button {
  font: inherit; color: var(--fg);
  background: var(--bg-elev);
  border: 1px solid var(--border); border-radius: var(--radius);
  padding: 7px 12px;
}
input:focus, select:focus { border-color: var(--fg); outline: none; }
button { cursor: pointer; letter-spacing: 0.15em; text-transform: uppercase; font-size: 11px; }
button:hover:not(:disabled) { border-color: var(--fg-muted); }
button:disabled { opacity: 0.4; cursor: default; }
button.primary {
  background: var(--accent); color: var(--accent-fg, var(--bg));
  border-color: var(--accent); font-weight: 600;
}
button.small { padding: 3px 8px; font-size: 10px; }
input[type="checkbox"] { width: 13px; height: 13px; accent-color: var(--accent); padding: 0; }
input[type="number"] { width: 64px; }

/* ---- 主区 ---- */
main { max-width: 1060px; margin: 0 auto; padding: 8px 24px 96px; }
section { margin-top: 36px; }
.section-head { display: flex; align-items: baseline; gap: 12px; margin-bottom: 14px; }
.section-title {
  font-size: 11px; font-weight: 600; letter-spacing: 0.35em;
  text-transform: uppercase; color: var(--fg-muted);
}

/* ---- 概览卡片 ---- */
.cards {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1px; background: var(--border);
  border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden;
}
.card { background: var(--bg); padding: 16px 18px; }
.card .label { font-size: 10px; letter-spacing: 0.3em; text-transform: uppercase; color: var(--fg-muted); }
.card .value { font-family: var(--font-mono); font-size: 28px; line-height: 1.2; margin-top: 6px; }
.card .value.off { color: var(--fg-muted); }
.card .sub { font-family: var(--font-mono); font-size: 11px; color: var(--fg-muted); margin-top: 4px; word-break: break-all; }
.card .rows { margin-top: 8px; font-family: var(--font-mono); font-size: 11px; color: var(--fg-muted); }
.card .rows > div { display: flex; justify-content: space-between; gap: 12px; }
.card .rows b { font-weight: 400; color: var(--fg); }

/* ---- 检索 ---- */
.searchbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.searchbar input[type="search"] { flex: 1; min-width: 240px; }
.inline {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--fg-muted);
}
.meta-line { font-size: 11px; color: var(--fg-muted); margin: 10px 0 2px; min-height: 1.2em; letter-spacing: 0.05em; }
.result {
  border: 1px solid var(--border); border-radius: var(--radius);
  padding: 12px 16px; margin-top: 10px; cursor: pointer;
}
.result:hover { border-color: var(--fg-muted); background: var(--bg-elev); }
.result .r-head { display: flex; align-items: baseline; gap: 12px; }
.result .r-title { font-weight: 600; }
.result .r-score { font-family: var(--font-mono); font-size: 12px; color: var(--fg-muted); margin-left: auto; }
.badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.badges .badge { font-size: 10px; }
.result .r-text { margin-top: 8px; font-size: 13px; color: var(--fg-muted); white-space: pre-wrap; word-break: break-word; }

/* ---- 文件表 ---- */
.files { width: 100%; border-collapse: collapse; }
.files th {
  text-align: left; font-size: 10px; font-weight: 600; letter-spacing: 0.25em;
  text-transform: uppercase; color: var(--fg-muted);
  padding: 8px 10px; border-bottom: 1px solid var(--border);
}
.files td { padding: 8px 10px; border-bottom: 1px solid var(--border); font-size: 13px; }
.files tbody tr { cursor: pointer; }
.files tbody tr:hover { background: var(--bg-elev); }
.files .mono { font-size: 12px; }
.files .cell-path { color: var(--fg-muted); font-family: var(--font-mono); font-size: 12px; word-break: break-all; }
.files tr.deleted .cell-title { color: var(--fg-muted); }
.st { font-size: 10px; letter-spacing: 0.15em; text-transform: uppercase; }
.st-deleted { color: var(--danger); }
.empty { color: var(--fg-muted); padding: 18px 10px; font-size: 12px; letter-spacing: 0.1em; }

/* ---- 弹层 ---- */
.modal {
  position: fixed; inset: 0; z-index: 50;
  background: rgba(0, 0, 0, 0.72);
  display: flex; align-items: flex-start; justify-content: center;
  padding: 8vh 16px 16px;
}
.modal-box {
  width: 640px; max-width: 100%; max-height: 80vh; overflow: auto;
  background: var(--bg-elev);
  border: 1px solid var(--border); border-radius: var(--radius);
}
.modal-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 16px; border-bottom: 1px solid var(--border);
  position: sticky; top: 0; background: var(--bg-elev);
}
.modal-head .mono { font-size: 12px; letter-spacing: 0.1em; }
.kv { padding: 6px 16px; }
.kv > div { display: flex; gap: 16px; padding: 6px 0; border-bottom: 1px solid var(--border); }
.kv > div:last-child { border-bottom: none; }
.kv dt {
  width: 110px; flex-shrink: 0; padding-top: 2px;
  font-size: 10px; letter-spacing: 0.2em; text-transform: uppercase; color: var(--fg-muted);
}
.kv dd { font-family: var(--font-mono); font-size: 12px; word-break: break-all; white-space: pre-wrap; }
.kv dd.danger { color: var(--danger); }

/* ---- toast / noscript ---- */
.toast {
  position: fixed; right: 20px; bottom: 20px; z-index: 60; max-width: 480px;
  background: var(--bg-elev);
  border: 1px solid var(--border); border-radius: var(--radius);
  color: var(--fg); font-size: 11px; padding: 10px 14px;
  white-space: pre-wrap; word-break: break-all;
}
.noscript { padding: 20px; color: var(--danger); font-family: var(--font-mono); }
</style>
</head>
<body>
<header>
  <div class="brand">knowledge<em>&middot;</em>vault</div>
  <span class="badge mono" id="badge-version"></span>
  <span class="badge mono" id="badge-schema"></span>
  <div class="spacer"></div>
  <label class="theme-label" for="theme-select">theme</label>
  <select id="theme-select" aria-label="选择皮肤"></select>
  <button id="btn-optimize" class="ghost">optimize</button>
</header>
<main>
  <section>
    <h2 class="section-title">概览</h2>
    <div class="cards">
      <div class="card">
        <div class="label">documents</div>
        <div class="value" id="stat-doc-total">&mdash;</div>
        <div class="sub"><span id="stat-doc-active">&mdash;</span> active &middot; <span id="stat-doc-deleted">&mdash;</span> deleted</div>
      </div>
      <div class="card">
        <div class="label">chunks</div>
        <div class="value" id="stat-chunks-total">&mdash;</div>
        <div class="rows" id="stat-chunk-kinds"></div>
      </div>
      <div class="card">
        <div class="label">edges</div>
        <div class="value" id="stat-edges">&mdash;</div>
        <div class="sub">graph links</div>
      </div>
      <div class="card">
        <div class="label">vector</div>
        <div class="value" id="stat-vector">&mdash;</div>
        <div class="sub" id="stat-vector-path"></div>
      </div>
      <div class="card">
        <div class="label">db size</div>
        <div class="value" id="stat-db-size">&mdash;</div>
        <div class="sub" id="stat-state-dir"></div>
      </div>
    </div>
  </section>

  <section>
    <h2 class="section-title">检索</h2>
    <div class="searchbar">
      <input id="q" type="search" placeholder="查询知识库，回车检索&hellip;" autocomplete="off" aria-label="查询">
      <label class="inline">top-j <input id="topj" class="mono" type="number" min="1" max="100" value="10"></label>
      <label class="inline"><input id="rerank" type="checkbox" checked> rerank</label>
      <button id="btn-search" class="primary">search</button>
    </div>
    <div id="search-meta" class="meta-line mono"></div>
    <div id="results"></div>
  </section>

  <section>
    <div class="section-head">
      <h2 class="section-title">最近登记</h2>
      <button id="btn-files-refresh" class="small">refresh</button>
    </div>
    <table class="files">
      <thead><tr><th>id</th><th>title</th><th>status</th><th>path</th><th>mtime</th></tr></thead>
      <tbody id="files-body"></tbody>
    </table>
  </section>
</main>

<div id="modal" class="modal hidden" role="dialog" aria-modal="true" aria-label="文件详情">
  <div class="modal-box">
    <div class="modal-head">
      <span class="mono" id="modal-title"></span>
      <button id="modal-close" aria-label="关闭">&times;</button>
    </div>
    <div id="modal-body"></div>
  </div>
</div>

<div id="toast" class="toast mono hidden"></div>
<noscript><div class="noscript">此看板需要启用 JavaScript。</div></noscript>

<script>
"use strict";
const BOOT = __BOOT_JSON__;
const $ = (id) => document.getElementById(id);

/* ---------- 皮肤：拉取 /ui/theme.css 替换 :root 变量块，localStorage 记忆 ---------- */
async function applyTheme(name, persist) {
  const resp = await fetch("/ui/theme.css?name=" + encodeURIComponent(name));
  if (!resp.ok) throw new Error("HTTP " + resp.status);
  document.getElementById("theme-style").textContent = await resp.text();
  if (persist) { try { localStorage.setItem("kv-theme", name); } catch (e) {} }
  BOOT.currentTheme = name;
  const sel = $("theme-select");
  if (sel && sel.value !== name) sel.value = name;
}

function initTheme() {
  const sel = $("theme-select");
  BOOT.themes.forEach((t) => {
    const opt = document.createElement("option");
    opt.value = t.name;
    opt.textContent = t.displayName;
    sel.appendChild(opt);
  });
  let saved = null;
  try { saved = localStorage.getItem("kv-theme"); } catch (e) {}
  if (saved && !BOOT.themes.some((t) => t.name === saved)) {
    try { localStorage.removeItem("kv-theme"); } catch (e) {}
    saved = null;
  }
  const active = saved || BOOT.currentTheme;
  if (active !== BOOT.currentTheme) applyTheme(active, false).catch(() => {});
  sel.value = active;
  sel.addEventListener("change", () => {
    applyTheme(sel.value, true).catch((err) => toast("换肤失败: " + err));
  });
}

/* ---------- 工具 ---------- */
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}
function fmtBytes(n) {
  if (typeof n !== "number" || !isFinite(n) || n < 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n, i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return (i === 0 ? String(v) : v.toFixed(1)) + " " + units[i];
}
function fmtWhen(s) { return s ? String(s).replace("T", " ") : "—"; }

let toastTimer = null;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 6000);
}

/* ---------- 概览：30s 轮询 /stats ---------- */
async function refreshStats() {
  let s;
  try {
    const resp = await fetch("/stats");
    if (!resp.ok) return;
    s = await resp.json();
  } catch (e) { return; } /* 静默：下一次轮询再试 */
  $("stat-doc-total").textContent = String(s.documents.total);
  $("stat-doc-active").textContent = String(s.documents.active);
  $("stat-doc-deleted").textContent = String(s.documents.deleted);
  $("stat-chunks-total").textContent = String(s.chunks.total);
  const kinds = $("stat-chunk-kinds");
  kinds.textContent = "";
  const names = Object.keys(s.chunks.by_kind || {}).sort();
  if (!names.length) {
    kinds.appendChild(el("div", null, "—"));
  } else {
    names.forEach((k) => {
      const row = el("div");
      row.appendChild(el("span", null, k));
      row.appendChild(el("b", null, String(s.chunks.by_kind[k])));
      kinds.appendChild(row);
    });
  }
  $("stat-edges").textContent = String(s.edges);
  const present = !!(s.vector && s.vector.present);
  const vecEl = $("stat-vector");
  vecEl.textContent = present ? "ACTIVE" : "ABSENT";
  vecEl.className = "value" + (present ? "" : " off");
  $("stat-vector-path").textContent = s.vector ? s.vector.path : "";
  $("stat-db-size").textContent = fmtBytes(s.db_size_bytes);
  $("stat-state-dir").textContent = s.state_dir || "";
  $("badge-version").textContent = "v" + s.version;
  $("badge-schema").textContent = "schema v" + BOOT.schemaVersion;
}

/* ---------- 最近登记（file_id 倒序） ---------- */
const FILES_LIMIT = 20;
async function refreshFiles() {
  let data;
  try {
    const resp = await fetch("/files?limit=" + FILES_LIMIT);
    if (!resp.ok) return;
    data = await resp.json();
  } catch (e) { return; }
  const tbody = $("files-body");
  tbody.textContent = "";
  if (!data.files || !data.files.length) {
    const tr = el("tr");
    const td = el("td", "empty", "暂无登记记录");
    td.colSpan = 5;
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }
  data.files.forEach((f) => {
    const tr = el("tr", f.deleted_at ? "deleted" : "");
    tr.appendChild(el("td", "mono", "#" + f.file_id));
    tr.appendChild(el("td", "cell-title", f.title));
    const stTd = el("td");
    stTd.appendChild(el("span", "st" + (f.deleted_at ? " st-deleted" : ""),
      f.deleted_at ? "deleted" : f.status));
    tr.appendChild(stTd);
    tr.appendChild(el("td", "cell-path", f.file_path));
    tr.appendChild(el("td", "mono", fmtWhen(f.mtime)));
    tr.addEventListener("click", () => showFile(f.file_id));
    tbody.appendChild(tr);
  });
}

/* ---------- 检索：POST /search ---------- */
async function doSearch() {
  const q = $("q").value.trim();
  if (!q) return;
  const topJ = Math.max(1, parseInt($("topj").value, 10) || 10);
  const noRerank = !$("rerank").checked;
  const btn = $("btn-search");
  btn.disabled = true;
  try {
    const resp = await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, top_j: topJ, no_rerank: noRerank }),
    });
    const out = await resp.json();
    const results = $("results");
    results.textContent = "";
    const meta = $("search-meta");
    if (!resp.ok || out.error) {
      meta.textContent = "错误: " + (out.error || "HTTP " + resp.status);
      return;
    }
    meta.textContent = out.items.length + " 条结果 · rerank "
      + (out.meta && out.meta.reranked ? "on" : "off") + " · top_j " + topJ;
    if (!out.items.length) {
      results.appendChild(el("div", "meta-line", "无命中"));
      return;
    }
    out.items.forEach((item) => {
      const box = el("div", "result");
      const head = el("div", "r-head");
      head.appendChild(el("span", "r-title", item.title || ("#" + item.file_id)));
      head.appendChild(el("span", "r-score", Number(item.score).toFixed(4)));
      box.appendChild(head);
      const badges = el("div", "badges");
      badges.appendChild(el("span", "badge", item.kind));
      (item.sources || []).forEach((s) => badges.appendChild(el("span", "badge", s)));
      badges.appendChild(el("span", "badge", "#" + item.file_id));
      box.appendChild(badges);
      const text = String(item.text || "");
      box.appendChild(el("div", "r-text", text.length > 240 ? text.slice(0, 240) + "…" : text));
      box.addEventListener("click", () => showFile(item.file_id));
      results.appendChild(box);
    });
  } catch (err) {
    $("search-meta").textContent = "请求失败: " + err;
  } finally {
    btn.disabled = false;
  }
}

/* ---------- 文件详情弹层：GET /files/{id} ---------- */
const DETAIL_FIELDS = [
  ["file_id", "file id", null],
  ["title", "title", null],
  ["status", "status", null],
  ["file_type", "type", null],
  ["size_bytes", "size", "bytes"],
  ["file_path", "path", null],
  ["content_hash", "hash", null],
  ["context_tag", "tags", "list"],
  ["summary", "summary", null],
  ["created_at", "created", "when"],
  ["mtime", "mtime", "when"],
  ["registered_at", "registered", "when"],
  ["updated_at", "updated", "when"],
  ["deleted_at", "deleted", "when"],
];
async function showFile(fileId) {
  let doc;
  try {
    const resp = await fetch("/files/" + encodeURIComponent(fileId));
    doc = await resp.json();
    if (!resp.ok || doc.error) {
      toast("打开详情失败: " + (doc.error || "HTTP " + resp.status));
      return;
    }
  } catch (err) { toast("打开详情失败: " + err); return; }
  $("modal-title").textContent = "#" + doc.file_id + " · " + (doc.title || "");
  const body = $("modal-body");
  body.textContent = "";
  const dl = el("dl", "kv");
  DETAIL_FIELDS.forEach(([key, label, mode]) => {
    let value = doc[key];
    if (mode === "bytes") value = fmtBytes(value) + " (" + value + ")";
    else if (mode === "list") value = (value && value.length) ? value.join(", ") : null;
    else if (mode === "when") value = fmtWhen(value);
    if (value === null || value === undefined || value === "") value = "—";
    const row = el("div");
    row.appendChild(el("dt", null, label));
    row.appendChild(el("dd", key === "deleted_at" && doc.deleted_at ? "danger" : "", String(value)));
    dl.appendChild(row);
  });
  body.appendChild(dl);
  $("modal").classList.remove("hidden");
}
function closeModal() { $("modal").classList.add("hidden"); }

/* ---------- optimize：confirm 后 POST /optimize ---------- */
async function runOptimize() {
  if (!window.confirm("触发索引优化？\n（FTS 合并 + SQLite checkpoint + 向量库 optimize）")) return;
  const btn = $("btn-optimize");
  btn.disabled = true;
  try {
    const resp = await fetch("/optimize", { method: "POST" });
    const out = await resp.json();
    if (!resp.ok || out.error) {
      toast("optimize 失败: " + (out.error || "HTTP " + resp.status));
    } else {
      toast("optimize 完成\n" + JSON.stringify(out, null, 2));
      refreshStats();
      refreshFiles();
    }
  } catch (err) {
    toast("optimize 请求失败: " + err);
  } finally {
    btn.disabled = false;
  }
}

/* ---------- 启动 ---------- */
function main() {
  initTheme();
  refreshStats();
  refreshFiles();
  setInterval(refreshStats, 30000);
  $("btn-search").addEventListener("click", doSearch);
  $("q").addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });
  $("btn-files-refresh").addEventListener("click", refreshFiles);
  $("btn-optimize").addEventListener("click", runOptimize);
  $("modal-close").addEventListener("click", closeModal);
  $("modal").addEventListener("click", (e) => { if (e.target === $("modal")) closeModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
}
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", main);
} else {
  main();
}
</script>
</body>
</html>
"""


# ------------------------------------------------------------------ 渲染入口


def render_index(theme: str = DEFAULT_THEME) -> str:
    """渲染单页看板 HTML（内嵌基础样式 + 当前皮肤 CSS + 启动数据）。

    ``theme`` 未知时回退 :data:`DEFAULT_THEME`（与 ``GET /ui?theme=`` 语义
    一致）。所有动态数据经 JS 走只读路由获取；页面本体只内嵌皮肤 CSS 与
    ``BOOT`` 启动对象（版本 / schema 版本 / 皮肤清单），无任何外链。
    """
    resolved = resolve_theme(theme)
    boot = json.dumps(_boot_payload(resolved), ensure_ascii=False).replace("</", "<\\/")
    return (
        _INDEX_TEMPLATE.replace("__THEME_CSS__", render_theme_css(resolved))
        .replace("__BOOT_JSON__", boot)
    )
