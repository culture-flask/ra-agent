/* 从 index.html 提取真实 loadFiles/loadKbDocs/viewFile/downloadFile，验证渲染与交互 */
const fs = require("fs");
const path = require("node:path");
const src = fs.readFileSync(path.join(__dirname, "..", "ra-web", "index.html"), "utf8");
const start = src.indexOf("/* ================= 笔记和文件 =================");
const end = src.indexOf("/* ================= 用量统计 =================");
if (start < 0 || end < 0) { console.error("extract markers not found"); process.exit(1); }
const chunk = src.slice(start, end);

const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let apiCalls = [];
const apiResponses = {};
const api = async (url) => { apiCalls.push(url); if (!(url in apiResponses)) throw new Error("no stub: " + url); return apiResponses[url]; };
const fmtTime = () => "T";
const modal = { html: null };
const openModal = h => { modal.html = h; };
const toasts = [];
const toast = (m, k) => toasts.push(m + "|" + k);
const store = { base: "http://backend", token: "tok" };
const fetchCalls = [];
const fetchStub = async (url, opts) => {
  fetchCalls.push({ url, headers: opts.headers });
  if (!opts.headers.Authorization) throw new Error("missing auth header");
  const resp = { ok: true, text: async () => "FILE<CONTENT>", blob: async () => "BLOB" };
  if (!resp.ok) throw new Error("http");
  return resp;
};
const createdObjects = [], clicks = [];
const URLStub = { createObjectURL: () => { createdObjects.push(1); return "blob:fake"; }, revokeObjectURL: () => {} };
const windowStub = { open: (u) => clicks.push("open:" + u) };
const documentStub = { createElement: () => ({ click: () => clicks.push("a-click") }) };

const els = {
  "#outputs-list": { innerHTML: "" },
  "#kbdocs-list": { innerHTML: "" },
  "#files-kb": { value: "", innerHTML: "", onchange: null },
  "#files-refresh": { onclick: null },
};
const $ = sel => els[sel];
/* (sel, root) → 从 root.innerHTML 解析交互元素桩。
   按 root 缓存：同一份 innerHTML 只解析一次，绑定与断言共享同一批对象 */
const parseBtns = (html, attr, label) =>
  [...html.matchAll(new RegExp('<button class="btn sm" ' + attr + '="([^"]*)" data-\\w+-name="([^"]*)">' + label + "</button>", "g"))]
    .map(([, p, n]) => ({ dataset: { [attr === "data-view-path" ? "viewPath" : "dlPath"]: p, [attr === "data-view-path" ? "viewName" : "dlName"]: n }, onclick: null }));
const parseTiles = html =>
  [...html.matchAll(/<div class="file-tile" data-tile data-can-view="(\d)"\s*data-view-name="([^"]*)" data-dl-path="([^"]*)" data-dl-name="([^"]*)"/g)]
    .map(([, v, vn, dp, dn]) => ({ dataset: { canView: v, viewName: vn, dlPath: dp, dlName: dn }, onclick: null }));
const stubCache = new Map();
const $$ = (sel, root) => {
  const html = (root && root.innerHTML) || "";
  let c = stubCache.get(root);
  if (!c || c.html !== html) {
    c = { html, view: parseBtns(html, "data-view-path", "查看"), dl: parseBtns(html, "data-dl-path", "下载"),
          tiles: parseTiles(html) };
    stubCache.set(root, c);
  }
  if (sel.includes("[data-view-path]")) return c.view;
  if (sel.includes("[data-dl-path]")) return c.dl;
  if (sel.includes("[data-tile]")) return c.tiles;
  return [];
};

const factory = new Function("api", "$", "$$", "esc", "fmtTime", "openModal", "toast", "store",
  "fetch", "URL", "window", "document",
  chunk + "\nreturn { loadFiles, loadKbDocs, viewFile, downloadFile, fileTile, fileIconSVG };");
const { loadFiles, loadKbDocs, viewFile, downloadFile, fileTile, fileIconSVG } =
  factory(api, $, $$, esc, fmtTime, openModal, toast, store, fetchStub, URLStub, windowStub, documentStub);

let pass = 0, fail = 0;
const check = (name, cond, extra) => {
  if (cond) { pass++; console.log("  ok  " + name); }
  else { fail++; console.log("FAIL  " + name + (extra ? "\n      → " + String(extra).slice(0, 300) : "")); }
};

(async () => {

/* 1. 产出文件列表渲染 */
apiResponses["/api/v1/outputs/all"] = [
  { name: "20260901-1010-调研报告.md", size: 1536, mtime: 1788800000 },
  { name: "refs<b>.bib", size: 10, mtime: 1788700000 },
];
await loadFiles();
check("产出文件改为图标网格：瓦片数正确", els["#outputs-list"].innerHTML.includes('class="file-grid"')
  && (els["#outputs-list"].innerHTML.match(/file-tile/g) || []).length === 2,
  els["#outputs-list"].innerHTML.slice(0, 200));
check("瓦片带类型图标（md 角标）", els["#outputs-list"].innerHTML.includes('class="file-ic"')
  && els["#outputs-list"].innerHTML.includes(">MD<"), els["#outputs-list"].innerHTML.slice(0, 400));
check("元信息行（大小·时间）", els["#outputs-list"].innerHTML.includes("1.5 KB · T"));
check("文件名 XSS 转义", els["#outputs-list"].innerHTML.includes("refs&lt;b&gt;.bib"));
check("下载路径编码", els["#outputs-list"].innerHTML.includes("/api/v1/outputs/download?name=" + encodeURIComponent("20260901-1010-调研报告.md")));
check("查看/下载按钮就位并已绑定", els["#outputs-list"].innerHTML.includes(">查看</button>")
  && els["#outputs-list"].innerHTML.includes(">下载</button>"));

/* 2. 文本预览走弹窗（带凭证 fetch + 转义内容） */
const viewBtn = $$("[data-view-path]", els["#outputs-list"])[0];
await viewBtn.onclick({ stopPropagation() {} });
check("文本文件弹窗预览（内容转义）", modal.html && modal.html.includes("查看 · 20260901-1010-调研报告.md")
  && modal.html.includes("FILE&lt;CONTENT&gt;"), modal.html && modal.html.slice(0, 120));
check("预览请求带 Bearer 凭证", fetchCalls.at(-1).headers.Authorization === "Bearer tok"
  && fetchCalls.at(-1).url === "http://backend" + viewBtn.dataset.viewPath);

/* 3. 下载：blob URL + a[download] 点击 */
const dlBtn = $$("[data-dl-path]", els["#outputs-list"])[0];
await dlBtn.onclick({ stopPropagation() {} });
check("下载触发 blob + a.click", createdObjects.length === 1 && clicks.includes("a-click"), clicks);

/* 3.5 瓦片点击 = 查看（可预览类型） */
modal.html = null;
const tile = $$("[data-tile]", els["#outputs-list"])[0];
await tile.onclick();
check("点击瓦片直接预览", modal.html && modal.html.includes("查看 · 20260901-1010-调研报告.md"), modal.html);

/* 4. 非文本非 pdf → 转下载并提示 */
toasts.length = 0;
await viewFile("/api/v1/outputs/download?name=x.docx", "x.docx");
check("docx 转为下载", toasts.some(t => t.includes("不支持在线预览")) && createdObjects.length === 2, toasts);

/* 5. 知识库下拉：可见库 + 公共/私人标注，旧选中保持 */
apiResponses["/api/v1/kbs"] = [
  { kb_id: "kb1", name: "公共库", scope: "public", kind: "user" },
  { kb_id: "kb2", name: "私人库", scope: "private", kind: "archive" },
];
els["#files-kb"].value = "";
await loadFiles();
check("下拉带公共/私人/沉淀库标注", els["#files-kb"].innerHTML.includes("公共库（公共）")
  && els["#files-kb"].innerHTML.includes("私人库（私人 · 沉淀库）"), els["#files-kb"].innerHTML);
els["#files-kb"].value = "kb1";
await loadFiles();
check("旧选中保持并自动加载该库文档", els["#files-kb"].value === "kb1"
  && apiCalls.some(u => u.includes("/kbs/kb1/files")), apiCalls.at(-1));

/* 6. 源文档列表：文件名/片段/页码；无文件名的旧数据不显示查看按钮 */
apiResponses["/api/v1/kbs/kb1/files"] = [
  { doc_id: "3cf7839a", filename: "论文.pdf", chunks: 12, pages: [1, 3] },
  { doc_id: "abc123ff", filename: null, chunks: 2, pages: [] },
];
await loadKbDocs("kb1");
check("源文档表格渲染", els["#kbdocs-list"].innerHTML.includes("论文.pdf")
  && els["#kbdocs-list"].innerHTML.includes("1-3") && els["#kbdocs-list"].innerHTML.includes("12"), els["#kbdocs-list"].innerHTML);
check("旧数据（无文件名）回退 doc_id 且不给查看按钮", els["#kbdocs-list"].innerHTML.includes("abc123ff（旧数据无文件名）")
  && (els["#kbdocs-list"].innerHTML.match(/>查看<\/button>/g) || []).length === 1, els["#kbdocs-list"].innerHTML);
check("下载端点路径正确", els["#kbdocs-list"].innerHTML.includes("/api/v1/kbs/kb1/docs/3cf7839a/download"));

/* 7. 空态与错误 */
apiResponses["/api/v1/outputs/all"] = [];
await loadFiles();
check("产出目录空提示", els["#outputs-list"].innerHTML.includes("暂无产出文件"));
delete apiResponses["/api/v1/kbs/kb1/files"];
els["#kbdocs-list"].innerHTML = "";
await loadKbDocs("kb1");
check("源文档加载失败显示错误", els["#kbdocs-list"].innerHTML.includes("no stub"), els["#kbdocs-list"].innerHTML);

/* 8. canView=false：瓦片无查看按钮、点击瓦片走下载 */
const noViewTile = fileTile("n.bin", "1 B", "/x", false);
check("canView=false 仅下载按钮", !noViewTile.includes(">查看</button>")
  && noViewTile.includes(">下载</button>") && noViewTile.includes('data-can-view="0"'), noViewTile);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

})();
