/* 从 index.html 提取真实 loadMemories，验证核心/短期分层渲染与批量删除逻辑 */
const fs = require("fs");
const path = require("node:path");
const src = fs.readFileSync(path.join(__dirname, "..", "ra-web", "index.html"), "utf8");
const start = src.indexOf("/* ================= 长期记忆 ================= */");
const end = src.indexOf("/* ================= 调用追踪 ================= */");
if (start < 0 || end < 0) { console.error("extract markers not found"); process.exit(1); }
const chunk = src.slice(start, end);

let apiCalls = [];
let apiRows = [];
const api = async (url, opts) => {
  apiCalls.push({ url, opts });
  if (url === "/api/v1/memories") return apiRows;
  return {};
};
const confirm = () => true;
const toast = () => {};
const box = { innerHTML: "" };
const mkBtn = () => ({ disabled: true, textContent: "删除选中", onclick: null });
const btns = { "#mem-del-core": mkBtn(), "#mem-del-short": mkBtn() };
const $ = sel => sel === "#memory-list" ? box : btns[sel];
let cbStubs = [], delStubs = [], tierStubs = [], parsedHTML = null;
/* $$ 在 loadMemories 绑定事件时才被调用（此时 innerHTML 已渲染）。
   同一份 innerHTML 只解析一次：三次 $$ 查询共享同一批元素对象，
   绑定的处理函数才能被测试读到；innerHTML 变化（重新 load）后再重新解析 */
const parseAll = () => {
  if (box.innerHTML === parsedHTML) return;
  parsedHTML = box.innerHTML;
  cbStubs = [...box.innerHTML.matchAll(/<input type="checkbox" data-mtier="(\w+)" data-mkey="([^"]+)">/g)]
    .map(([, t, k]) => ({ checked: false, dataset: { mtier: t, mkey: k }, onchange: null }));
  delStubs = [...box.innerHTML.matchAll(/data-del="([^"]+)"/g)]
    .map(([, k]) => ({ dataset: { del: k }, onclick: null }));
  tierStubs = [...box.innerHTML.matchAll(/data-(promote|demote)="([^"]+)"/g)]
    .map(([, kind, k]) => ({ dataset: { [kind]: k }, onclick: null }));
};
const $$ = sel => {
  parseAll();
  if (sel.includes("[data-mkey]")) return cbStubs;
  if (sel.includes("[data-del]")) return delStubs;
  if (sel.includes("[data-promote],[data-demote]")) return tierStubs;
  return [];
};
const fmtTime = () => "T";
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

const factory = new Function("api", "confirm", "toast", "$", "$$", "fmtTime", "esc",
  chunk + "\nreturn { loadMemories };");
const { loadMemories } = factory(api, confirm, toast, $, $$, fmtTime, esc);

/* 从渲染出的 HTML 解析出交互元素桩（模拟浏览器 querySelectorAll） */

let pass = 0, fail = 0;
const check = (name, cond, extra) => {
  if (cond) { pass++; console.log("  ok  " + name); }
  else { fail++; console.log("FAIL  " + name + (extra ? "\n      → " + String(extra).slice(0, 300) : "")); }
};
const R = (key, tier) => ({ key, tier, value: "v-" + key, topic: "", last_used_at: null });

(async () => {

const freshBtns = () => { btns["#mem-del-core"] = mkBtn(); btns["#mem-del-short"] = mkBtn(); };

/* 场景 1：混合层级 → 两张分层卡，核心在前 */
freshBtns(); apiCalls = [];
apiRows = [R("a", "core"), R("b", "short"), R("c", "core")];
await loadMemories();
const corePos = box.innerHTML.indexOf("核心记忆"), shortPos = box.innerHTML.indexOf("短期记忆");
check("两张分层卡且核心在前", corePos >= 0 && shortPos > corePos, box.innerHTML.slice(0, 200));
check("各节带条数描述", /核心记忆<\/span>\s*<span[^>]*>2 条 ·/.test(box.innerHTML)
  && /短期记忆<\/span>\s*<span[^>]*>1 条 ·/.test(box.innerHTML), box.innerHTML);
check("不再有混合的层级列", !box.innerHTML.includes("<th") || !/>层级</.test(box.innerHTML));
check("核心行带「降为短期」、短期行带「置顶核心」", /data-demote="a"/.test(box.innerHTML)
  && /data-promote="b"/.test(box.innerHTML));

/* 批量删除：各节独立计数与提交 */
cbStubs.find(c => c.dataset.mkey === "a").checked = true; cbStubs.find(c => c.dataset.mkey === "a").onchange();
cbStubs.find(c => c.dataset.mkey === "b").checked = true; cbStubs.find(c => c.dataset.mkey === "b").onchange();
check("各节独立计数", btns["#mem-del-core"].textContent === "删除选中(1)"
  && btns["#mem-del-short"].textContent === "删除选中(1)"
  && !btns["#mem-del-core"].disabled && !btns["#mem-del-short"].disabled);
apiCalls = [];
await btns["#mem-del-core"].onclick();
const delCall = apiCalls.find(c => c.url === "/api/v1/memories/delete");
check("按节提交删除 keys", delCall && JSON.parse(delCall.opts.body).keys.length === 1
  && JSON.parse(delCall.opts.body).keys[0] === "a", apiCalls);

/* 场景 2：某层级为空 → 显示空提示而非消失 */
freshBtns();   // 真实 DOM 中每次 innerHTML 重写后按钮均为全新元素
apiRows = [R("a", "core")];
await loadMemories();
check("空层级显示提示", box.innerHTML.includes("该层级暂无记忆")
  && box.innerHTML.indexOf("该层级暂无记忆") > box.innerHTML.indexOf("核心记忆"));
check("空层级按钮存在且禁用", btns["#mem-del-short"] && btns["#mem-del-short"].disabled);

/* 场景 3：全部为空 → 保留全局空态 */
freshBtns();
apiRows = [];
await loadMemories();
check("无任何记忆走全局空态", box.innerHTML.includes("暂无长期记忆"));

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

})();
