/* 多 Agent 视图总 token 消耗显示：从 index.html 提取真实实现验证 */
const fs = require("fs");
const path = require("node:path");
const src = fs.readFileSync(path.join(__dirname, "..", "ra-web", "index.html"), "utf8");
const start = src.indexOf("/* 总 token 消耗：整场结束或回放完成后展示");
const end = src.indexOf("function bsResetButtons");
if (start < 0 || end < 0) { console.error("extract markers not found"); process.exit(1); }
const chunk = src.slice(start, end);

const els = { "#bs-token-total": { style: {}, innerHTML: "" } };
const $ = sel => els[sel];
let apiCalls = [];
const apiResponses = {};
const api = async url => { apiCalls.push(url); return apiResponses[url]; };
const bsApi = p => "/mock" + p;

const factory = new Function("$", "api", "bsApi",
  chunk + "\nreturn { bsRenderTokenTotal, bsShowTokenTotal };");
const { bsRenderTokenTotal, bsShowTokenTotal } = factory($, api, bsApi);

let pass = 0, fail = 0;
const check = (name, cond, extra) => {
  if (cond) { pass++; console.log("  ok  " + name); }
  else { fail++; console.log("FAIL  " + name + (extra ? "\n      → " + String(extra).slice(0, 200) : "")); }
};

(async () => {
  bsRenderTokenTotal({ tokens: 1234567, turns: 5 });
  check("有消耗时显示千分位合计", els["#bs-token-total"].style.display === ""
    && els["#bs-token-total"].innerHTML.includes("1,234,567"), els["#bs-token-total"].innerHTML);
  bsRenderTokenTotal({});
  check("无统计则隐藏", els["#bs-token-total"].style.display === "none");
  bsRenderTokenTotal(null);
  check("stats 缺失隐藏", els["#bs-token-total"].style.display === "none");

  apiResponses["/mock/bs-1"] = { stats: { tokens: 999 } };
  await bsShowTokenTotal("bs-1");
  check("直播结束拉详情并显示", apiCalls.pop() === "/mock/bs-1"
    && els["#bs-token-total"].innerHTML.includes("999"));
  await bsShowTokenTotal(null);
  check("无 sid 直接隐藏且不发请求", els["#bs-token-total"].style.display === "none"
    && apiCalls.length === 0);

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
})();
