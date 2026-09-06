/* 重试 UI 渲染测试：从 index.html 提取真实的 handleChatEvent / stepsHTML /
   bsHandleEvent，验证 retry 倒计时与 agent_fail/agent_end 失败原因展示。 */
const fs = require("fs");
const path = require("node:path");
const src = fs.readFileSync(path.join(__dirname, "..", "ra-web", "index.html"), "utf8");

function cut(startMarker, endMarker) {
  const a = src.indexOf(startMarker);
  const b = src.indexOf(endMarker, a);
  if (a < 0 || b < 0) throw new Error("markers not found: " + startMarker);
  return src.slice(a, b);
}

const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

/* ---- 块 A+B：STEP_STYLE + resolveRetrySteps + stepsHTML ---- */
const chunkSteps = cut("const STEP_STYLE = {", "/* 检索模式切换") +
  cut("/* 重试步骤收尾", "/* 思考面板");
const fakeSetInterval = () => 0;   // 不真正起心跳
const stepsMod = new Function("setInterval", "$$",
  chunkSteps + "\nreturn { STEP_STYLE, resolveRetrySteps, stepsHTML };");
const { STEP_STYLE, resolveRetrySteps, stepsHTML } = stepsMod(fakeSetInterval, () => []);

/* ---- 块 C：handleChatEvent ---- */
const chunkChat = cut("function handleChatEvent(ev, asst, stepByTrace) {", "/* ================= 知识库");
const chatMod = new Function("esc", "shortId", "state", "ctxSeq", "renderUsageBar", "resolveRetrySteps",
  chunkChat + "\nreturn { handleChatEvent };");
const chat = chatMod(esc, () => "sid", {}, 0, () => {}, resolveRetrySteps);

/* ---- 块 D：bsHandleEvent ---- */
const a0 = src.indexOf("function bsHandleEvent(ev) {");
const b0 = src.indexOf("\n}\n", a0);
const chunkBs = src.slice(a0, b0 + 3);
const bsMod = new Function("state", "$", "$$", "esc", "document", "bsStatus", "bsAgentMeta",
  "bsPhaseLabel", "bsLiveOf", "bsScheduleFlush", "bsScroll", "openModal", "toast", "fmtTime",
  "BS_INSIGHT_KIND", "DR_META",
  chunkBs + "\nreturn { bsHandleEvent };");

let pass = 0, fail = 0;
const check = (name, cond, extra) => {
  if (cond) { pass++; console.log("  ok  " + name); }
  else { fail++; console.log("FAIL  " + name + (extra ? "\n      → " + String(extra).slice(0, 300) : "")); }
};

/* ---------- 主对话 ---------- */
const asst = { steps: [], content: "" };
chat.handleChatEvent({ type: "retry", attempt: 2, max: 4, delay: 5, error: "read timed out" },
                     asst, {});
check("retry 事件入步骤条", asst.steps.length === 1 && asst.steps[0].kind === "retry"
  && asst.steps[0].status === "running", asst.steps);
check("步骤文案含 次数/原因", /超时重试 2\/4/.test(asst.steps[0].desc)
  && asst.steps[0].desc.includes("timed out"), asst.steps[0].desc);
check("deadline ≈ now+5s", Math.abs(asst.steps[0].deadline - Date.now() - 5000) < 1500);

const html = stepsHTML(asst.steps);
check("重试中面板自动展开 + 倒计时 span", html.includes("<details class=\"steps\" open>")
  && /data-cd="\d+"/.test(html), html);

chat.handleChatEvent({ type: "token", content: "答" }, asst, {});
check("token 到达 → 重试步骤标成功", asst.steps[0].status === "ok"
  && asst.steps[0].desc === "重试成功，继续生成" && asst.steps[0].deadline === 0, asst.steps[0]);
const html2 = stepsHTML(asst.steps);
check("成功后面板折叠、无倒计时", !html2.includes("steps\" open") && !html2.includes("data-cd")
  && html2.includes("完成"), html2);

/* ---------- 多 Agent ---------- */
function mkHarness() {
  const inserted = [], removed = [], flushed = [];
  const state = { bs: { live: {}, streamWrap: {} } };
  const mkEl = () => ({ tag: "div", innerHTML: "", style: {}, hidden: false,
    classList: { add() {}, remove() {} },
    remove() { this.removed = true; },
  });
  const mkItem = () => {
    const itemEl = mkEl(); itemEl._q = {};
    const bodyEl = mkEl();
    const item = { agent: "a1", phase: "research", content: "", streaming: true,
                   reasoning: "", el: itemEl, bodyEl,
                   reasonWrap: mkEl(), reasonEl: mkEl(), toolsEl: mkEl() };
    bodyEl.parentNode = { insertBefore(el) { itemEl._q[".bs-retry"] = el; inserted.push(el); } };
    return item;
  };
  const globalQ = {};
  const $ = (sel, root) => (root && root._q && !root._q[sel]?.removed ? root._q[sel] : null)
    || globalQ[sel] || null;
  const deps = {
    state, $, $$: () => [], esc,
    document: { createElement: () => mkEl() },
    bsStatus: () => {}, bsAgentMeta: id => ({ name: id, color: "#000" }),
    bsPhaseLabel: {}, bsScheduleFlush: it => flushed.push(it),
    bsScroll: () => {}, openModal: () => {}, toast: () => {}, fmtTime: () => "T",
    BS_INSIGHT_KIND: {}, DR_META: {},
  };
  const bs = bsMod(deps.state, deps.$, deps.$$, deps.esc, deps.document,
    deps.bsStatus, deps.bsAgentMeta, deps.bsPhaseLabel,
    a => state.bs.live[a], deps.bsScheduleFlush, deps.bsScroll,
    deps.openModal, deps.toast, deps.fmtTime, deps.BS_INSIGHT_KIND, deps.DR_META);
  return { state, mkItem, inserted, removed, flushed, handle: bs.bsHandleEvent, globalQ };
}

/* retry：横幅插入气泡，含次数/倒计时/原因 */
const h1 = mkHarness();
const it1 = h1.mkItem();
h1.state.bs.live.a1 = it1;
h1.handle({ type: "bs_retry", agent: "a1", attempt: 2, max: 5, delay: 8, error: "read timed out" });
check("多Agent：重试横幅插入", h1.inserted.length === 1
  && /超时重试 2\/5/.test(h1.inserted[0].innerHTML)
  && /data-cd="\d+"/.test(h1.inserted[0].innerHTML)
  && h1.inserted[0].innerHTML.includes("timed out"), h1.inserted[0] && h1.inserted[0].innerHTML);
h1.handle({ type: "bs_retry", agent: "a1", attempt: 3, max: 5, delay: 16, error: "again" });
check("重复重试：复用同一横幅更新次数", h1.inserted.length === 1
  && /超时重试 3\/5/.test(h1.inserted[0].innerHTML));

/* token：横幅撤除 */
h1.handle({ type: "bs_token", agent: "a1", content: "答" });
check("token 恢复 → 横幅撤除 + 内容继续", h1.inserted[0].removed === true
  && it1.content === "答" && h1.flushed.length === 1,
  { content: it1.content, flushed: h1.flushed.length, removed: h1.inserted[0].removed });

/* agent_fail：红字失败原因，气泡结束 */
const h2 = mkHarness();
const it2 = h2.mkItem();
h2.state.bs.live.a1 = it2;
h2.handle({ type: "bs_retry", agent: "a1", attempt: 5, max: 5, delay: 16, error: "x" });
h2.handle({ type: "bs_agent_fail", agent: "a1", error: "连接被重置" });
check("agent_fail：红字失败原因", it2.bodyEl.innerHTML.includes("生成失败")
  && it2.bodyEl.innerHTML.includes("连接被重置")
  && it2.bodyEl.innerHTML.includes("var(--danger)"), it2.bodyEl.innerHTML);
check("agent_fail：结束流式并清横幅/占位", it2.streaming === false
  && h2.inserted[0].removed === true && h2.state.bs.live.a1 === undefined);

/* agent_end：全阶段失败原因（辩论发言等原先不显示） */
const h3 = mkHarness();
const it3 = h3.mkItem();
h3.state.bs.live.a1 = it3;
h3.handle({ type: "bs_agent_end", agent: "a1", phase: "debate", reason: "限流 429" });
check("agent_end 辩论失败：显示原因", it3.bodyEl.innerHTML.includes("生成失败：限流 429"),
  it3.bodyEl.innerHTML);
const h4 = mkHarness();
const it4 = h4.mkItem();
h4.state.bs.live.a1 = it4;
h4.handle({ type: "bs_agent_end", agent: "a1", phase: "research", failed: true });
check("agent_end 调研失败（无原因）→ 缺席文案",
  it4.bodyEl.innerHTML.includes("调研失败，本角色缺席"), it4.bodyEl.innerHTML);
const h5 = mkHarness();
const it5 = h5.mkItem();
h5.state.bs.live.a1 = it5;
h5.handle({ type: "bs_agent_end", agent: "a1", phase: "research", failed: true, reason: "额度用尽" });
check("agent_end 调研失败：显示原因", it5.bodyEl.innerHTML.includes("调研失败：额度用尽"));

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
