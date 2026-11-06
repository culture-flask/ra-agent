/* 网络重试撤回：抽取真实前端函数，覆盖普通对话、三团队、追问和断流收尾。 */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const src = fs.readFileSync(path.join(__dirname, "..", "ra-web", "index.html"), "utf8");

function extract(marker) {
  const start = src.indexOf(marker);
  const end = src.indexOf("\n}\n", start);
  assert.ok(start >= 0 && end > start, "missing function: " + marker);
  return src.slice(start, end + 3);
}

function compile(marker, name, deps = {}) {
  return new Function(...Object.keys(deps), extract(marker) + "\nreturn " + name + ";")
    (...Object.values(deps));
}

const trimStreamSuffix = compile("function trimStreamSuffix(", "trimStreamSuffix");
assert.equal(trimStreamSuffix("前轮🧭失败😀", 3), "前轮🧭", "Python 字符计数需按码点撤回 emoji");
assert.equal(trimStreamSuffix("保留", 0), "保留");
assert.equal(trimStreamSuffix("保留", -1), "保留");
assert.equal(trimStreamSuffix("保留", undefined), "保留");

const handleChatEvent = compile("function handleChatEvent(", "handleChatEvent", {
  trimStreamSuffix, resolveRetrySteps() {}, esc: String,
});
const reset = {
  type: "stream_reset", attempt_id: "next", previous_attempt_id: "failed",
  token_event: "token", reasoning_event: "reasoning", token_chars: 3, reasoning_chars: 3,
};
const chat = { content: "前轮🧭失败😀", reasoning: "前轮思考错误🧪", reasoningState: "thinking", steps: [] };
handleChatEvent(reset, chat, {});
assert.equal(chat.content, "前轮🧭", "撤回当前尝试，保留之前成功工具轮正文");
assert.equal(chat.reasoning, "前轮思考", "撤回当前尝试推理后缀");
handleChatEvent({ type: "token", attempt_id: "next", content: "成功" }, chat, {});
handleChatEvent({ type: "reasoning", attempt_id: "next", content: "新推理" }, chat, {});
assert.equal(chat.content, "前轮🧭成功");
assert.equal(chat.reasoning, "前轮思考新推理");

const empty = { content: "失败😀", reasoning: "错误🧪", reasoningState: "thinking", steps: [] };
handleChatEvent(reset, empty, {});
assert.equal(empty.content, "");
assert.equal(empty.reasoning, "");
assert.equal(empty.reasoningState, "discarded");
const routing = { content: "答案", reasoning: "答案推理", routeThinking: "路由错误🧪" };
handleChatEvent({ ...reset, token_event: "other_token", reasoning_event: "routing_reasoning" }, routing, {});
assert.equal(routing.content, "答案");
assert.equal(routing.reasoning, "答案推理");
assert.equal(routing.routeThinking, "路由");

function teamHarness() {
  const frames = [];
  const state = { bs: { live: {} } };
  const bsScheduleFlush = compile("function bsScheduleFlush(", "bsScheduleFlush", {
    requestAnimationFrame(callback) { frames.push(callback); return frames.length; },
    md: text => text ? "<p>" + text + "</p>" : "", bsScroll() {},
  });
  const handle = compile("function bsHandleEvent(", "bsHandleEvent", {
    trimStreamSuffix, bsLiveOf: agent => state.bs.live[agent], bsScheduleFlush, $: () => null,
  });
  const item = (content, reasoning) => ({
    content, reasoning, streaming: true, el: {},
    bodyEl: { innerHTML: "stale answer" }, reasonEl: { textContent: "stale reasoning", scrollHeight: 0 },
    reasonWrap: { hidden: false },
  });
  return { state, handle, item, flush() { frames.splice(0).forEach(fn => fn()); } };
}

for (const prefix of ["bs_", "sem_", "dr_"]) {
  const h = teamHarness();
  const first = h.state.bs.live.first = h.item("前轮失败😀", "旧推理错误🧪");
  const other = h.state.bs.live.other = h.item("其他角色答案", "其他角色推理");
  h.handle({ ...reset, type: prefix + "stream_reset", agent: "first",
    token_event: prefix + "token", reasoning_event: prefix + "reasoning" });
  h.flush();
  assert.equal(first.content, "前轮", prefix + "保留当前角色前轮正文");
  assert.equal(first.reasoning, "旧推理");
  assert.equal(first.bodyEl.innerHTML, "<p>前轮</p>");
  assert.equal(first.reasonEl.textContent, "旧推理");
  assert.equal(other.content, "其他角色答案", "不同 agent 的并行输出不受影响");
  assert.equal(other.reasoning, "其他角色推理");
  h.handle({ ...reset, type: prefix + "stream_reset", agent: "first", token_chars: 2, reasoning_chars: 3 });
  h.flush();
  assert.equal(first.bodyEl.innerHTML, "", "撤回至空串也要清掉已渲染正文");
  assert.equal(first.reasonEl.textContent, "", "撤回至空串也要清掉已渲染推理");
  assert.equal(first.reasonWrap.hidden, true);
  h.handle({ type: prefix + "token", agent: "first", content: "重试成功" });
  h.handle({ type: prefix + "reasoning", agent: "first", content: "新推理" });
  h.flush();
  assert.equal(first.bodyEl.innerHTML, "<p>重试成功</p>");
  assert.equal(first.reasonEl.textContent, "新推理");
  assert.equal(first.reasonWrap.hidden, false);
  first.streaming = false;
  h.handle({ ...reset, type: prefix + "stream_reset", agent: "first" });
  assert.equal(first.content, "重试成功", "结束的气泡忽略迟到 reset");
  h.handle({ ...reset, type: prefix + "stream_reset", agent: "missing" });
}

let qaFlushes = 0;
const bsHandleAskEvent = compile("function bsHandleAskEvent(", "bsHandleAskEvent", {
  trimStreamSuffix, bsRenderQA() {}, bsFlushQA() { qaFlushes++; },
});
const turn = { a: "前轮失败😀" };
bsHandleAskEvent({ ...reset, reasoning_event: "ask_reasoning" }, turn);
assert.equal(turn.a, "前轮", "追问撤回失败尝试正文");
bsHandleAskEvent({ type: "token", content: "成功" }, turn);
assert.equal(turn.a, "前轮成功");
assert.equal(qaFlushes, 2);

async function interruptedChat() {
  const state = { active: "sid", sessions: [{ id: "sid", title: "existing" }] };
  const msgs = [], elements = new Map();
  let saved;
  let reads = 0;
  const deps = {
    state, store: { base: "", token: "test" }, getMsgs: () => msgs,
    saveMsgs(value) { saved = JSON.parse(JSON.stringify(value)); },
    renderMessages() {}, renderSessions() {}, saveSessions() {}, patchLastMsg() {},
    loadSessionUsage() {}, notifyBackground() {}, handleChatEvent, esc: String,
    $(selector) {
      if (!elements.has(selector)) elements.set(selector, { style: {}, focus() {} });
      return elements.get(selector);
    },
    async fetch() {
      return { ok: true, body: { getReader: () => ({ async read() {
        if (reads++ === 0) return { done: false, value: new TextEncoder().encode(
          'data: {"type":"reasoning","content":"未收到收尾事件"}\n\n') };
        throw new Error("connection lost");
      } }) } };
    },
  };
  const sendMessage = compile("async function sendMessage(", "sendMessage", deps);
  await sendMessage("问题");
  const last = saved[saved.length - 1];
  assert.equal(last.reasoning, "未收到收尾事件");
  assert.equal(last.reasoningState, "done", "断流前未收到 end 时也要结束思考中外观");
  assert.equal(last.interrupted, true);
  assert.equal(last._streaming, false);
}

interruptedChat().then(() => console.log("stream reset render regressions passed"))
  .catch(error => { console.error(error); process.exitCode = 1; });
