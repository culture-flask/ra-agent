/* 抽取真实前端实现，回归开关、上传、账户切换及普通/三团队请求协议。无需浏览器或数据库。 */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const source = fs.readFileSync(path.join(__dirname, "..", "ra-web", "index.html"), "utf8");
const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[ch]));

function extract(marker) {
  const start = source.indexOf(marker), end = source.indexOf("\n}\n", start);
  assert.ok(start >= 0 && end > start, "missing function: " + marker);
  return source.slice(start, end + 3);
}
function compile(marker, name, deps) {
  return new Function(...Object.keys(deps), extract(marker) + "\nreturn " + name + ";")(...Object.values(deps));
}

class Element {
  constructor() { this._html = ""; this.style = {}; this.value = ""; this.controls = []; }
  set innerHTML(html) {
    this._html = html;
    this.controls = [...html.matchAll(/<input type="checkbox" data-mm-kind="([^"]+)"( checked)?>/g)]
      .map(([, kind, checked]) => ({ dataset: { mmKind: kind }, checked: !!checked }));
  }
  get innerHTML() { return this._html; }
  focus() {}
}
const elements = new Map();
const $ = selector => {
  if (!elements.has(selector)) elements.set(selector, new Element());
  return elements.get(selector);
};
const $$ = (selector, root) => selector === "[data-mm-kind]" ? root.controls : [];
const state = {
  user: { id: "alice", username: "Alice" }, pendingFiles: [], bsFiles: [],
  multimodal: { image: false, audio: false, video: false },
  active: "chat-1", sessions: [{ id: "chat-1", title: "existing" }], bs: { team: "debate" },
};
const store = { token: "token-alice", base: "http://backend" };
const saved = new Map(), notices = [], uploads = [], sentChats = [], launched = [];
const localStorage = { getItem: key => saved.get(key) ?? null, setItem: (key, value) => saved.set(key, value) };
let uploadResponse = { files: [] };
const api = async (url, opts) => {
  uploads.push({ url, opts });
  return typeof uploadResponse === "function" ? uploadResponse() : uploadResponse;
};
class FormDataStub {
  constructor() { this.items = []; }
  append(key, file, filename) { this.items.push({ key, file, filename }); }
}
const prefixes = { debate: "/api/v1/brainstorm", seminar: "/api/v1/seminar", deep_research: "/api/v1/deep-research" };
const bsApi = suffix => prefixes[state.bs.team] + suffix;
const noOp = () => {};
const deps = {
  state, store, $, $$, esc, localStorage, api, FormData: FormDataStub,
  toast: (message, kind) => notices.push({ message, kind }), ensureNotifyPermission: noOp,
  sendMessage: (text, opts) => sentChats.push({ text, opts }),
  bsQAHint: noOp, bsRenderTokenTotal: noOp, bsSyncRerun: noOp, bsRenderTopic: noOp, bsStatus: noOp,
  bsApi, async bsStream(sid, endpoint, body) { launched.push({ sid, endpoint, body }); },
  bsResetButtons() { state.bs.running = false; },
};
const start = source.indexOf("/* ---------------- 多模态输入："), end = source.indexOf("/* 发送消息：", start);
assert.ok(start >= 0 && end > start);
const mod = new Function(...Object.keys(deps), "let bsPinned;\n" + source.slice(start, end) +
  extract("function bsRenderFiles(") + extract("async function uploadBsFiles(") +
  extract("async function bsLaunch(") +
  "\nreturn { multimodalInputOptions, resetMultimodalInputs, renderMultimodalInputs, inputFileKind," +
  "attachmentDisabled, attachmentChip, validatePendingAttachments, uploadChatFiles, uploadBsFiles, sendChat, bsLaunch };")
  (...Object.values(deps));

function setInput(kind, checked, pane = "chat") {
  const cb = $("#" + pane + "-multimodal-controls").controls.find(input => input.dataset.mmKind === kind);
  assert.ok(cb, "missing " + kind + " control");
  cb.checked = checked; cb.onchange();
}
const file = (name, type = "", size = 42) => ({ name, type, size });

async function chatRequest(files = []) {
  const requests = [], messages = [];
  const send = compile("async function sendMessage(", "sendMessage", {
    state, store, $, esc, toast: deps.toast, getMsgs: () => messages, saveMsgs: noOp,
    renderMessages: noOp, renderSessions: noOp, saveSessions: noOp, patchLastMsg: noOp,
    loadSessionUsage: noOp, notifyBackground: noOp, handleChatEvent: noOp,
    fetch: async (url, opts) => {
      requests.push({ url, body: JSON.parse(opts.body) });
      return { ok: true, body: { getReader: () => ({ read: async () => ({ done: true }) }) } };
    },
  });
  await send("查看附件", { files });
  return { requests, messages };
}

(async () => {
  mod.resetMultimodalInputs("alice");
  assert.deepEqual(mod.multimodalInputOptions(), { image: false, audio: false, video: false });
  assert.equal($("#chat-file").accept, ".txt,.md,.pdf,.docx");
  assert.equal($("#bs-file").accept, $("#chat-file").accept);
  setInput("image", true);
  assert.ok($("#chat-file").accept.includes(".png"));
  assert.ok(!$("#chat-file").accept.includes(".mp3"));
  assert.equal($("#bs-multimodal-controls").controls[0].checked, true);
  assert.deepEqual(JSON.parse(saved.get("ra.multimodal.alice")), { image: true, audio: false, video: false });
  setInput("audio", true, "bs");
  assert.ok($("#chat-file").accept.includes(".mp3"));
  assert.ok(!$("#chat-file").accept.includes(".mp4"));
  assert.ok($("#bs-multimodal-controls").innerHTML.includes("声音请另传音频"));

  assert.equal(mod.inputFileKind(file("PHOTO.PNG")), "image");
  assert.equal(mod.inputFileKind(file("note.pdf")), "text");
  assert.equal(mod.inputFileKind(file("voice.webm", "audio/webm")), "audio");
  assert.equal(mod.inputFileKind(file("movie.webm", "video/webm")), "video");
  assert.equal(mod.inputFileKind(file("movie.avi")), "video");
  assert.equal(mod.inputFileKind(file("unsafe.exe", "image/png")), "unknown");

  uploadResponse = { files: [{ file_id: "img-1", filename: '<img src=x onerror="bad">.png', kind: "image",
    mime_type: "image/png", size_bytes: 42, base64: "NEVER_STORE_THIS" }] };
  await mod.uploadChatFiles([file("original.png", "image/png")]);
  assert.equal(uploads.at(-1).url, "/api/v1/chat/files");
  assert.deepEqual(uploads.at(-1).opts.body.items.map(item => item.key), ["files"]);
  assert.equal(state.pendingFiles[0].status, "ok");
  assert.equal(state.pendingFiles[0].kind, "image");
  assert.equal(state.pendingFiles[0].mime_type, "image/png");
  assert.ok(!JSON.stringify(state.pendingFiles).includes("NEVER_STORE_THIS"));
  assert.ok($("#attach-bar").innerHTML.includes("&lt;img src=x onerror=&quot;bad&quot;&gt;.png"));
  assert.ok(!$("#attach-bar").innerHTML.includes('<img src=x'));
  setInput("image", false);
  assert.equal(state.pendingFiles.length, 1, "关闭输入保留附件，用户可选择启用或移除");
  assert.ok($("#attach-bar").innerHTML.includes("（已关闭）"));
  $("#chat-input").value = "请分析";
  mod.sendChat();
  assert.equal(sentChats.length, 0, "关闭的附件禁止发送");
  assert.equal($("#chat-input").value, "请分析", "拦截时保留问题");
  setInput("image", true);
  mod.sendChat();
  assert.equal(sentChats.at(-1).opts.files[0].file_id, "img-1");
  assert.equal(state.pendingFiles.length, 0);

  const beforeDisabledUpload = uploads.length;
  await mod.uploadChatFiles([file("clip.mp4", "video/mp4")]);
  assert.equal(uploads.length, beforeDisabledUpload);
  assert.equal(state.pendingFiles[0].status, "err");
  assert.ok(state.pendingFiles[0].error.includes("已关闭"));
  assert.equal(mod.validatePendingAttachments(state.pendingFiles), false);
  state.pendingFiles = [];
  await mod.uploadChatFiles([file("large.pdf", "application/pdf", 20 * 1024 * 1024 + 1)]);
  assert.equal(uploads.length, beforeDisabledUpload);
  assert.ok(state.pendingFiles[0].error.includes("20MiB"));

  state.pendingFiles = [];
  uploadResponse = { files: [{ file_id: "doc-1", filename: "one.txt" }] };
  await mod.uploadChatFiles([file("one.txt"), file("two.txt")]);
  assert.equal(state.pendingFiles[0].kind, "text", "旧文档响应无 kind 仍兼容");
  assert.equal(state.pendingFiles[1].status, "err", "缺少逐文件响应不会永久显示上传中");
  assert.equal(mod.validatePendingAttachments(state.pendingFiles), false, "部分失败不能静默丢文件发送");

  // WebM 的实际内容可能与浏览器 MIME 不同，后端重判后仍须服从新类型的开关。
  state.pendingFiles = [];
  uploadResponse = { files: [{ file_id: "actual-video", filename: "voice.webm", kind: "video", mime_type: "video/webm" }] };
  await mod.uploadChatFiles([file("voice.webm", "audio/webm")]);
  assert.equal(state.pendingFiles[0].status, "ok");
  assert.equal(state.pendingFiles[0].kind, "video");
  assert.ok($("#attach-bar").innerHTML.includes("视频"));
  assert.ok($("#attach-bar").innerHTML.includes("（已关闭）"));
  assert.equal(mod.validatePendingAttachments(state.pendingFiles), false, "后端重判为关闭类型时禁止发送");

  state.pendingFiles = [];
  const beforeTooMany = uploads.length;
  await mod.uploadChatFiles(Array.from({ length: 9 }, (_, i) => file(i + ".txt")));
  assert.equal(uploads.length, beforeTooMany);
  assert.equal(state.pendingFiles.length, 0);
  assert.ok(notices.at(-1).message.includes("本次未添加"));

  // 旧账户的上传即使在新账户登录后返回，也不能修改新账户队列或首选项。
  let resolveOld;
  uploadResponse = () => new Promise(resolve => { resolveOld = resolve; });
  const oldUpload = mod.uploadChatFiles([file("alice.txt")]);
  assert.equal(state.pendingFiles[0].status, "uploading");
  state.user = { id: "bob" }; store.token = "token-bob";
  mod.resetMultimodalInputs("bob");
  assert.equal(state.pendingFiles.length, 0);
  assert.deepEqual(mod.multimodalInputOptions(), { image: false, audio: false, video: false });
  resolveOld({ files: [{ file_id: "alice-secret", kind: "text" }] });
  await oldUpload;
  assert.equal(state.pendingFiles.length, 0);
  state.user = { id: "alice" }; store.token = "token-alice";
  mod.resetMultimodalInputs("alice");
  assert.equal(state.multimodal.image, true, "用户偏好按账户恢复");
  saved.set("ra.multimodal.broken", "not-json");
  mod.resetMultimodalInputs("broken");
  assert.equal(state.multimodal.image, false);
  saved.set("ra.multimodal.null", "null");
  mod.resetMultimodalInputs("null");
  assert.deepEqual(mod.multimodalInputOptions(), { image: false, audio: false, video: false });

  // 无媒体时请求明确全关；服务端可据此过滤已有 checkpoint 的历史媒体。
  const plain = await chatRequest();
  assert.deepEqual(plain.requests[0].body.multimodal, { image: false, audio: false, video: false });
  assert.ok(!("attachments" in plain.requests[0].body));
  state.multimodal = { image: true, audio: false, video: true };
  const image = { name: "plot.png", file_id: "img-2", status: "ok", kind: "image", mime_type: "image/png" };
  const chat = await chatRequest([image]);
  assert.deepEqual(chat.requests[0].body.attachments, ["img-2"]);
  assert.deepEqual(chat.requests[0].body.multimodal, { image: true, audio: false, video: true });
  assert.deepEqual(chat.messages[0].files, [{ name: "plot.png", kind: "image", mime_type: "image/png" }]);
  state.multimodal.image = false;
  assert.equal((await chatRequest([image])).requests.length, 0, "直接发送入口也不能绕过关闭状态");
  state.multimodal.image = true;

  for (const team of ["debate", "seminar", "deep_research"]) {
    state.bs = { team };
    uploadResponse = { files: [{ file_id: "team-image", filename: "ref.png", kind: "image", mime_type: "image/png" }] };
    await mod.uploadBsFiles([file("ref.png", "image/png")]);
    assert.equal(state.bsFiles[0].kind, "image");
    await mod.bsLaunch("科研议题", [], "session-" + team, 3, 10000);
    const launch = launched.at(-1);
    assert.equal(launch.endpoint, prefixes[team] + "/stream");
    assert.deepEqual(launch.body.multimodal, { image: true, audio: false, video: true });
    assert.deepEqual(launch.body.attachments, ["team-image"]);
    assert.equal(state.bsFiles.length, 0);
    assert.equal("max_rounds" in launch.body, team === "debate");
  }
  state.bsFiles = [image]; state.multimodal.image = false;
  const launchCount = launched.length;
  await mod.bsLaunch("拒绝发送关闭附件", [], "blocked", 3, 10000);
  assert.equal(launched.length, launchCount);
  assert.equal(state.bsFiles.length, 1);

  const asks = [];
  for (const team of ["debate", "seminar", "deep_research"]) {
    state.bs = { team, sid: "ask-session", qa: [], running: false };
    $("#bs-qa-input").value = "结合之前附件补充";
    // bsAsk 运行真实函数时从注入模块获取同一组规范化开关。
    const ask = compile("async function bsAsk(", "bsAsk", {
      state, store, $, toast: deps.toast, bsApi, bsRenderQA: noOp, bsHandleAskEvent: noOp,
      cancelAnimationFrame: noOp, multimodalInputOptions: mod.multimodalInputOptions,
      fetch: async (url, opts) => {
        asks.push({ url, body: JSON.parse(opts.body) });
        return { ok: true, body: { getReader: () => ({ read: async () => ({ done: true }) }) } };
      },
    });
    await ask();
    assert.equal(asks.at(-1).url, store.base + prefixes[team] + "/ask-session/ask");
    assert.deepEqual(asks.at(-1).body.multimodal, { image: false, audio: false, video: true });
  }
  assert.ok(source.includes('bsApi("/rerun-writer"), { session_id: sid, multimodal: multimodalInputOptions() }'),
    "执笔人重跑也携带当前开关，供后端过滤已有媒体");
  console.log("multimodal input render regressions passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
