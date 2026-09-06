/* UX 增强测试：代码块复制、消息复制/继续生成按钮、后台通知、回到底部。
   从 index.html 提取真实实现，DOM/定时器全部用桩。 */
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

/* ---- 通知助手 + 代码复制委托 ---- */
const notifyChunk = cut("/* ---------- 长任务后台提醒 ---------- */", "function fmtTime");
let intervals = [], cleared = [], notifPerm = "default", notifications = [], docState;
const docStub = {
  get hidden() { return docState.hidden; },
  get title() { return docState.title; },
  set title(v) { docState.title = v; },
  addEventListener(type, fn) { (docState.handlers[type] ||= []).push(fn); },
};
const notifyMod = new Function("document", "window", "setInterval", "clearInterval",
  "NotificationCtor",
  notifyChunk
    .replace(/new Notification\(/g, "new NotificationCtor(")
    .replace(/Notification\.requestPermission/g, "NotificationCtor.requestPermission")
    .replace(/Notification\.permission/g, "NotificationCtor.permission")
    + "\nreturn { notifyBackground, ensureNotifyPermission };");
function mkNotify(hidden, perm) {
  docState = { hidden, title: "T", handlers: {} };
  intervals = []; cleared = []; notifications = []; notifPerm = perm;
  const win = {};
  win.Notification = function (title, opts) { notifications.push({ title, opts }); };
  Object.defineProperty(win.Notification, "permission", { get: () => notifPerm });
  win.Notification.requestPermission = function () { notifPerm = "granted"; return Promise.resolve(); };
  const mod = notifyMod(docStub, win,
    (fn, ms) => { intervals.push({ fn, ms }); return 42; },
    id => cleared.push(id), win.Notification);
  return mod;
}

/* ---- msgHTML + continueGeneration ---- */
const msgChunk = cut("function msgHTML(m, i, isLast) {", "/* ---------------- 滚动跟随");
const stubs = {
  feedbackHTML: () => "", citesHTML: () => "", stepsHTML: () => "",
  routeThinkingHTML: () => "", thinkingHTML: () => "",
};
const msgMod = new Function("esc", "md", "state", "feedbackHTML", "citesHTML", "stepsHTML",
  "routeThinkingHTML", "thinkingHTML",
  msgChunk + "\nreturn { msgHTML };");
const msgState = { user: { username: "tester" } };
const { msgHTML } = msgMod(esc, m => esc(m.content), msgState, stubs.feedbackHTML, stubs.citesHTML,
  stubs.stepsHTML, stubs.routeThinkingHTML, stubs.thinkingHTML);

const contChunk = cut("/* 继续生成：上一条回答", "function regenerateLast");
let sent = null;
const contMod = new Function("state", "sendMessage",
  contChunk + "\nreturn { continueGeneration };");
const { continueGeneration } = contMod({ streaming: false }, t => { sent = t; });

let pass = 0, fail = 0;
const check = (name, cond, extra) => {
  if (cond) { pass++; console.log("  ok  " + name); }
  else { fail++; console.log("FAIL  " + name + (extra ? "\n      → " + String(extra).slice(0, 260) : "")); }
};

(async () => {

/* 1. 代码块复制按钮（md() 输出，从 frontend_md_render 的方式提取太重，这里直接验证标记存在） */
check("md() 代码块带复制按钮", src.includes('data-copy-code title="复制代码"')
  && src.includes('<pre class="md-pre">'));
check("复制为事件委托且读 code 文本", notifyChunk === "" ? false :
  src.includes('e.target.closest("[data-copy-code]")') && src.includes('pre.querySelector("code")'));

/* 2. 后台通知 */
const n1 = mkNotify(true, "granted");
n1.notifyBackground("科研助手", "回答已生成");
check("后台时弹系统通知", notifications.length === 1 && notifications[0].title === "科研助手"
  && notifications[0].opts.body === "回答已生成", notifications);
check("后台时标题闪烁启动", intervals.length === 1 && intervals[0].ms === 1000, intervals);
n1.notifyBackground("科研助手", "再次完成");
check("闪烁定时器不重复启动", intervals.length === 1);

const n2 = mkNotify(true, "default");
n2.ensureNotifyPermission();
check("default 权限下预请求授权", notifPerm === "granted", notifPerm);
const n2b = mkNotify(true, "denied");   // 用户拒绝过：仅标题闪烁，不发通知
n2b.notifyBackground("科研助手", "完成");
check("未授权时仅标题闪烁", notifications.length === 0 && intervals.length === 1);

const n3 = mkNotify(false, "granted");
n3.notifyBackground("科研助手", "完成");
check("页面前台时不打扰", notifications.length === 0 && intervals.length === 0);

const n4 = mkNotify(true, "granted");
n4.notifyBackground("科研助手", "完成");
docState.hidden = false;
docState.handlers.visibilitychange.forEach(fn => fn());
check("回到页面：标题复原、闪烁停止", docState.title === "T" && cleared.length === 1, {
  title: docState.title, cleared,
});

/* 3. 消息操作条：复制 + 继续生成 */
const base = { role: "assistant", content: "答案", steps: [], retrievals: [] };
const h1 = msgHTML({ ...base }, 3, true);
check("助手消息带复制按钮", h1.includes('data-copymsg="3"'), h1);
check("未中断：无继续生成按钮", !h1.includes("data-continue"), h1);
const h2 = msgHTML({ ...base, interrupted: true }, 4, true);
check("中断的末条消息带继续生成按钮", h2.includes("data-continue=\"1\"")
  && h2.includes("继续生成"), h2);
const h3 = msgHTML({ ...base, interrupted: true }, 1, false);
check("非末条中断消息不给继续生成", !h3.includes("data-continue"), h3);
const h4 = msgHTML({ role: "user", content: "q" }, 0, false);
check("用户消息无复制/继续按钮", !h4.includes("data-copymsg") && !h4.includes("data-continue"), h4);

/* 4. 继续生成：发送固定续写指令 */
continueGeneration();
check("continueGeneration 发送续写指令", sent !== null
  && sent.includes("继续") && sent.includes("不要重复已有部分"), sent);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

})();
