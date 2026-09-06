/* 从 index.html 提取真实 renderUsageCharts，用 Chart 桩验证聚合与降级逻辑 */
const fs = require("fs");
const path = require("node:path");
const src = fs.readFileSync(path.join(__dirname, "..", "ra-web", "index.html"), "utf8");
const start = src.indexOf("let usageLastData");
const end = src.indexOf("/* ================= 弹窗 ================= */");
if (start < 0 || end < 0) { console.error("extract markers not found"); process.exit(1); }
const chunk = src.slice(start, end);

const captured = [];
let destroyCount = 0;
class ChartStub {
  static defaults = { font: {} };
  constructor(canvas, config) { this.canvas = canvas; this.config = config; captured.push(this); }
  destroy() { destroyCount++; }
}

const els = {
  "#usage-charts": { hidden: true },
  "#usage-chart-trend": { id: "trend" },
  "#usage-chart-pie": { id: "pie" },
  "#usage-chart-calls": { id: "calls" },
};
const $ = sel => els[sel];
const getComputedStyle = () => ({ getPropertyValue: () => "", fontFamily: "" });
const document = { documentElement: {}, body: {} };

const factory = new Function("window", "$", "getComputedStyle", "document", "Chart",
  chunk + "\nreturn { renderUsageCharts, charts: () => usageCharts };");
const { renderUsageCharts, charts } = factory({ Chart: ChartStub }, $, getComputedStyle, document, ChartStub);

let pass = 0, fail = 0;
const check = (name, cond, extra) => {
  if (cond) { pass++; console.log("  ok  " + name); }
  else { fail++; console.log("FAIL  " + name + (extra ? "\n      → " + String(extra).slice(0, 400) : "")); }
};

/* 样例：3 天（乱序）、8 个模型（触发环形图「其他」合并） */
const M = (model, input, cached, output, calls) =>
  ({ model, input_tokens: input, cached_tokens: cached, output_tokens: output, total_tokens: input + output, calls });
const data = {
  days: 30, grand: {},
  items: [
    { date: "2026-09-05", models: [M("m1", 100, 20, 30, 2), M("m2", 50, 10, 5, 1)] },
    { date: "2026-08-22", models: [M("m1", 200, 60, 40, 3), M("m3", 10, 0, 1, 1)] },
    { date: "2026-08-25", models: Array.from({ length: 7 }, (_, i) => M("x" + i, 10, 0, 2, 4)) },
  ],
};

captured.length = 0; destroyCount = 0;
renderUsageCharts(data);
check("创建 3 张图表", captured.length === 3, captured.length);
check("图表区显示", els["#usage-charts"].hidden === false);

/* 每日趋势：日期升序、未命中 = 输入 - 缓存 */
const trend = captured.find(c => c.canvas.id === "trend");
check("趋势图日期升序", JSON.stringify(trend.config.data.labels) === '["08-22","08-25","09-05"]', trend.config.data.labels);
const ds = trend.config.data.datasets;
check("趋势图三组堆叠", ds.length === 3 && ds[0].label === "输入·未命中" && ds[1].label === "输入·缓存命中" && ds[2].label === "输出");
check("未命中 = 输入-缓存（按天求和）", JSON.stringify(ds[0].data) === JSON.stringify([150, 70, 120]), ds[0].data);
check("缓存命中求和", JSON.stringify(ds[1].data) === JSON.stringify([60, 0, 30]), ds[1].data);
check("输出求和", JSON.stringify(ds[2].data) === JSON.stringify([41, 14, 35]), ds[2].data);

/* 环形图：按 token 降序 Top7 + 其他 */
const pie = captured.find(c => c.canvas.id === "pie");
check("环形图 Top7 + 其他", pie.config.data.labels.length === 8 && pie.config.data.labels[7] === "其他"
  && pie.config.data.labels[0] === "m1", pie.config.data.labels);
check("其他 = 剩余模型合计（x5+x6+m3）", pie.config.data.datasets[0].data[7] === 35, pie.config.data.datasets[0].data);

/* 调用条形图：按调用次数降序 Top10（同次数按插入序稳定排序） */
const calls = captured.find(c => c.canvas.id === "calls");
check("调用图按次数降序", JSON.stringify(calls.config.data.labels) === JSON.stringify(["m1","x0","x1","x2","x3","x4","x5","x6","m2","m3"]),
  calls.config.data.labels.map((l, i) => l + ":" + calls.config.data.datasets[0].data[i]));
check("m1 调用次数 = 各天求和", calls.config.data.datasets[0].data[0] === 5);

/* tooltip 回调可执行（含百分比计算） */
const tooltipLabel = pie.config.options.plugins.tooltip.callbacks.label;
check("环形图 tooltip 回调", /tokens · .*% · \d+ 次/.test(tooltipLabel({ parsed: 330, dataIndex: 0 })), tooltipLabel({ parsed: 330, dataIndex: 0 }));

/* 重复渲染：先销毁旧实例 */
captured.length = 0;
renderUsageCharts(data);
check("重渲染前销毁旧实例", destroyCount === 3, destroyCount);

/* 空 data / Chart 缺失 → 图表区隐藏 */
renderUsageCharts(null);
check("无数据隐藏图表区", els["#usage-charts"].hidden === true);
const { renderUsageCharts: renderNoChart } = factory({ }, $, getComputedStyle, document, undefined);
els["#usage-charts"].hidden = false;
renderNoChart(data);
check("Chart 未加载（CDN 失败）隐藏图表区", els["#usage-charts"].hidden === true);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
