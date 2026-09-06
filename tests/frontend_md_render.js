/* 从 index.html 提取真实的 esc/md 实现，跑回归用例 */
const fs = require("fs");
const path = require("node:path");
const src = fs.readFileSync(path.join(__dirname, "..", "ra-web", "index.html"), "utf8");
const start = src.indexOf("const esc = s =>");
const end = src.indexOf("/* ================= 鉴权 ================= */");
if (start < 0 || end < 0) { console.error("extract markers not found"); process.exit(1); }
const chunk = src.slice(start, end);

const factory = new Function("window", "hljs", chunk + "\nreturn { md, esc, katexHTML };");
const rendered = [];
const fakeWindow = {
  katex: {
    renderToString: (tex, opts) => {
      rendered.push({ tex, display: !!(opts && opts.displayMode) });
      return '<span class="katex-stub" data-display="' + !!(opts && opts.displayMode) + '"></span>';
    },
  },
};
const { md } = factory(fakeWindow, undefined);   // hljs 缺失（CDN 失败场景）

let pass = 0, fail = 0;
function check(name, cond, extra) {
  if (cond) { pass++; console.log("  ok  " + name); }
  else { fail++; console.log("FAIL  " + name + (extra ? "\n      → " + String(extra).slice(0, 300) : "")); }
}
const texOf = i => rendered[i] && rendered[i].tex;

/* 1. 多行 display 公式：不再被 <br> 拆断，& 对齐符保真 */
rendered.length = 0;
let h = md("前文\n\n$$\n\\begin{aligned}\na &= b \\\\\nc &= d\n\\end{aligned}\n$$\n\n后文");
check("多行 aligned 渲染为 1 个 display 公式", rendered.length === 1 && rendered[0].display, h);
check("公式体保真（无 &amp; / <br>）", /a &= b/.test(texOf(0)) && !/&amp;/.test(texOf(0)) && !/<br/.test(texOf(0)), texOf(0));
check("前文/后文各自成段，公式独立成块", /<p>前文<\/p>/.test(h) && /katex-stub[^]*?<p>后文<\/p>/.test(h), h);
check("正文无残留 $$", !h.includes("$$"), h);

/* 2. \[ \] 与独立 \begin{align} 环境 */
rendered.length = 0;
h = md("\\[\nE = mc^2\n\\]");
check("\\[\\] display", rendered.length === 1 && rendered[0].display, h);
rendered.length = 0;
h = md("\\begin{align}\nx &= 1 \\\\\ny &= 2\n\\end{align}");
check("独立 align 环境 display", rendered.length === 1 && /x &= 1/.test(texOf(0)), h);
rendered.length = 0;
h = md("满足 \\(x > 0\\) 且 $y < 1$ 时成立");
check("\\(\\) 与 $ $ 行内公式", rendered.length === 2 && !rendered[0].display && !rendered[1].display, h);
check("行内公式后接中文", h.includes("时成立"), h);

/* 3. 货币 $ 与 \$ 字面量不误判 */
rendered.length = 0;
h = md("价格$5，成本$10 之间");
check("货币 $ 不误判为公式", rendered.length === 0, h);
h = md("花了 \\$5 and \\$10");
check("\\$ 字面量还原为 $", h.includes("$5 and $10") && !h.includes("\\u0001"), h);

/* 4. 行内 code / 围栏代码里的 $ 与公式语法不被处理 */
rendered.length = 0;
h = md("运行 `$HOME/bin/$USER` 查看");
check("行内 code 内 $ 不误判", rendered.length === 0 && h.includes("<code>$HOME/bin/$USER</code>"), h);
h = md("```\n$$x$$ \\alpha\n```");
check("围栏代码内公式不处理", rendered.length === 0 && h.includes("<pre><code>") && h.includes("\\alpha"), h);

/* 5. hljs 缺失时代码内容仍被转义（防 XSS） */
h = md("```html\n<script>alert(1)</script>\n```");
check("无 hljs 时代码转义", !h.includes("<script>alert") && h.includes("&lt;script&gt;"), h);
h = md("正文 <img src=x onerror=alert(1)> 注入");
check("正文 HTML 转义", !h.includes("<img src=x"), h);

/* 6. 表格：单元格内行内公式、code、加粗、对齐 */
rendered.length = 0;
h = md("| 符号 | 含义 | 值 |\n| --- | :-: | --: |\n| $x^2$ | **平方** | `v1` |");
check("表格渲染", h.includes("<table") && h.includes("md-table"), h);
check("单元格行内公式回填", rendered.length === 1 && /<td[^>]*><span class="katex-stub"/.test(h), h);
check("单元格加粗", /<th[^>]*>符号<\/th>/.test(h) && /<b>平方<\/b>/.test(h), h);
check("单元格对齐", h.includes("text-align:center") && h.includes("text-align:right"), h);

/* 7. 标题层级 + 内嵌格式 */
h = md("# 一级\n## **二级**标题\n### 三级\n#### 四级\n##### 五级");
check("标题渲染为 h1-h4", h.includes("<h1>一级</h1>") && h.includes("<h2><b>二级</b>标题</h2>")
  && h.includes("<h3>三级</h3>") && h.includes("<h4>四级</h4>") && !h.includes("<h5"), h);

/* 8. 列表：嵌套 / 有序 / 任务清单 / 混排 */
h = md("- 顶层一\n  - 子项\n  - 子项2\n- 顶层二\n1. 第一\n2. 第二");
check("嵌套无序列表", h.includes("<ul><li>顶层一<ul><li>子项</li><li>子项2</li></ul>") && h.includes("<li>顶层二</li>"), h);
check("有序列表与类型切换", h.includes("</ul><ol>") && h.includes("<li>第一</li><li>第二</li></ol>"), h);
h = md("3. 第三\n4. 第四");
check("有序列表起始编号", h.includes('<ol start="3">'), h);
h = md("- [ ] 待办\n- [x] 已完成");
check("任务清单", h.includes('class="task"') && h.includes("☐") && h.includes("☑") && h.includes("待办"), h);

/* 9. 引用块 / 分隔线 / 段内换行 */
h = md("> 引用一行\n> 第二行\n\n正文A\n正文B");
check("引用块", h.includes("<blockquote><p>引用一行<br>第二行</p></blockquote>"), h);
check("段内换行 <br>，跨段 <p>", h.includes("<p>正文A<br>正文B</p>"), h);
h = md("上\n\n---\n\n下");
check("分隔线", h.includes("<hr>"), h);

/* 10. 行内格式：粗体/斜体/粗斜体/删除线/链接/图片/自动链接 */
h = md("**粗** *斜* ***粗斜*** ~~删~~");
check("粗/斜/粗斜/删除线", h.includes("<b>粗</b> <i>斜</i> <b><i>粗斜</i></b> <del>删</del>"), h);
h = md("参见 [论文](https://arxiv.org/abs/1234.5678) 与 https://example.com/a。");
check("markdown 链接 + 裸链接", h.includes('<a href="https://arxiv.org/abs/1234.5678"') && h.includes(">https://example.com/a</a>。"), h);
h = md("图：![架构](https://example.com/a.png)");
check("图片", h.includes('<img class="md-img" src="https://example.com/a.png"'), h);
h = md("[危险](javascript:alert(1))");
check("javascript: 链接不渲染", !h.includes("<a "), h);

/* 11. 流式中间态：未闭合公式/代码不崩、不吞正文 */
rendered.length = 0;
h = md("推导如下\n\n$$\n\\begin{aligned}\nx &= 1");
check("未闭合 $$ 流式态保留原文", rendered.length === 0 && h.includes("x &amp;= 1") && h.includes("推导如下"), h);
h = md("```python\nprint('hi')");
check("未闭合代码围栏（流式态）", h.includes("<pre><code") && h.includes("print"), h);

/* 12. KaTeX 缺失时降级为原文 */
const factoryNoKatex = new Function("window", "hljs", chunk + "\nreturn { md };");
const { md: mdNoKatex } = factoryNoKatex({}, undefined);
h = mdNoKatex("质能方程 $E = mc^2$ 与\n\n$$\\int_0^1 x\\,dx$$");
check("无 KaTeX 时公式降级为原文", h.includes("E = mc^2") && h.includes("\\int_0^1") && !h.includes("katex"), h);

/* 13. 多个公式混排 + 段落结构（display 优先提取，按回填位置断言） */
rendered.length = 0;
h = md("已知 $a$ 与 $b$。\n\n$$c = d$$\n\n又见 \\(e\\)。");
const stubs = (h.match(/katex-stub/g) || []).length;
check("混排公式计数", rendered.length === 4 && stubs === 4, h + " | rendered=" + JSON.stringify(rendered));
check("display 判定：仅 c=d 为 display", rendered.filter(r => r.display).length === 1
  && rendered.some(r => r.tex === "c = d" && r.display), JSON.stringify(rendered));
check("公式回填位置正确", /<p>已知 <span class="katex-stub"[^>]*><\/span> 与 <span class="katex-stub"[^>]*><\/span>。<\/p><span class="katex-stub" data-display="true"><\/span><p>又见 <span/.test(h), h);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
