# 单向量检索 vs 混合检索：详细测试报告

## 1. 实验目的

本项目知识库支持两种检索方式：

1. **单向量检索（vector）**
2. **混合检索（hybrid）**

本实验的目的是：

- 用可复现的测试用例比较两种检索方式的效果。
- 解释测试指标的含义。
- 给出当前知识库配置下应该选择哪种检索方式的建议。
- 保留测试会话记录，方便后续复查。

---

## 2. 测试环境

- 服务地址：`http://127.0.0.1:8000`
- 测试用户：`pyp`
- 知识库：
  - kb_id：`ad4a296fda7c`
  - 名称：密码学-神经区分器
  - 文档数：81 个 PDF
  - 片段数：7342 chunks
- 嵌入配置：
  - provider：`llama.cpp`
  - model：`qwen3-embedding:8b`
  - base_url：`http://127.0.0.1:18080/v1`
- LLM 配置：
  - 使用 pyp 的用户级配置
  - provider：`custom`
  - base_url：`https://opencode.ai/zen/go/v1`
  - model：`deepseek-v4-flash`

---

## 3. 概念讲解

### 3.1 单向量检索（vector）

把用户问题和知识库片段都转换成向量，然后找语义上最接近的片段。

- 优点：能处理同义词、 paraphrase、语义相关的问题。
- 缺点：对精确缩写、论文名、专有名词可能不够敏感。

### 3.2 混合检索（hybrid）

同时做向量检索和 BM25 关键词检索，再用 RRF 合并结果。

- 优点：既保留语义匹配，又能用关键词精确命中。
- 缺点：实现更复杂，可能混入一些只看关键词但不那么相关的片段。

### 3.3 BM25

一种经典的关键词检索算法。

- 根据词频、文档长度、稀有程度计算相关性。
- 适合精确匹配缩写、论文名、术语。

### 3.4 RRF（Reciprocal Rank Fusion）

混合检索把两路排名合并的方法。

- 每一路结果中，排名越靠前，贡献分越高。
- 公式思想：`得分 += 1 / (60 + 排名)`。
- 最终按融合分从高到低排序。

### 3.5 Precision@k（精确率@k）

返回的前 k 个结果里，有多少比例是真正相关的。

- 公式：`Precision@k = 前 k 个结果中相关结果数 / k`
- 例子：`Precision@5 = 0.4` 表示返回 5 条里只有 2 条是相关的。
- 它衡量“返回结果干不干净”。

### 3.6 Recall@k（召回率@k）

所有应该被找到的相关文档里，前 k 个结果找回了多少比例。

- 公式：`Recall@k = 前 k 个结果中相关结果数 / 全部相关文档数`
- 例子：某个问题有 3 篇相关论文，前 5 条结果找回了其中 1 篇，则 `Recall@5 = 1/3 ≈ 0.33`。
- 它衡量“有没有漏掉重要资料”。

### 3.7 MRR（Mean Reciprocal Rank，平均倒数排名）

第一个正确答案出现在第几位。

- 排第 1：得 1 分
- 排第 2：得 1/2
- 排第 3：得 1/3
- MRR 是所有测试问题的平均值。
- 它衡量“用户能不能很快看到第一个正确答案”。

### 3.8 金标准（Ground Truth）

为了计算 Precision/Recall/MRR，需要先人工确定“哪些文档对这个问题是相关的”。

- 这些人工标注的正确答案集合就叫“金标准”。
- 本次测试按“文档级”判断：只要某个 PDF 被检索到，就算该文档相关；同一文档的多个片段只算一次。

---

## 4. 实验步骤

### 4.1 第一步：阅读项目代码

确认检索相关实现：

- `app/services/kb_service.py` 中的 `search()` 方法。
- `app/abstractions/vectorstore.py`：Chroma 向量检索。
- `app/abstractions/bm25.py`：BM25 和 RRF 融合。
- `app/api/kbs.py`：检索测试接口。
- `app/api/chat.py`：对话接口，支持 `retrieval_mode` 参数。

### 4.2 第二步：确认配置

- 确认知识库嵌入配置和向量库实际写入模型一致。
- 确认 pyp 的用户级 LLM 配置存在，并在对话测试中使用该配置，避免系统默认模型限流。

### 4.3 第三步：离线检索测试

使用接口：

```text
GET /api/v1/kbs/ad4a296fda7c/search?query=...&k=10&mode=vector
GET /api/v1/kbs/ad4a296fda7c/search?query=...&k=10&mode=hybrid
```

对每组测试用例分别用两种模式检索，记录：

- 返回的前 10 个文档。
- 每个结果的 `method`（vector / bm25 / hybrid）。
- 每个结果的距离、BM25 分数、RRF 融合分。

然后按金标准计算：

- Precision@5
- Precision@10
- Recall@5
- Recall@10
- MRR

### 4.4 第四步：对话测试

建立两个持久化会话：

- 纯向量：`eval2-vector-20260816`
- 混合检索：`eval2-hybrid-20260816`

两个会话发送完全相同的 6 个问题，参数统一为：

```text
retrieval_mode = vector 或 hybrid
per_kb_k = 5
total_k = 5
parent_groups = 0
temperature = 0
user_id = pyp 的内部用户 ID
```

对话接口会返回：

- 最终回答。
- 实际用于回答的检索片段。

测试结束后保留会话记录，不删除。

### 4.5 第五步：汇总结果并给出结论

- 对比两种模式的平均指标。
- 对比具体用例中的检索来源。
- 给出建议。

---

## 5. 测试用例

### 5.1 离线检索用例

以下 13 组问题用于离线检索评测，`gold` 列是人工标注的强相关文档。

| 编号 | 测试问题 | gold 相关文档 |
|---|---|---|
| 1 | SMS4 线性密码分析 | 2023Yu-SM4.pdf、2023Wang-SM4.pdf、2022DiffAttacks.pdf |
| 2 | Gohr 神经区分器 2019 | 2019Gohr.pdf |
| 3 | HDND 是什么 | HDND.pdf |
| 4 | ASCON 算法 神经网络区分器 | 2023Wang-LW.pdf、2023Baksi.pdf、2023Bao-MI.pdf |
| 5 | SPECK 差分攻击 | 2024Chen-SPECK.pdf、2024Yuan-Simeck.pdf |
| 6 | SIMON 矩形攻击 | 2025Sun.pdf、2024Yuan-Simeck.pdf |
| 7 | 残差网络 密码分析 | 2022ARX.pdf、2023Shan.pdf、2023Chen-LW.pdf |
| 8 | 轻量级分组密码 深度学习 | 2023Chen-LW.pdf、2023Shan.pdf、2024Chen-DL.pdf |
| 9 | Gohr's neural distinguisher | 2019Gohr.pdf |
| 10 | Biryukov 相关密钥 | 2025RelatedKey.pdf、2024Wang Related-Key.pdf |
| 11 | SALSA 密码 | 2024Bellini-SALSA.pdf |
| 12 | Gimli 置换 | 2023Wang-LW.pdf |
| 13 | CPDI-ND 是什么 | CPDI-ND.docx |

### 5.2 对话测试用例

以下 6 个问题用于两个持久化会话：

1. 请检索知识库回答：SMS4 的线性密码分析主要有哪些方法？
2. 请检索知识库回答：HDND 是什么？
3. 请检索知识库回答：Gohr 的神经区分器方法是什么？
4. 请检索知识库回答：SPECK 算法的差分攻击有哪些研究？
5. 请检索知识库回答：残差网络在密码分析中如何应用？
6. 请检索知识库回答：CPDI-ND 是什么？

---

## 6. 离线检索结果

### 6.1 平均指标

| 模式 | P@5 | R@5 | P@10 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| vector | 0.162 | 0.346 | 0.162 | 0.372 | 0.487 |
| hybrid | 0.160 | 0.397 | 0.161 | 0.423 | 0.515 |

解读：

- hybrid 的 Recall@5/10 更高：更不容易漏掉相关文档。
- hybrid 的 MRR 更高：第一个正确答案平均更靠前。
- vector 的 Precision 略高：返回结果更干净，但漏召回更多。

### 6.2 分用例明细

| 问题 | 模式 | P@5 | R@5 | P@10 | R@10 | MRR |
|---|---|---:|---:|---:|---:|---:|
| SMS4 线性密码分析 | vector | 0.600 | 1.000 | 0.600 | 1.000 | 1.000 |
| SMS4 线性密码分析 | hybrid | 0.400 | 0.667 | 0.500 | 1.000 | 1.000 |
| Gohr 神经区分器 2019 | vector | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Gohr 神经区分器 2019 | hybrid | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| HDND 是什么 | vector | 0.200 | 1.000 | 0.125 | 1.000 | 0.333 |
| HDND 是什么 | hybrid | 0.200 | 1.000 | 0.200 | 1.000 | 1.000 |
| ASCON 算法 神经网络区分器 | vector | 0.500 | 0.333 | 0.500 | 0.333 | 1.000 |
| ASCON 算法 神经网络区分器 | hybrid | 0.250 | 0.333 | 0.250 | 0.333 | 1.000 |
| SPECK 差分攻击 | vector | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| SPECK 差分攻击 | hybrid | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| SIMON 矩形攻击 | vector | 0.200 | 0.500 | 0.200 | 0.500 | 1.000 |
| SIMON 矩形攻击 | hybrid | 0.250 | 0.500 | 0.250 | 0.500 | 1.000 |
| 残差网络 密码分析 | vector | 0.200 | 0.333 | 0.286 | 0.667 | 1.000 |
| 残差网络 密码分析 | hybrid | 0.200 | 0.333 | 0.167 | 0.333 | 1.000 |
| 轻量级分组密码 深度学习 | vector | 0.200 | 0.333 | 0.200 | 0.333 | 1.000 |
| 轻量级分组密码 深度学习 | hybrid | 0.250 | 0.333 | 0.250 | 0.333 | 0.500 |
| Gohr's neural distinguisher | vector | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Gohr's neural distinguisher | hybrid | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Biryukov 相关密钥 | vector | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Biryukov 相关密钥 | hybrid | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| SALSA 密码 | vector | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| SALSA 密码 | hybrid | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Gimli 置换 | vector | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Gimli 置换 | hybrid | 0.200 | 1.000 | 0.143 | 1.000 | 0.200 |
| CPDI-ND 是什么 | vector | 0.200 | 1.000 | 0.200 | 1.000 | 1.000 |
| CPDI-ND 是什么 | hybrid | 0.333 | 1.000 | 0.333 | 1.000 | 1.000 |

说明：

- 有些问题两种模式都没有在 top10 中找到金标准文档，说明当前检索仍有提升空间，也可能是我的人工金标准范围偏严。
- 在 `HDND`、`Gimli` 等问题上，hybrid 通过 BM25 能把精确关键词文档排到更前。
- 在 `CPDI-ND` 用例中，两种模式都能稳定召回 `CPDI-ND.docx`；hybrid 的 Precision@5 更高（0.333 vs 0.200），因为它的前 5 条结果里 CPDI-ND.docx 的占比更高。

---

## 7. 对话测试结果

### 7.1 会话信息

| 模式 | 会话 ID |
|---|---|
| 纯向量 | `eval2-vector-20260816` |
| 混合检索 | `eval2-hybrid-20260816` |

### 7.2 检索来源对比

| 问题 | vector 检索到的来源 | hybrid 检索到的来源 |
|---|---|---|
| SMS4 线性密码分析 | 2023Yu-SM4.pdf、2023Wang-LW.pdf、2023Shan.pdf | 2023Yu-SM4.pdf、2023Wang-LW.pdf |
| HDND 是什么 | HDND.pdf、2026Mirzaali.pdf | HDND.pdf |
| Gohr 神经区分器方法 | 2024Li-LBlock.pdf、2025Yuan-Rethink.pdf、2024Chen-DL.pdf、2025Shen-MiF.pdf、2024Wang.pdf | 2024Li-LBlock.pdf、2025Shen-MiF.pdf、2025Yuan-Rethink.pdf、2023KeyRecovery.pdf、2024Chen-DL.pdf |
| SPECK 差分攻击 | 2024Yue.pdf、2023Yue.pdf、2025Shen-MiF.pdf、2023Shan.pdf、2024Chen-SPECK.pdf | 2024Yue.pdf、2023Yu-SM4.pdf、2023Yue.pdf、2024Chen-DL.pdf、2025Shen-MiF.pdf |
| 残差网络在密码分析中应用 | 2024Yang-XDU.pdf、2023Wang-LW.pdf、2025Li-XDU.pdf、2023Tcydenova.pdf、2021Polytope.pdf | 2024Yang-XDU.pdf、2023Wang-LW.pdf、2023Yu-SM4.pdf、2025Li-XDU.pdf |
| CPDI-ND 是什么 | CPDI-ND.docx | CPDI-ND.docx |

从对话检索看：

- hybrid 的返回结果中经常带有 `method=bm25` 的片段，说明它确实把关键词命中的内容带进来了。
- vector 的返回结果全部是 `method=vector`，没有关键词补充。
- 两种模式在部分问题上都能检索到核心文档，但 hybrid 在 `HDND` 上更集中，全部命中 `HDND.pdf`。

---

## 8. 结论

在当前查询配置和知识库下：

1. **混合检索整体更好。**
   - 召回率更高，不容易漏掉相关资料。
   - MRR 更高，第一个正确答案平均更靠前。
   - 能通过 BM25 补充纯向量容易漏掉的精确缩写和专有名词。

2. **单向量检索的精确率略高。**
   - 返回结果更“干净”，但会漏掉一些关键词强相关的片段。
   - 如果知识库规模小、问题都是语义型，可能够用；但在这种论文/术语密集的科研知识库中，漏召回风险更大。

3. **建议继续使用 hybrid 模式。**

---

## 9. 测试产物

- `ra-agent/eval_retrieval_results.json`：第一轮离线检索结果
- `ra-agent/eval_retrieval_results2.json`：本轮离线检索结果
- `ra-agent/eval_retrieval_chat_results.json`：第一轮对话结果
- `ra-agent/eval_retrieval_chat_results2.json`：本轮对话结果
- `ra-agent/docs/retrieval_eval_results2.json`：本轮离线检索结果副本
- `ra-agent/docs/retrieval_eval_chat_results2.json`：本轮对话结果副本
- `ra-agent/docs/retrieval_eval_report.md`：本报告

所有测试会话均保留在服务端，未删除。
