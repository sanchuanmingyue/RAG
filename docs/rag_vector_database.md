# RAG 向量数据库学习笔记

这份笔记配合项目里的 `backend/vector_store.py` 和 Streamlit 页面 `pages/1_向量数据库学习.py` 阅读。

## 1. 向量数据库存什么

在这个项目里，Chroma 每条记录对应一个论文 chunk。

一条记录包含四类信息：

```text
id         唯一主键，例如 DQN_2015_page_3_chunk_2
document   chunk 原文
embedding  document 对应的向量
metadata   paper_id、file_name、page、chunk_id
```

RAG 里非常重要的一点是：向量数据库不只存向量，也要存来源信息。

如果不存 `page`，大模型回答时就无法告诉用户依据来自哪一页。

## 2. 入库流程

项目里的入库流程是：

```text
PDF
→ 按页解析
→ 文本切分为 chunk
→ 每个 chunk 调 embedding 模型
→ 写入 Chroma
```

对应代码在：

```text
backend/pdf_loader.py
backend/text_splitter.py
backend/embeddings.py
backend/vector_store.py
```

关键方法：

```python
vector_store.index_chunks(chunks, embedding_client)
```

它最终会调用：

```python
collection.upsert(
    ids=[...],
    documents=[...],
    embeddings=[...],
    metadatas=[...],
)
```

`upsert` 表示：如果 id 已存在就更新，不存在就插入。

## 3. 检索流程

用户提问时，系统不会直接把问题发给大模型。

真正流程是：

```text
用户问题
→ 问题转 embedding
→ 在 Chroma 中找最近的 top_k 个 chunk
→ 把 chunk 拼进 Prompt
→ LLM 基于 chunk 回答
```

关键方法：

```python
vector_store.query(
    query_text="这篇论文用了什么数据集？",
    embedding_client=llm_client,
    top_k=5,
    paper_id="DQN_2015",
)
```

## 4. top_k 是什么

`top_k=5` 表示取最相关的前 5 个 chunk。

常见问题：

```text
top_k 太小：可能漏掉关键证据
top_k 太大：上下文变长，噪声变多，成本变高
```

初学建议：

```text
问答：top_k = 4~6
总结：top_k = 10~15
多论文对比：先生成每篇论文卡片，再比较卡片
```

## 5. metadata filtering 是什么

metadata filtering 是按元数据过滤检索范围。

例如只在某一篇论文中查：

```python
where = {"paper_id": "DQN_2015"}
```

这很重要。

如果你上传了 10 篇论文，用户问“这篇论文的方法是什么”，但没有过滤 `paper_id`，系统可能从其他论文里检索到相似片段，导致回答混乱。

## 6. distance 怎么看

Chroma 检索结果里有 `distance`。

简单理解：

```text
distance 越小，通常越相关
```

但不要死记某个固定阈值，因为它受这些因素影响：

```text
embedding 模型
向量距离算法
chunk 长度
文本语言
问题写法
```

更可靠的学习方式是打开“向量数据库学习”页面，输入问题，观察 top 结果的文本是否真的相关。

## 7. 初学时最容易踩的坑

### chunk 切得太大

问题：

```text
一个 chunk 里混入太多主题，检索命中后上下文不够精准。
```

解决：

```text
chunk_size 先用 500~800
chunk_overlap 先用 100~150
```

### chunk 切得太小

问题：

```text
单个 chunk 缺少上下文，大模型读不懂。
```

解决：

```text
保留 overlap
必要时按段落或章节切分
```

### metadata 不完整

问题：

```text
回答无法显示来源页码，也无法按论文过滤。
```

解决：

```text
每个 chunk 至少保存 paper_id、file_name、page、chunk_id。
```

### 只看大模型回答，不看检索结果

问题：

```text
你不知道错在检索，还是错在生成。
```

解决：

```text
先看检索到的 chunk，再看 LLM 回答。
```

## 8. 你可以怎么练习

1. 上传一篇 PDF，建立索引。
2. 打开“向量数据库学习”页面。
3. 输入问题：“这篇论文用了哪些数据集？”
4. 观察 top 5 的 chunk 是否包含 dataset、experiment、evaluation 等信息。
5. 改成 top_k=2、top_k=8，对比结果变化。
6. 回主页再让 LLM 回答，判断回答是否真的来自检索片段。

