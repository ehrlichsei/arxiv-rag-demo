# AI 论文 RAG

这个项目会直接从 arXiv API 收集 AI 论文的标题、作者、日期、分类和摘要，用本地 embedding 模型建立向量库。提问时先从论文摘要中检索，再交给本地聊天模型回答，并显示匹配论文和链接。

默认范围是 `cs.AI`、`cs.CL`、`cs.LG`，覆盖人工智能、计算语言学和机器学习。默认每次收集最新的 3,000 篇；可调到 10,000 或更多。arXiv 有数十万篇以上的相关论文，建议先用几千篇跑通完整流程，再逐渐扩大。下载的是元数据和摘要，不是论文全文，因此适合问研究方向、论文主题和摘要中明确写出的贡献；实验细节、具体结果需要全文语料。

论文内容和版权归各自作者或权利人所有。此 Demo 仅收集元数据和摘要供个人检索学习，不能据此假设论文全文可自由再发布或用于任意训练用途。

## 准备

## 在 Windows 主机运行

先安装 [Ollama](https://ollama.com/)，启动后在 PowerShell 下载模型：

```bash
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

`nomic-embed-text` 负责语义向量，`qwen3:8b` 负责依据检索到的摘要回答。可用环境变量 `RAG_EMBED_MODEL`、`RAG_CHAT_MODEL` 指定其他已下载模型；Ollama 在其他机器时设置 `OLLAMA_URL` 指向其 API 地址。

## 收集论文并提问

在本目录打开 PowerShell 运行：

```bash
py rag_demo.py ingest --max-papers 3000
py rag_demo.py ask "RAG 系统中有哪些方法可以减少检索错误？"
```

扩展到一万篇：

```bash
py rag_demo.py ingest --max-papers 10000
```

也可以自定义 arXiv 查询式，例如只收集语言模型相关论文：

```bash
py rag_demo.py ingest --query 'cat:cs.CL AND (ti:language model OR abs:language model)' --max-papers 5000
```

查看库规模：

```bash
py rag_demo.py status
```

数据保存在本地 `arxiv_rag.sqlite3`。重复运行会跳过已存论文并补充新论文。采集器每页最多请求 2,000 条，并在分页之间等待 3 秒。[arXiv API 手册](https://github.com/arXiv/arxiv-docs/blob/develop/source/help/api/user-manual.md)

## Demo 的 RAG 流程

`arXiv API → 元数据和摘要 → 文本分块 → embedding → SQLite 向量存储 → 相似度检索 → LLM 根据检索到的摘要回答`。

向量检索目前用纯 Python 计算余弦相似度，方便理解流程；几千到一万篇论文可以用于入门实验，但每次提问都需要扫描本地向量。语料更大时，可以将向量索引换成 FAISS、Qdrant 等专用方案。这个项目不会训练或修改模型权重，知识更新来自新增论文和回答时的检索。当前语料只有摘要，不适合回答摘要没有写出的实验细节。
