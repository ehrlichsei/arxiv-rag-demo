#!/usr/bin/env python3
"""Small, local-first arXiv abstract RAG demo (Python standard library only)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


HERE = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("RAG_DB", HERE / "arxiv_rag.sqlite3"))
ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "http://www.w3.org/2005/Atom"
ARXIV = "http://arxiv.org/schemas/atom"
DEFAULT_QUERY = "cat:cs.AI OR cat:cs.CL OR cat:cs.LG"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "qwen3:8b")
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 180


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY, title TEXT NOT NULL, authors TEXT NOT NULL,
            published TEXT NOT NULL, categories TEXT NOT NULL, abstract TEXT NOT NULL,
            url TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT NOT NULL REFERENCES papers(paper_id),
            part INTEGER NOT NULL, content TEXT NOT NULL, embedding TEXT NOT NULL,
            UNIQUE(paper_id, part)
        );
        CREATE INDEX IF NOT EXISTS chunks_paper_id ON chunks(paper_id);
        """
    )
    return db


def http_json(url: str, payload: dict, timeout: int = 180) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"无法连接 {url}：{exc}。请确认 Ollama 已安装并正在运行。"
        ) from exc


def embed(texts: list[str]) -> list[list[float]]:
    result = http_json(
        f"{OLLAMA_URL}/api/embed",
        {"model": EMBED_MODEL, "input": texts, "truncate": True},
    )
    vectors = result.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise RuntimeError("Ollama 返回的 embedding 数量与输入不一致。")
    return vectors


def split_text(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + CHUNK_CHARS // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def fetch_page(query: str, start: int, count: int) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": start,
            "max_results": count,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    req = urllib.request.Request(
        f"{ARXIV_API}?{params}",
        headers={"User-Agent": "arxiv-rag-personal-demo/0.1 (metadata and abstracts)"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                root = ET.fromstring(response.read())
            papers = []
            for entry in root.findall(f"{{{ATOM}}}entry"):
                raw_id = (entry.findtext(f"{{{ATOM}}}id") or "").strip()
                paper_id = raw_id.rstrip("/").split("/abs/")[-1]
                if not paper_id:
                    continue
                authors = [
                    a.findtext(f"{{{ATOM}}}name", default="").strip()
                    for a in entry.findall(f"{{{ATOM}}}author")
                ]
                categories = [
                    node.attrib.get("term", "")
                    for node in entry.findall(f"{{{ARXIV}}}primary_category")
                ]
                categories.extend(
                    node.attrib.get("term", "")
                    for node in entry.findall(f"{{{ATOM}}}category")
                    if node.attrib.get("term") not in categories
                )
                papers.append(
                    {
                        "paper_id": paper_id,
                        "title": " ".join((entry.findtext(f"{{{ATOM}}}title") or "").split()),
                        "authors": ", ".join(authors),
                        "published": (entry.findtext(f"{{{ATOM}}}published") or "")[:10],
                        "categories": ", ".join(filter(None, categories)),
                        "abstract": " ".join((entry.findtext(f"{{{ATOM}}}summary") or "").split()),
                        "url": f"https://arxiv.org/abs/{paper_id}",
                    }
                )
            return papers
        except urllib.error.HTTPError as exc:
            if attempt == 3:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(
                    f"arXiv API 返回 HTTP {exc.code}：{detail or exc.reason}"
                ) from exc
            retry_after = exc.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = max(10, int(retry_after))
            elif exc.code == 429:
                delay = 30 * (2**attempt)
            else:
                delay = 5 * (attempt + 1)
            print(f"arXiv API 返回 HTTP {exc.code}，等待 {delay} 秒后重试…", file=sys.stderr)
            time.sleep(delay)
        except (urllib.error.URLError, ET.ParseError) as exc:
            if attempt == 3:
                raise RuntimeError(f"arXiv API 请求失败：{exc}") from exc
            time.sleep(5 * (attempt + 1))
    return []


def ingest(max_papers: int, query: str, batch_size: int) -> None:
    db = connect()
    added = 0
    start = 0
    page_size = min(2000, max_papers)
    try:
        while added < max_papers:
            count = min(page_size, max_papers - added)
            page = fetch_page(query, start, count)
            if not page:
                break
            start += len(page)
            new = []
            for paper in page:
                exists = db.execute(
                    "SELECT 1 FROM papers WHERE paper_id=?", (paper["paper_id"],)
                ).fetchone()
                if not exists and paper["abstract"]:
                    new.append(paper)
            for offset in range(0, len(new), batch_size):
                batch = new[offset : offset + batch_size]
                pending: list[tuple[dict, int, str]] = []
                for paper in batch:
                    for part, content in enumerate(split_text(paper["abstract"])):
                        pending.append((paper, part, content))
                vectors: list[list[float]] = []
                for offset2 in range(0, len(pending), 32):
                    vectors.extend(
                        embed([row[2] for row in pending[offset2 : offset2 + 32]])
                    )
                with db:
                    for paper in batch:
                        db.execute(
                            """INSERT OR IGNORE INTO papers
                            (paper_id,title,authors,published,categories,abstract,url)
                            VALUES (:paper_id,:title,:authors,:published,:categories,:abstract,:url)""",
                            paper,
                        )
                    for (paper, part, content), vector in zip(pending, vectors):
                        db.execute(
                            "INSERT OR IGNORE INTO chunks(paper_id,part,content,embedding) VALUES(?,?,?,?)",
                            (paper["paper_id"], part, content, json.dumps(vector)),
                        )
                added += len(batch)
                print(f"已入库 {added}/{max_papers} 篇（当前查询偏移 {start}）")
            if len(page) < count:
                break
            time.sleep(3)
    finally:
        db.close()
    print(f"完成：本次新增 {added} 篇；数据库：{DB_PATH}")


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def retrieve(question: str, top_k: int) -> list[dict]:
    query_vector = embed([question])[0]
    db = connect()
    try:
        rows = db.execute(
            """SELECT c.content,c.embedding,p.paper_id,p.title,p.authors,p.published,
                      p.categories,p.url
               FROM chunks c JOIN papers p USING(paper_id)"""
        ).fetchall()
    finally:
        db.close()
    ranked = sorted(
        (
            (cosine(query_vector, json.loads(row["embedding"])), dict(row))
            for row in rows
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    return [dict(score=score, **row) for score, row in ranked[:top_k]]


def ask(question: str, top_k: int) -> None:
    matches = retrieve(question, top_k)
    if not matches:
        raise RuntimeError("向量库还没有论文。请先运行 ingest 命令。")
    context = "\n\n".join(
        f"[论文 {i}] {m['title']} ({m['published']})\n"
        f"作者：{m['authors']}；分类：{m['categories']}\n摘要：{m['content']}\n链接：{m['url']}"
        for i, m in enumerate(matches, 1)
    )
    prompt = (
        "请只根据下面提供的论文摘要回答问题。引用论文时使用 [论文 1] 这样的编号。"
        "如果摘要不足以支持结论，请明确说明证据不足；不要编造论文结论。\n\n"
        f"问题：{question}\n\n检索到的论文：\n{context}"
    )
    result = http_json(
        f"{OLLAMA_URL}/api/chat",
        {
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=300,
    )
    print("\n回答\n" + result.get("message", {}).get("content", "（模型未返回回答）"))
    print("\n检索来源")
    for i, match in enumerate(matches, 1):
        print(f"[{i}] {match['title']} — {match['published']} — {match['url']} (相似度 {match['score']:.3f})")


def status() -> None:
    if not DB_PATH.exists():
        print(f"数据库尚未创建：{DB_PATH}")
        return
    db = connect()
    try:
        papers = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        db.close()
    print(f"论文：{papers}\n文本块：{chunks}\n数据库：{DB_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description="arXiv 论文摘要本地 RAG")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest_parser = sub.add_parser("ingest", help="抓取 arXiv 元数据和摘要并建立向量库")
    ingest_parser.add_argument("--max-papers", type=int, default=3000)
    ingest_parser.add_argument("--query", default=DEFAULT_QUERY)
    ingest_parser.add_argument("--batch-size", type=int, default=32)
    ask_parser = sub.add_parser("ask", help="检索论文并让本地 LLM 回答")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=6)
    sub.add_parser("status", help="查看本地向量库规模")
    args = parser.parse_args()
    try:
        if args.command == "ingest":
            if args.max_papers < 1 or args.batch_size < 1:
                parser.error("--max-papers 和 --batch-size 必须大于 0")
            ingest(args.max_papers, args.query, args.batch_size)
        elif args.command == "ask":
            ask(args.question, args.top_k)
        else:
            status()
        return 0
    except (RuntimeError, urllib.error.URLError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
