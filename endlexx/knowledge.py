import re
from collections import defaultdict

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import delete, select

from .config import settings
from .db import Chunk, Document, Session, record
from .providers import runtime_call


def tokenize(value: str) -> list[str]:
    # Overlapping Han bigrams preserve names and short quotes not in jieba's lexicon.
    words = [part.lower() for part in jieba.lcut(value) if re.search(r"\w", part)]
    for segment in re.findall(r"[\u4e00-\u9fff]{2,}", value):
        words.extend(segment[i : i + 2] for i in range(len(segment) - 1))
    return words


def split_document(value: str, limit: int = 700) -> list[str]:
    blocks = re.split(r"\n\s*\n", value.strip())
    chunks, current = [], ""
    for block in blocks:
        pieces = [block[i : i + limit] for i in range(0, len(block), limit)] or [""]
        for piece in pieces:
            if len(current) + len(piece) > limit and current:
                chunks.append(current.strip())
                current = ""
            current += ("\n\n" if current else "") + piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def validate_vectors(vectors, count):
    matrix = np.asarray(vectors, dtype=float)
    if matrix.shape != (count, 1024) or not np.isfinite(matrix).all():
        raise ValueError("向量响应必须与输入逐项对应，且为有限的 1024 维向量")
    return vectors


def index_document(document_id: str) -> dict:
    with Session() as db:
        doc = db.get(Document, document_id)
        if doc is None:
            raise ValueError("文档不存在")
        texts = split_document(doc.content)
        if not texts:
            raise ValueError("文档为空")
        try:
            embedded = runtime_call("/embed", {"texts": texts})
            vectors, model = embedded["vectors"], embedded["model"]
            validate_vectors(vectors, len(texts))
            warning = None
        except Exception:
            vectors, model = [None] * len(texts), None
            warning = "向量模型尚不可用，文档已建立 BM25 索引；模型就绪后可重新索引"
        db.execute(delete(Chunk).where(Chunk.document_id == doc.id))
        db.add_all(
            [
                Chunk(
                    document_id=doc.id,
                    character_id=doc.character_id,
                    ordinal=i,
                    content=text,
                    embedding=vector,
                    embedding_model=model,
                )
                for i, (text, vector) in enumerate(zip(texts, vectors))
            ]
        )
        db.commit()
        return {"chunks": len(texts), "vector_indexed": model is not None, "warning": warning}


def retrieve(
    character_id: str | None,
    query: str,
    version: str | None = None,
    limit: int = 5,
    strategy: str = "hybrid_rerank",
) -> dict:
    if strategy not in {"bm25", "dense", "hybrid", "hybrid_rerank"}:
        raise ValueError("未知检索策略")
    if not character_id:
        return {"citations": [], "method": "无角色范围", "warnings": []}
    with Session() as db:
        statement = select(Chunk, Document).join(Document).where(Chunk.character_id == character_id)
        if version:
            statement = statement.where(Document.version.in_([version, "默认"]))
        rows = db.execute(statement).all()
        if not rows:
            return {"citations": [], "method": "空知识库", "warnings": []}
        tokens = [tokenize(chunk.content) for chunk, _ in rows]
        bm25 = BM25Okapi(tokens, epsilon=0.25)
        lexical_scores = bm25.get_scores(tokenize(query))
        # BM25 can be zero/negative for tiny corpora; exact token overlap is still useful.
        query_tokens = set(tokenize(query))
        lexical = sorted(
            range(len(rows)),
            key=lambda i: float(lexical_scores[i]) + len(query_tokens.intersection(tokens[i])) / 50,
            reverse=True,
        )
        lexical = [i for i in lexical[:20] if query_tokens.intersection(tokens[i])]
        rankings, warnings = ([] if strategy == "dense" else [lexical]), []
        dense_scores = {}
        if strategy != "bm25":
            try:
                vector = validate_vectors(runtime_call("/embed", {"texts": [query]})["vectors"], 1)[0]
                candidates = [
                    i
                    for i, (chunk, _) in enumerate(rows)
                    if chunk.embedding is not None and chunk.embedding_model == settings().embedding_model
                ]
                if db.bind.dialect.name == "postgresql":
                    dense_statement = (
                        select(Chunk.id, Chunk.embedding.cosine_distance(vector))
                        .join(Document)
                        .where(
                            Chunk.character_id == character_id,
                            Chunk.embedding_model == settings().embedding_model,
                            Chunk.embedding.is_not(None),
                        )
                    )
                    if version:
                        dense_statement = dense_statement.where(Document.version.in_([version, "默认"]))
                    distances = db.execute(
                        dense_statement.order_by(Chunk.embedding.cosine_distance(vector)).limit(20)
                    ).all()
                    by_id = {chunk.id: i for i, (chunk, _) in enumerate(rows)}
                    dense_scores = {
                        by_id[cid]: 1 - float(distance) for cid, distance in distances if cid in by_id
                    }
                else:
                    dense_scores = {
                        i: float(np.dot(np.asarray(rows[i][0].embedding), vector)) for i in candidates
                    }
                dense = [
                    i
                    for i in sorted(dense_scores, key=dense_scores.get, reverse=True)[:20]
                    if dense_scores[i] > 0.35
                ]
                rankings.append(dense)
            except Exception:
                warnings.append("向量召回暂不可用" + ("" if strategy == "dense" else "，本次使用 BM25"))
        fused = defaultdict(float)
        for ranking in rankings:
            for rank, i in enumerate(ranking):
                fused[i] += 1 / (60 + rank + 1)
        selected = sorted(fused, key=fused.get, reverse=True)[:12]
        method = {
            "bm25": "BM25",
            "dense": "向量",
            "hybrid": "BM25 + 向量 + RRF",
            "hybrid_rerank": "BM25 + 向量 + RRF",
        }[strategy]
        if strategy in {"hybrid", "hybrid_rerank"} and len(rankings) != 2:
            method = "BM25"
        if selected and strategy == "hybrid_rerank":
            try:
                scores = runtime_call(
                    "/rerank", {"query": query, "texts": [rows[i][0].content for i in selected]}
                )["scores"]
                if len(scores) != len(selected) or not np.isfinite(np.asarray(scores, dtype=float)).all():
                    raise ValueError("重排响应数量或分值无效")
                selected = [i for _, i in sorted(zip(scores, selected), reverse=True)]
                method += " + Reranker"
            except Exception:
                warnings.append("重排模型暂不可用，保留融合排名")
        citations = []
        seen = set()
        for i in selected:
            chunk, doc = rows[i]
            signature = (chunk.content, doc.version, doc.source)
            if signature in seen:
                continue
            seen.add(signature)
            citations.append(
                {
                    "number": len(citations) + 1,
                    "chunk_id": chunk.id,
                    "document_id": doc.id,
                    "title": doc.title,
                    "source": doc.source,
                    "version": doc.version,
                    "category": doc.category,
                    "ordinal": chunk.ordinal,
                    "text": chunk.content,
                    "score": round(fused[i], 5),
                }
            )
            if len(citations) >= limit:
                break
        return {"citations": citations, "method": method, "warnings": warnings}


def list_documents(character_id: str):
    with Session() as db:
        docs = db.scalars(
            select(Document).where(Document.character_id == character_id).order_by(Document.created_at.desc())
        ).all()
        return [
            record(doc)
            | {
                "vector_indexed": bool(
                    db.scalar(
                        select(Chunk.id)
                        .where(Chunk.document_id == doc.id, Chunk.embedding.is_not(None))
                        .limit(1)
                    )
                )
            }
            for doc in docs
        ]
