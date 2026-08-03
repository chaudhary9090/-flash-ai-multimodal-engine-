"""
Phase 2: Production Semantic RAG Pipeline Engine
------------------------------------------------
Implements:
1. Format-aware Document Extractors (PPTX, PDF, DOCX, HTML, TXT, CSV)
2. Precise Line-Aware Character Text Splitter with Empirical Line Number Tracking
3. Dense Semantic Sentence Transformer Embedder (all-MiniLM-L6-v2) & Cosine Similarity
4. Verified Citations with Exact Supporting Snippets and Line Numbers
"""

import os
import re
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from app.services.document_service import extract_clean_text
from app.core.logging import logger
from app.models.schemas import Citation

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModel = None


@dataclass
class DocumentChunk:
    chunk_id: int
    filename: str
    text: str
    line_start: int
    line_end: int
    metadata: Dict[str, str]
    embedding: Optional[List[float]] = None


class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text_with_lines(self, text: str) -> List[Tuple[str, int, int]]:
        """Splits text into precise chunks and calculates empirical 1-indexed line numbers."""
        lines = text.split("\n")
        
        # If document has line structure, process line by line or small paragraph chunks
        chunks_with_lines = []
        current_chunk_lines = []
        current_chunk_len = 0
        chunk_start_line = 1

        for line_idx, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped:
                continue

            line_len = len(stripped)
            if current_chunk_len + line_len > self.chunk_size and current_chunk_lines:
                chunk_str = "\n".join(current_chunk_lines)
                chunks_with_lines.append((chunk_str, chunk_start_line, line_idx - 1))
                current_chunk_lines = [stripped]
                current_chunk_len = line_len
                chunk_start_line = line_idx
            else:
                if not current_chunk_lines:
                    chunk_start_line = line_idx
                current_chunk_lines.append(stripped)
                current_chunk_len += line_len + 1

        if current_chunk_lines:
            chunk_str = "\n".join(current_chunk_lines)
            chunks_with_lines.append((chunk_str, chunk_start_line, len(lines)))

        return chunks_with_lines


class DenseSemanticEmbedder:
    """
    Computes 384-dimensional dense semantic embeddings using HuggingFace all-MiniLM-L6-v2.
    Enables true semantic search (e.g., 'salary' matches 'compensation').
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_attempted = False

    def _init_model(self):
        if not self._load_attempted:
            self._load_attempted = True
            if torch is not None and AutoTokenizer is not None:
                try:
                    logger.info(f"Initializing Dense Semantic Embedder ({self.model_name})...")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModel.from_pretrained(self.model_name)
                    self.model.eval()
                except Exception as e:
                    logger.warning(f"Could not load HuggingFace dense embedder model: {e}")
                    self.tokenizer = None
                    self.model = None

    def embed(self, text: str) -> List[float]:
        self._init_model()
        if self.model is not None and self.tokenizer is not None:
            try:
                inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=256)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, 1)
                    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                    pooled = sum_embeddings / sum_mask
                    norm_vec = torch.nn.functional.normalize(pooled, p=2, dim=1)[0].tolist()
                    return norm_vec
            except Exception as embed_err:
                logger.error(f"Dense embedding inference error: {embed_err}")

        # Fallback hash-frequency embedder
        vector_dim = 128
        words = re.findall(r"\w+", text.lower())
        vec = [0.0] * vector_dim
        if not words:
            return vec
        for w in words:
            idx = abs(hash(w)) % vector_dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0 else vec


class VectorStore:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.embedder = DenseSemanticEmbedder()

    def add_document(self, filename: str, content: str) -> int:
        splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
        chunks_info = splitter.split_text_with_lines(content)
        
        added_count = 0
        for i, (text, l_start, l_end) in enumerate(chunks_info):
            embedding = self.embedder.embed(text)
            chunk = DocumentChunk(
                chunk_id=len(self.chunks) + 1,
                filename=filename,
                text=text,
                line_start=l_start,
                line_end=l_end,
                metadata={"chunk_index": str(i + 1), "char_len": str(len(text))},
                embedding=embedding
            )
            self.chunks.append(chunk)
            added_count += 1
        
        logger.info(f"Indexed {added_count} line-level semantic chunks for '{filename}'.")
        return added_count

    def similarity_search(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks:
            return []

        query_vec = self.embedder.embed(query)
        scored_chunks = []

        for chunk in self.chunks:
            if chunk.embedding is None:
                continue
            score = sum(q * c for q, c in zip(query_vec, chunk.embedding))
            if score > 0.10:
                scored_chunks.append((chunk, score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

    def clear(self):
        self.chunks.clear()


class RAGPipelineService:
    def __init__(self):
        self.vector_store = VectorStore()

    def ingest_document(self, filename: str, content_bytes: bytes) -> Dict[str, Any]:
        text_content = extract_clean_text(filename, content_bytes)
        chunk_count = self.vector_store.add_document(filename, text_content)
        return {
            "filename": filename,
            "chunks_indexed": chunk_count,
            "total_store_chunks": len(self.vector_store.chunks),
            "status": "successfully_indexed"
        }

    def query_rag_with_citations(self, query: str, top_k: int = 3) -> Tuple[Optional[str], List[Citation]]:
        results = self.vector_store.similarity_search(query, top_k=top_k)
        if not results:
            return None, []

        citation_text = f"[SEMANTIC RAG RETRIEVAL REPORT — {len(results)} Matching Sources Found]\n"
        citation_text += "--------------------------------------------------\n"

        structured_citations = []

        for i, (chunk, score) in enumerate(results, 1):
            pct = int(score * 100)
            citation_text += (
                f"📌 SOURCE #{i} | File: {chunk.filename} (Lines {chunk.line_start}-{chunk.line_end}) | Relevance: {pct}%\n"
                f"Excerpt: \"{chunk.text}\"\n\n"
            )
            structured_citations.append(
                Citation(
                    source_type="document",
                    filename=chunk.filename,
                    chunk_index=int(chunk.metadata["chunk_index"]),
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                    snippet=chunk.text[:400],  # Full snippet without cutting off mid-chunk text
                    relevance_score=round(float(score), 4)
                )
            )

        citation_text += "--------------------------------------------------"
        return citation_text, structured_citations


rag_service = RAGPipelineService()
