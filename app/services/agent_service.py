"""
Phase B: Unified Agent Router with Multi-Tool Chaining & Reasoning Trace
-----------------------------------------------------------------------
Orchestrates intelligent function routing, multi-tool chaining, 
1-line Reasoning Trace generation, and structured citation tracking.
"""

import re
from typing import Tuple, List, Dict, Any, Optional
from app.core.logging import logger
from app.models.schemas import Citation
from app.services.gpt_service import gpt_service
from app.services.rag_service import rag_service
from app.services.weather_service import fetch_global_weather
from app.services.youtube_service import fetch_youtube_knowledge
from app.services.book_service import fetch_book_knowledge
from app.services.code_executor_service import code_executor_service
from app.services.memory_service import memory_service


class UnifiedAgentRouter:
    """Intelligent Agent Router with tool chaining and reasoning trace generation."""

    def route_and_execute(
        self,
        prompt: str,
        engine_mode: str = "flash_reasoning",
        session_id: str = "default_session"
    ) -> Tuple[str, str, List[str], List[Citation]]:
        """
        Evaluates prompt, chains tools if required, builds reasoning trace,
        and returns (response_text, primary_source_tool, reasoning_trace, citations).
        """
        clean_prompt = prompt.strip()
        lower_prompt = clean_prompt.lower()
        reasoning_trace = []
        tool_results = []
        citations: List[Citation] = []
        primary_tool = "gpt_model"

        # 1. Check for Python Code Execution Intent (Phase E)
        code_match = re.search(r"```python(.*?)```", clean_prompt, re.DOTALL)
        if not code_match and any(kw in lower_prompt for kw in ["calculate", "python code", "run code", "plot", "math calculation"]):
            math_expr = re.search(r"(?:calculate|compute|math|what is)?\s*([0-9\s\+\-\*\/\(\)\.\*\*]+)", lower_prompt)
            if math_expr and len(math_expr.group(1).strip()) >= 3 and any(c in math_expr.group(1) for c in "+-*/"):
                code_text = f"print({math_expr.group(1).strip()})"
            else:
                code_text = None
        elif code_match:
            code_text = code_match.group(1).strip()
        else:
            code_text = None

        if code_text:
            reasoning_trace.append("Executing tool: execute_python_code (AST-verified Python Sandbox)")
            py_res = code_executor_service.execute_python(code_text)
            tool_results.append(py_res)
            primary_tool = "python_code_executor"

        # 2. Check for Weather Tool
        if any(w in lower_prompt for w in ["weather", "temperature", "temp in", "climate"]):
            reasoning_trace.append("Executing tool: get_weather (Open-Meteo Live Geocoding API)")
            w_res = fetch_global_weather(clean_prompt)
            if w_res:
                tool_results.append(w_res)
                if primary_tool == "gpt_model":
                    primary_tool = "weather_service"

        # 3. Check for YouTube Tool
        if "youtube.com" in lower_prompt or "youtu.be" in lower_prompt or "youtube" in lower_prompt:
            reasoning_trace.append("Executing tool: get_youtube_summary (oEmbed Knowledge Extractor)")
            yt_res = fetch_youtube_knowledge(clean_prompt)
            if yt_res:
                tool_results.append(yt_res)
                if primary_tool == "gpt_model":
                    primary_tool = "youtube_service"

        # 4. Check for Book Tool
        if "book" in lower_prompt or "title" in lower_prompt:
            book_match = re.search(r"(?:book\s+title|summary\s+of|book)\s+[\"']?([a-zA-Z0-9\s]+)[\"']?", lower_prompt)
            if book_match:
                book_title = book_match.group(1).strip()
                reasoning_trace.append(f"Executing tool: get_book_summary (OpenLibrary API for '{book_title.title()}')")
                b_res = fetch_book_knowledge(book_title)
                if b_res:
                    tool_results.append(b_res)
                    if primary_tool == "gpt_model":
                        primary_tool = "book_service"

        # 5. Check for RAG Vector Store Context (Semantic Dense Embeddings + Structured Citations)
        doc_intents = ["document", "file", "resume", "pdf", "pptx", "slide", "uploaded", "quest", "kindness", "experience", "skill"]
        if rag_service.vector_store.chunks and any(di in lower_prompt for di in doc_intents):
            reasoning_trace.append(f"Executing tool: query_uploaded_document (Semantic Dense RAG with {len(rag_service.vector_store.chunks)} chunks)")
            rag_res, rag_citations = rag_service.query_rag_with_citations(clean_prompt, top_k=2)
            if rag_res:
                tool_results.append(rag_res)
                citations.extend(rag_citations)
                if primary_tool == "gpt_model":
                    primary_tool = "rag_service"

        # 6. Combine Tool Results + Memory Context + Model Inference
        if tool_results:
            combined_context = "\n\n".join(tool_results)
            augmented_prompt = f"Context Data:\n{combined_context}\n\nUser Question: {clean_prompt}"
            final_prompt = memory_service.get_context_prompt(session_id, augmented_prompt)
            model_res, source = gpt_service.generate(final_prompt, engine_mode=engine_mode)
            
            final_text = f"{combined_context}\n\n{model_res}" if any(kw in combined_context for kw in ["WEATHER", "PYTHON", "RAG RETRIEVAL"]) else model_res
            memory_service.add_turn(session_id, clean_prompt, final_text)
            return final_text, primary_tool, reasoning_trace, citations

        # Default: Pure Chat Reasoning with Memory
        reasoning_trace.append(f"Executing engine: {engine_mode} (Multi-turn Conversational Memory)")
        memory_prompt = memory_service.get_context_prompt(session_id, clean_prompt)
        response_text, source = gpt_service.generate(memory_prompt, engine_mode=engine_mode)
        
        memory_service.add_turn(session_id, clean_prompt, response_text)
        return response_text, source, reasoning_trace, citations


agent_router = UnifiedAgentRouter()
