"""
Phase C: Per-Session Conversation Memory Service
------------------------------------------------
Manages rolling conversation history per session_id for follow-up query context.
"""

from typing import Dict, List, Tuple
from collections import defaultdict
from app.core.logging import logger


class MemoryService:
    """Manages rolling conversation history keyed by session_id."""

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        # session_id -> list of (user_text, assistant_text)
        self.sessions: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    def add_turn(self, session_id: str, user_text: str, assistant_text: str):
        """Appends a completed conversation turn to the session history."""
        if not session_id:
            session_id = "default_session"
        
        history = self.sessions[session_id]
        history.append((user_text, assistant_text))
        
        # Maintain rolling window
        if len(history) > self.max_turns:
            self.sessions[session_id] = history[-self.max_turns:]
        logger.debug(f"Session '{session_id}' updated (total turns: {len(self.sessions[session_id])})")

    def get_context_prompt(self, session_id: str, current_prompt: str) -> str:
        """Constructs prompt augmented with conversation history context."""
        if not session_id or session_id not in self.sessions:
            return current_prompt

        history = self.sessions[session_id]
        if not history:
            return current_prompt

        formatted_history = []
        for user_msg, bot_msg in history[-3:]:
            formatted_history.append(f"User: {user_msg}\nAssistant: {bot_msg}")

        history_str = "\n".join(formatted_history)
        return f"Previous Conversation Context:\n{history_str}\n\nCurrent User Question: {current_prompt}"

    def clear_session(self, session_id: str):
        """Clears memory for a specific session."""
        if session_id in self.sessions:
            del self.sessions[session_id]


memory_service = MemoryService()
