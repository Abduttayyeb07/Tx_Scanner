"""
Chat Session Manager
Manages per-TX chat context with token-budget-aware history.
TX data is the real context, not the chat.
"""

from src.query_engine import QueryContext


class ChatSession:
    """Manages a single transaction chat session."""

    # Token budget for chat history (tx data is always resent separately)
    MAX_HISTORY_CHARS = 6000  # ~1500 tokens at ~4 chars/token

    def __init__(self, tx_hash: str, normalized_data: dict, interpretation: dict):
        self.tx_hash = tx_hash
        self.normalized_data = normalized_data
        self.interpretation = interpretation
        self.query_ctx = QueryContext(normalized_data, interpretation)
        self.history = []

    @property
    def complexity(self) -> str:
        return self.query_ctx.complexity

    def add_user_message(self, message: str):
        self.history.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        self.history.append({"role": "assistant", "content": message})

    def get_context_history(self) -> list:
        """Return recent history trimmed to token budget.

        Walks backwards from newest, keeps what fits.
        TX data is always resent by the caller — this is just chat history.
        """
        if not self.history:
            return []

        result = []
        total_chars = 0

        for msg in reversed(self.history):
            msg_chars = len(msg["content"])
            if total_chars + msg_chars > self.MAX_HISTORY_CHARS:
                break
            result.insert(0, msg)
            total_chars += msg_chars

        return result

    def get_stats(self) -> dict:
        return {
            "tx_hash": self.tx_hash,
            "messages": len(self.history),
            "tx_type": self.interpretation.get("tx_type", "unknown"),
            "complexity": self.interpretation.get("complexity", "unknown"),
        }
