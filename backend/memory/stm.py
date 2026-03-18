"""
memory/stm.py — Short-Term Memory (STM) context builder.

WHAT IT DOES:
  Builds the "conversation so far" block injected into every RAG prompt,
  giving the LLM awareness of the current conversation thread.

HOW IT WORKS:
  1. Check if there's a stored summary for this session (old turns, > STM_TURNS ago)
  2. Fetch the last STM_TURNS user+assistant pairs from DB
  3. Format them as a readable dialogue block
  4. Enforce a token budget (STM_MAX_TOKENS) — trim oldest if too long

RESULT INJECTED INTO PROMPT:
  [CONVERSATION SO FAR]
  [Summary of earlier turns: ...]   ← only if session is long
  User: What is a Python list?
  Assistant: A Python list is an ordered, mutable...
  User: How do I sort it?
  [END CONVERSATION HISTORY]
"""

from typing import Optional
from logger import get_logger
from config import STM_TURNS, STM_MAX_TOKENS
import memory.db as db

log = get_logger("memory")

# Rough chars-per-token estimate (faster than running a tokenizer)
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def build_stm_block(session_id: Optional[str]) -> str:
    """
    Build the STM context block for a given session.

    Returns "" (empty string) if:
      - session_id is None (new session, no history yet)
      - Session exists but has no prior messages

    Returns a formatted string like:
      [CONVERSATION SO FAR]
      User: ...
      Assistant: ...
      [END CONVERSATION HISTORY]
    """
    if not session_id:
        return ""

    # ── Fetch last N turns ────────────────────────────────────────────────────
    recent_messages = db.get_recent_messages(session_id, n=STM_TURNS)
    if not recent_messages:
        return ""

    lines = ["[CONVERSATION SO FAR]"]

    # ── Prepend summary if session has older context ──────────────────────────
    stored = db.get_summary(session_id)
    if stored:
        # Only include summary if there are messages AFTER the summarized point
        # (means old turns were summarized and recent turns are separate)
        oldest_recent_id = recent_messages[0]["id"]
        if stored["summarized_up_to_msg_id"] < oldest_recent_id:
            lines.append(f"[Summary of earlier conversation: {stored['summary']}]")

    # ── Format recent messages ────────────────────────────────────────────────
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].strip()
        # Truncate very long individual messages to avoid budget blowout
        if len(content) > 600:
            content = content[:600] + "..."
        lines.append(f"{role}: {content}")

    lines.append("[END CONVERSATION HISTORY]")
    block = "\n".join(lines)

    # ── Enforce token budget — trim from oldest if over limit ─────────────────
    if _estimate_tokens(block) > STM_MAX_TOKENS:
        # Drop messages from top (oldest) until within budget
        while len(recent_messages) > 1 and _estimate_tokens(block) > STM_MAX_TOKENS:
            recent_messages.pop(0)
            lines = ["[CONVERSATION SO FAR]"]
            if stored:
                lines.append(f"[Summary of earlier conversation: {stored['summary']}]")
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role}: {msg['content'].strip()[:600]}")
            lines.append("[END CONVERSATION HISTORY]")
            block = "\n".join(lines)

    log.debug("stm_block_built",
        session_id=session_id,
        turns=len(recent_messages),
        estimated_tokens=_estimate_tokens(block),
        has_summary=stored is not None
    )

    return block


def should_summarize(session_id: str) -> bool:
    """
    Return True if the session has enough messages that old turns
    should be summarized to keep prompts efficient.

    Triggers when total messages > STM_TURNS * 2 (i.e., the session
    has more turns than the STM window can hold).
    """
    count = db.get_message_count(session_id)
    return count > STM_TURNS * 2
