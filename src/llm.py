"""
Layer 4 - LLM Wrapper

The LLM does NOT think. It translates structured truth into human explanation.
Uses a single Ollama-style /api/generate model for all requests.
"""

import json

import requests

from src.config import (
    LLM_API_PASSWORD,
    LLM_API_URL,
    LLM_API_USER,
    LLM_MAX_TOKENS,
    LLM_MODEL_NAME,
    LLM_TIMEOUT,
    LLM_TEMPERATURE,
    LLM_TOP_P,
)


def _get_auth() -> tuple | None:
    """Return HTTP Basic Auth tuple or None."""
    if LLM_API_USER and LLM_API_PASSWORD:
        return (LLM_API_USER, LLM_API_PASSWORD)
    return None


def warmup_models(verbose: bool = True) -> dict[str, bool]:
    """Warm only the selected production model."""
    try:
        payload = {
            "model": LLM_MODEL_NAME,
            "prompt": "hello",
            "system": "respond with ok",
            "stream": False,
            "options": {"num_predict": 1},
        }
        resp = requests.post(
            LLM_API_URL,
            json=payload,
            auth=_get_auth(),
            timeout=30,
        )
        resp.raise_for_status()
        if verbose:
            print(f"  loaded {LLM_MODEL_NAME}")
        return {LLM_MODEL_NAME: True}
    except Exception as exc:
        if verbose:
            print(f"  failed {LLM_MODEL_NAME} ({type(exc).__name__})")
        return {LLM_MODEL_NAME: False}


SYSTEM_PROMPT = """You are a blockchain transaction analysis engine for ZigChain.

Your job is to explain a SINGLE transaction using ONLY the structured data provided.

STRICT RULES:

1. You MUST NOT invent or assume anything not present in the data.
2. You MUST NOT guess intent beyond what is explicitly inferable.
3. If the transaction failed, clearly state it and explain why.
4. Events in failed transactions are NOT final state changes - clarify this.
5. Always distinguish between:
   - attempted actions
   - successful state changes
6. Always extract and highlight:
   - transaction status (success/failed)
   - gas usage and fee
   - signer / fee payer
   - transfers (if valid)
7. Be precise, not verbose.
8. Do NOT use markdown formatting. No bold (**), no italic (*), no headers (#). Output plain text only. This is a terminal application.

OUTPUT FORMAT:

TITLE:
<One-line summary of what happened>

KEY POINTS:
- Bullet points of important actions

DETAILED EXPLANATION:
Explain step-by-step what occurred based on the structured data.

ADDITIONAL CONTEXT:
Explain technical meaning (e.g., failure reason, gas usage).

---

CHAT MODE RULES (IMPORTANT):

You may receive follow-up questions.

You MUST:
- Answer ONLY using the transaction context
- Refuse unrelated questions

If a question is unrelated, respond:
"This assistant only answers questions about this specific transaction."

Do NOT break this rule."""


def call_llm(
    normalized_data: dict,
    interpretation: dict,
    user_question: str = "Explain this transaction",
    chat_history: list = None,
    complexity: str = "moderate",
    prompt_directive: str = None,
) -> str:
    """Call the selected LLM with structured tx data and get an explanation."""
    prompt_parts = [
        f"Transaction Data:\n{json.dumps(normalized_data, indent=2)}",
        f"\nDeterministic Analysis:\n{json.dumps(interpretation, indent=2)}",
    ]

    if chat_history:
        prompt_parts.append("\nPrevious conversation:")
        for msg in chat_history:
            prompt_parts.append(f"  {msg['role']}: {msg['content']}")

    if prompt_directive:
        prompt_parts.append(f"\n[System directive: {prompt_directive}]")

    prompt_parts.append(f"\nUser Question:\n{user_question}")
    full_prompt = "\n".join(prompt_parts)

    payload = {
        "model": LLM_MODEL_NAME,
        "prompt": full_prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "num_predict": LLM_MAX_TOKENS,
        },
    }

    try:
        resp = requests.post(
            LLM_API_URL,
            json=payload,
            auth=_get_auth(),
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
        return result.get("response", "").strip()
    except requests.ConnectionError:
        return "[ERROR] Cannot reach LLM API. Check your connection and LLM_API_URL."
    except requests.Timeout:
        return "[ERROR] LLM request timed out. The model may be overloaded."
    except requests.HTTPError:
        return f"[ERROR] LLM API returned HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as exc:
        return f"[ERROR] LLM call failed: {str(exc)}"
