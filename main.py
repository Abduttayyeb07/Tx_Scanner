"""
ZigChain TX Reasoning Engine — CLI Interface

TX -> Normalized State -> Deterministic Interpretation -> LLM Explanation -> Guarded Chat

Pipeline:
  Layer 1 (Fetch) → Layer 2 (Normalize) → Layer 3 (Interpret) → Layer 4 (LLM)
  + Query Intelligence Layer (tx-aware intent gate)
  + Tiered model routing (simple/moderate/complex → fast/default/powerful)
  + Token registry (exponent-aware amount formatting)
"""

import sys
import json
import threading
from colorama import init, Fore, Style

from src.fetcher import fetch_tx
from src.normalizer import normalize_tx
from src.interpreter import interpret
from src.llm import call_llm, warmup_models
from src.query_engine import check_question, QueryContext, REJECTION_MESSAGE
from src.chat import ChatSession
from src.tokens import registry as token_registry

init(autoreset=True)

BANNER = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║          ZigChain Transaction Reasoning Engine           ║
║                                                          ║
║  Deterministic analysis + LLM explanation for ZigChain   ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""

HELP_TEXT = f"""
{Fore.YELLOW}Commands:{Style.RESET_ALL}
  {Fore.GREEN}/tx <hash>{Style.RESET_ALL}      — Analyze a new transaction
  {Fore.GREEN}/raw{Style.RESET_ALL}            — Show raw normalized JSON for current tx
  {Fore.GREEN}/interpret{Style.RESET_ALL}      — Show deterministic interpretation
  {Fore.GREEN}/stats{Style.RESET_ALL}          — Show current session stats
  {Fore.GREEN}/help{Style.RESET_ALL}           — Show this help message
  {Fore.GREEN}/quit{Style.RESET_ALL}           — Exit

  Or just type a question about the current transaction.
"""

# ──────────────────────────────────────────────
# Background initialization
# ──────────────────────────────────────────────

_init_done = threading.Event()
_init_results = {}


def _background_init():
    """Run model warmup + token loading in background.
    The CLI is usable immediately — this runs in parallel.
    """
    global _init_results

    # Load token registry
    token_count = token_registry.load(verbose=True)

    # Warmup LLM models
    model_results = warmup_models(verbose=True)
    loaded = sum(1 for v in model_results.values() if v)
    total = len(model_results)

    _init_results = {
        "tokens": token_count,
        "models_loaded": loaded,
        "models_total": total,
    }
    _init_done.set()


# ──────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────

def print_error(msg: str):
    print(f"\n{Fore.RED}! {msg}{Style.RESET_ALL}")


def print_info(msg: str):
    print(f"\n{Fore.YELLOW}> {msg}{Style.RESET_ALL}")


def print_llm_response(text: str):
    print(f"\n{Fore.WHITE}{text}{Style.RESET_ALL}\n")


def print_warnings(warnings: list):
    for w in warnings:
        level = w.get("level", "info")
        msg = w.get("message", "")
        if level == "critical":
            print(f"  {Fore.RED}[!] {msg}{Style.RESET_ALL}")
        elif level == "warning":
            print(f"  {Fore.YELLOW}[!] {msg}{Style.RESET_ALL}")
        else:
            print(f"  {Fore.CYAN}[i] {msg}{Style.RESET_ALL}")


# ──────────────────────────────────────────────
# Core pipeline
# ──────────────────────────────────────────────

def analyze_tx(tx_hash: str) -> ChatSession | None:
    """Full pipeline: Fetch -> Normalize -> Interpret -> LLM Explain."""

    # Layer 1 — Fetch
    print_info(f"Fetching tx {tx_hash[:16]}...")
    try:
        raw = fetch_tx(tx_hash)
    except Exception as e:
        print_error(str(e))
        return None

    # Layer 2 — Normalize
    print_info("Normalizing transaction data...")
    normalized = normalize_tx(raw)

    # Layer 3 — Interpret
    print_info("Running deterministic analysis...")
    interpretation = interpret(normalized)
    complexity = interpretation["complexity"]

    # Print interpretation summary
    print(f"\n{Fore.CYAN}{'─' * 60}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Type:{Style.RESET_ALL}       {interpretation['tx_type']}")
    print(f"  {Fore.WHITE}Status:{Style.RESET_ALL}     {normalized['status'].upper()}")
    print(f"  {Fore.WHITE}Complexity:{Style.RESET_ALL} {complexity}")
    print(f"  {Fore.WHITE}Model:{Style.RESET_ALL}      {_model_label(complexity)}")
    print(f"  {Fore.WHITE}Summary:{Style.RESET_ALL}    {interpretation['summary']}")

    if interpretation["warnings"]:
        print(f"\n  {Fore.YELLOW}Warnings:{Style.RESET_ALL}")
        print_warnings(interpretation["warnings"])
    print(f"{Fore.CYAN}{'─' * 60}{Style.RESET_ALL}")

    # Layer 4 — LLM Explanation (routed to appropriate model tier)
    print_info(f"Generating explanation ({_model_label(complexity)})...")
    explanation = call_llm(normalized, interpretation, complexity=complexity)
    print_llm_response(explanation)

    # Create chat session
    session = ChatSession(tx_hash, normalized, interpretation)
    session.add_assistant_message(explanation)

    return session


def _model_label(complexity: str) -> str:
    """Human-readable label for the model tier."""
    labels = {
        "simple": "⚡ fast",
        "moderate": "🧠 standard",
        "complex": "🔬 powerful",
    }
    return labels.get(complexity, complexity)


def handle_question(session: ChatSession, question: str):
    """Handle a follow-up question through the query intelligence layer."""

    # Query intelligence gate — tx-aware reasoning
    allowed, note, directive = check_question(question, session.query_ctx)
    if not allowed:
        print(f"\n{Fore.YELLOW}{note}{Style.RESET_ALL}\n")
        return

    session.add_user_message(question)
    history = session.get_context_history()

    # If gate returned a context note, append it for the LLM
    effective_question = question
    if note:
        effective_question = f"{question}\n\n[System note: {note}]"

    print_info(f"Thinking ({_model_label(session.complexity)})...")
    response = call_llm(
        session.normalized_data,
        session.interpretation,
        user_question=effective_question,
        chat_history=history[:-1],
        complexity=session.complexity,
        prompt_directive=directive,
    )

    session.add_assistant_message(response)
    print_llm_response(response)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print(BANNER)

    # Start background init (models + tokens) — CLI is usable immediately
    print(f"{Fore.YELLOW}> Initializing (tokens + models loading in background)...{Style.RESET_ALL}")
    init_thread = threading.Thread(target=_background_init, daemon=True)
    init_thread.start()

    print(HELP_TEXT)

    session = None

    if len(sys.argv) > 1:
        # Wait for init to finish before first analysis (needs token registry)
        if not _init_done.wait(timeout=60):
            print_info("Background init still running, proceeding anyway...")
        session = analyze_tx(sys.argv[1])

    while True:
        try:
            prompt_color = Fore.GREEN if session else Fore.YELLOW
            prompt_label = f"tx:{session.tx_hash[:8]}..." if session else "no-tx"
            user_input = input(f"{prompt_color}[{prompt_label}]>{Style.RESET_ALL} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Fore.CYAN}Goodbye.{Style.RESET_ALL}")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd in ("/quit", "/exit", "/q"):
                print(f"\n{Fore.CYAN}Goodbye.{Style.RESET_ALL}")
                break

            elif cmd in ("/help", "/h"):
                print(HELP_TEXT)

            elif cmd == "/tx":
                if len(parts) < 2:
                    print_error("Usage: /tx <hash>")
                else:
                    session = analyze_tx(parts[1])

            elif cmd == "/raw":
                if not session:
                    print_error("No transaction loaded. Use /tx <hash> first.")
                else:
                    print(f"\n{json.dumps(session.normalized_data, indent=2)}\n")

            elif cmd == "/interpret":
                if not session:
                    print_error("No transaction loaded. Use /tx <hash> first.")
                else:
                    print(f"\n{json.dumps(session.interpretation, indent=2)}\n")

            elif cmd == "/stats":
                if not session:
                    print_error("No transaction loaded. Use /tx <hash> first.")
                else:
                    stats = session.get_stats()
                    print(f"\n  TX: {stats['tx_hash']}")
                    print(f"  Type: {stats['tx_type']}")
                    print(f"  Complexity: {stats['complexity']}")
                    print(f"  Model tier: {_model_label(stats['complexity'])}")
                    print(f"  Messages: {stats['messages']}")
                    print(f"  Tokens loaded: {token_registry.token_count}\n")

            else:
                print_error(f"Unknown command: {cmd}. Type /help for available commands.")

        else:
            if not session:
                print_error("No transaction loaded. Use /tx <hash> first.")
            else:
                handle_question(session, user_input)


if __name__ == "__main__":
    main()
