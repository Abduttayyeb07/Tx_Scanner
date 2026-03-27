"""
ZigChain TX Reasoning Engine - CLI Interface

TX -> Normalized State -> Deterministic Interpretation -> LLM Explanation -> Guarded Chat
"""

import json
import sys
import threading

from colorama import Fore, Style, init

from src.chat import ChatSession
from src.config import LLM_MODEL_NAME
from src.fetcher import fetch_tx
from src.interpreter import interpret
from src.llm import call_llm, warmup_models
from src.normalizer import normalize_tx
from src.query_engine import check_question
from src.tokens import registry as token_registry

init(autoreset=True)

BANNER = f"""
{Fore.CYAN}============================================================
          ZigChain Transaction Reasoning Engine

  Deterministic analysis + LLM explanation for ZigChain
============================================================{Style.RESET_ALL}
"""

HELP_TEXT = f"""
{Fore.YELLOW}Commands:{Style.RESET_ALL}
  {Fore.GREEN}/tx <hash>{Style.RESET_ALL}      - Analyze a new transaction
  {Fore.GREEN}/raw{Style.RESET_ALL}            - Show raw normalized JSON for current tx
  {Fore.GREEN}/interpret{Style.RESET_ALL}      - Show deterministic interpretation
  {Fore.GREEN}/stats{Style.RESET_ALL}          - Show current session stats
  {Fore.GREEN}/help{Style.RESET_ALL}           - Show this help message
  {Fore.GREEN}/quit{Style.RESET_ALL}           - Exit

  Or just type a question about the current transaction.
"""

_init_done = threading.Event()
_init_results = {}


def _background_init():
    """Run model warmup + token loading in background."""
    global _init_results

    token_count = token_registry.load(verbose=True)
    model_results = warmup_models(verbose=True)
    loaded = sum(1 for ok in model_results.values() if ok)

    _init_results = {
        "tokens": token_count,
        "models_loaded": loaded,
        "models_total": len(model_results),
    }
    _init_done.set()


def print_error(msg: str):
    print(f"\n{Fore.RED}! {msg}{Style.RESET_ALL}")


def print_info(msg: str):
    print(f"\n{Fore.YELLOW}> {msg}{Style.RESET_ALL}")


def print_llm_response(text: str):
    print(f"\n{Fore.WHITE}{text}{Style.RESET_ALL}\n")


def print_warnings(warnings: list):
    for warning in warnings:
        level = warning.get("level", "info")
        msg = warning.get("message", "")
        if level == "critical":
            print(f"  {Fore.RED}[!] {msg}{Style.RESET_ALL}")
        elif level == "warning":
            print(f"  {Fore.YELLOW}[!] {msg}{Style.RESET_ALL}")
        else:
            print(f"  {Fore.CYAN}[i] {msg}{Style.RESET_ALL}")


def analyze_tx(tx_hash: str) -> ChatSession | None:
    """Full pipeline: Fetch -> Normalize -> Interpret -> LLM Explain."""
    print_info(f"Fetching tx {tx_hash[:16]}...")
    try:
        raw = fetch_tx(tx_hash)
    except Exception as exc:
        print_error(str(exc))
        return None

    print_info("Normalizing transaction data...")
    normalized = normalize_tx(raw)

    print_info("Running deterministic analysis...")
    interpretation = interpret(normalized)
    complexity = interpretation["complexity"]

    print(f"\n{Fore.CYAN}{'-' * 60}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Type:{Style.RESET_ALL}       {interpretation['tx_type']}")
    print(f"  {Fore.WHITE}Status:{Style.RESET_ALL}     {normalized['status'].upper()}")
    print(f"  {Fore.WHITE}Complexity:{Style.RESET_ALL} {complexity}")
    print(f"  {Fore.WHITE}Model:{Style.RESET_ALL}      {LLM_MODEL_NAME}")
    print(f"  {Fore.WHITE}Summary:{Style.RESET_ALL}    {interpretation['summary']}")

    if interpretation["warnings"]:
        print(f"\n  {Fore.YELLOW}Warnings:{Style.RESET_ALL}")
        print_warnings(interpretation["warnings"])
    print(f"{Fore.CYAN}{'-' * 60}{Style.RESET_ALL}")

    print_info(f"Generating explanation ({LLM_MODEL_NAME})...")
    explanation = call_llm(normalized, interpretation, complexity=complexity)
    print_llm_response(explanation)

    session = ChatSession(tx_hash, normalized, interpretation)
    session.add_assistant_message(explanation)
    return session


def handle_question(session: ChatSession, question: str):
    """Handle a follow-up question through the query intelligence layer."""
    allowed, note, directive = check_question(question, session.query_ctx)
    if not allowed:
        print(f"\n{Fore.YELLOW}{note}{Style.RESET_ALL}\n")
        return

    session.add_user_message(question)
    history = session.get_context_history()

    effective_question = question
    if note:
        effective_question = f"{question}\n\n[System note: {note}]"

    print_info(f"Thinking ({LLM_MODEL_NAME})...")
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


def main():
    print(BANNER)

    print(f"{Fore.YELLOW}> Initializing (tokens + model loading in background)...{Style.RESET_ALL}")
    init_thread = threading.Thread(target=_background_init, daemon=True)
    init_thread.start()

    print(HELP_TEXT)

    session = None

    if len(sys.argv) > 1:
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
            if cmd in ("/help", "/h"):
                print(HELP_TEXT)
                continue
            if cmd == "/tx":
                if len(parts) < 2:
                    print_error("Usage: /tx <hash>")
                else:
                    session = analyze_tx(parts[1])
                continue
            if cmd == "/raw":
                if not session:
                    print_error("No transaction loaded. Use /tx <hash> first.")
                else:
                    print(f"\n{json.dumps(session.normalized_data, indent=2)}\n")
                continue
            if cmd == "/interpret":
                if not session:
                    print_error("No transaction loaded. Use /tx <hash> first.")
                else:
                    print(f"\n{json.dumps(session.interpretation, indent=2)}\n")
                continue
            if cmd == "/stats":
                if not session:
                    print_error("No transaction loaded. Use /tx <hash> first.")
                else:
                    stats = session.get_stats()
                    print(f"\n  TX: {stats['tx_hash']}")
                    print(f"  Type: {stats['tx_type']}")
                    print(f"  Complexity: {stats['complexity']}")
                    print(f"  Model: {LLM_MODEL_NAME}")
                    print(f"  Messages: {stats['messages']}")
                    print(f"  Tokens loaded: {token_registry.token_count}\n")
                continue

            print_error(f"Unknown command: {cmd}. Type /help for available commands.")
            continue

        if not session:
            print_error("No transaction loaded. Use /tx <hash> first.")
            continue

        handle_question(session, user_input)


if __name__ == "__main__":
    main()
