#!/usr/bin/env python3
"""
LoCoMo-MC10 Benchmark Evaluation for Shodh-Memory

This script evaluates Shodh-Memory on the LoCoMo-MC10 benchmark,
a 1,986-item multiple-choice test for long-term conversational memory.

Supports multiple LLM backends:
  - OpenAI (gpt-4o-mini, gpt-4o)
  - Anthropic (claude-3-haiku, claude-3-sonnet)
  - Ollama (llama3.2, mistral, etc.) - FREE, local
  - Baseten (hosted OSS models) - FREE tier available
  - Any OpenAI-compatible API (Together, Groq, etc.)

Requirements:
    pip install datasets tqdm shodh-memory
    pip install openai  # For OpenAI/compatible APIs
    pip install anthropic  # For Anthropic
    pip install ollama  # For Ollama (local)

Usage:
    # OpenAI
    python locomo_mc10_eval.py --provider openai --model gpt-4o-mini --limit 50

    # Ollama (FREE - local)
    python locomo_mc10_eval.py --provider ollama --model llama3.2 --limit 100

    # Baseten (FREE tier)
    python locomo_mc10_eval.py --provider baseten --model llama-3-8b --limit 100

    # Together.ai (OpenAI-compatible)
    python locomo_mc10_eval.py --provider openai-compatible \\
        --api-base https://api.together.xyz/v1 \\
        --model meta-llama/Llama-3-8b-chat-hf --limit 100

    # Groq (FREE tier, very fast)
    python locomo_mc10_eval.py --provider openai-compatible \\
        --api-base https://api.groq.com/openai/v1 \\
        --model llama-3.1-8b-instant --limit 100
"""

import argparse
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

# Dataset loaded from local cache (no need for datasets library)

try:
    from tqdm import tqdm
except ImportError:
    print("Install tqdm: pip install tqdm")
    exit(1)

import requests

# HTTP API Client for shodh-memory server
class ShodhMemoryClient:
    """HTTP client for shodh-memory server API."""

    def __init__(self, base_url: str = "http://127.0.0.1:3030", api_key: str = "sk-shodh-dev-local-testing-key", user_id: str = "locomo-eval"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.user_id = user_id
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-API-Key": api_key
        })

    def remember(self, content: str, memory_type: str = "Observation", tags: list = None):
        payload = {"user_id": self.user_id, "content": content, "memory_type": memory_type}
        if tags:
            payload["tags"] = tags
        resp = self.session.post(f"{self.base_url}/api/remember", json=payload)
        resp.raise_for_status()
        return resp.json()

    def recall(self, query: str, limit: int = 5, mode: str = "hybrid"):
        payload = {"user_id": self.user_id, "query": query, "limit": limit, "mode": mode}
        resp = self.session.post(f"{self.base_url}/api/recall", json=payload)
        resp.raise_for_status()
        return resp.json().get("memories", [])


# ============================================================================
# LLM Provider Abstraction
# ============================================================================

# Completion budget for the judge's answer.
#
# This was hardcoded at 10 for every provider. A reasoning model spends its
# first tokens thinking, so at 10 it returns `content: None` with
# `finish_reason: "length"` and never emits the digit — measured against
# `stealth/ox-alpha`, which answers correctly at 64 (29 completion tokens,
# 19 of them internal) and returns nothing at 10. Ten tokens silently
# disqualifies an entire class of judge. Override with
# SHODH_JUDGE_MAX_TOKENS.
JUDGE_MAX_TOKENS = int(os.environ.get("SHODH_JUDGE_MAX_TOKENS", "64"))


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Generate completion for prompt."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=0
        )
        # `content` is None when the budget was spent on internal reasoning
        # tokens; return "" so the caller reports an abstention, not a crash.
        return (response.choices[0].message.content or "").strip()


class OpenAICompatibleProvider(LLMProvider):
    """Provider for OpenAI-compatible APIs (Together, Groq, Baseten, etc.)."""

    def __init__(self, model: str, api_base: str, api_key: Optional[str] = None):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
        )

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=0
        )
        # `content` is None when the budget was spent on internal reasoning
        # tokens; return "" so the caller reports an abstention, not a crash.
        return (response.choices[0].message.content or "").strip()


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        from anthropic import Anthropic
        self.model = model
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=JUDGE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}]
        )
        return (response.content[0].text or "").strip() if response.content else ""


class OllamaProvider(LLMProvider):
    """Ollama local provider - FREE!"""

    def __init__(self, model: str, host: str = "http://localhost:11434"):
        try:
            import ollama
            self.model = model
            self.host = host
        except ImportError:
            print("Install ollama: pip install ollama")
            print("Also ensure Ollama is running: ollama serve")
            exit(1)

    def complete(self, prompt: str) -> str:
        import ollama
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": JUDGE_MAX_TOKENS}
        )
        return (response["message"]["content"] or "").strip()


class BasetenProvider(LLMProvider):
    """Baseten hosted models provider."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("BASETEN_API_KEY")
        if not self.api_key:
            print("Set BASETEN_API_KEY environment variable")
            exit(1)

    def complete(self, prompt: str) -> str:
        import requests

        # Baseten model deployment URL format
        url = f"https://model-{self.model}.api.baseten.co/production/predict"

        response = requests.post(
            url,
            headers={"Authorization": f"Api-Key {self.api_key}"},
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": JUDGE_MAX_TOKENS,
                "temperature": 0
            }
        )

        result = response.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"].strip()
        elif "output" in result:
            return result["output"].strip()
        else:
            return str(result)


def create_provider(
    provider: str,
    model: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None
) -> LLMProvider:
    """Factory function to create LLM provider."""

    if provider == "openai":
        return OpenAIProvider(model, api_key)
    elif provider == "openai-compatible":
        if not api_base:
            raise ValueError("--api-base required for openai-compatible provider")
        return OpenAICompatibleProvider(model, api_base, api_key)
    elif provider == "anthropic":
        return AnthropicProvider(model, api_key)
    elif provider == "ollama":
        return OllamaProvider(model)
    elif provider == "baseten":
        return BasetenProvider(model, api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


@dataclass
class EvalResult:
    question_id: str
    question_type: str
    correct: bool
    predicted_idx: Optional[int]
    correct_idx: int
    # False when the judge returned no usable answer (API error / unparseable).
    # Accuracy is computed over judged items only; see `select_answer_with_llm`.
    judged: bool
    latency_store_ms: float
    latency_recall_ms: float
    num_memories_stored: int
    # Debug fields for failure analysis
    question_text: str = ""
    retrieved_context: str = ""
    correct_answer: str = ""
    predicted_answer: str = ""
    all_choices: list = None


def store_conversations(
    client: ShodhMemoryClient,
    sessions: list,
    summaries: list[str],
    datetimes: list[str] = None
) -> tuple[int, float]:
    """Store conversation sessions into Shodh-Memory via HTTP API.

    Uses semantic chunking: keeps dialogue turns together instead of arbitrary splits.
    Includes session datetime for temporal reasoning (resolving "last Saturday" etc).
    """
    start = time.perf_counter()
    count = 0
    datetimes = datetimes or []

    for i, summary in enumerate(summaries):
        session = sessions[i] if i < len(sessions) else []
        session_dt = datetimes[i] if i < len(datetimes) else None

        # Format datetime for context (e.g., "2023-05-25T13:14:00" -> "May 25, 2023")
        dt_prefix = ""
        if session_dt:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(session_dt.replace('Z', '+00:00'))
                dt_prefix = f"[Conversation on {dt.strftime('%B %d, %Y at %I:%M %p')}] "
            except:
                dt_prefix = f"[{session_dt}] "

        # Store the summary with datetime context
        if summary and isinstance(summary, str) and summary.strip():
            client.remember(
                content=f"{dt_prefix}Session {i+1} Summary: {summary[:2000]}",
                memory_type="Context",
                tags=[f"session_{i+1}", "summary"]
            )
            count += 1

        # SEMANTIC CHUNKING: Keep dialogue turns together, don't split mid-sentence
        if session and isinstance(session, list):
            # Group turns into semantic chunks (~800 chars target, but don't break turns)
            current_chunk = []
            current_len = 0
            TARGET_CHUNK_SIZE = 800

            for turn in session:
                if isinstance(turn, dict):
                    content = turn.get("content", "").strip()
                    if not content:
                        continue

                    turn_len = len(content)

                    # If adding this turn exceeds target and we have content, flush chunk
                    if current_len > 0 and current_len + turn_len > TARGET_CHUNK_SIZE:
                        chunk_text = "\n".join(current_chunk)
                        client.remember(
                            content=f"{dt_prefix}Session {i+1}:\n{chunk_text}",
                            memory_type="Conversation",
                            tags=[f"session_{i+1}", "dialogue"]
                        )
                        count += 1
                        current_chunk = []
                        current_len = 0

                    current_chunk.append(content)
                    current_len += turn_len

            # Don't forget the last chunk
            if current_chunk:
                chunk_text = "\n".join(current_chunk)
                client.remember(
                    content=f"{dt_prefix}Session {i+1}:\n{chunk_text}",
                    memory_type="Conversation",
                    tags=[f"session_{i+1}", "dialogue"]
                )
                count += 1

    elapsed_ms = (time.perf_counter() - start) * 1000
    return count, elapsed_ms


def recall_context(client: ShodhMemoryClient, question: str, limit: int = 5) -> tuple[str, float]:
    """Retrieve relevant context for a question via HTTP API."""
    start = time.perf_counter()

    results = client.recall(query=question, limit=limit)

    elapsed_ms = (time.perf_counter() - start) * 1000

    if results:
        # API returns JSON with experience.content
        context = "\n\n".join([f"[Memory {i+1}]: {r.get('experience', {}).get('content', '')}" for i, r in enumerate(results)])
    else:
        context = "(No relevant memories found)"

    return context, elapsed_ms


def select_answer_with_llm(
    provider: LLMProvider,
    question: str,
    choices: list[str],
    context: str
) -> Optional[int]:
    """Use LLM to select the best answer given retrieved context.

    Returns the chosen option index, or ``None`` when no judgement was
    obtained — the API call failed, or the response contained no parseable
    option digit. `None` MUST NOT be coerced to an index by callers: a
    defaulted answer is indistinguishable from a real prediction, so every
    arm scores the rate at which the gold label happens to sit at that
    index and every arm scores it identically. That reads as "no layer
    changes the answer", which is the hypothesis under test. Abstain
    instead, and let the caller report the judged count alongside accuracy.
    """

    choices_text = "\n".join([f"{i}. {choice}" for i, choice in enumerate(choices)])

    prompt = f"""Based on the following conversation memories, answer the question by selecting the correct option.

RETRIEVED MEMORIES:
{context}

QUESTION: {question}

OPTIONS:
{choices_text}

Instructions:
- Analyze the memories to find relevant information
- Select the option that best answers the question
- Respond with ONLY the option number (0-9)
- If unsure, make your best guess based on available information

Your answer (single digit 0-9):"""

    try:
        answer_text = provider.complete(prompt)
        if not answer_text:
            # Reasoning models exhaust the completion budget on internal
            # tokens and return an empty body with finish_reason="length".
            print(
                "LLM Empty: no completion body — raise SHODH_JUDGE_MAX_TOKENS "
                f"(currently {JUDGE_MAX_TOKENS})"
            )
            return None
        # Extract digit from response
        for char in answer_text:
            if char.isdigit():
                idx = int(char)
                if 0 <= idx <= 9:
                    return idx
        print(f"LLM Unparseable: no option digit in {answer_text!r:.200}")
        return None

    except Exception as e:
        print(f"LLM Error: {e}")
        return None


def evaluate_single_item(
    item: dict,
    provider: LLMProvider,
    client: ShodhMemoryClient
) -> EvalResult:
    """Evaluate a single LoCoMo-MC10 item."""

    # Set user_id for this conversation (isolation)
    client.user_id = f"locomo_{item['question_id']}"

    # Store conversation sessions with timestamps for temporal reasoning
    sessions = item.get("haystack_sessions", [])
    summaries = item.get("haystack_session_summaries", [])
    datetimes = item.get("haystack_session_datetimes", [])
    num_stored, store_latency = store_conversations(client, sessions, summaries, datetimes)

    # Recall relevant context
    context, recall_latency = recall_context(client, item["question"], limit=5)

    # Select answer using LLM
    predicted_idx = select_answer_with_llm(
        provider=provider,
        question=item["question"],
        choices=item["choices"],
        context=context
    )

    correct_idx = item["correct_choice_index"]
    judged = predicted_idx is not None
    is_correct = judged and predicted_idx == correct_idx

    return EvalResult(
        question_id=item["question_id"],
        question_type=item["question_type"],
        correct=is_correct,
        predicted_idx=predicted_idx,
        correct_idx=correct_idx,
        judged=judged,
        latency_store_ms=store_latency,
        latency_recall_ms=recall_latency,
        num_memories_stored=num_stored,
        question_text=item["question"],
        retrieved_context=context,
        correct_answer=item["choices"][correct_idx],
        predicted_answer=(
            item["choices"][predicted_idx] if judged else "<no judgement>"
        ),
        all_choices=item["choices"]
    )


def run_evaluation(
    provider_name: str = "openai",
    model: str = "gpt-4o-mini",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    limit: Optional[int] = None,
    shodh_url: str = "http://127.0.0.1:3030",
    shodh_api_key: str = "sk-shodh-dev-local-testing-key",
    output_file: str = "locomo_results.json"
):
    """Run the full LoCoMo-MC10 evaluation."""

    print("=" * 60)
    print("LoCoMo-MC10 Benchmark for Shodh-Memory")
    print("=" * 60)

    # Create HTTP client for shodh-memory
    print(f"\nConnecting to Shodh-Memory at {shodh_url}")
    client = ShodhMemoryClient(base_url=shodh_url, api_key=shodh_api_key)

    # Test connection
    try:
        resp = requests.get(f"{shodh_url}/health")
        if resp.status_code != 200:
            print(f"ERROR: Shodh-Memory server not responding at {shodh_url}")
            exit(1)
        print("[OK] Connected to Shodh-Memory server")
    except Exception as e:
        print(f"ERROR: Cannot connect to Shodh-Memory server: {e}")
        exit(1)

    # Create LLM provider
    print(f"Initializing {provider_name} provider with model: {model}")
    provider = create_provider(provider_name, model, api_base, api_key)

    # Load dataset from local cache
    print("Loading LoCoMo-MC10 dataset...")
    mc10_path = os.path.expanduser('~/.cache/huggingface/hub/datasets--Percena--locomo-mc10/snapshots/7d59a0463d83f97b042684310c0b3d17553004cd/data/locomo_mc10.json')

    if not os.path.exists(mc10_path):
        print(f"Dataset not found at {mc10_path}")
        print("Please download: from datasets import load_dataset; load_dataset('Percena/locomo-mc10')")
        exit(1)

    with open(mc10_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    total_items = len(dataset)
    if limit:
        dataset = dataset[:min(limit, total_items)]

    print(f"Evaluating {len(dataset)} / {total_items} items")
    print(f"Provider: {provider_name}")
    print(f"Model: {model}")
    print(f"Shodh URL: {shodh_url}")
    print()

    # Run evaluation
    results: list[EvalResult] = []

    for item in tqdm(dataset, desc="Evaluating"):
        result = evaluate_single_item(
            item=item,
            provider=provider,
            client=client
        )
        results.append(result)

    # Calculate metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Overall accuracy — over JUDGED items only. An item the judge never
    # answered is an abstention, not a wrong answer: scoring it as wrong
    # silently converts an outage into a quality number.
    judged_results = [r for r in results if r.judged]
    unjudged = len(results) - len(judged_results)
    if not judged_results:
        print(
            f"\nFATAL: the judge returned no usable answer for any of "
            f"{len(results)} items. No accuracy number exists for this run — "
            f"check the LLM provider (credits, key, rate limits) and re-run."
        )
        sys.exit(2)
    if unjudged:
        print(
            f"\nWARNING: {unjudged}/{len(results)} items unjudged "
            f"({unjudged / len(results) * 100:.1f}%) — accuracy below is over "
            f"the {len(judged_results)} judged items only."
        )

    total_correct = sum(1 for r in judged_results if r.correct)
    overall_accuracy = total_correct / len(judged_results) * 100

    print(
        f"\nOverall Accuracy: {overall_accuracy:.2f}% "
        f"({total_correct}/{len(judged_results)} judged, {unjudged} unjudged)"
    )

    # Per-category accuracy
    by_type = defaultdict(list)
    for r in judged_results:
        by_type[r.question_type].append(r.correct)

    print("\nAccuracy by Question Type:")
    print("-" * 40)
    for qtype, correct_list in sorted(by_type.items()):
        acc = sum(correct_list) / len(correct_list) * 100
        print(f"  {qtype:20s}: {acc:6.2f}% ({sum(correct_list)}/{len(correct_list)})")

    # Latency stats
    avg_store = sum(r.latency_store_ms for r in results) / len(results)
    avg_recall = sum(r.latency_recall_ms for r in results) / len(results)
    avg_memories = sum(r.num_memories_stored for r in results) / len(results)

    print(f"\nLatency (avg):")
    print(f"  Store:  {avg_store:.1f} ms")
    print(f"  Recall: {avg_recall:.1f} ms")
    print(f"  Memories stored per conversation: {avg_memories:.1f}")

    # Save detailed results
    output_data = {
        "provider": provider_name,
        "model": model,
        "total_items": len(results),
        "judged_items": len(judged_results),
        "unjudged_items": unjudged,
        "overall_accuracy": overall_accuracy,
        "accuracy_by_type": {
            qtype: sum(correct_list) / len(correct_list) * 100
            for qtype, correct_list in by_type.items()
        },
        "latency_store_ms_avg": avg_store,
        "latency_recall_ms_avg": avg_recall,
        "results": [
            {
                "question_id": r.question_id,
                "question_type": r.question_type,
                "correct": r.correct,
                "judged": r.judged,
                "predicted_idx": r.predicted_idx,
                "correct_idx": r.correct_idx,
                "latency_store_ms": r.latency_store_ms,
                "latency_recall_ms": r.latency_recall_ms,
                "question_text": r.question_text,
                "retrieved_context": r.retrieved_context,
                "correct_answer": r.correct_answer,
                "predicted_answer": r.predicted_answer,
                "all_choices": r.all_choices
            }
            for r in results
        ]
    }

    # Print detailed failure analysis
    failures = [r for r in results if not r.correct]
    if failures:
        print(f"\n{'='*60}")
        print(f"FAILURE ANALYSIS ({len(failures)} failures)")
        print('='*60)
        for i, f in enumerate(failures[:10]):  # Show first 10 failures
            print(f"\n--- Failure {i+1}: {f.question_id} ({f.question_type}) ---")
            print(f"QUESTION: {f.question_text}")
            print(f"\nRETRIEVED CONTEXT:")
            print(f"{f.retrieved_context[:1500]}..." if len(f.retrieved_context) > 1500 else f.retrieved_context)
            print(f"\nCORRECT ANSWER [{f.correct_idx}]: {f.correct_answer}")
            print(f"PREDICTED [{f.predicted_idx}]: {f.predicted_answer}")
            print("-" * 40)

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    # Random baseline
    print(f"\nRandom baseline (10 choices): 10.00%")
    print(f"Improvement over random: {overall_accuracy - 10:.2f}%")

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoCoMo-MC10 Benchmark for Shodh-Memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OpenAI (requires OPENAI_API_KEY)
  python locomo_mc10_eval.py --provider openai --model gpt-4o-mini --limit 50

  # Baseten (OpenAI-compatible, uses your API key)
  python locomo_mc10_eval.py --provider openai-compatible \\
      --api-base https://inference.baseten.co/v1 \\
      --api-key YOUR_KEY --model openai/gpt-oss-120b --limit 100

  # Groq (FREE tier, very fast)
  python locomo_mc10_eval.py --provider openai-compatible \\
      --api-base https://api.groq.com/openai/v1 \\
      --model llama-3.1-8b-instant --limit 100

  # Ollama (FREE - local)
  python locomo_mc10_eval.py --provider ollama --model llama3.2 --limit 100
        """
    )
    parser.add_argument("--provider", default="openai",
                        choices=["openai", "openai-compatible", "anthropic", "ollama", "baseten"],
                        help="LLM provider to use")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name")
    parser.add_argument("--api-base", default=None, help="API base URL (required for openai-compatible)")
    parser.add_argument("--api-key", default=None, help="API key (or use env vars: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items (for testing)")
    parser.add_argument("--shodh-url", default="http://127.0.0.1:3030", help="Shodh-Memory server URL")
    parser.add_argument("--shodh-api-key", default="sk-shodh-dev-local-testing-key", help="Shodh-Memory API key")
    parser.add_argument("--output", default="locomo_results.json", help="Output file for results")
    parser.add_argument("--full", action="store_true", help="Run full evaluation (all 1986 items)")

    args = parser.parse_args()

    limit = None if args.full else (args.limit or 50)  # Default to 50 for quick test

    run_evaluation(
        provider_name=args.provider,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        limit=limit,
        shodh_url=args.shodh_url,
        shodh_api_key=args.shodh_api_key,
        output_file=args.output
    )
