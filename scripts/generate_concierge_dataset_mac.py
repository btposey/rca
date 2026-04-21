"""generate_concierge_dataset_mac.py — pipeline-driven Concierge dataset generation.

Derivative of generate_concierge_dataset.py designed to run on the macOS dev machine:
  - Calls the GPU machine's Dispatcher (vLLM) and Librarian (pgvector) over the LAN
  - Uses the macOS AWS credentials context for Bedrock (Claude gold response generation)

Prerequisites:
  - deploy_native_vllm.sh running on GPU machine (192.168.200.100)
  - Ports 8000 (dispatcher) and 5432 (postgres) reachable from Mac

Usage:
  uv run scripts/generate_concierge_dataset_mac.py
  uv run scripts/generate_concierge_dataset_mac.py --train-size 500 --eval-size 60
"""
import json
import os
import random
from pathlib import Path

import boto3
import instructor
import typer
from pydantic import BaseModel
from rich.console import Console
from rich.progress import track

# Point the app config at the GPU machine before importing services
GPU_HOST = os.environ.get("GPU_HOST", "192.168.200.100")
os.environ.setdefault("VLLM_DISPATCHER_BASE_URL", f"http://{GPU_HOST}:8000/v1")
os.environ.setdefault("VLLM_CONCIERGE_BASE_URL",  f"http://{GPU_HOST}:8001/v1")
os.environ.setdefault("DATABASE_URL", f"postgresql+asyncpg://rca:rca@{GPU_HOST}:5432/rca")

from app.config import get_settings  # noqa: E402 — must come after env override
from app.schema import AgentState, DispatcherOutput  # noqa: E402
from app.services import librarian  # noqa: E402
from scripts.generate_concierge_dataset import (  # noqa: E402
    HAIKU_MODEL,
    OPUS_MODEL,
    NO_RESULTS_PHRASE,
    PERSONA_INSTRUCTIONS,
    QUERY_SEEDS,
    SYSTEM_PROMPT_TEMPLATE,
    TIER_LEGEND,
    ConciergeResponse,
    NaturalQuery,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"

console = Console()
app = typer.Typer()


async def run_dispatcher(user_query: str) -> DispatcherOutput:
    from openai import AsyncOpenAI
    from app.services.dispatcher import SYSTEM_PROMPT
    settings = get_settings()
    client = AsyncOpenAI(base_url=settings.vllm_dispatcher_base_url, api_key="not-needed")
    response = await client.chat.completions.create(
        model=settings.vllm_dispatcher_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        temperature=settings.dispatcher_temperature,
        max_tokens=settings.dispatcher_max_tokens,
    )
    raw = (response.choices[0].message.content or "").strip()
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw)
        return DispatcherOutput.model_validate(obj)
    except Exception:
        return DispatcherOutput()


async def generate_sample(
    bedrock_client: instructor.Instructor,
    persona: str,
    seed: str,
) -> dict | None:
    # Step 1: Generate natural query via Haiku
    natural: NaturalQuery = bedrock_client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=96,
        messages=[{
            "role": "user",
            "content": (
                f"Write exactly ONE natural restaurant request sentence from a {persona} diner. "
                f"Base it on this theme: '{seed}'. "
                f"{'Use culinary terminology.' if persona == 'foodie' else 'Keep it casual and natural.'} "
                f"Output only the sentence, no quotes."
            ),
        }],
        response_model=NaturalQuery,
    )
    query = natural.query

    # Step 2: Dispatcher (GPU machine vLLM)
    dispatcher_output = await run_dispatcher(query)
    state = AgentState(user_query=query)
    state.persona = dispatcher_output.persona or persona
    state.attack = dispatcher_output.attack
    state.search_predicate = dispatcher_output.search_predicate
    state.semantic_query = dispatcher_output.semantic_query

    if state.attack:
        return None  # skip attack-classified queries for concierge training

    # Step 3: Librarian (GPU machine postgres)
    state = await librarian.search(state)
    if not state.retrieved_results:
        return None

    # Step 4: Build prompt (identical to inference)
    system = SYSTEM_PROMPT_TEMPLATE.format(
        persona_instruction=PERSONA_INSTRUCTIONS[persona],
        tier_legend=TIER_LEGEND,
        no_results=NO_RESULTS_PHRASE,
    )
    user_msg = (
        f"User request: {query}\n\n"
        f"Restaurant candidates (ONLY recommend from this list — no other options exist):\n"
        f"{json.dumps(state.retrieved_results, indent=2)}"
    )

    # Step 5: Gold response via Opus (Mac Bedrock auth)
    response: ConciergeResponse = bedrock_client.messages.create(
        model=OPUS_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": f"{system}\n\n{user_msg}"}],
        response_model=ConciergeResponse,
    )

    return {
        "system": system,
        "input": user_msg,
        "output": response.model_dump(),
        "persona": persona,
        "retrieved_count": len(state.retrieved_results),
    }


@app.command()
def main(
    train_size: int = typer.Option(500, help="Training samples"),
    eval_size: int = typer.Option(60, help="Held-out eval samples"),
    region: str = typer.Option("us-east-1"),
) -> None:
    import asyncio
    asyncio.run(_generate(train_size, eval_size, region))


async def _generate(train_size: int, eval_size: int, region: str) -> None:
    from collections import Counter

    bedrock = boto3.client("bedrock-runtime", region_name=region)
    client = instructor.from_bedrock(bedrock)

    settings = get_settings()
    console.print(f"[bold]Generating {train_size + eval_size} pipeline-driven samples[/bold]")
    console.print(f"  Dispatcher: {settings.vllm_dispatcher_base_url}")
    console.print(f"  DB:         {settings.database_url}")

    total = train_size + eval_size
    persona_list = (["foodie", "normie", "neutral"] * (total // 3 + 1))[:total]
    random.shuffle(persona_list)

    train_path = DATA_DIR / "concierge_train.jsonl"
    eval_path  = DATA_DIR / "concierge_eval.jsonl"
    train_path.parent.mkdir(parents=True, exist_ok=True)

    train_written = eval_written = errors = skipped = 0
    i = 0

    with open(train_path, "w") as f_train, open(eval_path, "w") as f_eval:
        for persona in track(persona_list, description="Generating..."):
            seed = random.choice(QUERY_SEEDS[persona])
            try:
                sample = await generate_sample(client, persona, seed)
                if sample is None:
                    skipped += 1
                    continue
                dest = f_train if i < train_size else f_eval
                dest.write(json.dumps(sample) + "\n")
                dest.flush()
                if i < train_size:
                    train_written += 1
                else:
                    eval_written += 1
                i += 1
            except Exception as e:
                console.print(f"[yellow]Warning ({persona}): {e}[/yellow]")
                errors += 1

    console.print(f"\n{train_written} train, {eval_written} eval ({errors} errors, {skipped} skipped).")
    with open(train_path) as f:
        samples = [json.loads(l) for l in f if l.strip()]
    console.print(f"Persona dist: {dict(Counter(s['persona'] for s in samples))}")
    console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    app()
