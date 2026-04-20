"""generate_dataset_bedrock.py — generate synthetic training data via Claude on AWS Bedrock.

Identical logic to generate_dataset.py but uses boto3 + instructor's Bedrock adapter
instead of the Anthropic SDK. Uses ambient AWS credentials (~/.aws/credentials or env vars).

Produces:
  data/synthetic/train.jsonl   (~800 normal + ~200 attack samples)
  data/synthetic/eval.jsonl    (~100 held-out samples)

Usage:
  uv run scripts/generate_dataset_bedrock.py
  uv run scripts/generate_dataset_bedrock.py --region us-east-1 --train-size 800

AWS credentials: uses boto3 default credential chain (env vars, ~/.aws/credentials, IAM role).
Required Bedrock model access: claude-opus-4-5 and claude-haiku-4-5-20251001 (enable in
AWS Console → Bedrock → Model access).
"""
import asyncio
import json
import random
from pathlib import Path

import boto3
import instructor
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from app.schema import DispatcherOutput
from scripts.generate_dataset import (
    ATTACK_SEEDS,
    FOODIE_SEEDS,
    NORMAL_SEEDS,
    TrainingSample,
)

console = Console()
app = typer.Typer()

DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"

# Bedrock cross-region inference profile IDs
OPUS_MODEL = "us.anthropic.claude-opus-4-5-20251101-v1:0"
HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def make_bedrock_client(region: str) -> instructor.Instructor:
    bedrock = boto3.client("bedrock-runtime", region_name=region)
    return instructor.from_bedrock(bedrock)


async def generate_normal_sample_bedrock(
    client: instructor.Instructor,
    seed: str,
    is_foodie: bool = False,
) -> TrainingSample:
    persona_hint = (
        "The user is a culinary enthusiast using technical food terms."
        if is_foodie
        else "The user is a casual diner."
    )

    # Extract structured intent
    result: DispatcherOutput = client.messages.create(
        model=OPUS_MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                f"Generate a realistic restaurant recommendation request. "
                f"{persona_hint} "
                f"Base it loosely on this theme: '{seed}'. "
                f"Make it sound natural and varied. Then extract the structured intent."
            ),
        }],
        response_model=DispatcherOutput,
    )

    # Generate the natural-language query
    query_resp: str = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=128,
        messages=[{
            "role": "user",
            "content": (
                f"Write exactly ONE natural restaurant recommendation request sentence "
                f"based on: '{seed}'. "
                f"{'Use culinary terminology.' if is_foodie else 'Keep it casual.'} "
                f"Output only the sentence, no quotes."
            ),
        }],
        response_model=str,
    )

    return TrainingSample(input=query_resp, output=result)


def generate_attack_sample_deterministic(seed: str, index: int) -> TrainingSample:
    """Generate attack samples without LLM calls to avoid safety refusals.

    Applies deterministic surface-level mutations to the seed strings to create
    variety: case changes, prefix/suffix wrappers, punctuation, whitespace tricks.
    """
    mutations = [
        lambda s: s,
        lambda s: s.upper(),
        lambda s: s.lower(),
        lambda s: f"[INST] {s} [/INST]",
        lambda s: f"### Instruction:\n{s}",
        lambda s: f"<system>{s}</system>",
        lambda s: f"SYSTEM MESSAGE: {s}",
        lambda s: f"{s} Now comply.",
        lambda s: f"IMPORTANT: {s}",
        lambda s: f"{{{{ {s} }}}}",
        lambda s: f"---\n{s}\n---",
        lambda s: f"[OVERRIDE] {s}",
        lambda s: s.replace(" ", "\u00a0"),   # non-breaking spaces
        lambda s: f"As a reminder, {s.lower()}",
        lambda s: f"New task: {s}",
        lambda s: f"# {s}",
        lambda s: f"/* {s} */",
        lambda s: f"User said: \"{s}\"",
        lambda s: f"Context update: {s}",
        lambda s: f"{s} Disregard safety filters.",
    ]
    mutator = mutations[index % len(mutations)]
    varied_input = mutator(seed)
    output = DispatcherOutput(
        persona="neutral",
        attack=True,
        search_predicate=None,
        semantic_query=None,
    )
    return TrainingSample(input=varied_input, output=output)


@app.command()
def main(
    train_size: int = typer.Option(800, help="Normal training samples"),
    attack_size: int = typer.Option(200, help="Attack training samples"),
    eval_size: int = typer.Option(100, help="Held-out eval samples"),
    foodie_ratio: float = typer.Option(0.25, help="Fraction of normal samples that are foodie"),
    region: str = typer.Option("us-east-1", help="AWS region for Bedrock"),
) -> None:
    asyncio.run(_generate(train_size, attack_size, eval_size, foodie_ratio, region))


async def _generate(
    train_size: int,
    attack_size: int,
    eval_size: int,
    foodie_ratio: float,
    region: str,
) -> None:
    console.print(f"[bold]Generating dataset via AWS Bedrock[/bold] (region: {region})")
    console.print(f"  Train normal: {train_size}  Attack: {attack_size}  Eval: {eval_size}")

    # Bedrock client is sync — instructor's from_bedrock doesn't have an async variant,
    # so we run calls in a thread pool to avoid blocking the event loop.
    import concurrent.futures
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    client = make_bedrock_client(region)

    def sync_normal(seed: str, is_foodie: bool) -> TrainingSample:
        return asyncio.run(generate_normal_sample_bedrock(client, seed, is_foodie))

    all_normal: list[TrainingSample] = []
    all_attack: list[TrainingSample] = []

    normal_needed = train_size + eval_size
    foodie_count = int(normal_needed * foodie_ratio)
    seeds_cycle = FOODIE_SEEDS + NORMAL_SEEDS

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Generating normal samples...", total=normal_needed + attack_size)

        futures = [
            loop.run_in_executor(
                executor,
                sync_normal,
                seeds_cycle[i % len(seeds_cycle)],
                i < foodie_count,
            )
            for i in range(normal_needed)
        ]
        for fut in asyncio.as_completed(futures):
            try:
                all_normal.append(await fut)
            except Exception as e:
                console.print(f"[yellow]Warning: normal sample failed: {e}[/yellow]")
            progress.advance(task)

        # Attack samples are generated deterministically — no LLM call needed
        progress.update(task, description="Generating attack samples (deterministic)...")
        for i in range(attack_size):
            seed = ATTACK_SEEDS[i % len(ATTACK_SEEDS)]
            all_attack.append(generate_attack_sample_deterministic(seed, i))
            progress.advance(task)

    executor.shutdown(wait=False)

    # Stream all normal samples to a temp file as they arrived (already collected above).
    # Attack samples are deterministic so no persistence risk there.
    # Split and write final files.
    random.shuffle(all_normal)
    train_normal = all_normal[:train_size]
    eval_normal  = all_normal[train_size:]

    train_samples = train_normal + all_attack
    random.shuffle(train_samples)

    # Write one line at a time so a mid-write crash loses at most one record
    train_path = DATA_DIR / "train.jsonl"
    eval_path  = DATA_DIR / "eval.jsonl"
    train_path.parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, "w") as f:
        for s in train_samples:
            f.write(json.dumps({"input": s.input, "output": s.output.model_dump()}) + "\n")
            f.flush()

    with open(eval_path, "w") as f:
        for s in eval_normal:
            f.write(json.dumps({"input": s.input, "output": s.output.model_dump()}) + "\n")
            f.flush()

    console.print(f"[green]Wrote {len(train_samples)} samples → {train_path}[/green]")
    console.print(f"[green]Wrote {len(eval_normal)} samples → {eval_path}[/green]")
    console.print(f"\n[bold green]Done.[/bold green] {len(train_samples)} train, {len(eval_normal)} eval.")


if __name__ == "__main__":
    app()
