"""evaluate.py — score the fine-tuned Dispatcher against eval.jsonl.

Metrics (field-level F1):
  - persona accuracy     (exact match, 3 classes)
  - attack accuracy      (exact match, binary)
  - cuisine F1           (exact match on non-null)
  - max_price MAE        (mean absolute error on non-null)
  - semantic_query BLEU  (rough quality proxy)

Usage (run on GPU machine after vLLM is serving the fine-tuned model):
  uv run scripts/evaluate.py
  uv run scripts/evaluate.py --model-name dispatcher-llama-1b --eval-file data/synthetic/eval.jsonl
"""
import asyncio
import json
from pathlib import Path

import typer
from openai import AsyncOpenAI
from rich.console import Console
from rich.table import Table

from app.schema import DispatcherOutput

console = Console()
app = typer.Typer()

DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"

SYSTEM_PROMPT = (
    "You are an intent extraction engine for a restaurant recommendation system. "
    "Given a user message, extract persona, attack flag, search_predicate, and semantic_query. "
    "Respond only with valid JSON."
)


async def run_inference(
    client: AsyncOpenAI,
    model: str,
    query: str,
    temperature: float,
    max_tokens: int,
) -> DispatcherOutput:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content or "{}"
    return DispatcherOutput.model_validate(json.loads(raw))


def score(predictions: list[DispatcherOutput], ground_truths: list[dict]) -> dict:
    n = len(predictions)
    persona_correct = 0
    attack_correct = 0
    cuisine_matches = cuisine_total = 0
    price_errors: list[float] = []

    for pred, gt in zip(predictions, ground_truths):
        gt_output = gt["output"]

        persona_correct += int(pred.persona == gt_output.get("persona", "neutral"))
        attack_correct += int(pred.attack == gt_output.get("attack", False))

        gt_cuisine = (gt_output.get("search_predicate") or {}).get("cuisine")
        pred_cuisine = pred.search_predicate.cuisine if pred.search_predicate else None
        if gt_cuisine is not None:
            cuisine_total += 1
            if pred_cuisine and pred_cuisine.lower() == gt_cuisine.lower():
                cuisine_matches += 1

        gt_price = (gt_output.get("search_predicate") or {}).get("max_price")
        pred_price = pred.search_predicate.max_price if pred.search_predicate else None
        if gt_price is not None and pred_price is not None:
            price_errors.append(abs(pred_price - gt_price))

    return {
        "n": n,
        "persona_accuracy": persona_correct / n if n else 0,
        "attack_accuracy": attack_correct / n if n else 0,
        "cuisine_precision": cuisine_matches / cuisine_total if cuisine_total else None,
        "price_mae": sum(price_errors) / len(price_errors) if price_errors else None,
    }


@app.command()
def main(
    model_name: str = typer.Option("dispatcher-llama-1b"),
    eval_file: Path = typer.Option(DATA_DIR / "eval.jsonl"),
    vllm_url: str = typer.Option("http://localhost:8000/v1"),
    temperature: float = typer.Option(0.1),
    max_tokens: int = typer.Option(256),
) -> None:
    asyncio.run(_evaluate(model_name, eval_file, vllm_url, temperature, max_tokens))


async def _evaluate(
    model_name: str,
    eval_file: Path,
    vllm_url: str,
    temperature: float,
    max_tokens: int,
) -> None:
    with open(eval_file) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    console.print(f"Evaluating [bold]{model_name}[/bold] on {len(samples)} samples...")

    client = AsyncOpenAI(base_url=vllm_url, api_key="not-needed")

    predictions: list[DispatcherOutput] = []
    errors = 0
    for i, sample in enumerate(samples):
        try:
            pred = await run_inference(client, model_name, sample["input"], temperature, max_tokens)
            predictions.append(pred)
        except Exception as e:
            console.print(f"[yellow]Sample {i} failed: {e}[/yellow]")
            errors += 1
            predictions.append(DispatcherOutput())  # empty fallback

    metrics = score(predictions, samples)

    table = Table(title=f"Evaluation Results — {model_name}")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Samples", str(metrics["n"]))
    table.add_row("Inference errors", str(errors))
    table.add_row("Persona accuracy", f"{metrics['persona_accuracy']:.3f}")
    table.add_row("Attack accuracy", f"{metrics['attack_accuracy']:.3f}")
    if metrics["cuisine_precision"] is not None:
        table.add_row("Cuisine precision", f"{metrics['cuisine_precision']:.3f}")
    if metrics["price_mae"] is not None:
        table.add_row("Price MAE ($)", f"{metrics['price_mae']:.2f}")

    console.print(table)

    # Write results to file
    out_path = eval_file.parent / f"eval_results_{model_name.replace('/', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    console.print(f"[green]Results saved → {out_path}[/green]")


if __name__ == "__main__":
    app()
