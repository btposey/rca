"""evaluate_concierge.py — score the fine-tuned Concierge against concierge_eval.jsonl.

Metrics:
  - json_parse_rate    : fraction of responses that parse as valid JSON
  - suggestion_present : fraction with non-empty suggestion field
  - elaboration_present: fraction with non-empty elaboration field
  - avg_suggestion_len : average character length of suggestion
  - avg_elaboration_len: average character length of elaboration
  - persona_adherence  : heuristic — foodie responses should contain culinary terms,
                         normie responses should not
  - tier_mention_rate  : fraction of award-present scenarios where response mentions
                         accolade language (award, Michelin, James Beard, acclaimed, etc.)

Usage (run on GPU machine after vLLM is serving the fine-tuned Concierge):
  uv run scripts/evaluate_concierge.py
  uv run scripts/evaluate_concierge.py --model-name concierge-llama-3b
"""
import asyncio
import json
import re
from pathlib import Path

import instructor
import typer
from openai import AsyncOpenAI
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer()

DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"

FOODIE_TERMS = {
    "maillard", "umami", "terroir", "mise en place", "tasting menu", "omakase",
    "charcuterie", "tartare", "gastrique", "confit", "braise", "fond", "roux",
    "emulsion", "reduction", "sear", "render", "blanch", "julienne", "brunoise",
    "ferment", "cured", "koji", "dashi", "umami", "amuse", "bouche", "deglaze",
}

ACCOLADE_TERMS = {
    "award", "michelin", "james beard", "acclaimed", "celebrated", "starred",
    "recognition", "distinguished", "renowned", "destination", "exceptional",
}


class ConciergeOutput(BaseModel):
    suggestion: str = ""
    elaboration: str = ""


async def run_inference(
    client: instructor.AsyncInstructor,
    model: str,
    sample: dict,
    temperature: float,
    max_tokens: int,
) -> tuple[ConciergeOutput, bool]:
    """Returns (output, json_parsed_ok)."""
    try:
        result: ConciergeOutput = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sample["system"]},
                {"role": "user", "content": sample["input"]},
            ],
            response_model=ConciergeOutput,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return result, True
    except Exception:
        return ConciergeOutput(), False


def check_persona_adherence(output: ConciergeOutput, persona: str) -> bool:
    text = (output.suggestion + " " + output.elaboration).lower()
    has_foodie_term = any(term in text for term in FOODIE_TERMS)
    if persona == "foodie":
        return has_foodie_term
    elif persona == "normie":
        return not has_foodie_term
    return True  # neutral — no strong expectation


def check_tier_mention(output: ConciergeOutput, scenario: str) -> bool:
    if "award" not in scenario and "mostly_award" not in scenario:
        return True  # not applicable
    text = (output.suggestion + " " + output.elaboration).lower()
    return any(term in text for term in ACCOLADE_TERMS)


@app.command()
def main(
    model_name: str = typer.Option("concierge-llama-3b"),
    eval_file: Path = typer.Option(DATA_DIR / "concierge_eval.jsonl"),
    vllm_url: str = typer.Option("http://localhost:8000/v1"),
    temperature: float = typer.Option(0.7),
    max_tokens: int = typer.Option(512),
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

    raw_client = AsyncOpenAI(base_url=vllm_url, api_key="not-needed")
    client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)

    outputs: list[ConciergeOutput] = []
    json_ok: list[bool] = []
    inference_errors = 0

    for sample in samples:
        try:
            out, parsed = await run_inference(client, model_name, sample, temperature, max_tokens)
            outputs.append(out)
            json_ok.append(parsed)
        except Exception as e:
            console.print(f"[yellow]Inference error: {e}[/yellow]")
            outputs.append(ConciergeOutput())
            json_ok.append(False)
            inference_errors += 1

    n = len(samples)
    json_parse_rate      = sum(json_ok) / n
    suggestion_present   = sum(1 for o in outputs if o.suggestion.strip()) / n
    elaboration_present  = sum(1 for o in outputs if o.elaboration.strip()) / n
    avg_sug_len          = sum(len(o.suggestion) for o in outputs) / n
    avg_elab_len         = sum(len(o.elaboration) for o in outputs) / n

    persona_checks = [
        check_persona_adherence(o, s["persona"])
        for o, s in zip(outputs, samples)
    ]
    persona_adherence = sum(persona_checks) / n

    tier_checks = [
        check_tier_mention(o, s["tier_scenario"])
        for o, s in zip(outputs, samples)
        if "award" in s.get("tier_scenario", "")
    ]
    tier_mention_rate = sum(tier_checks) / len(tier_checks) if tier_checks else None

    table = Table(title=f"Concierge Evaluation — {model_name}")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Samples", str(n))
    table.add_row("Inference errors", str(inference_errors))
    table.add_row("JSON parse rate", f"{json_parse_rate:.3f}")
    table.add_row("Suggestion present", f"{suggestion_present:.3f}")
    table.add_row("Elaboration present", f"{elaboration_present:.3f}")
    table.add_row("Avg suggestion length", f"{avg_sug_len:.0f} chars")
    table.add_row("Avg elaboration length", f"{avg_elab_len:.0f} chars")
    table.add_row("Persona adherence", f"{persona_adherence:.3f}")
    if tier_mention_rate is not None:
        table.add_row("Tier-4 accolade mention rate", f"{tier_mention_rate:.3f}")

    console.print(table)

    metrics = {
        "model": model_name,
        "n": n,
        "json_parse_rate": json_parse_rate,
        "suggestion_present": suggestion_present,
        "elaboration_present": elaboration_present,
        "avg_suggestion_len": avg_sug_len,
        "avg_elaboration_len": avg_elab_len,
        "persona_adherence": persona_adherence,
        "tier_mention_rate": tier_mention_rate,
    }
    out_path = eval_file.parent / f"eval_results_{model_name.replace('/', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    console.print(f"[green]Results saved → {out_path}[/green]")


if __name__ == "__main__":
    app()
