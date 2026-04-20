"""train_concierge.py — run Concierge training pipeline on the GPU machine.

Prerequisite: run generate_concierge_dataset.py first to produce concierge_train.jsonl and concierge_eval.jsonl.

Steps:
  1. Install train dependencies
  2. Fine-tune Llama 3.2 3B via QLoRA
  3. Evaluate fine-tuned Concierge against concierge_eval.jsonl

Usage (run directly on GPU machine):
  uv run scripts/train_concierge.py

  # Re-run eval only (model already fine-tuned):
  uv run scripts/train_concierge.py --skip-finetune
"""
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

console = Console()
app = typer.Typer()

PROJECT_PATH = Path(__file__).parent.parent


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    return subprocess.run(cmd, check=check, cwd=PROJECT_PATH)


@app.command()
def main(
    skip_finetune: bool = typer.Option(False, help="Skip fine-tuning (re-run eval only)"),
    skip_eval: bool = typer.Option(False, help="Skip evaluation"),
) -> None:
    if not skip_finetune:
        console.rule("[bold]Step 1: Install train dependencies[/bold]")
        venv_uv = Path(sys.executable).parent / "uv"
        uv_cmd = str(venv_uv) if venv_uv.exists() else "uv"
        run([uv_cmd, "pip", "install", "-e", ".[train,serve]", "--quiet"])

        console.rule("[bold]Step 2: Fine-tune Concierge (Llama 3.2 3B)[/bold]")
        run([sys.executable, "-m", "scripts.finetune_concierge"])

        console.rule("[bold]Step 2b: Patch tokenizer for vLLM compatibility[/bold]")
        run([sys.executable, "-m", "scripts.patch_tokenizers"])

    if not skip_eval:
        console.rule("[bold]Step 3: Evaluate Concierge[/bold]")
        run([sys.executable, "-m", "scripts.evaluate_concierge"])

    console.print("\n[bold green]Concierge training pipeline complete.[/bold green]")
    console.print("Serve with: vllm serve models/concierge-llama-3b --served-model-name concierge-llama-3b")


if __name__ == "__main__":
    app()
