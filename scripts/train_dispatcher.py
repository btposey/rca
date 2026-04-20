"""train_dispatcher.py — run Dispatcher training pipeline on the GPU machine.

Prerequisite: run generate_dataset_bedrock.py first to produce train.jsonl and eval.jsonl.

Steps:
  1. Install train dependencies
  2. Fine-tune Llama 3.2 1B via QLoRA
  3. Evaluate fine-tuned Dispatcher against eval.jsonl

Usage (run directly on GPU machine):
  uv run scripts/train_dispatcher.py

  # Re-run eval only (model already fine-tuned):
  uv run scripts/train_dispatcher.py --skip-finetune
"""
import subprocess
import sys
from pathlib import Path
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

        console.rule("[bold]Step 2: Fine-tune Dispatcher (Llama 3.2 1B)[/bold]")
        run([sys.executable, "-m", "scripts.finetune"])

        console.rule("[bold]Step 2b: Patch tokenizer for vLLM compatibility[/bold]")
        run([sys.executable, "-m", "scripts.patch_tokenizers"])

    if not skip_eval:
        console.rule("[bold]Step 3: Evaluate Dispatcher[/bold]")
        run([sys.executable, "-m", "scripts.evaluate"])

    console.print("\n[bold green]Dispatcher training pipeline complete.[/bold green]")
    console.print("Serve with: vllm serve models/dispatcher-llama-1b --served-model-name dispatcher-llama-1b")


if __name__ == "__main__":
    app()
