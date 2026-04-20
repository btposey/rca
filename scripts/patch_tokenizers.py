"""patch_tokenizers.py — fix tokenizer files in fine-tuned model directories.

Unsloth saves a custom TokenizersBackend tokenizer class that vLLM cannot load.
This script replaces the tokenizer files in each fine-tuned model directory with
the standard tokenizer from the corresponding base model, which is identical in
vocabulary and behavior but compatible with vLLM and standard transformers.

Run once on the GPU machine after finetune.py / finetune_concierge.py complete:
  uv run scripts/patch_tokenizers.py

Safe to re-run — only tokenizer files are touched, weights are not modified.
"""
import os
from pathlib import Path

import typer
from rich.console import Console

console = Console()
app = typer.Typer()

MODEL_DIR = Path(os.environ.get("HF_HOME", "./models"))

PATCH_MAP = {
    "dispatcher-llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "concierge-llama-3b":  "meta-llama/Llama-3.2-3B-Instruct",
}


@app.command()
def main(
    model_dir: Path = typer.Option(MODEL_DIR, help="Directory containing fine-tuned models"),
) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        console.print("[red]transformers not installed. Run: uv pip install -e '.[train]'[/red]")
        raise typer.Exit(1)

    for model_name, base_model in PATCH_MAP.items():
        target = model_dir / model_name
        if not target.exists():
            console.print(f"[yellow]Skipping {model_name} — not found at {target}[/yellow]")
            continue

        # Check if patch is needed
        cfg_path = target / "tokenizer_config.json"
        if cfg_path.exists():
            import json
            with open(cfg_path) as f:
                cfg = json.load(f)
            if cfg.get("tokenizer_class") not in ("TokenizersBackend", None) and \
               (target / "tokenizer.json").exists():
                console.print(f"[green]Already patched:[/green] {model_name}")
                continue

        console.print(f"Patching tokenizer for [bold]{model_name}[/bold] from {base_model}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.save_pretrained(str(target))
        console.print(f"[green]Done:[/green] {model_name}")

    console.print("\n[bold green]All tokenizers patched.[/bold green]")


if __name__ == "__main__":
    app()
