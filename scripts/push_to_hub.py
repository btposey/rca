"""push_to_hub.py — upload fine-tuned models to HuggingFace Hub.

Pushes:
  - models/dispatcher-llama-1b  → bposey/rca-dispatcher-llama-1b
  - models/concierge-llama-3b   → bposey/rca-concierge-llama-3b

Reads HF token from ./HF_TOKEN file (KEY=value format) or HF_TOKEN env var.

Usage (run on GPU machine):
  uv run scripts/push_to_hub.py
  uv run scripts/push_to_hub.py --dispatcher-only
  uv run scripts/push_to_hub.py --concierge-only
"""
import os
import re
from pathlib import Path

import typer
from rich.console import Console

console = Console()
app = typer.Typer()

HF_USERNAME   = "bposey-flexion"
DISPATCHER_REPO = f"{HF_USERNAME}/rca-dispatcher-llama-1b"
CONCIERGE_REPO  = f"{HF_USERNAME}/rca-concierge-llama-3b"

MODEL_DIR = Path(__file__).parent.parent / "models"
TOKEN_FILE = Path(__file__).parent.parent / "HF_TOKEN"


def get_hf_token() -> str:
    # Try env var first
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    # Try HF_TOKEN file (KEY=value or bare token)
    if TOKEN_FILE.exists():
        content = TOKEN_FILE.read_text().strip()
        match = re.match(r"^HF_TOKEN\s*=\s*(.+)$", content)
        if match:
            return match.group(1).strip()
        return content
    raise ValueError(
        "No HuggingFace token found. Set HF_TOKEN env var or create an HF_TOKEN file."
    )


def push_model(local_path: Path, repo_id: str, token: str) -> None:
    from huggingface_hub import HfApi, create_repo

    console.print(f"[bold]Pushing {local_path.name} → {repo_id}[/bold]")

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, token=token, exist_ok=True, private=False)
        console.print(f"  Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        console.print(f"  [yellow]Repo creation note: {e}[/yellow]")

    # Upload all files in the model directory
    console.print(f"  Uploading from {local_path} ...")
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {local_path.name} fine-tuned model",
    )
    console.print(f"  [green]Done → https://huggingface.co/{repo_id}[/green]")


@app.command()
def main(
    dispatcher_only: bool = typer.Option(False, "--dispatcher-only"),
    concierge_only: bool = typer.Option(False, "--concierge-only"),
    username: str = typer.Option(HF_USERNAME, help="HuggingFace username"),
) -> None:
    try:
        from huggingface_hub import HfApi  # noqa: F401
    except ImportError:
        console.print("[red]huggingface_hub not installed. Run: uv pip install huggingface_hub[/red]")
        raise typer.Exit(1)

    token = get_hf_token()
    console.print(f"Authenticated as token: {token[:12]}...")

    dispatcher_repo = f"{username}/rca-dispatcher-llama-1b"
    concierge_repo  = f"{username}/rca-concierge-llama-3b"

    if not concierge_only:
        path = MODEL_DIR / "dispatcher-llama-1b"
        if not path.exists():
            console.print(f"[red]Dispatcher model not found at {path}[/red]")
        else:
            push_model(path, dispatcher_repo, token)

    if not dispatcher_only:
        path = MODEL_DIR / "concierge-llama-3b"
        if not path.exists():
            console.print(f"[red]Concierge model not found at {path}[/red]")
        else:
            push_model(path, concierge_repo, token)

    console.print("\n[bold green]Upload complete.[/bold green]")
    console.print(f"  Dispatcher: https://huggingface.co/{dispatcher_repo}")
    console.print(f"  Concierge:  https://huggingface.co/{concierge_repo}")


if __name__ == "__main__":
    app()
