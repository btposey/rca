"""finetune.py — QLoRA fine-tuning of Llama 3.2 1B on the dispatcher dataset.

Run on the GPU machine (192.168.200.100) with the [train] extras installed:
  uv pip install -e ".[train]"
  uv run scripts/finetune.py

The fine-tuned adapter is saved to models/dispatcher-lora/
Then merged and served via vLLM as dispatcher-llama-1b.
"""
import json
import os
from pathlib import Path

# Suppress transformers deprecation warnings from older Unsloth internals
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import typer
from rich.console import Console

console = Console()
app = typer.Typer()

DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"
MODEL_DIR = Path(os.environ.get("HF_HOME", "./models"))
OUTPUT_DIR = MODEL_DIR / "dispatcher-lora"
MERGED_DIR = MODEL_DIR / "dispatcher-llama-1b"

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

SYSTEM_PROMPT = (
    "You are an intent extraction engine for a restaurant recommendation system. "
    "Given a user message, extract persona, attack flag, search_predicate, and semantic_query. "
    "Respond only with valid JSON."
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_sample(sample: dict) -> str:
    """Format a training sample as a chat string for Unsloth."""
    user_msg = sample["input"]
    assistant_msg = json.dumps(sample["output"])
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n{assistant_msg}<|eot_id|>"
    )


@app.command()
def main(
    base_model: str = typer.Option(BASE_MODEL, help="HuggingFace model ID"),
    train_file: Path = typer.Option(DATA_DIR / "train.jsonl", help="Training data"),
    output_dir: Path = typer.Option(OUTPUT_DIR, help="LoRA adapter output"),
    merged_dir: Path = typer.Option(MERGED_DIR, help="Merged model output"),
    epochs: int = typer.Option(3),
    batch_size: int = typer.Option(4),
    grad_accum: int = typer.Option(4),
    lr: float = typer.Option(2e-4),
    max_seq_len: int = typer.Option(512),
    lora_r: int = typer.Option(16),
    lora_alpha: int = typer.Option(32),
) -> None:
    # Import here so the file is importable without GPU deps installed
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset
    except ImportError:
        console.print("[red]Train dependencies not installed. Run: uv pip install -e '.[train]'[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Loading base model:[/bold] {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_len,
        dtype=None,          # auto-detect
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )

    console.print(f"[bold]Loading training data:[/bold] {train_file}")
    raw = load_jsonl(train_file)
    texts = [format_sample(s) for s in raw]
    dataset = Dataset.from_dict({"text": texts})

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        args=training_args,
    )

    console.print("[bold green]Starting fine-tuning...[/bold green]")
    trainer.train()

    console.print(f"[bold]Saving LoRA adapter →[/bold] {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    console.print(f"[bold]Merging and saving full model →[/bold] {merged_dir}")
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    console.print("[bold green]Fine-tuning complete.[/bold green]")
    console.print(f"Serve with vLLM: vllm serve {merged_dir} --served-model-name dispatcher-llama-1b")


if __name__ == "__main__":
    app()
