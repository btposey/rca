"""ingest_bedrock.py — generate synthetic restaurants via Bedrock and write to JSON.

Derivative of ingest.py using boto3 instead of the Anthropic SDK.
Uses ambient AWS credentials. DB ingestion unchanged — handled by ingest.py once
the pod is running.

Usage:
  uv run scripts/ingest_bedrock.py                        # regenerate 200 restaurants
  uv run scripts/ingest_bedrock.py --generate-count 300
  uv run scripts/ingest_bedrock.py --region us-west-2
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import boto3
import instructor
import typer
from pydantic import BaseModel
from rich.console import Console
from rich.progress import track

DATA_DIR = Path(__file__).parent.parent / "data" / "restaurant_data"

HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Evenly distributed so each cuisine gets ~14-17 restaurants across 200
CUISINES = [
    "Italian", "Japanese", "Mexican", "American", "Thai", "Indian",
    "Chinese", "French", "Korean", "Mediterranean", "Ethiopian", "Vietnamese",
]

# Name style inspirations per cuisine — steers the model away from generic patterns
NAME_STYLES = {
    "Italian":       ["trattoria", "osteria", "named after a founder", "a neighborhood", "a saint", "an ingredient"],
    "Japanese":      ["a season or nature element", "a district or town", "a craft or technique", "a poetic phrase"],
    "Mexican":       ["a family name", "a local landmark", "a regional ingredient", "a color or feeling"],
    "American":      ["a street or neighborhood", "a founder's name", "a local landmark", "an era or craft"],
    "Thai":          ["a Thai word for a flavor or place", "a flower or herb", "a river or region"],
    "Indian":        ["a spice or ingredient", "a region or city", "a historical figure or dynasty"],
    "Chinese":       ["a dynasty or era", "a lucky symbol", "a regional specialty", "a philosopher"],
    "French":        ["a French village", "a cooking technique", "a French word for a feeling or season"],
    "Korean":        ["a Korean word for a place or craft", "a neighborhood in Seoul", "a traditional object"],
    "Mediterranean": ["a coastal town", "a sea or wind", "an herb or ingredient", "a myth or legend"],
    "Ethiopian":     ["an Amharic word", "a region or city", "a historical figure"],
    "Vietnamese":    ["a Vietnamese word for nature", "a city or river", "a herb or flower"],
}

console = Console()
app = typer.Typer()


class RestaurantRaw(BaseModel):
    name: str
    cuisine: str
    price_range: float      # avg entree USD
    atmosphere: str         # 1-2 sentences: setting, vibe, decor
    dish_notes: str         # 1-2 sentences: specific dishes a reviewer tried, named dishes only


def build_prompt(cuisine: str, price: float, used_names: list[str], style_hint: str) -> str:
    avoid = ""
    if used_names:
        sample = used_names[-8:]  # last 8 to keep prompt short
        avoid = f" Do NOT use any of these already-used names: {', '.join(sample)}."
    return (
        f"Generate a fictional but realistic {cuisine} restaurant with an average entree price of ${price:.0f}. "
        f"Name it using inspiration from: {style_hint}.{avoid} "
        f"The name must be unique and specific — avoid generic words like 'Ember', 'Lantern', 'Nocturne', 'Twilight', 'Mekong', 'Siam', 'Lotus', 'Dragon', 'Jade', 'Maharaja', 'Étoile', 'Luminosa', 'Maredolce', 'Stella', 'Hanok'. "
        f"For atmosphere: describe the physical space and vibe in 1-2 sentences. "
        f"For dish_notes: name 1-2 specific dishes a reviewer ordered and describe what made them memorable, in 1-2 sentences."
    )


def generate_synthetic_restaurants(n: int, region: str) -> list[dict]:
    bedrock = boto3.client("bedrock-runtime", region_name=region)
    client = instructor.from_bedrock(bedrock)

    # Distribute cuisines evenly
    cuisine_list = (CUISINES * (n // len(CUISINES) + 1))[:n]
    random.shuffle(cuisine_list)

    used_names_by_cuisine: dict[str, list[str]] = defaultdict(list)
    restaurants = []

    console.print(f"Generating {n} synthetic restaurants via Bedrock ({region})...")
    for cuisine in track(cuisine_list, description="Generating..."):
        price = round(random.uniform(8, 95), 0)
        style_hint = random.choice(NAME_STYLES.get(cuisine, ["a local landmark"]))
        prompt = build_prompt(cuisine, price, used_names_by_cuisine[cuisine], style_hint)

        try:
            r: RestaurantRaw = client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=384,
                messages=[{"role": "user", "content": prompt}],
                response_model=RestaurantRaw,
            )
            used_names_by_cuisine[cuisine].append(r.name)
            # Merge atmosphere + dish_notes into a single description field
            restaurants.append({
                "name": r.name,
                "cuisine": r.cuisine,
                "price_range": r.price_range,
                "description": f"{r.atmosphere} {r.dish_notes}".strip(),
            })
        except Exception as e:
            console.print(f"[yellow]Warning ({cuisine}): {e}[/yellow]")

    out_path = DATA_DIR / "synthetic_restaurants.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(restaurants, f, indent=2)
    console.print(f"[green]Saved {len(restaurants)} restaurants → {out_path}[/green]")
    return restaurants


@app.command()
def main(
    generate_count: int = typer.Option(200, help="Number of synthetic restaurants to generate"),
    region: str = typer.Option("us-east-1", help="AWS region for Bedrock"),
) -> None:
    generate_synthetic_restaurants(generate_count, region)
    console.print(
        "[bold green]Done.[/bold green] "
        "Run ingest.py --source data/restaurant_data/synthetic_restaurants.json when the pod is up."
    )


if __name__ == "__main__":
    app()
