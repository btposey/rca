"""ingest.py — embed restaurant records and load into pgvector.

Reads JSON files from data/restaurant_data/ or generates synthetic restaurants.
Each record: {name, cuisine, price_range, description}

Usage:
  uv run scripts/ingest.py                        # generate + ingest synthetic data
  uv run scripts/ingest.py --source path/to.json  # ingest from file
  uv run scripts/ingest.py --generate-only 200    # only generate synthetic data
"""
import asyncio
import json
import random
from pathlib import Path

import anthropic
import instructor
import typer
from rich.console import Console
from rich.progress import track
from sentence_transformers import SentenceTransformer
import asyncpg

from app.config import get_settings

console = Console()
app = typer.Typer()

DATA_DIR = Path(__file__).parent.parent / "data" / "restaurant_data"

CUISINES = [
    "Italian", "Japanese", "Mexican", "American", "Thai", "Indian",
    "Chinese", "French", "Korean", "Mediterranean", "Ethiopian", "Vietnamese",
]


async def generate_synthetic_restaurants(n: int = 200) -> list[dict]:
    settings = get_settings()
    raw_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    client = instructor.from_anthropic(raw_client)

    from pydantic import BaseModel

    class Restaurant(BaseModel):
        name: str
        cuisine: str
        price_range: float  # avg entree USD
        description: str    # 2-3 sentences, vivid

    restaurants = []
    console.print(f"Generating {n} synthetic restaurants via Claude...")
    for i in track(range(n), description="Generating..."):
        cuisine = random.choice(CUISINES)
        price = round(random.uniform(8, 95), 0)
        try:
            r: Restaurant = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Generate a fictional but realistic {cuisine} restaurant. "
                        f"Average entree price: ${price}. "
                        f"Include a creative name, and a vivid 2-sentence description "
                        f"highlighting the atmosphere and signature dishes."
                    ),
                }],
                response_model=Restaurant,
            )
            restaurants.append(r.model_dump())
        except Exception as e:
            console.print(f"[yellow]Warning: {e}[/yellow]")

    out_path = DATA_DIR / "synthetic_restaurants.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(restaurants, f, indent=2)
    console.print(f"[green]Saved {len(restaurants)} restaurants → {out_path}[/green]")
    return restaurants


async def ingest_to_db(restaurants: list[dict]) -> None:
    settings = get_settings()
    embedder = SentenceTransformer(settings.embedding_model)

    # asyncpg uses a different URL format
    db_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(db_url)

    try:
        await conn.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE TABLE IF NOT EXISTS restaurants (
                id          SERIAL PRIMARY KEY,
                name        TEXT NOT NULL,
                cuisine     TEXT,
                price_range FLOAT,
                tier        SMALLINT DEFAULT 3,
                description TEXT,
                embedding   VECTOR(384)
            );
        """)

        console.print(f"Embedding and inserting {len(restaurants)} restaurants...")
        for r in track(restaurants, description="Ingesting..."):
            text = f"{r['name']}. {r.get('description', '')} Cuisine: {r.get('cuisine', '')}."
            embedding = embedder.encode(text).tolist()
            await conn.execute(
                """INSERT INTO restaurants (name, cuisine, price_range, tier, description, embedding)
                   VALUES ($1, $2, $3, $4, $5, $6::vector)
                   ON CONFLICT DO NOTHING""",
                r["name"], r.get("cuisine"), r.get("price_range"), r.get("tier", 3),
                r.get("description"), str(embedding),
            )

        count = await conn.fetchval("SELECT COUNT(*) FROM restaurants")
        console.print(f"[green]Database now has {count} restaurants.[/green]")

        # Build IVFFlat index if enough rows
        if count >= 100:
            console.print("Building IVFFlat index...")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS restaurants_embedding_idx "
                "ON restaurants USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);"
            )
    finally:
        await conn.close()


@app.command()
def main(
    source: str = typer.Option("", help="Path to JSON file with restaurant records"),
    generate_count: int = typer.Option(200, help="Number of synthetic restaurants to generate"),
    generate_only: bool = typer.Option(False, help="Generate data file without ingesting"),
) -> None:
    asyncio.run(_main(source, generate_count, generate_only))


async def _main(source: str, generate_count: int, generate_only: bool) -> None:
    if source:
        with open(source) as f:
            restaurants = json.load(f)
        console.print(f"Loaded {len(restaurants)} restaurants from {source}")
    else:
        restaurants = await generate_synthetic_restaurants(generate_count)

    if not generate_only:
        await ingest_to_db(restaurants)


if __name__ == "__main__":
    app()
