"""backfill_tiers.py — assign quality tier to each restaurant in synthetic_restaurants.json.

Reads the existing JSON, asks Claude to assign a tier based on the description,
then writes the result back in place with a `tier` field added.

Tier scale:
  4 = Award-winning (Michelin / James Beard caliber, destination dining)
  3 = Solid (reliable neighborhood favorite, good value)
  2 = Mixed (one strong element offset by a clear flaw)
  1 = Poor (actively bad: overpriced, declining quality, rude staff, etc.)

Target distribution across 200 restaurants: ~30×4, ~110×3, ~40×2, ~20×1

Usage:
  uv run scripts/backfill_tiers.py
  uv run scripts/backfill_tiers.py --input data/restaurant_data/synthetic_restaurants.json
"""
import json
import random
from pathlib import Path

import boto3
import instructor
import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import track

DATA_DIR = Path(__file__).parent.parent / "data" / "restaurant_data"
HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Target counts per tier for 200 restaurants
TIER_TARGETS = {4: 30, 3: 110, 2: 40, 1: 20}

# Tier descriptions injected into the LLM prompt
TIER_DESCRIPTIONS = {
    4: "Award-winning: exceptional technique, destination dining, Michelin/James Beard caliber. Description reads as truly special.",
    3: "Solid: reliable, good value, neighborhood favorite. A place people return to but wouldn't call life-changing.",
    2: "Mixed: one strong element offset by a notable flaw — great food but slow service, amazing vibe but inconsistent kitchen, excellent value but cramped and loud.",
    1: "Poor: actively bad in some dimension — overpriced for what it is, food quality has declined, rude or inattentive staff, disappointing execution of dishes.",
}

console = Console()
app = typer.Typer()


class TierAssignment(BaseModel):
    tier: int = Field(ge=1, le=4)
    tier_rationale: str  # one sentence explaining the assignment


def assign_tier_llm(
    client: instructor.Instructor,
    restaurant: dict,
    quota_remaining: dict[int, int],
) -> TierAssignment:
    """Ask the LLM to assign a tier, steering it toward quota targets."""
    available = [t for t, q in quota_remaining.items() if q > 0]
    available_str = ", ".join(
        f"{t} ({TIER_DESCRIPTIONS[t]})" for t in sorted(available, reverse=True)
    )
    # Weighted hint so the model distributes naturally
    weights = {t: quota_remaining[t] for t in available}
    total = sum(weights.values())
    hint_tier = random.choices(list(weights.keys()), weights=list(weights.values()))[0]

    prompt = (
        f"You are assigning quality tiers to restaurant records for a recommendation dataset.\n\n"
        f"Restaurant: {restaurant['name']} ({restaurant['cuisine']}, ${restaurant['price_range']:.0f} avg)\n"
        f"Description: {restaurant['description']}\n\n"
        f"Available tiers (assign one): {available_str}\n\n"
        f"Based on the description's tone and content, assign the most appropriate tier. "
        f"Consider tier {hint_tier} as a strong candidate given current distribution needs. "
        f"Be realistic — not every restaurant should be tier 3. "
        f"Provide your assignment and a one-sentence rationale."
    )
    return client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
        response_model=TierAssignment,
    )


def rewrite_description_for_tier(
    client: instructor.Instructor,
    restaurant: dict,
    tier: int,
) -> str:
    """Rewrite description to be consistent with the assigned tier."""
    tier_tone = {
        4: "The tone should convey genuine excellence — specific, evocative praise that sounds like a serious food critic.",
        3: "The tone should be warm but measured — a reliable spot worth visiting, not life-changing.",
        2: "The tone should highlight one strength clearly, then note a specific flaw that tempers the recommendation.",
        1: "The tone should convey honest disappointment — something about the food, service, or value falls short.",
    }

    class RewrittenDescription(BaseModel):
        description: str

    result: RewrittenDescription = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=384,
        messages=[{
            "role": "user",
            "content": (
                f"Rewrite this restaurant description to be consistent with a tier {tier} quality rating.\n\n"
                f"Restaurant: {restaurant['name']} ({restaurant['cuisine']}, ${restaurant['price_range']:.0f} avg)\n"
                f"Original description: {restaurant['description']}\n\n"
                f"{tier_tone[tier]} "
                f"Keep the same restaurant name, cuisine, dishes, and setting — only adjust the tone and emphasis. "
                f"Keep the same length (3-5 sentences)."
            ),
        }],
        response_model=RewrittenDescription,
    )
    return result.description


def deduplicate_names(restaurants: list[dict]) -> list[dict]:
    """Rename duplicate restaurant names by appending a suffix to later occurrences.
    Uses list position as the discriminator — no field comparison."""
    seen: dict[str, int] = {}
    for r in restaurants:
        name = r["name"]
        if name in seen:
            seen[name] += 1
            # Build a suffix that is cuisine-aware and readable
            suffix_words = [w for w in r["cuisine"].split() if len(w) > 3]
            suffix = suffix_words[0] if suffix_words else str(seen[name])
            r["name"] = f"{name} {suffix}"
        else:
            seen[name] = 0
    return restaurants


@app.command()
def main(
    input_path: Path = typer.Option(DATA_DIR / "synthetic_restaurants.json"),
    region: str = typer.Option("us-east-1"),
    rewrite: bool = typer.Option(True, help="Rewrite descriptions to match assigned tier"),
) -> None:
    with open(input_path) as f:
        restaurants: list[dict] = json.load(f)

    restaurants = deduplicate_names(restaurants)
    console.print(f"Names deduplicated. Unique: {len(set(r['name'] for r in restaurants))}/{len(restaurants)}")

    console.print(f"[bold]Backfilling tiers for {len(restaurants)} restaurants...[/bold]")

    # Scale targets to actual dataset size
    n = len(restaurants)
    scale = n / 200
    quota: dict[int, int] = {t: max(1, round(c * scale)) for t, c in TIER_TARGETS.items()}
    # Adjust rounding error to hit exact total
    while sum(quota.values()) < n:
        quota[3] += 1
    while sum(quota.values()) > n:
        quota[3] -= 1
    console.print(f"Tier targets: {dict(sorted(quota.items(), reverse=True))}")

    bedrock = boto3.client("bedrock-runtime", region_name=region)
    client = instructor.from_bedrock(bedrock)

    quota_remaining = dict(quota)
    updated = []

    for r in track(restaurants, description="Assigning tiers..."):
        if not quota_remaining or all(v == 0 for v in quota_remaining.values()):
            # Quota exhausted — default to 3
            r["tier"] = 3
            updated.append(r)
            continue
        try:
            assignment = assign_tier_llm(client, r, quota_remaining)
            tier = assignment.tier
            # If assigned tier is exhausted, pick the closest available
            if quota_remaining.get(tier, 0) == 0:
                available = [t for t, q in quota_remaining.items() if q > 0]
                tier = min(available, key=lambda t: abs(t - tier))
            quota_remaining[tier] -= 1
            if quota_remaining[tier] == 0:
                del quota_remaining[tier]
            r["tier"] = tier
        except Exception as e:
            console.print(f"[yellow]Tier assign failed for {r['name']}: {e}[/yellow]")
            r["tier"] = 3

        updated.append(r)

    if rewrite:
        console.print("\n[bold]Rewriting descriptions to match tiers...[/bold]")
        rewritten = []
        for r in track(updated, description="Rewriting..."):
            try:
                r["description"] = rewrite_description_for_tier(client, r, r["tier"])
            except Exception as e:
                console.print(f"[yellow]Rewrite failed for {r['name']}: {e}[/yellow]")
            rewritten.append(r)
        updated = rewritten

    with open(input_path, "w") as f:
        json.dump(updated, f, indent=2)

    # Summary
    from collections import Counter
    dist = Counter(r["tier"] for r in updated)
    console.print(f"\n[bold green]Done.[/bold green] Tier distribution: {dict(sorted(dist.items(), reverse=True))}")
    console.print(f"Saved → {input_path}")


if __name__ == "__main__":
    app()
