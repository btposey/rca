"""Stage 3: Concierge

Calls Llama 3.2 3B via vLLM to synthesize retrieved results into a
persona-adaptive narrative recommendation.
"""
import json
from openai import AsyncOpenAI

from app.config import get_settings
from app.schema import AgentState

SAFE_RESPONSE = (
    "I'm here to help with restaurant recommendations. "
    "Please ask me about places to eat and I'll be happy to help!"
)

PERSONA_INSTRUCTIONS = {
    "foodie": (
        "The user is a culinary enthusiast. Use rich, sensory language. "
        "Highlight technique, ingredients, chef philosophy, and flavor profiles. "
        "Reference culinary terms naturally."
    ),
    "normie": (
        "The user wants a straightforward recommendation. "
        "Use plain, casual language. Focus on vibe, value, wait times, and consensus. "
        "Avoid culinary jargon."
    ),
    "neutral": (
        "Provide a balanced, factual recommendation. "
        "Be concise and informative without assuming culinary expertise."
    ),
}

NO_RESULTS_RESPONSE = (
    "I don't have any restaurants in my database that I'd confidently recommend for your request. "
    "Try broadening your search — different cuisine, price range, or vibe."
)

TIER_LEGEND = (
    "Each candidate has a quality tier (1-4): 4=award-winning, 3=solid neighborhood favorite, "
    "2=mixed (notable flaw). Prefer tier 4 and 3. For tier 2, name the trade-off."
)

SYSTEM_PROMPT_TEMPLATE = (
    "You are a restaurant concierge. {persona_instruction}\n\n"
    "CRITICAL: You may ONLY recommend restaurants from the candidate list provided. "
    "The candidate list is complete — there are no other options available to you. "
    "Never invent, hallucinate, or suggest any restaurant not explicitly listed. "
    "Draw all details exclusively from the candidate descriptions.\n\n"
    "{tier_legend}\n\n"
    "Produce:\n"
    "1. suggestion: Name your primary pick and core reason in 1-2 sentences. "
    "If a suitable alternate exists, add: 'Alternatively, [name] is worth considering because [reason].' "
    "If no candidates are suitable, set suggestion to exactly: \"{no_results}\"\n"
    "2. elaboration: 2-4 sentences of detail drawn only from the candidate descriptions. "
    "Empty string if no suitable candidates.\n\n"
    "Respond as JSON: {{\"suggestion\": \"...\", \"elaboration\": \"...\"}}"
)


async def synthesize(state: AgentState) -> AgentState:
    if state.attack:
        state.suggestion = SAFE_RESPONSE
        state.elaboration = ""
        return state

    settings = get_settings()
    client = AsyncOpenAI(
        base_url=settings.vllm_concierge_base_url,
        api_key="not-needed",
    )

    if not state.retrieved_results:
        state.suggestion = NO_RESULTS_RESPONSE
        state.elaboration = ""
        return state

    persona_instruction = PERSONA_INSTRUCTIONS[state.persona]
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        persona_instruction=persona_instruction,
        tier_legend=TIER_LEGEND,
        no_results=NO_RESULTS_RESPONSE,
    )

    candidates = json.dumps(state.retrieved_results, indent=2)
    user_message = (
        f"User request: {state.user_query}\n\n"
        f"Restaurant candidates (ONLY recommend from this list — no other options exist):\n{candidates}"
    )

    response = await client.chat.completions.create(
        model=settings.vllm_concierge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=settings.concierge_temperature,
        top_p=settings.concierge_top_p,
        max_tokens=settings.concierge_max_tokens,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    parsed = json.loads(content)
    state.suggestion = parsed.get("suggestion", "")
    state.elaboration = parsed.get("elaboration", "")

    # If the model returned the no-results phrase, suppress any elaboration —
    # the model sometimes continues generating advice after it despite training.
    if state.suggestion.startswith(NO_RESULTS_RESPONSE[:40]):
        state.suggestion = NO_RESULTS_RESPONSE
        state.elaboration = ""

    return state
