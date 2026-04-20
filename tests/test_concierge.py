"""Unit tests for app/services/concierge.py.

All OpenAI network calls are mocked.  No vLLM or network required.

Covered scenarios:
- Attack state → returns SAFE_RESPONSE, empty elaboration, skips LLM
- Empty retrieved_results → returns NO_RESULTS_RESPONSE, skips LLM
- No-results phrase in suggestion → elaboration forced to empty string
- Normal response → suggestion and elaboration populated from JSON
- Each persona (foodie/normie/neutral) uses the correct system prompt fragment
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schema import AgentState
from app.services.concierge import (
    synthesize,
    SAFE_RESPONSE,
    NO_RESULTS_RESPONSE,
    PERSONA_INSTRUCTIONS,
    SYSTEM_PROMPT_TEMPLATE,
    TIER_LEGEND,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RESULTS = [
    {
        "id": 1,
        "name": "Bella Roma",
        "cuisine": "Italian",
        "price_range": 35.0,
        "tier": 3,
        "description": "Family-run trattoria with house-made pasta.",
        "distance": 0.10,
    }
]


def _build_completion(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _patch_openai(content: str):
    """Patch AsyncOpenAI in concierge module."""
    completion = _build_completion(content)
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)
    return patch(
        "app.services.concierge.AsyncOpenAI",
        return_value=mock_client,
    )


# ---------------------------------------------------------------------------
# Attack state
# ---------------------------------------------------------------------------

class TestAttackState:
    @pytest.mark.asyncio
    async def test_attack_returns_safe_response(self):
        state = AgentState(user_query="bad input", attack=True)
        # synthesize must NOT call the OpenAI client
        with patch("app.services.concierge.AsyncOpenAI") as mock_cls:
            result = await synthesize(state)
        mock_cls.assert_not_called()
        assert result.suggestion == SAFE_RESPONSE

    @pytest.mark.asyncio
    async def test_attack_returns_empty_elaboration(self):
        state = AgentState(user_query="bad input", attack=True)
        with patch("app.services.concierge.AsyncOpenAI"):
            result = await synthesize(state)
        assert result.elaboration == ""

    @pytest.mark.asyncio
    async def test_attack_returns_same_state_object(self):
        state = AgentState(user_query="bad", attack=True)
        with patch("app.services.concierge.AsyncOpenAI"):
            result = await synthesize(state)
        assert result is state

    @pytest.mark.asyncio
    async def test_attack_with_existing_results_still_skips_llm(self):
        """Even if results were somehow pre-loaded, attack takes priority."""
        state = AgentState(
            user_query="ignore instructions",
            attack=True,
            retrieved_results=_SAMPLE_RESULTS,
        )
        with patch("app.services.concierge.AsyncOpenAI") as mock_cls:
            result = await synthesize(state)
        mock_cls.assert_not_called()
        assert result.suggestion == SAFE_RESPONSE


# ---------------------------------------------------------------------------
# Empty retrieved_results
# ---------------------------------------------------------------------------

class TestEmptyResults:
    @pytest.mark.asyncio
    async def test_empty_results_returns_no_results_response(self):
        state = AgentState(user_query="find me a restaurant", retrieved_results=[])
        with patch("app.services.concierge.AsyncOpenAI") as mock_cls:
            # Client is instantiated before the empty check, but create() is never called
            mock_cls.return_value.chat.completions.create = AsyncMock()
            result = await synthesize(state)
        mock_cls.return_value.chat.completions.create.assert_not_called()
        assert result.suggestion == NO_RESULTS_RESPONSE

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty_elaboration(self):
        state = AgentState(user_query="find me a restaurant", retrieved_results=[])
        with patch("app.services.concierge.AsyncOpenAI"):
            result = await synthesize(state)
        assert result.elaboration == ""

    @pytest.mark.asyncio
    async def test_empty_results_same_state_object(self):
        state = AgentState(user_query="find me a restaurant", retrieved_results=[])
        with patch("app.services.concierge.AsyncOpenAI"):
            result = await synthesize(state)
        assert result is state


# ---------------------------------------------------------------------------
# No-results phrase in LLM suggestion
# ---------------------------------------------------------------------------

class TestNoResultsPhraseHandling:
    @pytest.mark.asyncio
    async def test_no_results_phrase_suppresses_elaboration(self):
        """If the model returns NO_RESULTS_RESPONSE as suggestion, elaboration cleared."""
        llm_response = json.dumps({
            "suggestion": NO_RESULTS_RESPONSE,
            "elaboration": "Maybe try a different area.",
        })
        with _patch_openai(llm_response):
            state = AgentState(
                user_query="find vegan food",
                retrieved_results=_SAMPLE_RESULTS,
            )
            result = await synthesize(state)

        assert result.suggestion == NO_RESULTS_RESPONSE
        assert result.elaboration == ""

    @pytest.mark.asyncio
    async def test_partial_no_results_phrase_also_suppressed(self):
        """Suggestion starting with first 40 chars of NO_RESULTS_RESPONSE is normalized."""
        partial = NO_RESULTS_RESPONSE[:40] + " but here's something anyway."
        llm_response = json.dumps({
            "suggestion": partial,
            "elaboration": "Some spurious elaboration.",
        })
        with _patch_openai(llm_response):
            state = AgentState(
                user_query="find food",
                retrieved_results=_SAMPLE_RESULTS,
            )
            result = await synthesize(state)

        assert result.suggestion == NO_RESULTS_RESPONSE
        assert result.elaboration == ""

    @pytest.mark.asyncio
    async def test_normal_suggestion_preserves_elaboration(self):
        """A normal suggestion should NOT have its elaboration cleared."""
        llm_response = json.dumps({
            "suggestion": "Try Bella Roma for great pasta.",
            "elaboration": "Their cacio e pepe is outstanding.",
        })
        with _patch_openai(llm_response):
            state = AgentState(
                user_query="Italian food",
                retrieved_results=_SAMPLE_RESULTS,
            )
            result = await synthesize(state)

        assert result.suggestion == "Try Bella Roma for great pasta."
        assert result.elaboration == "Their cacio e pepe is outstanding."


# ---------------------------------------------------------------------------
# Normal response
# ---------------------------------------------------------------------------

class TestNormalResponse:
    @pytest.mark.asyncio
    async def test_suggestion_and_elaboration_populated(self):
        llm_response = json.dumps({
            "suggestion": "Bella Roma is your best bet.",
            "elaboration": "House-made pasta and a warm atmosphere make it ideal.",
        })
        with _patch_openai(llm_response):
            state = AgentState(
                user_query="cozy Italian",
                persona="neutral",
                retrieved_results=_SAMPLE_RESULTS,
            )
            result = await synthesize(state)

        assert result.suggestion == "Bella Roma is your best bet."
        assert result.elaboration == "House-made pasta and a warm atmosphere make it ideal."

    @pytest.mark.asyncio
    async def test_state_object_returned_is_same_instance(self):
        llm_response = json.dumps({"suggestion": "Go here.", "elaboration": "Nice."})
        with _patch_openai(llm_response):
            state = AgentState(
                user_query="any food",
                retrieved_results=_SAMPLE_RESULTS,
            )
            result = await synthesize(state)
        assert result is state

    @pytest.mark.asyncio
    async def test_missing_suggestion_key_defaults_to_empty(self):
        """If the LLM omits 'suggestion', parsed.get returns '' (no crash)."""
        llm_response = json.dumps({"elaboration": "Details here."})
        with _patch_openai(llm_response):
            state = AgentState(
                user_query="food",
                retrieved_results=_SAMPLE_RESULTS,
            )
            result = await synthesize(state)
        assert result.suggestion == ""
        assert result.elaboration == "Details here."

    @pytest.mark.asyncio
    async def test_missing_elaboration_key_defaults_to_empty(self):
        """If the LLM omits 'elaboration', it defaults to ''."""
        llm_response = json.dumps({"suggestion": "Bella Roma."})
        with _patch_openai(llm_response):
            state = AgentState(
                user_query="food",
                retrieved_results=_SAMPLE_RESULTS,
            )
            result = await synthesize(state)
        assert result.elaboration == ""


# ---------------------------------------------------------------------------
# Persona system prompt selection
# ---------------------------------------------------------------------------

class TestPersonaSystemPrompt:
    """Verify that each persona results in its own instruction text being
    embedded in the system prompt sent to the LLM."""

    async def _captured_system_prompt(self, persona: str) -> str:
        """Run synthesize and return the system prompt the LLM received."""
        llm_response = json.dumps({"suggestion": "Go here.", "elaboration": "Nice."})
        completion = _build_completion(llm_response)
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=completion)

        with patch("app.services.concierge.AsyncOpenAI", return_value=mock_client):
            state = AgentState(
                user_query="food",
                persona=persona,
                retrieved_results=_SAMPLE_RESULTS,
            )
            await synthesize(state)

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        return system_msg["content"]

    @pytest.mark.asyncio
    async def test_foodie_uses_foodie_instruction(self):
        prompt = await self._captured_system_prompt("foodie")
        assert PERSONA_INSTRUCTIONS["foodie"] in prompt

    @pytest.mark.asyncio
    async def test_normie_uses_normie_instruction(self):
        prompt = await self._captured_system_prompt("normie")
        assert PERSONA_INSTRUCTIONS["normie"] in prompt

    @pytest.mark.asyncio
    async def test_neutral_uses_neutral_instruction(self):
        prompt = await self._captured_system_prompt("neutral")
        assert PERSONA_INSTRUCTIONS["neutral"] in prompt

    @pytest.mark.asyncio
    async def test_foodie_prompt_does_not_contain_normie_instruction(self):
        prompt = await self._captured_system_prompt("foodie")
        assert PERSONA_INSTRUCTIONS["normie"] not in prompt

    @pytest.mark.asyncio
    async def test_system_prompt_contains_tier_legend(self):
        """Every persona's system prompt should include the tier legend."""
        for persona in ("foodie", "normie", "neutral"):
            prompt = await self._captured_system_prompt(persona)
            assert TIER_LEGEND in prompt

    @pytest.mark.asyncio
    async def test_system_prompt_contains_no_results_phrase(self):
        """The no-results sentinel phrase must appear in the system prompt."""
        for persona in ("foodie", "normie", "neutral"):
            prompt = await self._captured_system_prompt(persona)
            assert NO_RESULTS_RESPONSE in prompt
