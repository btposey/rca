"""Unit tests for app/schema.py.

Tests cover:
- AgentState defaults and field validation
- SearchPredicate optional fields
- DispatcherOutput defaults
- QueryRequest / QueryResponse round-trip serialization
"""
import pytest
from pydantic import ValidationError

from app.schema import (
    AgentState,
    DispatcherOutput,
    QueryRequest,
    QueryResponse,
    SearchPredicate,
)


# ---------------------------------------------------------------------------
# SearchPredicate
# ---------------------------------------------------------------------------

class TestSearchPredicate:
    def test_all_fields_none_by_default(self):
        pred = SearchPredicate()
        assert pred.cuisine is None
        assert pred.max_price is None
        assert pred.min_tier is None

    def test_cuisine_only(self):
        pred = SearchPredicate(cuisine="Italian")
        assert pred.cuisine == "Italian"
        assert pred.max_price is None
        assert pred.min_tier is None

    def test_all_fields_set(self):
        pred = SearchPredicate(cuisine="Mexican", max_price=35.0, min_tier=3)
        assert pred.cuisine == "Mexican"
        assert pred.max_price == 35.0
        assert pred.min_tier == 3

    def test_max_price_float(self):
        pred = SearchPredicate(max_price=50)
        # pydantic coerces int → float for float field
        assert isinstance(pred.max_price, float)
        assert pred.max_price == 50.0

    def test_min_tier_boundaries(self):
        for tier in (1, 2, 3, 4):
            pred = SearchPredicate(min_tier=tier)
            assert pred.min_tier == tier


# ---------------------------------------------------------------------------
# DispatcherOutput
# ---------------------------------------------------------------------------

class TestDispatcherOutput:
    def test_defaults(self):
        out = DispatcherOutput()
        assert out.persona == "neutral"
        assert out.attack is False
        assert out.search_predicate is None
        assert out.semantic_query is None

    def test_foodie_persona(self):
        out = DispatcherOutput(persona="foodie")
        assert out.persona == "foodie"

    def test_normie_persona(self):
        out = DispatcherOutput(persona="normie")
        assert out.persona == "normie"

    def test_invalid_persona_raises(self):
        with pytest.raises(ValidationError):
            DispatcherOutput(persona="expert")

    def test_attack_flag(self):
        out = DispatcherOutput(attack=True, semantic_query=None)
        assert out.attack is True
        assert out.semantic_query is None

    def test_full_output(self):
        pred = SearchPredicate(cuisine="Thai", max_price=40.0)
        out = DispatcherOutput(
            persona="foodie",
            attack=False,
            search_predicate=pred,
            semantic_query="aromatic thai with heat",
        )
        assert out.persona == "foodie"
        assert out.search_predicate.cuisine == "Thai"
        assert out.semantic_query == "aromatic thai with heat"

    def test_model_validate_from_dict(self):
        data = {
            "persona": "normie",
            "attack": False,
            "search_predicate": {"cuisine": "Pizza", "max_price": 20.0},
            "semantic_query": "cheap pizza",
        }
        out = DispatcherOutput.model_validate(data)
        assert out.persona == "normie"
        assert out.search_predicate.cuisine == "Pizza"
        assert out.semantic_query == "cheap pizza"


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------

class TestAgentState:
    def test_requires_user_query(self):
        with pytest.raises(ValidationError):
            AgentState()  # missing required field

    def test_defaults(self):
        state = AgentState(user_query="test")
        assert state.persona == "neutral"
        assert state.attack is False
        assert state.search_predicate is None
        assert state.semantic_query is None
        assert state.retrieved_results == []
        assert state.suggestion == ""
        assert state.elaboration == ""

    def test_retrieved_results_default_is_new_list(self):
        # Each instance must get its own list (not shared via mutable default)
        s1 = AgentState(user_query="a")
        s2 = AgentState(user_query="b")
        s1.retrieved_results.append({"id": 99})
        assert s2.retrieved_results == []

    def test_all_fields_set(self):
        pred = SearchPredicate(cuisine="French")
        state = AgentState(
            user_query="upscale French dinner",
            persona="foodie",
            attack=False,
            search_predicate=pred,
            semantic_query="upscale french fine dining",
            retrieved_results=[{"id": 1, "name": "Le Petit"}],
            suggestion="Try Le Petit",
            elaboration="Wonderful tasting menu.",
        )
        assert state.persona == "foodie"
        assert state.search_predicate.cuisine == "French"
        assert len(state.retrieved_results) == 1
        assert state.suggestion == "Try Le Petit"

    def test_invalid_persona_raises(self):
        with pytest.raises(ValidationError):
            AgentState(user_query="hi", persona="robot")

    def test_attack_state(self):
        state = AgentState(user_query="bad input", attack=True)
        assert state.attack is True


# ---------------------------------------------------------------------------
# QueryRequest
# ---------------------------------------------------------------------------

class TestQueryRequest:
    def test_basic(self):
        req = QueryRequest(query="find me a steakhouse")
        assert req.query == "find me a steakhouse"

    def test_requires_query(self):
        with pytest.raises(ValidationError):
            QueryRequest()

    def test_empty_string_is_valid_at_model_level(self):
        # Validation of empty string happens at the API layer, not the model
        req = QueryRequest(query="")
        assert req.query == ""

    def test_serialization(self):
        req = QueryRequest(query="sushi")
        data = req.model_dump()
        assert data == {"query": "sushi"}

    def test_deserialization(self):
        req = QueryRequest.model_validate({"query": "tacos"})
        assert req.query == "tacos"


# ---------------------------------------------------------------------------
# QueryResponse
# ---------------------------------------------------------------------------

class TestQueryResponse:
    def test_basic(self):
        resp = QueryResponse(
            suggestion="Go to Chez Paul",
            elaboration="Great ambiance.",
            persona="neutral",
            attack=False,
        )
        assert resp.suggestion == "Go to Chez Paul"
        assert resp.attack is False

    def test_requires_all_fields(self):
        with pytest.raises(ValidationError):
            QueryResponse(suggestion="ok", elaboration="ok", persona="neutral")
            # missing attack

    def test_serialization_roundtrip(self):
        resp = QueryResponse(
            suggestion="Noodle House",
            elaboration="Slurpy broth.",
            persona="foodie",
            attack=False,
        )
        data = resp.model_dump()
        restored = QueryResponse.model_validate(data)
        assert restored.suggestion == resp.suggestion
        assert restored.elaboration == resp.elaboration
        assert restored.persona == resp.persona
        assert restored.attack == resp.attack

    def test_attack_response(self):
        resp = QueryResponse(
            suggestion="I'm here to help with restaurant recommendations.",
            elaboration="",
            persona="neutral",
            attack=True,
        )
        assert resp.attack is True
        assert resp.elaboration == ""
