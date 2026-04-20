"""test_evaluate.py — unit tests for evaluate.py and evaluate_concierge.py.

Tests cover:
  - score()                   from evaluate.py
  - check_persona_adherence() from evaluate_concierge.py
  - check_tier_mention()      from evaluate_concierge.py

No network, no GPU, no vLLM calls.
"""
import pytest

from app.schema import DispatcherOutput, SearchPredicate
from scripts.evaluate import score
from scripts.evaluate_concierge import (
    ConciergeOutput,
    check_persona_adherence,
    check_tier_mention,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_pred(
    persona="neutral",
    attack=False,
    cuisine=None,
    max_price=None,
) -> DispatcherOutput:
    sp = SearchPredicate(cuisine=cuisine, max_price=max_price) if (cuisine or max_price) else None
    return DispatcherOutput(persona=persona, attack=attack, search_predicate=sp)


def _make_gt(
    persona="neutral",
    attack=False,
    cuisine=None,
    max_price=None,
) -> dict:
    sp = {}
    if cuisine is not None:
        sp["cuisine"] = cuisine
    if max_price is not None:
        sp["max_price"] = max_price
    return {
        "input": "some query",
        "output": {
            "persona": persona,
            "attack": attack,
            "search_predicate": sp if sp else None,
        },
    }


# ===========================================================================
# score() — perfect predictions
# ===========================================================================

class TestScorePerfect:
    def test_persona_accuracy_is_one(self):
        preds = [_make_pred(persona="foodie"), _make_pred(persona="normie")]
        gts = [_make_gt(persona="foodie"), _make_gt(persona="normie")]
        metrics = score(preds, gts)
        assert metrics["persona_accuracy"] == 1.0

    def test_attack_accuracy_is_one(self):
        preds = [_make_pred(attack=True), _make_pred(attack=False)]
        gts = [_make_gt(attack=True), _make_gt(attack=False)]
        metrics = score(preds, gts)
        assert metrics["attack_accuracy"] == 1.0

    def test_cuisine_precision_is_one(self):
        preds = [_make_pred(cuisine="Italian"), _make_pred(cuisine="Japanese")]
        gts = [_make_gt(cuisine="Italian"), _make_gt(cuisine="Japanese")]
        metrics = score(preds, gts)
        assert metrics["cuisine_precision"] == 1.0

    def test_price_mae_is_zero(self):
        preds = [_make_pred(max_price=30.0), _make_pred(max_price=50.0)]
        gts = [_make_gt(max_price=30.0), _make_gt(max_price=50.0)]
        metrics = score(preds, gts)
        assert metrics["price_mae"] == 0.0

    def test_n_matches_input_length(self):
        preds = [_make_pred(), _make_pred(), _make_pred()]
        gts = [_make_gt(), _make_gt(), _make_gt()]
        metrics = score(preds, gts)
        assert metrics["n"] == 3


# ===========================================================================
# score() — all-wrong predictions
# ===========================================================================

class TestScoreAllWrong:
    def test_persona_accuracy_is_zero(self):
        preds = [_make_pred(persona="foodie"), _make_pred(persona="normie")]
        gts = [_make_gt(persona="normie"), _make_gt(persona="foodie")]
        metrics = score(preds, gts)
        assert metrics["persona_accuracy"] == 0.0

    def test_attack_accuracy_is_zero(self):
        preds = [_make_pred(attack=False), _make_pred(attack=True)]
        gts = [_make_gt(attack=True), _make_gt(attack=False)]
        metrics = score(preds, gts)
        assert metrics["attack_accuracy"] == 0.0

    def test_cuisine_precision_is_zero(self):
        preds = [_make_pred(cuisine="Italian"), _make_pred(cuisine="Mexican")]
        gts = [_make_gt(cuisine="Japanese"), _make_gt(cuisine="French")]
        metrics = score(preds, gts)
        assert metrics["cuisine_precision"] == 0.0


# ===========================================================================
# score() — mixed / fractional
# ===========================================================================

class TestScoreMixed:
    def test_persona_half_correct(self):
        preds = [_make_pred(persona="foodie"), _make_pred(persona="normie")]
        gts = [_make_gt(persona="foodie"), _make_gt(persona="neutral")]
        metrics = score(preds, gts)
        assert metrics["persona_accuracy"] == pytest.approx(0.5)

    def test_attack_three_quarters_correct(self):
        preds = [
            _make_pred(attack=True),
            _make_pred(attack=True),
            _make_pred(attack=True),
            _make_pred(attack=False),  # wrong
        ]
        gts = [
            _make_gt(attack=True),
            _make_gt(attack=True),
            _make_gt(attack=True),
            _make_gt(attack=True),
        ]
        metrics = score(preds, gts)
        assert metrics["attack_accuracy"] == pytest.approx(0.75)

    def test_cuisine_one_of_three_correct(self):
        preds = [
            _make_pred(cuisine="Italian"),
            _make_pred(cuisine="Wrong"),
            _make_pred(cuisine="Wrong"),
        ]
        gts = [
            _make_gt(cuisine="Italian"),
            _make_gt(cuisine="Japanese"),
            _make_gt(cuisine="Mexican"),
        ]
        metrics = score(preds, gts)
        assert metrics["cuisine_precision"] == pytest.approx(1 / 3)

    def test_price_mae_nonzero(self):
        preds = [_make_pred(max_price=30.0), _make_pred(max_price=60.0)]
        gts = [_make_gt(max_price=40.0), _make_gt(max_price=50.0)]
        metrics = score(preds, gts)
        # |30-40| + |60-50| = 10 + 10 = 20, avg = 10
        assert metrics["price_mae"] == pytest.approx(10.0)

    def test_cuisine_case_insensitive_match(self):
        """Cuisine comparison is case-insensitive per the implementation."""
        preds = [_make_pred(cuisine="ITALIAN")]
        gts = [_make_gt(cuisine="italian")]
        metrics = score(preds, gts)
        assert metrics["cuisine_precision"] == 1.0


# ===========================================================================
# score() — None returns for missing labels
# ===========================================================================

class TestScoreNoneMetrics:
    def test_cuisine_precision_none_when_no_gt_cuisine(self):
        """When ground-truth has no cuisine labels, precision must be None."""
        preds = [_make_pred(cuisine="Italian"), _make_pred()]
        gts = [_make_gt(), _make_gt()]  # no cuisine in ground-truth
        metrics = score(preds, gts)
        assert metrics["cuisine_precision"] is None

    def test_price_mae_none_when_no_gt_price(self):
        """When ground-truth has no price labels, MAE must be None."""
        preds = [_make_pred(max_price=30.0), _make_pred(max_price=50.0)]
        gts = [_make_gt(), _make_gt()]  # no price in ground-truth
        metrics = score(preds, gts)
        assert metrics["price_mae"] is None

    def test_price_mae_none_when_pred_has_no_price(self):
        """price_errors only appended when both pred AND gt have a price."""
        preds = [_make_pred()]  # no search_predicate
        gts = [_make_gt(max_price=30.0)]
        metrics = score(preds, gts)
        assert metrics["price_mae"] is None

    def test_empty_predictions_returns_zero_accuracy(self):
        metrics = score([], [])
        assert metrics["persona_accuracy"] == 0
        assert metrics["attack_accuracy"] == 0
        assert metrics["n"] == 0


# ===========================================================================
# check_persona_adherence()
# ===========================================================================

class TestCheckPersonaAdherence:
    def test_foodie_with_foodie_term_returns_true(self):
        output = ConciergeOutput(
            suggestion="The maillard reaction on this steak is perfection.",
            elaboration="A deep umami broth with a proper roux base.",
        )
        assert check_persona_adherence(output, "foodie") is True

    def test_foodie_without_foodie_term_returns_false(self):
        output = ConciergeOutput(
            suggestion="Nice burgers here.",
            elaboration="Cheap and cheerful.",
        )
        assert check_persona_adherence(output, "foodie") is False

    def test_normie_without_foodie_term_returns_true(self):
        output = ConciergeOutput(
            suggestion="Great pizza place downtown.",
            elaboration="Good value and friendly service.",
        )
        assert check_persona_adherence(output, "normie") is True

    def test_normie_with_foodie_term_returns_false(self):
        output = ConciergeOutput(
            suggestion="Outstanding omakase experience.",
            elaboration="The mise en place is immaculate.",
        )
        assert check_persona_adherence(output, "normie") is False

    def test_neutral_always_returns_true_regardless_of_content(self):
        foodie_output = ConciergeOutput(
            suggestion="Brilliant terroir-driven tasting menu.",
            elaboration="",
        )
        plain_output = ConciergeOutput(suggestion="Good tacos.", elaboration="")
        assert check_persona_adherence(foodie_output, "neutral") is True
        assert check_persona_adherence(plain_output, "neutral") is True

    def test_case_insensitive_term_detection(self):
        """FOODIE_TERMS are checked against lowercased text."""
        output = ConciergeOutput(
            suggestion="UMAMI levels are off the charts.",
            elaboration="",
        )
        assert check_persona_adherence(output, "foodie") is True

    def test_term_in_elaboration_counts(self):
        output = ConciergeOutput(
            suggestion="Nice place.",
            elaboration="The confit duck is exceptional.",
        )
        assert check_persona_adherence(output, "foodie") is True

    def test_empty_output_foodie_returns_false(self):
        output = ConciergeOutput(suggestion="", elaboration="")
        assert check_persona_adherence(output, "foodie") is False

    def test_empty_output_normie_returns_true(self):
        output = ConciergeOutput(suggestion="", elaboration="")
        assert check_persona_adherence(output, "normie") is True


# ===========================================================================
# check_tier_mention()
# ===========================================================================

class TestCheckTierMention:
    def test_award_scenario_with_accolade_term_returns_true(self):
        output = ConciergeOutput(
            suggestion="This Michelin-starred gem is a must-visit.",
            elaboration="",
        )
        assert check_tier_mention(output, "award_present") is True

    def test_award_scenario_without_accolade_term_returns_false(self):
        output = ConciergeOutput(
            suggestion="Good food, nothing special.",
            elaboration="",
        )
        assert check_tier_mention(output, "award_present") is False

    def test_mostly_award_scenario_with_accolade_returns_true(self):
        output = ConciergeOutput(
            suggestion="A celebrated destination with James Beard recognition.",
            elaboration="",
        )
        assert check_tier_mention(output, "mostly_award") is True

    def test_non_award_scenario_always_returns_true(self):
        """Non-award scenarios are not applicable — always return True."""
        empty_output = ConciergeOutput(suggestion="", elaboration="")
        assert check_tier_mention(empty_output, "mixed") is True
        assert check_tier_mention(empty_output, "no_accolades") is True
        assert check_tier_mention(empty_output, "") is True

    def test_accolade_term_in_elaboration_counts(self):
        output = ConciergeOutput(
            suggestion="Great food.",
            elaboration="This restaurant is acclaimed across the city.",
        )
        assert check_tier_mention(output, "award_present") is True

    def test_case_insensitive_accolade_detection(self):
        output = ConciergeOutput(
            suggestion="AWARD-WINNING kitchen.",
            elaboration="",
        )
        assert check_tier_mention(output, "award_present") is True

    def test_multiple_accolade_terms_all_qualify(self):
        accolade_terms = [
            "award", "michelin", "james beard", "acclaimed", "celebrated",
            "starred", "recognition", "distinguished", "renowned",
            "destination", "exceptional",
        ]
        for term in accolade_terms:
            output = ConciergeOutput(suggestion=f"A {term} restaurant.", elaboration="")
            assert check_tier_mention(output, "award_present") is True, (
                f"Expected True for accolade term '{term}'"
            )
