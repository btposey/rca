"""test_finetune.py — unit tests for finetune.py and finetune_concierge.py.

Covers only pure-Python logic (format_sample, load_jsonl).
No GPU, no HuggingFace model downloads, no heavy dependencies required.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Make scripts importable without GPU/train deps by mocking heavy modules
# before import.
# ---------------------------------------------------------------------------

# Mock out modules that are only imported inside the main() function bodies
# (they use guarded try/except imports), but also mock typer/rich so the
# module-level Typer() call doesn't fail if those aren't installed.
_MOCK_MODULES = {
    "unsloth": MagicMock(),
    "trl": MagicMock(),
    "datasets": MagicMock(),
}

for _mod, _mock in _MOCK_MODULES.items():
    sys.modules.setdefault(_mod, _mock)

# typer and rich are real dependencies listed in pyproject.toml, so we only
# patch them if they're genuinely absent to avoid masking real import errors.

# Now import the modules under test
import importlib
import scripts.finetune as finetune_mod
import scripts.finetune_concierge as concierge_mod


# ===========================================================================
# Helpers
# ===========================================================================

DISPATCHER_SAMPLE = {
    "input": "I want a cosy Italian restaurant under $40",
    "output": {
        "persona": "normie",
        "attack": False,
        "search_predicate": {"cuisine": "Italian", "max_price": 40.0, "min_tier": None},
        "semantic_query": "cosy Italian restaurant",
    },
}

CONCIERGE_SAMPLE = {
    "system": "You are a helpful restaurant concierge. Tier legend: 4=award-winning.",
    "input": "User request: cheap sushi\n\nRestaurant candidates:\n[]",
    "output": {"suggestion": "Try Tokyo Garden.", "elaboration": "Great value sushi."},
}


# ===========================================================================
# finetune.py — format_sample
# ===========================================================================

class TestDispatcherFormatSample:
    def test_begins_with_begin_of_text(self):
        result = finetune_mod.format_sample(DISPATCHER_SAMPLE)
        assert result.startswith("<|begin_of_text|>")

    def test_contains_system_header(self):
        result = finetune_mod.format_sample(DISPATCHER_SAMPLE)
        assert "<|start_header_id|>system<|end_header_id|>" in result

    def test_contains_user_header(self):
        result = finetune_mod.format_sample(DISPATCHER_SAMPLE)
        assert "<|start_header_id|>user<|end_header_id|>" in result

    def test_contains_assistant_header(self):
        result = finetune_mod.format_sample(DISPATCHER_SAMPLE)
        assert "<|start_header_id|>assistant<|end_header_id|>" in result

    def test_contains_eot_tokens(self):
        result = finetune_mod.format_sample(DISPATCHER_SAMPLE)
        assert result.count("<|eot_id|>") == 3

    def test_contains_user_message(self):
        result = finetune_mod.format_sample(DISPATCHER_SAMPLE)
        assert DISPATCHER_SAMPLE["input"] in result

    def test_assistant_turn_is_json_serialised_output(self):
        result = finetune_mod.format_sample(DISPATCHER_SAMPLE)
        expected_json = json.dumps(DISPATCHER_SAMPLE["output"])
        assert expected_json in result

    def test_system_prompt_present(self):
        result = finetune_mod.format_sample(DISPATCHER_SAMPLE)
        assert finetune_mod.SYSTEM_PROMPT in result

    def test_section_order(self):
        """system must appear before user which must appear before assistant."""
        result = finetune_mod.format_sample(DISPATCHER_SAMPLE)
        sys_pos = result.index("<|start_header_id|>system<|end_header_id|>")
        usr_pos = result.index("<|start_header_id|>user<|end_header_id|>")
        ast_pos = result.index("<|start_header_id|>assistant<|end_header_id|>")
        assert sys_pos < usr_pos < ast_pos

    def test_result_is_string(self):
        result = finetune_mod.format_sample(DISPATCHER_SAMPLE)
        assert isinstance(result, str)

    def test_non_empty_result(self):
        result = finetune_mod.format_sample(DISPATCHER_SAMPLE)
        assert len(result) > 0


# ===========================================================================
# finetune_concierge.py — format_sample
# ===========================================================================

class TestConciergeFormatSample:
    def test_begins_with_begin_of_text(self):
        result = concierge_mod.format_sample(CONCIERGE_SAMPLE)
        assert result.startswith("<|begin_of_text|>")

    def test_contains_system_header(self):
        result = concierge_mod.format_sample(CONCIERGE_SAMPLE)
        assert "<|start_header_id|>system<|end_header_id|>" in result

    def test_contains_user_header(self):
        result = concierge_mod.format_sample(CONCIERGE_SAMPLE)
        assert "<|start_header_id|>user<|end_header_id|>" in result

    def test_contains_assistant_header(self):
        result = concierge_mod.format_sample(CONCIERGE_SAMPLE)
        assert "<|start_header_id|>assistant<|end_header_id|>" in result

    def test_three_eot_tokens(self):
        result = concierge_mod.format_sample(CONCIERGE_SAMPLE)
        assert result.count("<|eot_id|>") == 3

    def test_uses_sample_system_field(self):
        """Concierge reads system from sample, not a module-level constant."""
        result = concierge_mod.format_sample(CONCIERGE_SAMPLE)
        assert CONCIERGE_SAMPLE["system"] in result

    def test_uses_sample_input_field(self):
        result = concierge_mod.format_sample(CONCIERGE_SAMPLE)
        assert CONCIERGE_SAMPLE["input"] in result

    def test_assistant_turn_is_json_serialised_output(self):
        result = concierge_mod.format_sample(CONCIERGE_SAMPLE)
        expected_json = json.dumps(CONCIERGE_SAMPLE["output"])
        assert expected_json in result

    def test_section_order(self):
        result = concierge_mod.format_sample(CONCIERGE_SAMPLE)
        sys_pos = result.index("<|start_header_id|>system<|end_header_id|>")
        usr_pos = result.index("<|start_header_id|>user<|end_header_id|>")
        ast_pos = result.index("<|start_header_id|>assistant<|end_header_id|>")
        assert sys_pos < usr_pos < ast_pos

    def test_different_system_per_sample(self):
        """Ensure the system field is read from the sample dict, not hardcoded."""
        alt_sample = {**CONCIERGE_SAMPLE, "system": "CUSTOM SYSTEM PROMPT XYZ"}
        result = concierge_mod.format_sample(alt_sample)
        assert "CUSTOM SYSTEM PROMPT XYZ" in result
        assert CONCIERGE_SAMPLE["system"] not in result


# ===========================================================================
# load_jsonl — shared between finetune.py and finetune_concierge.py
# ===========================================================================

class TestLoadJsonl:
    def test_reads_valid_jsonl(self, tmp_path):
        records = [
            {"input": "hello", "output": {"persona": "normie"}},
            {"input": "world", "output": {"persona": "foodie"}},
        ]
        p = tmp_path / "data.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        result = finetune_mod.load_jsonl(p)

        assert len(result) == 2
        assert result[0]["input"] == "hello"
        assert result[1]["output"]["persona"] == "foodie"

    def test_skips_blank_lines(self, tmp_path):
        records = [{"input": "a"}, {"input": "b"}]
        lines = json.dumps(records[0]) + "\n\n   \n" + json.dumps(records[1]) + "\n"
        p = tmp_path / "data.jsonl"
        p.write_text(lines)

        result = finetune_mod.load_jsonl(p)
        assert len(result) == 2

    def test_returns_list(self, tmp_path):
        p = tmp_path / "single.jsonl"
        p.write_text(json.dumps({"x": 1}) + "\n")
        result = finetune_mod.load_jsonl(p)
        assert isinstance(result, list)

    def test_empty_file_returns_empty_list(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        result = finetune_mod.load_jsonl(p)
        assert result == []

    def test_only_blank_lines_returns_empty_list(self, tmp_path):
        p = tmp_path / "blanks.jsonl"
        p.write_text("\n\n\n")
        result = finetune_mod.load_jsonl(p)
        assert result == []

    def test_concierge_load_jsonl_equivalent(self, tmp_path):
        """finetune_concierge.load_jsonl should behave identically."""
        records = [CONCIERGE_SAMPLE]
        p = tmp_path / "concierge.jsonl"
        p.write_text(json.dumps(records[0]) + "\n\n" + json.dumps(records[0]) + "\n")

        result = concierge_mod.load_jsonl(p)
        assert len(result) == 2
        assert result[0]["system"] == CONCIERGE_SAMPLE["system"]
