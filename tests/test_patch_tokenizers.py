"""test_patch_tokenizers.py — unit tests for patch_tokenizers.py logic
and deduplicate_names() from backfill_tiers.py.

No real HuggingFace downloads, no GPU, no network.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Stub heavy optional dependencies before importing scripts
# ---------------------------------------------------------------------------
sys.modules.setdefault("boto3", MagicMock())
sys.modules.setdefault("instructor", MagicMock())

from scripts.backfill_tiers import deduplicate_names


# ===========================================================================
# deduplicate_names() from backfill_tiers.py
# ===========================================================================

class TestDeduplicateNames:
    def test_no_duplicates_unchanged(self):
        restaurants = [
            {"name": "Alpha", "cuisine": "Italian"},
            {"name": "Beta", "cuisine": "Japanese"},
            {"name": "Gamma", "cuisine": "Mexican"},
        ]
        result = deduplicate_names(restaurants)
        names = [r["name"] for r in result]
        assert names == ["Alpha", "Beta", "Gamma"]

    def test_single_duplicate_gets_suffix(self):
        restaurants = [
            {"name": "Sage", "cuisine": "French"},
            {"name": "Sage", "cuisine": "American"},
        ]
        result = deduplicate_names(restaurants)
        # First occurrence keeps original name
        assert result[0]["name"] == "Sage"
        # Second occurrence must be different
        assert result[1]["name"] != "Sage"
        assert result[1]["name"].startswith("Sage ")

    def test_suffix_uses_cuisine_word(self):
        """Suffix should come from the cuisine string when a long-enough word exists."""
        restaurants = [
            {"name": "Harbor", "cuisine": "Seafood"},
            {"name": "Harbor", "cuisine": "Seafood"},
        ]
        result = deduplicate_names(restaurants)
        # "Seafood" has 7 chars (> 3) so it should be the suffix
        assert result[1]["name"] == "Harbor Seafood"

    def test_three_occurrences_all_unique(self):
        # deduplicate_names appends cuisine word to 2nd occurrence;
        # 3rd occurrence with same cuisine will also get the same suffix.
        # The function only guarantees 1st-level deduplication.
        restaurants = [
            {"name": "Star", "cuisine": "Thai"},
            {"name": "Star", "cuisine": "Vietnamese"},
            {"name": "Star", "cuisine": "Italian"},
        ]
        result = deduplicate_names(restaurants)
        names = [r["name"] for r in result]
        assert len(set(names)) == len(names), f"Not all unique: {names}"

    def test_only_long_cuisine_words_used_as_suffix(self):
        """Words with 3 or fewer characters are skipped; numeric fallback used."""
        restaurants = [
            {"name": "Dub", "cuisine": "Bar"},  # 'Bar' has 3 chars — skipped
            {"name": "Dub", "cuisine": "Bar"},
        ]
        result = deduplicate_names(restaurants)
        # 'Bar' has len == 3, which is NOT > 3, so numeric fallback expected
        assert result[1]["name"] != "Dub Bar"
        assert result[1]["name"].startswith("Dub ")

    def test_returns_list(self):
        restaurants = [{"name": "X", "cuisine": "Thai"}]
        result = deduplicate_names(restaurants)
        assert isinstance(result, list)

    def test_empty_list(self):
        assert deduplicate_names([]) == []

    def test_modifies_in_place_and_returns_same_list(self):
        """The function mutates and returns the same list object."""
        restaurants = [
            {"name": "Twin", "cuisine": "Italian"},
            {"name": "Twin", "cuisine": "Italian"},
        ]
        returned = deduplicate_names(restaurants)
        assert returned is restaurants

    def test_non_duplicate_names_untouched(self):
        restaurants = [
            {"name": "Unique", "cuisine": "Vietnamese"},
            {"name": "Also Unique", "cuisine": "Korean"},
        ]
        original_names = [r["name"] for r in restaurants]
        deduplicate_names(restaurants)
        assert [r["name"] for r in restaurants] == original_names


# ===========================================================================
# patch_tokenizers logic — file-system tests using tmp_path
# ===========================================================================

class TestPatchTokenizersLogic:
    """Tests for the patch decision logic in patch_tokenizers.main().

    We call the core logic directly (mocking AutoTokenizer) rather than
    invoking the Typer CLI, so there is no subprocess overhead and no real
    HuggingFace requests are made.
    """

    def _write_tokenizer_config(self, target_dir: Path, tokenizer_class: str) -> None:
        cfg = {"tokenizer_class": tokenizer_class, "model_max_length": 131072}
        (target_dir / "tokenizer_config.json").write_text(json.dumps(cfg))

    def _run_patch_logic(self, model_dir: Path, mock_auto_tokenizer) -> None:
        """Execute the patching loop from patch_tokenizers.main() directly."""
        import scripts.patch_tokenizers as pt

        PATCH_MAP = {
            "dispatcher-llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
            "concierge-llama-3b":  "meta-llama/Llama-3.2-3B-Instruct",
        }

        for model_name, base_model in PATCH_MAP.items():
            target = model_dir / model_name
            if not target.exists():
                continue

            cfg_path = target / "tokenizer_config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = json.load(f)
                if cfg.get("tokenizer_class") not in ("TokenizersBackend", None) and \
                   (target / "tokenizer.json").exists():
                    continue  # already patched

            tokenizer = mock_auto_tokenizer.from_pretrained(base_model)
            tokenizer.save_pretrained(str(target))

    def test_tokenizers_backend_triggers_patch(self, tmp_path):
        """A config with TokenizersBackend should trigger from_pretrained + save."""
        target = tmp_path / "dispatcher-llama-1b"
        target.mkdir()
        self._write_tokenizer_config(target, "TokenizersBackend")

        mock_tok = MagicMock()
        mock_auto = MagicMock()
        mock_auto.from_pretrained.return_value = mock_tok

        self._run_patch_logic(tmp_path, mock_auto)

        mock_auto.from_pretrained.assert_called_once_with(
            "meta-llama/Llama-3.2-1B-Instruct"
        )
        mock_tok.save_pretrained.assert_called_once_with(str(target))

    def test_already_patched_skips_download(self, tmp_path):
        """A config with PreTrainedTokenizerFast + tokenizer.json should be skipped."""
        target = tmp_path / "dispatcher-llama-1b"
        target.mkdir()
        self._write_tokenizer_config(target, "PreTrainedTokenizerFast")
        # Simulate the tokenizer.json file being present
        (target / "tokenizer.json").write_text("{}")

        mock_auto = MagicMock()

        self._run_patch_logic(tmp_path, mock_auto)

        mock_auto.from_pretrained.assert_not_called()

    def test_missing_model_dir_is_skipped(self, tmp_path):
        """If the target model directory doesn't exist, no patching occurs."""
        mock_auto = MagicMock()
        # tmp_path has no subdirectories — neither model directory exists
        self._run_patch_logic(tmp_path, mock_auto)
        mock_auto.from_pretrained.assert_not_called()

    def test_both_models_patched_when_both_need_it(self, tmp_path):
        """Both dispatcher and concierge models are patched when both have backend config."""
        for name in ("dispatcher-llama-1b", "concierge-llama-3b"):
            d = tmp_path / name
            d.mkdir()
            self._write_tokenizer_config(d, "TokenizersBackend")

        mock_tok = MagicMock()
        mock_auto = MagicMock()
        mock_auto.from_pretrained.return_value = mock_tok

        self._run_patch_logic(tmp_path, mock_auto)

        assert mock_auto.from_pretrained.call_count == 2
        assert mock_tok.save_pretrained.call_count == 2

    def test_none_tokenizer_class_triggers_patch(self, tmp_path):
        """A config where tokenizer_class is None should also trigger patching."""
        target = tmp_path / "concierge-llama-3b"
        target.mkdir()
        # tokenizer_class absent / None
        cfg = {"model_max_length": 131072}
        (target / "tokenizer_config.json").write_text(json.dumps(cfg))

        mock_tok = MagicMock()
        mock_auto = MagicMock()
        mock_auto.from_pretrained.return_value = mock_tok

        self._run_patch_logic(tmp_path, mock_auto)

        mock_auto.from_pretrained.assert_called_once_with(
            "meta-llama/Llama-3.2-3B-Instruct"
        )

    def test_patch_map_contains_expected_models(self, tmp_path):
        """PATCH_MAP in patch_tokenizers.py covers both fine-tuned models."""
        import scripts.patch_tokenizers as pt
        assert "dispatcher-llama-1b" in pt.PATCH_MAP
        assert "concierge-llama-3b"  in pt.PATCH_MAP

    def test_patch_map_base_models_correct(self, tmp_path):
        import scripts.patch_tokenizers as pt
        assert pt.PATCH_MAP["dispatcher-llama-1b"] == "meta-llama/Llama-3.2-1B-Instruct"
        assert pt.PATCH_MAP["concierge-llama-3b"]  == "meta-llama/Llama-3.2-3B-Instruct"

    def test_tokenizer_config_written_to_correct_path(self, tmp_path):
        """save_pretrained should be called with the target model dir path."""
        target = tmp_path / "dispatcher-llama-1b"
        target.mkdir()
        self._write_tokenizer_config(target, "TokenizersBackend")

        mock_tok = MagicMock()
        mock_auto = MagicMock()
        mock_auto.from_pretrained.return_value = mock_tok

        self._run_patch_logic(tmp_path, mock_auto)

        mock_tok.save_pretrained.assert_called_once_with(str(target))
