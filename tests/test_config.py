"""
Unit tests for lora/core/config.py -- verifying defaults match the
original scripts' hardcoded values exactly (this is a refactor, not a
retune), plus the one deliberate behavior change (mixed precision).
"""
from lora.core.config import (
    get_lora_settings,
    get_qa_settings,
    get_summarization_settings,
    get_translation_settings,
)


def test_lora_default_rank_matches_original():
    settings = get_lora_settings()
    assert settings.default_rank == 8


def test_translation_defaults_match_original_script():
    settings = get_translation_settings()
    assert settings.model_name == "google/flan-t5-base"
    assert settings.train_sample_size == 5000
    assert settings.batch_size == 16
    assert settings.learning_rate == 3e-4
    assert settings.freeze_decoder is True


def test_summarization_defaults_match_original_script():
    settings = get_summarization_settings()
    assert settings.model_name == "google/flan-t5-small"
    assert settings.train_sample_size == 1000
    assert settings.batch_size == 8
    assert settings.learning_rate == 5e-4


def test_summarization_mixed_precision_is_enabled_by_default():
    """This is the deliberate behavior fix: the original README claimed
    mixed_float16 was used for summarization, but the code never
    enabled it. The new default turns it on."""
    settings = get_summarization_settings()
    assert settings.use_mixed_precision is True


def test_qa_defaults_match_original_script():
    settings = get_qa_settings()
    assert settings.model_name == "google/flan-t5-large"
    assert settings.train_sample_size == 2000
    assert settings.batch_size == 4
    assert settings.learning_rate == 5e-5


def test_qa_has_configurable_unanswerable_target():
    settings = get_qa_settings()
    assert settings.unanswerable_target
    assert isinstance(settings.unanswerable_target, str)
