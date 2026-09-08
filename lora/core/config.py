"""
Centralized hyperparameters and model settings.

The original had model names, learning rates, batch sizes, and dataset
sample sizes hardcoded inline in each training script, with no single
place to see or override them. All defaults here match the original
scripts' values exactly — this is a refactor, not a retune — except
where explicitly noted (mixed_float16 is now actually enabled where the
README always claimed it was).
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class LoraSettings(BaseSettings):
    """Default LoRA hyperparameters, overridable via LORA_* environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="LORA_", extra="ignore")

    default_rank: int = 8


class TranslationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="TRANSLATION_", extra="ignore")

    model_name: str = "google/flan-t5-base"
    dataset_name: str = "wmt16"
    dataset_config: str = "de-en"
    train_sample_size: int = 5000
    max_input_length: int = 128
    max_target_length: int = 128
    batch_size: int = 16
    learning_rate: float = 3e-4
    lora_rank: int = 8
    freeze_decoder: bool = True


class SummarizationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SUMMARIZATION_", extra="ignore")

    model_name: str = "google/flan-t5-small"
    dataset_name: str = "cnn_dailymail"
    dataset_config: str = "3.0.0"
    train_sample_size: int = 1000
    max_input_length: int = 512
    max_target_length: int = 128
    batch_size: int = 8
    learning_rate: float = 5e-4
    lora_rank: int = 8
    use_mixed_precision: bool = True


class QaTransferSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="QA_", extra="ignore")

    model_name: str = "google/flan-t5-large"
    dataset_name: str = "squad_v2"
    train_sample_size: int = 2000
    max_input_length: int = 512
    max_target_length: int = 128
    batch_size: int = 4
    learning_rate: float = 5e-5
    lora_rank: int = 8
    unanswerable_target: str = "No answer available."


@lru_cache
def get_lora_settings() -> LoraSettings:
    return LoraSettings()


@lru_cache
def get_translation_settings() -> TranslationSettings:
    return TranslationSettings()


@lru_cache
def get_summarization_settings() -> SummarizationSettings:
    return SummarizationSettings()


@lru_cache
def get_qa_settings() -> QaTransferSettings:
    return QaTransferSettings()
