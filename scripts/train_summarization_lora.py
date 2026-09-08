"""Train a LoRA-adapted FLAN-T5 model for text summarization (CNN/DailyMail).

Uses mixed_float16 precision, as the project's README always described —
the original script imported the pieces needed for this but never
actually called tf.keras.mixed_precision.set_global_policy(), so no
mixed-precision training was ever happening despite the documentation.
"""
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

from lora.apply import apply_lora_to_t5_attention
from lora.core.config import get_summarization_settings


def build_dataset(tokenizer, settings):
    dataset = load_dataset(settings.dataset_name, settings.dataset_config)
    train_data = dataset["train"].shuffle(seed=42).select(range(settings.train_sample_size))

    def preprocess_fn(examples):
        inputs = tokenizer(
            examples["article"],
            max_length=settings.max_input_length,
            truncation=True,
            padding="max_length",
        )
        targets = tokenizer(
            examples["highlights"],
            max_length=settings.max_target_length,
            truncation=True,
            padding="max_length",
        )

        inputs["labels"] = targets["input_ids"]
        inputs["decoder_input_ids"] = targets["input_ids"]
        return inputs

    tokenized_data = train_data.map(preprocess_fn, batched=True, remove_columns=["id"])

    return tokenized_data.to_tf_dataset(
        columns=["input_ids", "attention_mask", "decoder_input_ids"],
        label_cols="labels",
        shuffle=True,
        batch_size=settings.batch_size,
    )


def main():
    settings = get_summarization_settings()

    if settings.use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision (mixed_float16) enabled.")

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(settings.model_name)
    model = apply_lora_to_t5_attention(model, rank=settings.lora_rank)

    tf_dataset = build_dataset(tokenizer, settings)

    model.compile(optimizer=Adam(learning_rate=settings.learning_rate))

    print("Starting Summarization LoRA training...")
    # model.fit(tf_dataset, epochs=1)


if __name__ == "__main__":
    main()
