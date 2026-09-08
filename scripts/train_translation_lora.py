"""Train a LoRA-adapted FLAN-T5 model for English -> German translation (WMT16)."""
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

from lora.apply import apply_lora_to_t5_attention
from lora.core.config import get_translation_settings


def build_dataset(tokenizer, settings):
    dataset = load_dataset(settings.dataset_name, settings.dataset_config)
    train_data = dataset["train"].select(range(settings.train_sample_size))

    def preprocess_fn(examples):
        inputs = ["translate English to German: " + ex["en"] for ex in examples["translation"]]
        targets = [ex["de"] for ex in examples["translation"]]

        model_inputs = tokenizer(
            inputs, max_length=settings.max_input_length, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            targets, max_length=settings.max_target_length, truncation=True, padding="max_length"
        )

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["decoder_input_ids"] = labels["input_ids"]
        return model_inputs

    tokenized_data = train_data.map(preprocess_fn, batched=True)

    return tokenized_data.to_tf_dataset(
        columns=["input_ids", "attention_mask", "decoder_input_ids"],
        label_cols="labels",
        shuffle=True,
        batch_size=settings.batch_size,
    )


def main():
    settings = get_translation_settings()

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(settings.model_name)
    model = apply_lora_to_t5_attention(model, rank=settings.lora_rank)

    tf_dataset = build_dataset(tokenizer, settings)

    if settings.freeze_decoder:
        model.get_layer("decoder").trainable = False

    model.compile(optimizer=Adam(learning_rate=settings.learning_rate))

    print("Starting FLAN-T5 Translation LoRA training...")
    # model.fit(tf_dataset, epochs=1)


if __name__ == "__main__":
    main()
