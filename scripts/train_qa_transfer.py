"""Train a LoRA-adapted FLAN-T5 model for question answering (SQuAD v2).

Two bugs fixed from the original:
1. The original never called apply_lora_to_t5_attention() here — it
   fine-tuned the *entire* flan-t5-large model with no freezing,
   contradicting the project's own parameter-efficient premise (and
   using far more memory/compute than the other two training scripts).
2. SQuAD v2's defining feature is ~50% unanswerable questions, where
   answers["text"] is an empty list. The original's
   `answer["text"][0] if len(...) > 0 else ""` produced an empty-string
   training target for every unanswerable example -- teaching the model
   to output nothing rather than to recognize and state that a question
   is unanswerable. Fixed via lora.data_prep.extract_answer_text, which
   returns an explicit refusal string instead (and is unit-tested
   independently of this script).
"""
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

from lora.apply import apply_lora_to_t5_attention
from lora.core.config import get_qa_settings
from lora.data_prep import extract_answer_text


def build_dataset(tokenizer, settings):
    dataset = load_dataset(settings.dataset_name)
    train_data = dataset["train"].select(range(settings.train_sample_size))

    def preprocess_fn(examples):
        inputs = [
            context + " question: " + question
            for question, context in zip(examples["question"], examples["context"])
        ]
        targets = [
            extract_answer_text(answer, settings.unanswerable_target)
            for answer in examples["answers"]
        ]

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
    settings = get_qa_settings()

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(settings.model_name)
    # Apply LoRA here too -- the original left this as full fine-tuning,
    # which contradicted the project's parameter-efficiency premise.
    model = apply_lora_to_t5_attention(model, rank=settings.lora_rank)

    tf_dataset = build_dataset(tokenizer, settings)

    model.compile(optimizer=Adam(learning_rate=settings.learning_rate))

    print("Starting FLAN-T5 Large Q&A LoRA transfer training...")
    # model.fit(tf_dataset, epochs=1)


if __name__ == "__main__":
    main()
