import tensorflow as tf
import importlib
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from tensorflow.keras.optimizers import Adam

# استيراد كلاس LoraLayer من الملف الأول
lora_module = importlib.import_module("01_lora_from_scratch")
LoraLayer = lora_module.LoraLayer

def main():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

    # تحميل داتا سيت التلخيص CNN/DailyMail
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    train_data = dataset["train"].shuffle(seed=42).select(range(1000))

    def preprocess_fn(examples):
        inputs = tokenizer(examples["article"], max_length=512, truncation=True, padding="max_length")
        targets = tokenizer(examples["highlights"], max_length=128, truncation=True, padding="max_length")
        
        inputs["labels"] = targets["input_ids"]
        inputs["decoder_input_ids"] = targets["input_ids"]
        return inputs

    tokenized_data = train_data.map(preprocess_fn, batched=True, remove_columns=["id"])

    tf_dataset = tokenized_data.to_tf_dataset(
        columns=["input_ids", "attention_mask", "decoder_input_ids"],
        label_cols="labels",
        shuffle=True,
        batch_size=8
    )

    model.compile(optimizer=Adam(learning_rate=5e-4))
    print("🚀 Starting Summarization LoRA Training...")
    # model.fit(tf_dataset, epochs=1)

if __name__ == "__main__":
    main()