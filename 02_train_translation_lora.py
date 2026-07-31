import tensorflow as tf
import importlib
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from tensorflow.keras.optimizers import Adam

# استيراد كلاس LoraLayer من الملف الأول بدون مشاكل الأرقام
lora_module = importlib.import_module("01_lora_from_scratch")
LoraLayer = lora_module.LoraLayer

def main():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

    # تحميل عينة من داتا سيت WMT16 للترجمة
    dataset = load_dataset("wmt16", "de-en")
    train_data = dataset["train"].select(range(5000))

    def preprocess_fn(examples):
        inputs = ["translate English to German: " + ex["en"] for ex in examples["translation"]]
        targets = [ex["de"] for ex in examples["translation"]]
        
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["decoder_input_ids"] = labels["input_ids"]
        return model_inputs

    tokenized_data = train_data.map(preprocess_fn, batched=True)
    
    tf_dataset = tokenized_data.to_tf_dataset(
        columns=["input_ids", "attention_mask", "decoder_input_ids"],
        label_cols="labels",
        shuffle=True,
        batch_size=16
    )

    # تجميد أوزان الـ Decoder وتجميع الموديل
    model.get_layer("decoder").trainable = False
    model.compile(optimizer=Adam(learning_rate=3e-4))
    
    print("🚀 Starting FLAN-T5 Translation LoRA Training...")
    # model.fit(tf_dataset, epochs=1)

if __name__ == "__main__":
    main()