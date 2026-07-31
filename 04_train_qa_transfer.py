import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from tensorflow.keras.optimizers import Adam

def main():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

    # تحميل داتا سيت الأسئلة والأجوبة SQuAD v2
    dataset = load_dataset("squad_v2")
    train_data = dataset["train"].select(range(2000))

    def preprocess_fn(examples):
        inputs = [context + " question: " + question for question, context in zip(examples["question"], examples["context"])]
        targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in examples["answers"]]
        
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["decoder_input_ids"] = labels["input_ids"]
        return model_inputs

    tokenized_data = train_data.map(preprocess_fn, batched=True)

    tf_dataset = tokenized_data.to_tf_dataset(
        columns=["input_ids", "attention_mask", "decoder_input_ids"],
        label_cols="labels",
        shuffle=True,
        batch_size=4
    )

    model.compile(optimizer=Adam(learning_rate=5e-5))
    print("🚀 Starting FLAN-T5 Large Q&A Fine-Tuning...")
    # model.fit(tf_dataset, epochs=1)

if __name__ == "__main__":
    main()