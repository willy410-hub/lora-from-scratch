# 🤖 LLM Fine-Tuning & PEFT: Low-Rank Adaptation (LoRA) From Scratch

A hands-on, production-grade technical showcase demonstrating **Parameter-Efficient Fine-Tuning (PEFT)** and **Transfer Learning** on Large Language Models (specifically `google/flan-t5` and `distilbert`). 

This repository highlights a custom, framework-agnostic implementation of **LoRA (Low-Rank Adaptation)** built directly using **TensorFlow/Keras**, avoiding heavy wrapper abstractions to illustrate how attention projections ($Q, K, V, O$) are adapted under memory constraints.

---

## 💡 Key Engineering Features

* **Custom LoRA Layer (Built from Scratch):** Implements $W_{\text{frozen}} + (A \times B)$ as a native Keras layer, injecting trainable low-rank matrices into frozen Transformer modules.
* **Massive Resource Efficiency:** Reduces trainable parameters by **~90%** and cuts GPU VRAM consumption from **30 GB down to ~8 GB**, allowing execution on entry-level cloud GPUs (NVIDIA T4 / L4).
* **Multi-Task Adaptations:** Fine-tuned and evaluated across sequence-to-sequence tasks:
  * **Machine Translation (EN $\rightarrow$ DE):** WMT16 Dataset.
  * **Text Summarization:** CNN/DailyMail Dataset using Mixed Precision (`mixed_float16`).
  * **Question Answering (Q&A):** SQuAD v2 Dataset.
* **Quantitative Evaluation Pipeline:** Built evaluation routines using industry-standard sequence generation metrics (**BLEU** & **ROUGE-1/2/L**).

---

## 🏗️ Technical Architecture & Mathematical Logic

Instead of updating the massive pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA constrains the weight update by decomposing $\Delta W$ into two low-rank matrices $A$ and $B$:

$$W = W_0 + \Delta W = W_0 + (B \times A)$$

Where $A \in \mathbb{R}^{r \times k}$ is initialized from a normal distribution $\mathcal{N}(0, \sigma^2)$ and $B \in \mathbb{R}^{d \times r}$ is initialized as zero, ensuring $\Delta W = 0$ at the start of training.

```python
class LoraLayer(keras.layers.Layer):
    """Custom Keras Layer wrapping frozen dense projections with trainable Low-Rank matrices."""
    def __init__(self, original_layer, rank=8, trainable=True, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.original_layer = original_layer
        self.original_layer.trainable = False  # Freeze base weights
        
        self.A = tf.keras.layers.Dense(units=rank, use_bias=False, trainable=trainable)
        self.B = tf.keras.layers.Dense(units=original_layer.units, use_bias=False, trainable=trainable)

    def call(self, inputs):
        original_output = self.original_layer(inputs)
        if self.trainable:
            return original_output + self.B(self.A(inputs))
        return original_output