# LoRA From Scratch

A custom, from-scratch implementation of **LoRA (Low-Rank Adaptation)**
as a native Keras layer — no PEFT library, no wrapper abstractions —
built to show exactly how attention projections (Q, K, V, O) are
adapted under memory constraints, applied to FLAN-T5 across three
transfer-learning tasks: translation, summarization, and question
answering.

> **Note on scope:** this is a from-scratch reference implementation of
> the LoRA mechanism, not a production training pipeline — all three
> `scripts/train_*.py` files build the model, dataset, and optimizer,
> then stop short of an actual multi-hour `model.fit()` call (commented
> out), matching the original project's intent as a technical showcase
> rather than a trained-and-shipped model. Running the scripts as
> written requires TensorFlow, `tf_keras`, and downloading FLAN-T5 +
> the relevant Hugging Face datasets. The unit tests, however, need
> none of that — see **Running Tests** below.

---

## The Math

Instead of updating the full pre-trained weight matrix `W0`, LoRA
constrains the update to a low-rank decomposition:

```
W = W0 + delta_W = W0 + (B @ A)
```

`A` is initialized from a normal distribution; `B` is initialized as
**zero**, so `delta_W = 0` at the start of training — the adapted model
is numerically identical to the frozen base model before a single
gradient step happens.

```python
class LoraLayer(keras.layers.Layer):
    def __init__(self, original_layer, rank=8, trainable=True, **kwargs):
        ...
        self.original_layer.trainable = False  # Freeze base weights
        self.A = keras.layers.Dense(units=rank, kernel_initializer="random_normal", ...)
        self.B = keras.layers.Dense(units=original_layer.units, kernel_initializer="zeros", ...)

    def call(self, inputs):
        original_output = self.original_layer(inputs)
        return original_output + self.B(self.A(inputs))
```

---

## What Changed From the Original Prototype

This is a rebuild of an earlier prototype. The core LoRA *idea* was
right; several implementation details didn't match what the project
claimed:

- **B was never actually zero-initialized** — the original's `A` and
  `B` both used Keras' default `kernel_initializer` (`glorot_uniform`,
  i.e. random), so the "B starts at zero, guaranteeing delta_W = 0"
  property described in the README was **not enforced in code**. The
  LoRA branch injected random noise into the frozen model's output from
  step one, undermining the whole point of the technique. Fixed by
  explicitly setting `kernel_initializer="zeros"` on `B`, with the
  initializer choice extracted into a small, framework-independent
  module (`lora/lora_spec.py`) specifically so this property is now
  directly unit-tested rather than just asserted in a docstring.
- **The Q&A script never applied LoRA** — `04_train_qa_transfer.py`
  fine-tuned the *entire* `flan-t5-large` model with no freezing and no
  low-rank adaptation, contradicting the project's own
  parameter-efficiency premise (and using far more memory than the
  other two tasks, defeating the stated purpose). LoRA is now applied
  here too, consistent with the translation and summarization scripts.
- **SQuAD v2's unanswerable questions were mishandled** — roughly half
  of SQuAD v2's examples are intentionally unanswerable, represented as
  an empty `answers["text"]` list. The original's
  `answer["text"][0] if len(...) > 0 else ""` silently trained the
  model to output an *empty string* for every one of those examples,
  rather than teaching it to recognize and state that a question is
  unanswerable. Fixed in `lora/data_prep.py`'s `extract_answer_text`,
  which returns an explicit, configurable refusal string instead.
- **`mixed_float16` was claimed but never enabled** — the README
  described the summarization task as using mixed precision, but the
  original script never called
  `tf.keras.mixed_precision.set_global_policy(...)`. Now actually
  enabled (and toggleable via `SUMMARIZATION_USE_MIXED_PRECISION`).
- **No tests existed, and the package had an eager-import trap** — the
  first attempt at this rebuild added `from lora.layer import LoraLayer`
  to `lora/__init__.py` for a more convenient `from lora import
  LoraLayer` call site — which meant importing *any* submodule,
  including framework-independent ones like `lora.metrics`, transitively
  required `tf_keras` to be installed. Fixed by keeping `lora/__init__.py`
  import-free; training scripts import directly from submodules
  (`from lora.layer import LoraLayer`). This is what makes **23 unit
  tests** — covering the zero-init fix, the SQuAD v2 fix, BLEU/ROUGE
  correctness, and config defaults — runnable with a handful of small
  packages, no TensorFlow required.
- **`requirements.txt` was missing `tf-keras`** — `01_lora_from_scratch.py`
  imports `tf_keras` directly (needed for Keras 2 compatibility with
  current `transformers`/TF versions), but the package was never listed.
  Added.

---

## Project Structure

```text
lora-from-scratch/
├── lora/
│   ├── layer.py                # LoraLayer (Keras) -- the core adapter
│   ├── lora_spec.py             # Pure init-choice logic, zero TF dependency (the critical fix)
│   ├── apply.py                  # Injects LoraLayer into a T5 model's attention blocks
│   ├── data_prep.py              # Pure preprocessing helpers (the SQuAD v2 fix)
│   ├── metrics.py                # BLEU / ROUGE evaluation
│   └── core/
│       └── config.py               # Centralized hyperparameters, env-overridable
├── scripts/
│   ├── train_translation_lora.py    # WMT16 en->de
│   ├── train_summarization_lora.py  # CNN/DailyMail (mixed_float16 now actually on)
│   └── train_qa_transfer.py         # SQuAD v2 (LoRA now actually applied)
├── tests/                        # 23 tests -- see Running Tests
├── requirements.txt               # Full stack (TensorFlow, transformers, datasets, ...)
├── requirements-test.txt          # Minimal deps for the offline test suite
├── requirements-dev.txt
├── .env.example
└── README.md
```

---

## Quick Start

### Install the full stack

```bash
pip install -r requirements.txt
```

### Run a training script

Each script builds the model, applies LoRA, prepares the dataset, and
compiles — the actual `model.fit(...)` call is left commented out, same
as the original, since this is a showcase of the adaptation mechanism
rather than a multi-hour training job:

```bash
python scripts/train_translation_lora.py
python scripts/train_summarization_lora.py
python scripts/train_qa_transfer.py
```

---

## Running Tests

Most of this project's logic — the LoRA initializer fix, the SQuAD v2
answer-extraction fix, BLEU/ROUGE correctness, and hyperparameter
defaults — is pure Python with no TensorFlow dependency, and is tested
that way on purpose:

```bash
pip install -r requirements-test.txt
pytest -v
```

`LoraLayer` and `apply_lora_to_t5_attention` themselves (the actual
Keras wiring in `lora/layer.py` / `lora/apply.py`) do require
`tensorflow` + `tf-keras` to import, since they subclass
`tf_keras.layers.Layer` directly — but the property that matters most
(does `B` actually get zero-initialized?) is verified against
`lora/lora_spec.py` without ever needing to import Keras at all.

---

## Extending

- **Add a new task:** create a new `scripts/train_*.py` following the
  existing pattern — load a model, call
  `apply_lora_to_t5_attention(model, rank=...)`, build a dataset, compile.
- **Add a new hyperparameter set:** add a `*Settings` class in
  `lora/core/config.py` with its own env prefix.
- **Adapt a non-T5 architecture:** `apply.py`'s traversal logic is T5-specific
  (`TFT5Attention`); `LoraLayer` itself is architecture-agnostic and can wrap
  any `Dense`-based projection directly.
