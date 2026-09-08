"""
LoRA From Scratch -- a custom Keras Low-Rank Adaptation implementation.

Deliberately does NOT eagerly import LoraLayer/apply_lora_to_t5_attention
here, even though that would be the more convenient `from lora import X`
call site used by the training scripts. Doing so would mean importing
*any* submodule of this package -- including the framework-independent
ones like lora.metrics, lora.data_prep, and lora.lora_spec -- transitively
requires tf_keras/TensorFlow to be installed, exactly the eager-loading
problem this rebuild fixes elsewhere. Training scripts import
`from lora.layer import LoraLayer` and
`from lora.apply import apply_lora_to_t5_attention` directly instead.
"""

__version__ = "2.0.0"
