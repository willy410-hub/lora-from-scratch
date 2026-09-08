"""Utility to inject LoraLayer into a T5 model's attention projections."""
from lora.layer import LoraLayer


def apply_lora_to_t5_attention(model, rank: int = 8):
    """
    Traverse a TF T5 model's encoder and decoder submodules and wrap
    each attention block's Q/K/V/O projections in a LoraLayer.

    Returns the same model instance, mutated in place (matching the
    original API), so callers can keep writing `model = apply_lora_to_t5_attention(model)`.
    """
    for sub_layer in model.encoder.submodules:
        if "TFT5Attention" in sub_layer.__class__.__name__:
            sub_layer.k = LoraLayer(sub_layer.k, rank=rank)
            sub_layer.v = LoraLayer(sub_layer.v, rank=rank)
            sub_layer.q = LoraLayer(sub_layer.q, rank=rank)
            sub_layer.o = LoraLayer(sub_layer.o, rank=rank)

    for sub_layer in model.decoder.submodules:
        if "TFT5Attention" in sub_layer.__class__.__name__:
            sub_layer.k = LoraLayer(sub_layer.k, rank=rank)
            sub_layer.v = LoraLayer(sub_layer.v, rank=rank)
            sub_layer.q = LoraLayer(sub_layer.q, rank=rank)
            sub_layer.o = LoraLayer(sub_layer.o, rank=rank)

    return model
