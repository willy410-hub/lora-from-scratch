"""
Unit tests for lora/lora_spec.py — the module that decides A and B's
initializers.

This is the single most important test in the project: it directly
verifies the bug fix (B must be zero-initialized) without requiring
TensorFlow/Keras to be installed at all, since lora_spec.py has zero
framework dependency.
"""
from lora.lora_spec import build_lora_specs


def test_b_is_zero_initialized():
    """
    The core LoRA guarantee: delta_W = 0 at initialization requires B's
    kernel_initializer to be exactly 'zeros'. This is the property the
    original code claimed in its README but never actually enforced.
    """
    _, spec_b = build_lora_specs(rank=8, output_units=64)
    assert spec_b.kernel_initializer == "zeros"


def test_a_is_not_zero_initialized():
    """
    A must NOT be zero-initialized -- if both A and B started at zero,
    the LoRA branch's gradient would also be zero everywhere and
    training could never move it away from zero (a dead branch).
    """
    spec_a, _ = build_lora_specs(rank=8, output_units=64)
    assert spec_a.kernel_initializer != "zeros"
    assert spec_a.kernel_initializer == "random_normal"


def test_a_units_equals_rank():
    spec_a, _ = build_lora_specs(rank=4, output_units=64)
    assert spec_a.units == 4


def test_b_units_equals_output_units():
    _, spec_b = build_lora_specs(rank=4, output_units=64)
    assert spec_b.units == 64


def test_specs_are_named_lora_a_and_lora_b():
    spec_a, spec_b = build_lora_specs(rank=8, output_units=64)
    assert spec_a.name == "lora_A"
    assert spec_b.name == "lora_B"


def test_different_ranks_produce_different_a_units():
    spec_a_rank2, _ = build_lora_specs(rank=2, output_units=64)
    spec_a_rank16, _ = build_lora_specs(rank=16, output_units=64)
    assert spec_a_rank2.units == 2
    assert spec_a_rank16.units == 16
    assert spec_a_rank2.units != spec_a_rank16.units
