"""
Pure specification for the LoraLayer's A/B sub-layers: no TensorFlow or
Keras import here at all.

This is what makes the critical "B must be zero-initialized" property
unit-testable without installing the ~500MB TensorFlow stack: the
initializer choice is a plain Python fact about two named projections,
decided here and then consumed by lora/layer.py (which does the actual
Keras wiring).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DenseSpec:
    """Specification for one of LoraLayer's internal Dense projections."""

    units: int
    kernel_initializer: str
    name: str


def build_lora_specs(rank: int, output_units: int) -> tuple[DenseSpec, DenseSpec]:
    """
    Return the (A, B) DenseSpec pair for a LoraLayer of the given rank
    and output width.

    A is normal-initialized (the "down" projection to rank-many units).
    B is zero-initialized (the "up" projection back to output_units) —
    this is the property that guarantees delta_W = 0 at the start of
    training, so the adapted model exactly matches the frozen base
    model before any gradient step happens.
    """
    spec_a = DenseSpec(units=rank, kernel_initializer="random_normal", name="lora_A")
    spec_b = DenseSpec(units=output_units, kernel_initializer="zeros", name="lora_B")
    return spec_a, spec_b
