"""
Custom Keras Layer implementing Low-Rank Adaptation (LoRA).

Wraps an existing Dense layer, freezes it, and injects trainable
low-rank matrices A and B: Output = Frozen_Dense(x) + B(A(x))

The math: W = W0 + delta_W = W0 + (B @ A), where A is initialized from
a normal distribution and B is initialized as zero, so that
delta_W = 0 at the start of training and the adapted model is
numerically identical to the base model before any training happens.

Bug fixed from the original: A and B both used Keras' default
kernel_initializer (glorot_uniform, i.e. random), so the "B starts at
zero" guarantee described in the original README was never actually
enforced in code -- the LoRA branch injected random noise into the
frozen model's output from step one. The initializer choice now lives
in lora/lora_spec.py (see build_lora_specs), a plain-Python module with
no TensorFlow dependency, so that property is directly unit-testable.
"""
import tf_keras as keras

from lora.lora_spec import build_lora_specs


class LoraLayer(keras.layers.Layer):
    """
    Wraps an existing Dense layer, freezes it, and injects trainable
    low-rank matrices A and B: Output = Frozen_Dense(x) + B(A(x))
    """

    def __init__(self, original_layer, rank: int = 8, trainable: bool = True, **kwargs):
        original_layer_config = original_layer.get_config()
        name = original_layer_config.get("name", None)
        kwargs.pop("name", None)
        super().__init__(name=name, trainable=trainable, **kwargs)

        self.rank = rank
        self.original_layer = original_layer
        self.original_layer.trainable = False  # Freeze base weights

        spec_a, spec_b = build_lora_specs(rank=rank, output_units=original_layer.units)

        self.A = keras.layers.Dense(
            units=spec_a.units,
            use_bias=False,
            trainable=trainable,
            kernel_initializer=spec_a.kernel_initializer,
            name=spec_a.name,
        )
        self.B = keras.layers.Dense(
            units=spec_b.units,
            use_bias=False,
            trainable=trainable,
            kernel_initializer=spec_b.kernel_initializer,
            name=spec_b.name,
        )

    def call(self, inputs):
        original_output = self.original_layer(inputs)
        if self.trainable:
            lora_output = self.B(self.A(inputs))
            return original_output + lora_output
        return original_output

    def get_config(self):
        config = super().get_config()
        config.update({"rank": self.rank})
        return config
