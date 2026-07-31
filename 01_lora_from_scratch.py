import tensorflow as tf
import tf_keras as keras

class LoraLayer(keras.layers.Layer):
    """
    Custom Keras Layer implementing Low-Rank Adaptation (LoRA).
    Wraps an existing Dense layer, freezes it, and injects trainable 
    low-rank matrices A and B: Output = Frozen_Dense(x) + B(A(x))
    """
    def __init__(self, original_layer, rank=8, trainable=True, **kwargs):
        original_layer_config = original_layer.get_config()
        name = original_layer_config.get("name", None)
        kwargs.pop("name", None)
        super().__init__(name=name, trainable=trainable, **kwargs)
        
        self.rank = rank
        self.original_layer = original_layer
        self.original_layer.trainable = False  # Freeze base weights

        # Low-rank matrices A and B
        self.A = keras.layers.Dense(units=rank, use_bias=False, trainable=trainable, name="lora_A")
        self.B = keras.layers.Dense(units=original_layer.units, use_bias=False, trainable=trainable, name="lora_B")

    def call(self, inputs):
        original_output = self.original_layer(inputs)
        if self.trainable:
            lora_output = self.B(self.A(inputs))
            return original_output + lora_output
        return original_output

def apply_lora_to_t5(model, rank=8):
    """
    Utility function to traverse T5 layers and inject LoRA into Attention projections.
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

if __name__ == "__main__":
    print("✅ LoRA Layer Module defined successfully!")