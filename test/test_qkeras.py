import pytest

import hls4ml

from qkeras import *
from tensorflow.keras.layers import Input

qactivation_list=['quantized_relu', 'quantized_tanh']
activation_list=['quantized_relu', 'quantized_tanh', 'binary_tanh', 'ternary_tanh', 'quantized_bits']
activation_bit_list=["1","2","3","4"]

@pytest.mark.parametrize("activation_bit", activation_bit_list)
def test_dense(activation_bit):
    x = x_in = Input(10)
    x = QDense(10,
        kernel_quantizer="quantized_bits("+activation_bit+",0,1)",
        bias_quantizer="quantized_bits("+activation_bit+")",
        name="Qdense")(x)
    x = QActivation("quantized_relu")(x)

    model = Model(inputs=x_in, outputs=x)
    hls_model=hls4ml.converters.convert_from_keras_model(model)

    assert(len(model.layers)+1==len(hls_model.get_layers()))


@pytest.mark.parametrize("activation", qactivation_list)
def test_activation(activation):
    x = x_in = Input(10)
    x = QDense(10,
        kernel_quantizer="quantized_bits(3,0,1)",
        bias_quantizer="quantized_bits(3)",
        name="Qdense")(x)
    x = QActivation(activation)(x)

    model = Model(inputs=x_in, outputs=x)
    hls_model=hls4ml.converters.convert_from_keras_model(model)
    
    assert(len(model.layers)+1==len(hls_model.get_layers()))
    assert(list(hls_model.get_layers())[-1].attributes['activation']==activation.split("_")[1])