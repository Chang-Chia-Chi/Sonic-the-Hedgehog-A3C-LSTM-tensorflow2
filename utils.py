import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, RNN, LSTMCell, Flatten, Dense, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Orthogonal

cnn_dict = {}
def register(fn_name):
    def mapping(func):
        cnn_dict[fn_name] = func
        return func
    return mapping

def normalized_col_init(std=1.0):
    """
    Initializer for policy and value dense output layer

    Fixing norm of weights (w = g/||v||*v) to be independent of orighinal matrix vectors,
    decoupling norm of weight vector from direction of weight vector,
    which will speed up convergence.

    it scales the weight gradient by g/||v||, and it projects the gradient away 
    from the current weight vector. Both effects help to bring the covariance
    matrix of the gradient closer to identity and benefit optimization.
    If the norm of the gradients is small, we get √1 + c^2 ≈ 1, and the norm of v will stop increasing.
    And so the scaled gradient self-stabilizes its norm.

    Reference:
    Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks
    https://arxiv.org/pdf/1602.07868.pdf
    """
    def initializer(shape, dtype=None, partition_info=None):
        weight = np.random.randn(*shape).astype(np.float32)
        weight *= std/np.sqrt(np.square(weight).sum(axis=0, keepdims=True))
        return tf.constant(weight)
    return initializer

def conv(scope, n_filter, kernal_size, strides, padding='same', gain=1.0, act='relu'):
    """
    convolution layer with orthogonal initializer

    params:
    scope: name scope
    n_filter: number of filters
    kernak_size: size of conv kernel
    strides: strides
    padding: padding method
    gain: scale factor of orthogonal initialzer

    Why orthogonal initialization used:
    Eigenvalues of an orthogonal matrix has absolute value 1, 
    which means, at least at early stage of training, 
    it could avoid gradient exploding/vanishing problem.
                                         
    Reference:
    https://smerity.com/articles/2016/orthogonal_init.html
    https://hjweide.github.io/orthogonal-initialization-in-convolutional-layers
    """
    with tf.name_scope(scope):
        layer = Conv2D(n_filter, kernal_size, strides=strides, padding=padding, 
                       kernel_initializer=Orthogonal(gain=gain), activation=act)
    return layer

class Res_Block(keras.layers.Layer):
    """
    residual block layer for impala cnn model
    """
    def __init__(self, n_filters, **conv_kwarg):
        super(Res_Block, self).__init__()
        self.relu_1 = ReLU()
        self.conv_1 = Conv2D(n_filters, (3,3), padding='same', **conv_kwarg)
        self.relu_2 = ReLU()
        self.conv_2 = Conv2D(n_filters, (3,3), padding='same', **conv_kwarg)
    
    def call(self, inputs):
        outputs = self.relu_1(inputs)
        outputs = self.conv_1(outputs)
        outputs = self.relu_2(outputs)
        outputs = self.conv_2(outputs)
        return keras.layers.add([inputs, outputs])

class Conv_Block(keras.layers.Layer):
    """
    convolution block layer for impala cnn model
    """
    def __init__(self, n_filters, **conv_kwarg):
        super(Conv_Block, self).__init__()
        self.conv_1 = Conv2D(n_filters, (3, 3), padding='same', **conv_kwarg)
        self.maxp_1 = MaxPooling2D((3,3), strides=2, padding='same')
        self.res_block_1 = self.residual_block(n_filters, **conv_kwarg)
        self.res_block_2 = self.residual_block(n_filters, **conv_kwarg)

    def residual_block(self, n_filters, **conv_kwarg):
        """
        build residual block

        params:
        n_filter: number of channels of block
        """
        return Res_Block(n_filters, **conv_kwarg)
    
    def call(self, inputs):
        outputs = self.conv_1(inputs)
        outputs = self.maxp_1(outputs)
        outputs = self.res_block_1(outputs)
        outputs = self.res_block_2(outputs)
        return outputs

@register("basic_cnn")
class Basic_CNN(keras.layers.Layer):
    """
    basic cnn model without lstm
    """
    def __init__(self, **conv_kwarg):
        super(Basic_CNN, self).__init__()
        self.conv_1 = conv('conv_1', 32, (8,8), strides=4, padding='same', gain=np.sqrt(2), act='relu', **conv_kwarg)
        self.conv_2 = conv('conv_2', 64, (4,4), strides=2, padding='same', gain=np.sqrt(2), act='relu', **conv_kwarg)
        self.conv_3 = conv('conv_3', 64, (1,1), strides=1, padding='same', gain=np.sqrt(2), act='relu', **conv_kwarg)
        self.flatten_1 = Flatten()
        self.dense_1 = Dense(256, activation='relu', name='fc_1', kernel_initializer=Orthogonal())

    def call(self, inputs):
        outputs = self.conv_1(inputs)
        outputs = self.conv_2(outputs)
        outputs = self.conv_3(outputs)
        outputs = self.flatten_1(outputs)
        outputs = self.dense_1(outputs)
        return outputs

@register("impala_cnn")
class Impala_CNN(keras.layers.Layer):
    """
    impala cnn model without lstm from "IMPALA: Importance Weighted Actor-Learner Architectures"
    https://arxiv.org/abs/1802.01561
    """
    def __init__(self, structure=[16, 32, 32], **conv_kwarg):
        super(Impala_CNN, self).__init__()
        self.conv_block_1 = self.conv_block(structure[0], **conv_kwarg)
        self.conv_block_2 = self.conv_block(structure[1], **conv_kwarg)
        self.conv_block_3 = self.conv_block(structure[2], **conv_kwarg)
        self.flatten_1 = Flatten()
        self.relu_1 = ReLU()
        self.dense_1 = Dense(256, activation='relu', kernel_initializer=Orthogonal())
    
    def conv_block(self, n_filters, **conv_kwarg):
        """
        build whole convolution block

        params:
        n_filter: number of channels of block        
        """
        return Conv_Block(n_filters, **conv_kwarg)

    def call(self, inputs):
        outputs = self.conv_block_1(inputs)
        outputs = self.conv_block_2(outputs)
        outputs = self.conv_block_3(outputs)
        outputs = self.flatten_1(outputs)
        outputs = self.relu_1(outputs)
        outputs = self.dense_1(outputs)
        return outputs

class RLconv(keras.Model):
    """
    build Reinforcement Learning model with cnn+lstm

    params:
    n_units: number of units of LSTMcell
    n_acts: number of actions could be taken from game
    conv_fn: selected convolution architecture
    """
    def __init__(self, n_units, n_acts, conv_fn="impala_cnn", **conv_kwarg):
        super(RLconv, self).__init__()
        self.n_units = n_units
        self.n_acts = n_acts
        self.cnn_model = cnn_dict[conv_fn](**conv_kwarg)
        self.lstmcell = LSTMCell(self.n_units)
        self.act_dense = Dense(self.n_acts, kernel_initializer=normalized_col_init(0.01))
        self.critic_dense = Dense(1, kernel_initializer=normalized_col_init(1.0))

    def call(self, inputs):
        inputs, (ht, ct) = inputs

        cnn_outputs = self.cnn_model(inputs)
        lstm_outputs, (ht, ct) = self.lstmcell(cnn_outputs, states=[ht, ct])

        actor = self.act_dense(lstm_outputs)
        critic = self.critic_dense(lstm_outputs)
        return actor, critic, (ht, ct)
