import os
import cv2
import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from parser import parser
from utils import *
from Actions import ACTIONS

args = parser.parse_args()
class ActorCritic:
    """
    A3C implement with tensorflow2
    Reference: 
    https://github.com/sadeqa/Super-Mario-Bros-RL (Pytorch)
    https://github.com/seungeunrho/minimalR (Pytorch)
    https://github.com/marload/DeepRL-TensorFlow2 (Tensorflow2)

    params:
    state_shape: input image shape (h, w, channel)
    n_acts: number of actions could be taken from game
    n_units: number of units of LSTMcell
    conv_fn: model architecture
    """
    def __init__(self, n_acts, n_units, pretrained=False, weights_path=None, conv_fn="impala_cnn"):
        # self.state_shape = state_shape
        self.n_acts = n_acts
        self.n_units = n_units
        self.conv_fn = conv_fn
        self.model = self.build_model(pretrained)
        self.opt = tf.keras.optimizers.Adam(args.lr)

    def build_model(self, pretrained):
        """
        build actor and critic model
        """
        if self.conv_fn not in cnn_dict.keys():
            raise ValueError("Unknown model architecture")
        
        model = RLconv(self.n_units, self.n_acts)
        if pretrained:
            model.load_weights(weights_path)
            print("Model load weights successfully")
        
        # model initialization
        (ht, ct) = (tf.zeros((1, self.n_units)), tf.zeros((1, self.n_units)))
        _, _, (_, _) = model((tf.random.normal([1, 84, 84, 4]), (ht, ct)))

        return model
