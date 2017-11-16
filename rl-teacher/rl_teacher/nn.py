import numpy as np
import tensorflow as tf

from keras.layers import Dense, Dropout, LeakyReLU, Activation
from keras.models import Sequential

class FullyConnectedMLP(object):
    """Vanilla two hidden layer multi-layer perceptron"""

    def __init__(self, obs_shape, act_shape, h_size=64):
        input_dim = np.prod(obs_shape) + np.prod(act_shape)

        self.model = Sequential()
        self.model.add(Dense(h_size, input_dim=input_dim))
        self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(h_size))
        self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))

    def run(self, obs, act):
        flat_obs = tf.contrib.layers.flatten(obs)
        x = tf.concat([flat_obs, act], axis=1)
        return self.model(x)



#
class FullyConnected_classifier(object):
    """Vanilla two hidden layer multi-layer perceptron"""

    def __init__(self, obs_shape, act_shape, segment_length, h_size=64):
        input_dim = int((np.prod(obs_shape) + np.prod(act_shape)) * segment_length)

        self.model = Sequential()
        self.model.add(Dense(h_size, input_dim=input_dim))
        self.model.add(LeakyReLU())

        self.model.add(Dense(h_size))
        self.model.add(LeakyReLU())

        self.model.add(Dense(2))
        # self.model.add(Activation('softmax'))

    def run(self, obs_act):
        return self.model(obs_act)
