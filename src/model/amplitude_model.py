
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten, LayerNormalization
from tensorflow.keras.regularizers import L2


class AmplitudeEatModel(tf.keras.Model):

    def __init__(self, num_classes: int):
        super(AmplitudeEatModel, self).__init__()

        dropout_rate = 0.4
        l2_factor = 2e-4

        # first convolution layer with batch normalization and max pooling
        self.nn_conv_1 = Conv1D(32, 4000, strides=1, padding='same', name='nn_conv_1')
        self.nn_bnorm_1 = BatchNormalization(name='nn_bnorm_1')  # tfa.layers.GroupNormalization(groups=8, axis=2)
        self.act_1 = Activation('relu', name='act_1')
        self.dropout_1 = Dropout(rate=dropout_rate, name='dropout_1')
        self.maxpool_1 = MaxPooling1D(pool_size=100, name='maxpool_1')

        # first convolution layer with batch normalization and max pooling
        self.nn_conv_2 = Conv1D(32, 64, strides=1, padding='same', name='nn_conv_2')
        self.nn_bnorm_2 = BatchNormalization(name='nn_bnorm_2')  # tfa.layers.GroupNormalization(groups=8, axis=2)
        self.act_2 = Activation('relu', name='act_2')
        self.dropout_2 = Dropout(rate=dropout_rate, name='dropout_2')
        self.maxpool_2 = MaxPooling1D(pool_size=4, name='maxpool_2')

        # self.nn_conv_3 = Conv1D(64, 64, strides=1, padding='same', name='nn_conv_3')
        # #self.nn_bnorm_3 = BatchNormalization(name='nn_bnorm_3')
        # self.act_3 = Activation('relu', name='act_3')
        # self.dropout_3 = Dropout(rate=dropout_rate, name='dropout_3')
        # self.maxpool_3 = MaxPooling1D(pool_size=4, name='maxpool_3')

        # output logits layer
        self.flatten = Flatten(name='flatten')
        self.nn_dense_out = Dense(num_classes, activation='softmax', name='nn_dense_out')

    def call(self, inputs, training=False):
        x = inputs

        # process the first convolution layer
        x = self.nn_conv_1(x)
        x = self.nn_bnorm_1(x)
        x = self.act_1(x)
        x = self.dropout_1(x, training)
        x = self.maxpool_1(x)

        # process the second convolution layer
        x = self.nn_conv_2(x)
        x = self.nn_bnorm_2(x)
        x = self.act_2(x)
        x = self.dropout_2(x, training)
        x = self.maxpool_2(x)

        # x = self.nn_conv_3(x)
        # x = self.nn_bnorm_3(x, training)
        # x = self.act_3(x)
        # x = self.dropout_3(x, training)
        # x = self.maxpool_3(x)

        # process the logits output layer
        x = self.flatten(x)
        x = self.nn_dense_out(x)

        return x