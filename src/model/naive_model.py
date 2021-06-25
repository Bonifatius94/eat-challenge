
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten
from tensorflow.python.keras.regularizers import L2


class NaiveEatModel(tf.keras.Model):

    def __init__(self, num_classes: int):
        super(NaiveEatModel, self).__init__()

        dropout_rate = 0.4

        # first convolution layer with batch normalization and max pooling
        self.nn_conv_1 = Conv2D(32, (5, 5), strides=1, padding='same', name='nn_conv_1')
        self.nn_bnorm_1 = BatchNormalization(name='nn_bnorm_1')
        self.act_1 = Activation('relu', name='act_1')
        self.dropout_1 = Dropout(rate=dropout_rate, name='dropout_1')
        self.maxpool_1 = MaxPooling2D(name='maxpool_1')

        # second convolution layer only with max pooling
        self.nn_conv_2 = Conv2D(32, (3, 3), strides=1, padding='same', name='nn_conv_2')
        #self.nn_bnorm_2 = BatchNormalization(name='nn_bnorm_2')
        self.act_2 = Activation('relu', name='act_2')
        self.dropout_2 = Dropout(rate=dropout_rate, name='dropout_2')
        self.maxpool_2 = MaxPooling2D(name='maxpool_2')

        # TODO: think of adding a 3rd Conv2D layer
        # TODO: think of increasing the amount of conv filters
        # TODO: think of add L2 loss regularization

        # output logits layer
        self.flatten = Flatten(name='flatten')
        self.nn_dense_3 = Dense(500, activation='relu', name='nn_dense_3')
        self.nn_dense_out = Dense(num_classes, activation='softmax', name='nn_dense_out')


    def call(self, inputs, training=False):

        x = inputs

        # process the first convolution layer
        x = self.nn_conv_1(x)
        x = self.nn_bnorm_1(x, training)
        x = self.act_1(x)
        x = self.dropout_1(x, training)
        x = self.maxpool_1(x)

        # process the second convolution layer
        x = self.nn_conv_2(x)
        #x = self.nn_bnorm_2(x, training)
        x = self.act_2(x)
        x = self.dropout_2(x, training)
        x = self.maxpool_2(x)

        # process the logits output layer
        x = self.flatten(x)
        x = self.nn_dense_3(x)
        x = self.nn_dense_out(x)

        return x