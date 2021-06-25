
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten
from tensorflow.python.keras.regularizers import L2


class NaiveEatModel(tf.keras.Model):

    def __init__(self, num_classes: int):
        super(NaiveEatModel, self).__init__()

        dropout_rate = 0.4

        # first convolution layer with batch normalization and max pooling
        self.nn_conv_1 = Conv2D(32, (5, 5), strides=1, padding='same')
        self.nn_bnorm_1 = BatchNormalization()
        self.act_1 = Activation('relu')
        self.dropout_1 = Dropout(rate=dropout_rate)
        self.maxpool_1 = MaxPooling2D()

        # second convolution layer only with max pooling
        self.nn_conv_2 = Conv2D(32, (3, 3), strides=1, padding='same')
        #self.nn_bnorm_2 = BatchNormalization()
        self.act_2 = Activation('relu')
        self.dropout_2 = Dropout(rate=dropout_rate)
        self.maxpool_2 = MaxPooling2D()

        # TODO: think of adding a 3rd Conv2D layer
        # TODO: think of increasing the amount of conv filters
        # TODO: think of add L2 loss regularization

        # output logits layer
        self.flatten = Flatten()
        self.nn_dense_3 = Dense(500, activation='relu')
        self.nn_dense_out = Dense(num_classes, activation='softmax')


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