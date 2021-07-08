
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras.regularizers import L2


class NaiveEatModel(tf.keras.Model):

    def __init__(self, num_classes: int):
        super(NaiveEatModel, self).__init__()

        dropout_rate = 0.4
        # l2_factor = 0 # 2e-4

        # first convolution layer with batch normalization and max pooling
        self.nn_conv_1 = Conv2D(16, (5, 5), strides=1, padding='same', name='nn_conv_1')
        # self.nn_norm_1 = BatchNormalization(name='nn_norm_1')
        self.nn_norm_1 = GroupNormalization(groups=4, name='nn_norm_1')
        self.act_1 = Activation('relu', name='act_1')
        self.dropout_1 = Dropout(rate=dropout_rate, name='dropout_1')
        self.maxpool_1 = MaxPooling2D(name='maxpool_1')

        # second convolution layer with batch normalization and max pooling
        self.nn_conv_2 = Conv2D(16, (3, 3), strides=1, padding='same', name='nn_conv_2')
        # self.nn_norm_2 = BatchNormalization(name='nn_norm_2')
        self.nn_norm_2 = GroupNormalization(groups=4, name='nn_norm_2')
        self.act_2 = Activation('relu', name='act_2')
        self.dropout_2 = Dropout(rate=dropout_rate, name='dropout_2')
        self.maxpool_2 = MaxPooling2D(name='maxpool_2')

        # third convolution layer with batch normalization and max pooling
        self.nn_conv_3 = Conv2D(16, (3, 3), strides=1, padding='same', name='nn_conv_3')
        self.act_3 = Activation('relu', name='act_3')
        self.maxpool_3 = MaxPooling2D(name='maxpool_3')

        # fourth convolution layer with batch normalization and max pooling
        self.nn_conv_4 = Conv2D(8, (3, 3), strides=1, padding='same', name='nn_conv_4')
        self.act_4 = Activation('relu', name='act_4')
        self.maxpool_4 = MaxPooling2D(name='maxpool_4')

        # output logits layer
        self.flatten = Flatten(name='flatten')
        self.nn_dense_out = Dense(num_classes, activation='softmax', name='nn_dense_out')


    def call(self, inputs, training=False):

        x = inputs

        # process the first convolution layer
        x = self.nn_conv_1(x)
        x = self.nn_norm_1(x)
        x = self.act_1(x)
        x = self.dropout_1(x, training)
        x = self.maxpool_1(x)

        # process the second convolution layer
        x = self.nn_conv_2(x)
        x = self.nn_norm_2(x)
        x = self.act_2(x)
        x = self.dropout_2(x, training)
        x = self.maxpool_2(x)

        # process the third convolution layer
        x = self.nn_conv_3(x)
        x = self.act_3(x)
        x = self.maxpool_3(x)

        # process the fourth convolution layer
        x = self.nn_conv_4(x)
        x = self.act_4(x)
        x = self.maxpool_4(x)

        # process the logits output layer
        x = self.flatten(x)
        x = self.nn_dense_out(x)

        return x