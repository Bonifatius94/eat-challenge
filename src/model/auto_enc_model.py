
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten
from tensorflow.keras.regularizers import L2


class AutoEncClassifModel(tf.keras.Model):

    def __init__(self, encoder: EncoderModel, num_classes: int=7):
        super(AutoEncEatModel, self).__init__()

        # assign pre-trained encoder model
        self.nn_encoder = encoder

        # TODO: think of adding some more dense layers if required

        # create the classification layer (fully-connected)
        self.flatten = Flatten(name='flatten_out')
        self.dropout = Dropout(rate=0.5, name='drouput_out')
        self.dense_out = Dense(units=num_classes, name='dense_out')


    def call(self, inputs, training=False):

        x = inputs

        # encode the inputs -> downsampling
        x = self.nn_encoder(x, training=training)

        # classify the down-sampled features
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        x = self.dense_out(x)

        return x



class AutoEncEatModel(tf.keras.Model):

    def __init__(self):
        super(AutoEncEatModel, self).__init__()

        # create encoder and decoder model
        self.nn_encoder = EncoderModel()
        self.nn_decoder = DecoderModel()


    def call(self, inputs, training=False):

        x = inputs

        # encode (downsampling) and decode (upsampling)
        x = self.nn_encoder(x, training=training)
        x = self.nn_decoder(x, training=training)

        return x


class EncoderModel(tf.keras.Model):

    def __init__(self, out_channels: int=32):

        dropout_rate = 0.4
        l2_factor = 2e-4

        # freq 16000/s, 4s shards -> input shape (128, 128, 1)
        # freq 48000/s, 4s shards -> input shape (128, 376, 1)

        # first convolution layer with batch normalization and max pooling
        self.nn_conv_1 = Conv2D(32, (5, 5), strides=1, padding='same', name='nn_conv_1',
                                kernel_regularizer=L2(l2_factor))
        self.nn_bnorm_1 = BatchNormalization(name='nn_bnorm_1')
        self.act_1 = Activation('relu', name='act_1')
        self.dropout_1 = Dropout(rate=dropout_rate, name='dropout_1')
        self.maxpool_1 = MaxPooling2D(name='maxpool_1')

        # second convolution layer with batch normalization and max pooling
        self.nn_conv_2 = Conv2D(32, (3, 3), strides=1, padding='same', name='nn_conv_2',
                                kernel_regularizer=L2(l2_factor))
        self.nn_bnorm_2 = BatchNormalization(name='nn_bnorm_2')
        self.act_2 = Activation('relu', name='act_2')
        self.dropout_2 = Dropout(rate=dropout_rate, name='dropout_2')
        self.maxpool_2 = MaxPooling2D(name='maxpool_2', padding='same')

        # third convolution layer with batch normalization and max pooling
        self.nn_conv_3 = Conv2D(out_channels, (3, 3), strides=1, padding='same', name='nn_conv_3',
                                kernel_regularizer=L2(l2_factor))
        self.act_3 = Activation('relu', name='act_3')
        self.maxpool_3 = MaxPooling2D(name='maxpool_3')

        # freq 16000/s -> output shape (16, 16, ?) -> repr. state of 256 * out_channels
        # freq 48000/s -> output shape (16, 47, ?) -> repr. state of 752 * out_channels

        # TODO: decrease out_channels -> smaller repr. state for classification


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
        x = self.nn_bnorm_2(x, training)
        x = self.act_2(x)
        x = self.dropout_2(x, training)
        x = self.maxpool_2(x)

        # process the third convolution layer
        x = self.nn_conv_3(x)
        x = self.act_3(x)
        x = self.maxpool_3(x)

        return x


class DecoderModel(tf.keras.Model):

    def __init__(self):

        dropout_rate = 0.4
        l2_factor = 2e-4

        # freq 16000/s -> output shape (16, 16, ?) -> repr. state of 256 * out_channels
        # freq 48000/s -> output shape (16, 47, ?) -> repr. state of 752 * out_channels

        # first convolution layer with batch normalization and max pooling
        self.nn_conv_1 = Conv2DTranspose(32, (5, 5), strides=1, padding='same', name='nn_conv_1',
                                kernel_regularizer=L2(l2_factor))
        self.act_1 = Activation('relu', name='act_1')
        self.dropout_1 = Dropout(rate=dropout_rate, name='dropout_1')
        self.maxpool_1 = UpSampling2D(name='upsampling_1')

        # second convolution layer with batch normalization and max pooling
        self.nn_conv_2 = Conv2DTranspose(32, (3, 3), strides=1, padding='same', name='nn_conv_2',
                                kernel_regularizer=L2(l2_factor))
        self.act_2 = Activation('relu', name='act_2')
        self.dropout_2 = Dropout(rate=dropout_rate, name='dropout_2')
        self.maxpool_2 = UpSampling2D(name='upsampling_2')

        # third convolution layer with batch normalization and max pooling
        self.nn_conv_3 = Conv2DTranspose(32, (3, 3), strides=1, padding='same', name='nn_conv_3',
                                kernel_regularizer=L2(l2_factor))
        self.act_3 = Activation('relu', name='act_3')
        self.maxpool_3 = UpSampling2D(name='upsampling_3')

        # convert the upsampled content to 1-channel grayscale image data
        self.nn_conv_out = Conv2D(1, (1, 1), strides=1, padding='same', name='nn_conv_4')

        # freq 16000/s, 4s shards -> input shape (128, 128, 1)
        # freq 48000/s, 4s shards -> input shape (128, 376, 1)


    def call(self, inputs, training=False):

        x = inputs

        # process the first convolution layer
        x = self.nn_conv_1(x)
        x = self.act_1(x)
        x = self.dropout_1(x, training)
        x = self.maxpool_1(x)

        # process the second convolution layer
        x = self.nn_conv_2(x)
        x = self.act_2(x)
        x = self.dropout_2(x, training)
        x = self.maxpool_2(x)

        # process the third convolution layer
        x = self.nn_conv_3(x)
        x = self.act_3(x)
        x = self.maxpool_3(x)

        # process the output convolution layer
        x = self.nn_conv_out(x)

        return x
