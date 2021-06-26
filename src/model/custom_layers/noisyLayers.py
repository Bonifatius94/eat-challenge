import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import InputSpec
from tensorflow.keras import initializers


# This is a custom noisy layer implementation extending the common Dense and Conv2D
# layers with gaussian noise generation facilitating the layer's denoise capabilitaties.
# The noisy layers should be 100% compatible to their common layer counterparts.


class NoisyDense(Dense):
    def __init__(self, units, **kwargs):
        # pass constructor args through
        self.output_dim = units
        super(NoisyDense, self).__init__(units, **kwargs)

    def build(self, input_shape):
        # make sure there was a batch dimension added to the input data
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]

        # create the standard kernel (as in common implementation)
        self.kernel = self.add_weight(
            shape=(self.input_dim, self.units), initializer=self.kernel_initializer,
            name='kernel', regularizer=None, constraint=None)

        # create the special noise generation kernel (unique to this implementation)
        self.kernel_sigma = self.add_weight(
            shape=(self.input_dim, self.units), initializer=initializers.Constant(0.017),
            name='sigma_kernel', regularizer=None, constraint=None)

        # create bias weights (default: with bias)
        self.bias = None
        if self.use_bias:

            # create the standard bias (as in common implementation)
            self.bias = self.add_weight(
                shape=(self.units,), initializer=self.bias_initializer,
                name='bias', regularizer=None, constraint=None)

            # create the special noise generation bias (unique to this implementation)
            self.bias_sigma = self.add_weight(
                shape=(self.units,), initializer=initializers.Constant(0.017),
                name='bias_sigma', regularizer=None, constraint=None)

        # finally, set the input specification and mark the layer as successfully built
        self.input_spec = InputSpec(min_ndim=2, axes={ -1: self.input_dim })
        self.built = True

    def call(self, inputs):
        # generate some gaussian noise for each kernel weight and apply 
        # the generated noise by elementwise multiplication with the kernel
        self.kernel_epsilon = tf.random.normal(shape=(self.input_dim, self.units))
        w = self.kernel + tf.multiply(self.kernel_sigma, self.kernel_epsilon)

        # apply the kernel to the inputs like in a common dense layer
        output = tf.keras.dot(inputs, w)

        # process the bias (if exists)
        if self.use_bias:

            # generate some gaussian noise for each bias weight and apply 
            # the generated noise by elementwise multiplication with the bias
            self.bias_epsilon = tf.random.normal(shape=(self.units,))
            b = self.bias + tf.multiply(self.bias_sigma, self.bias_epsilon)

            # apply the bias to the inputs like in a common dense layer
            output = output + b

        # process the activation function attached to this layer (if exists)
        if self.activation is not None:
            output = self.activation(output)

        return output


class NoisyConv2D(Conv2D):
    def __init__(self, filters, kernel_size, **kwargs):
        # pass constructor args through
        self.output_dim = filters
        super(NoisyConv2D, self).__init__(filters, kernel_size, **kwargs)

    def build(self, input_shape):
        # make sure to convert the padding to uppercase letters
        self.padding = self.padding.upper()

        # determine the channel axis
        is_channel_first = self.data_format == 'channels_first' or self.data_format == 'NCHW'
        channel_axis = 1 if is_channel_first else 3
        self.data_format = "NCHW" if is_channel_first else "NHWC"

        # make sure the channel axis exists
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        # define input dimension and kernel shape
        self.input_dim = input_shape[channel_axis]
        self.kernel_shape = self.kernel_size + (self.input_dim, self.filters)

        # create the standard kernel (as in common implementation)
        self.kernel = self.add_weight(
            shape=self.kernel_shape, initializer=self.kernel_initializer,
            name='kernel', regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        # create the special noise generation kernel (unique to this implementation)
        self.kernel_sigma = self.add_weight(
            shape=self.kernel_shape, initializer=initializers.Constant(0.017),
            name='kernel_sigma', regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        # create bias weights (default: with bias)
        self.bias = None
        if self.use_bias:

            # create the standard bias (as in common implementation)
            self.bias = self.add_weight(
                shape=(self.filters,), initializer=self.bias_initializer, name='bias',
                regularizer=self.bias_regularizer, constraint=self.bias_constraint)

            # create the special noise generation bias (unique to this implementation)
            self.bias_sigma = self.add_weight(
                shape=(self.filters,), initializer=initializers.Constant(0.017), name='bias_sigma',
                regularizer=self.bias_regularizer, constraint=self.bias_constraint)

        # finally, set the input specification and mark the layer as successfully built
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={ channel_axis: self.input_dim })
        self.built = True

    def call(self, inputs):
        # generate some gaussian noise for each kernel weight and apply 
        # the generated noise by elementwise multiplication with the kernel
        self.kernel_epsilon = tf.random.normal(shape=self.kernel_shape)
        w = self.kernel + tf.multiply(self.kernel_sigma, self.kernel_epsilon)

        # apply the kernel to the inputs like in a common conv2d layer
        outputs = tf.nn.conv2d(
            inputs, w, strides=self.strides, padding=self.padding,
            data_format=self.data_format, dilations=self.dilation_rate)

        # process the bias (if exists)
        if self.use_bias:

            # generate some gaussian noise for each bias weight and apply 
            # the generated noise by elementwise multiplication with the bias
            self.bias_epsilon = tf.random.normal(shape=(self.filters,))
            b = self.bias + tf.multiply(self.bias_sigma, self.bias_epsilon)

            # apply the bias to the inputs like in a common conv2d layer
            outputs = tf.nn.bias_add(outputs, b, data_format=self.data_format)

        # process the activation function attached to this layer (if exists)
        if self.activation is not None:
            return self.activation(outputs)

        return outputs
