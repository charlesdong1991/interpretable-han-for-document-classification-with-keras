import keras
from keras.engine.topology import Layer
from keras import backend as K

from utils import k_dot


class Attention(Layer):
    def __init__(self, bias=True, **kwargs):
        """The implementation of an attention mechanism layer.

        Args:
            bias: boolean, adapt a bias modification or not.
            kwargs: any keyword arguments that base layer accepts.
        """
        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        """Create a trainable weight variable.

        This attention layer accepts the tensor that has three
        dimensions: (batch_size, time_steps, input_dim).

        Args:
            input_shape: 3d Input, (batch_size, time_steps, input_dim).

        Raises:
            ValueError: if input_shape is not 3 dimensions.
        """
        assert len(input_shape) == 3

        self.W = self.add_weight(
            name='W', shape=(input_shape[-1], input_shape[-1],),
            initializer=keras.initializers.get('uniform')
        )

        self.u = self.add_weight(
            name='context_vector', shape=(input_shape[-1],),
            initializer=keras.initializers.get('uniform')
        )

        self.b = self.add_weight(
            name='bias', shape=(input_shape[-1],),
            initializer='zero'
        )

        super(Attention, self).build(input_shape)

    def call(self, x):

        att_weights = self._get_attention_weights(x)

        # Reshape the attention weights to match the dimensions of X
        att_weights = K.expand_dims(att_weights)
        # Multiply each input by its attention weights
        weighted_input = keras.layers.Multiply()([x, att_weights])

        # Sum in the direction of the time-axis.
        return K.sum(weighted_input, axis=1)

    @staticmethod
    def compute_output_shape(input_shape):
        return input_shape[0], input_shape[-1]

    def _get_attention_weights(self, x):
        """Calculate attention weights.

        Args:
            x: Input array.

        Returns:
            a: attention weights.
        """
        u_xw = k_dot(x, self.W)
        if self.bias:
            u_xw += self.b
        u_tw = K.tanh(u_xw)
        a_tw = k_dot(u_tw, self.u)

        # apply softmax to get probability for attention
        a = K.softmax(a_tw)

        return a
