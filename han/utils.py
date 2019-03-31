from keras import backend as K


def k_dot(vec_x, vec_y):
    """Dot operation for Keras compatible for theano and tensorflow.

    Args:
        vec_x: vector.
        vec_y: vector.
    Returns:
        A dot product operation by two vectors.
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(vec_x, K.expand_dims(vec_y)), axis=-1)
    else:
        return K.dot(vec_x, vec_y)
