import tensorflow as tf
import numpy as np

from tensorflow.python.ops import array_ops, sparse_ops
from tensorflow.python.framework import sparse_tensor
# SHAPE OPERATIONS
py_all = all

def is_sparse(tensor):
    """Returns whether a tensor is a sparse tensor.
    Arguments:
      tensor: A tensor instance.
    Returns:
      A boolean.
    Example:
    ```python
      >>> from keras import backend as K
      >>> a = K.placeholder((2, 2), sparse=False)
      >>> print(K.is_sparse(a))
      False
      >>> b = K.placeholder((2, 2), sparse=True)
      >>> print(K.is_sparse(b))
      True
    ```
    """
    return isinstance(tensor, sparse_tensor.SparseTensor)

def to_dense(tensor):
    """Converts a sparse tensor into a dense tensor and returns it.
    Arguments:
      tensor: A tensor instance (potentially sparse).
    Returns:
      A dense tensor.
    Examples:
    ```python
      >>> from keras import backend as K
      >>> b = K.placeholder((2, 2), sparse=True)
      >>> print(K.is_sparse(b))
      True
      >>> c = K.to_dense(b)
      >>> print(K.is_sparse(c))
      False
    ```
    """
    if is_sparse(tensor):
        return sparse_ops.sparse_tensor_to_dense(tensor)
    else:
        return tensor

def concatenate(tensors, axis=-1):
    """Concatenates a list of tensors alongside the specified axis.
    Arguments:
      tensors: list of tensors to concatenate.
      axis: concatenation axis.
    Returns:
      A tensor.
    """
    if axis < 0:
        rank = ndim(tensors[0])
        if rank:
            axis %= rank
        else:
            axis = 0

    if py_all([is_sparse(x) for x in tensors]):
        return sparse_ops.sparse_concat(axis, tensors)
    else:
        return array_ops.concat([to_dense(x) for x in tensors], axis)


def resize_volumes(x, depth_factor, height_factor, width_factor, data_format='channels_last'):
    """Resizes the volume contained in a 5D tensor.
    Arguments:
      x: Tensor or variable to resize.
      depth_factor: Positive integer.
      height_factor: Positive integer.
      width_factor: Positive integer.
      data_format: One of `"channels_first"`, `"channels_last"`.
    Returns:
      A tensor.
    Raises:
      ValueError: if `data_format` is neither
          `channels_last` or `channels_first`.
    """
    if data_format == 'channels_first':
        output = repeat_elements(x, depth_factor, axis=2)
        output = repeat_elements(output, height_factor, axis=3)
        output = repeat_elements(output, width_factor, axis=4)
        return output
    elif data_format == 'channels_last':
        output = repeat_elements(x, depth_factor, axis=1)
        output = repeat_elements(output, height_factor, axis=2)
        output = repeat_elements(output, width_factor, axis=3)
        return output
    else:
        raise ValueError('Invalid data_format:', data_format)


def repeat_elements(x, rep, axis):
    """Repeats the elements of a tensor along an axis, like `np.repeat`.
    If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
    will have shape `(s1, s2 * rep, s3)`.
    Arguments:
      x: Tensor or variable.
      rep: Python integer, number of times to repeat.
      axis: Axis along which to repeat.
    Raises:
      ValueError: In case `x.shape[axis]` is undefined.
    Returns:
      A tensor.
    """
    x_shape = x.get_shape().as_list()
    if x_shape[axis] is None:
        raise ValueError('Axis ' + str(axis) + ' of input tensor '
                         'should have a defined dimension, but is None. '
                         'Full tensor shape: ' + str(tuple(x_shape)) + '. '
                         'Typically you need to pass a fully-defined '
                         '`input_shape` argument to your first layer.')
    # slices along the repeat axis
    splits = array_ops.split(value=x, num_or_size_splits=x_shape[axis], axis=axis)
    # repeat each slice the given number of reps
    x_rep = [s for s in splits for _ in range(rep)]
    return concatenate(x_rep, axis)

def glorot_uniform(shape):
    fan_in, fan_out = _compute_fans(shape)
    scale = 1
    scale /= max(1., float(fan_in + fan_out) / 2)
    
    limit = np.sqrt(3. * scale)
    return tf.random_uniform(shape, -limit, limit)
        
def _compute_fans(shape):
    receptive_field_size = np.prod(shape[:2])
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out

def weight_variable(shape):
    return tf.Variable(glorot_uniform(shape))

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                        strides=[1, 2, 2, 2, 1], padding='SAME')