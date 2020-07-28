import tensorflow as tf
from tensorflow.keras import layers

class ResBlock_generator(layers.Layer):
  def __init__(self, out_shape, strides=1, residual_path=False):
      super(ResBlock_generator, self).__init__()

      
      self.bn_0 = layers.BatchNormalization()
      self.relu0 = layers.LeakyReLU()
      self.upSample = layers.UpSampling2D()
      self.conv_1 = SpectralNormalization(layers.Conv2D(out_shape,(3,3),strides=1,padding='same', name = 'rg_conv1'))
      self.bn_1 = layers.BatchNormalization()
      self.relu1 = layers.LeakyReLU()
      self.conv_2 = SpectralNormalization(layers.Conv2D(out_shape,(3,3),strides=1,padding='same', name = 'rg_conv2'))
      

      #identity
      self.upSample_identity = layers.UpSampling2D()
      self.conv_identity = SpectralNormalization(layers.Conv2D(out_shape,(1,1),strides=1))


  def call(self, inputs, training=None):

      x = self.bn_0(inputs)
      x = self.relu0(x)
      x = self.upSample(x)
      x = self.conv_1(x)
      x = self.bn_1(x)
      x = self.relu1(x)
      x = self.conv_2(x)
      
      
      identity = self.upSample_identity(inputs)
      identity = self.conv_identity(identity)
      outputs = layers.add([x,identity])

      return outputs

class ResBlock_discriminator(layers.Layer):
  def __init__(self, out_shape, strides=1, residual_path=False):
      super(ResBlock_discriminator, self).__init__()

      self.relu0 = layers.LeakyReLU()
      self.conv_1 = SpectralNormalization(layers.Conv2D(out_shape,(3,3),strides=1,padding='same', name = 'rd_conv1'))
      self.relu1 = layers.LeakyReLU()
      self.conv_2 = SpectralNormalization(layers.Conv2D(out_shape,(3,3),strides=1,padding='same', name = 'rd_conv2'))
      self.average_pool = layers.AveragePooling2D()

      #identity
     
      self.conv_identity = SpectralNormalization(layers.Conv2D(out_shape,(1,1),strides=1))


  def call(self, inputs, training=None):

      x = self.relu0(inputs)
      x = self.conv_1(x)
      x = self.relu1(x)
      x = self.conv_2(x)
      x = self.average_pool(x)
      

      identity = self.conv_identity(inputs)
      identity = self.average_pool(identity)
      outputs = layers.add([x,identity])
      
      return outputs

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
class SpectralNormalization(layers.Wrapper):
    """
    Attributes:
       layer: tensorflow keras layers (with kernel attribute)
    """

    def __init__(self, layer, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        """Build `Layer`"""

        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, 'kernel'):
                raise ValueError(
                    '`SpectralNormalization` must wrap a layer that'
                    ' contains a `kernel` for weights')

            self.w = self.layer.kernel
            self.w_shape = self.w.shape.as_list()
            self.u = self.add_variable(
                shape=tuple([1, self.w_shape[-1]]),
                initializer=initializers.TruncatedNormal(stddev=0.02),
                name='sn_u',
                trainable=False,
                dtype=dtypes.float32)

        super(SpectralNormalization, self).build()

    @def_function.function
    def call(self, inputs, training=None):
        """Call `Layer`"""
        if training==None:
            training = K.learning_phase()
        
        if training==True:
            # Recompute weights for each forward pass
            self._compute_weights()
        
        output = self.layer(inputs)
        return output

    def _compute_weights(self):
        """Generate normalized weights.
        This method will update the value of self.layer.kernel with the
        normalized value, so that the layer is ready for call().
        """
        w_reshaped = array_ops.reshape(self.w, [-1, self.w_shape[-1]])
        eps = 1e-12
        _u = array_ops.identity(self.u)
        _v = math_ops.matmul(_u, array_ops.transpose(w_reshaped))
        _v = _v / math_ops.maximum(math_ops.reduce_sum(_v**2)**0.5, eps)
        _u = math_ops.matmul(_v, w_reshaped)
        _u = _u / math_ops.maximum(math_ops.reduce_sum(_u**2)**0.5, eps)

        self.u.assign(_u)
        sigma = math_ops.matmul(math_ops.matmul(_v, w_reshaped), array_ops.transpose(_u))

        self.layer.kernel.assign(self.w / sigma)

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())