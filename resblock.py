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

