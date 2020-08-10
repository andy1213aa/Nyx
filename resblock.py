import tensorflow as tf
from tensorflow.keras import layers
from SpectralNormalization import SpectralNormalization
class ResBlock_generator(layers.Layer):
  def __init__(self, out_shape, strides=1, ksize = 3, ):
      super(ResBlock_generator, self).__init__()

      
      #self.bn_0 = layers.BatchNormalization()
      self.PRelu0 = layers.PReLU()
      self.upSample = layers.UpSampling3D()
      self.conv_1 = SpectralNormalization(layers.Conv3D(out_shape,kernel_size = ksize, strides=1,padding='same', name = 'rg_conv1'))
      #self.bn_1 = layers.BatchNormalization()
      self.PRelu1 = layers.PReLU()
      self.conv_2 = SpectralNormalization(layers.Conv3D(out_shape,kernel_size = ksize ,strides=1,padding='same', name = 'rg_conv2'))
      

      #identity
      self.upSample_identity = layers.UpSampling3D()
      self.conv_identity = layers.Conv3D(out_shape,kernel_size=1,strides=1)


  def call(self, inputs, training=None):

      #x = self.bn_0(inputs)
      x = self.PRelu0(inputs)
      x = self.upSample(x)
      x = self.conv_1(x)
      #x = self.bn_1(x)
      x = self.PRelu1(x)
      x = self.conv_2(x)
      
      
      identity = self.upSample_identity(inputs)
      identity = self.conv_identity(identity)
      outputs = layers.add([x,identity])

      return outputs

class ResBlock_discriminator(layers.Layer):
  def __init__(self, out_shape, strides=1):
      super(ResBlock_discriminator, self).__init__()

      self.PRelu0 = layers.PReLU()
      self.conv_1 = SpectralNormalization(layers.Conv3D(out_shape,kernel_size=3,strides=1,padding='same', name = 'rd_conv1'))
      self.PRelu1 = layers.PReLU()
      self.conv_2 = SpectralNormalization(layers.Conv3D(out_shape,kernel_size=3,strides=1,padding='same', name = 'rd_conv2'))
      self.average_pool1 = layers.AveragePooling3D()

      #identity
      self.conv_identity = layers.Conv3D(out_shape, kernel_size=1 ,strides=1)
      self.average_pool2 = layers.AveragePooling3D()


  def call(self, inputs, training=None):

      x = self.PRelu0(inputs)
      x = self.conv_1(x)
      x = self.PRelu1(x)
      x = self.conv_2(x)
      x = self.average_pool1(x)
      

      identity = self.conv_identity(inputs)
      identity = self.average_pool2(identity)
      outputs = layers.add([x,identity])
      
      return outputs

