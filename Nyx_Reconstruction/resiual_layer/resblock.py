import tensorflow as tf
from tensorflow.keras import layers, initializers
from .SpectralNormalization import SpectralNormalization
class ResBlock_generator(layers.Layer):
  def __init__(self, out_shape, strides=1, ksize = 3):
      super(ResBlock_generator, self).__init__()

      
      #self.bn_0 = layers.BatchNormalization()
      self.PRelu0 = layers.LeakyReLU()
      self.upSample = layers.UpSampling3D()
      self.conv_0 = layers.Conv3D(out_shape,kernel_size = ksize, strides=1,padding='same', name = 'rg_conv1',  use_bias=False, kernel_initializer=initializers.he_normal())
      #self.bn_1 = layers.BatchNormalization()
      #self.PRelu1 = layers.PReLU()
      #self.conv_1 = layers.Conv3D(out_shape,kernel_size = ksize ,strides=1,padding='same', name = 'rg_conv2',  use_bias=False, kernel_initializer=initializers.he_normal())
      

      #shortcut
      # self.upSample_shortcut = layers.UpSampling3D()
      # self.conv_shortcut = layers.Conv3D(out_shape,kernel_size=1,strides=1, padding='same', use_bias=False)
        

  def call(self, inputs, training=None):

      #x = self.bn_0(inputs)
      x = self.upSample(inputs)
      x = self.conv_0(x)
      x = self.PRelu0(x)
      #x = self.PRelu1(x)
      #x = self.conv_1(x)
      #x = self.upSample(x)
      #x = self.bn_1(x)
      
      
      # shortcut = self.upSample_shortcut(inputs)
      # shortcut = self.conv_shortcut(shortcut)
    #   outputs = layers.add([x,shortcut])

      return x #+ shortcut

class ResBlock_discriminator(layers.Layer):
  def __init__(self, out_shape, strides=1,ksize=3):
      super(ResBlock_discriminator, self).__init__()

      self.conv_0 = SpectralNormalization(layers.Conv3D(out_shape,kernel_size=ksize,strides=2 ,padding='same', name = 'rd_conv1', use_bias=False, kernel_initializer=initializers.he_normal()))
      self.PRelu0 = layers.LeakyReLU()
     # self.PRelu1 = layers.PReLU()
      #self.conv_1 = SpectralNormalization(layers.Conv3D(out_shape,kernel_size=ksize,strides=1,padding='same', name = 'rd_conv2',  use_bias=False, kernel_initializer=initializers.he_normal()))
      #self.average_pool0 = layers.AveragePooling3D()

      #shortcut
      # self.conv_shortcut = SpectralNormalization(layers.Conv3D(out_shape, kernel_size=1 ,strides=2, padding='same', use_bias=False))
      #self.average_pool2 = layers.AveragePooling3D()


  def call(self, inputs, training=None):

      x = self.conv_0(inputs)
      #x = self.average_pool0(x)
      x = self.PRelu0(x)
      #x = self.PRelu1(x)
      #x = self.conv_1(x)
      

      # shortcut = self.conv_shortcut(inputs)
    #   #shortcut = self.average_pool2(shortcut)
    #   outputs = layers.add([x,shortcut])
      
      return x #+ shortcut

