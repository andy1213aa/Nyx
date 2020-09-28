from ..resiual_layer.resblock import ResBlock_discriminator
import tensorflow as tf
from tensorflow.keras import layers, initializers
from ..resiual_layer.SpectralNormalization import SpectralNormalization

#from IPython.display import Image
class discriminator(tf.keras.Model):
    def __init__(self, ch= 16):
        super(discriminator, self).__init__()

        self.xD0 = SpectralNormalization(layers.Dense(512, name = 'parameter1_layer_1'))
        self.xR0 = layers.LeakyReLU()
        self.xD1 = SpectralNormalization(layers.Dense(512, name = 'parameter1_layer_2'))
        self.xR1 = layers.LeakyReLU()
        self.xD2 = SpectralNormalization(layers.Dense(512, name = 'parameter1_layer_3'))
        self.xR2 = layers.LeakyReLU()

        self.yD0 = SpectralNormalization(layers.Dense(512, name = 'parameter2_layer_1'))
        self.yR0 = layers.LeakyReLU()
        self.yD1 = SpectralNormalization(layers.Dense(512, name = 'parameter2_layer_2'))
        self.yR1 = layers.LeakyReLU()
        self.yD2 = SpectralNormalization(layers.Dense(512, name = 'parameter2_layer_3'))
        self.yR2 = layers.LeakyReLU()

        self.zD0 = SpectralNormalization(layers.Dense(512, name = 'parameter3_layer_1'))
        self.zR0 = layers.LeakyReLU()
        self.zD1 = SpectralNormalization(layers.Dense(512, name = 'parameter3_layer_2'))
        self.zR1 = layers.LeakyReLU()
        self.zD2 = SpectralNormalization(layers.Dense(512, name = 'parameter3_layer_3'))
        self.zR2 = layers.LeakyReLU()

        self.concateD = SpectralNormalization(layers.Dense(ch*16))
        self.concateR = layers.LeakyReLU()

        # self.conv0 = SpectralNormalization(layers.Conv3D(ch, kernel_size=3, strides=2, padding='same', use_bias=False))
        #self.resR0 = layers.LeakyReLU()
        # self.conv1 = SpectralNormalization(layers.Conv3D(2*ch, kernel_size=3, strides=2, padding='same', use_bias=False))
        # self.resR1 = layers.LeakyReLU()
        # self.conv2 = SpectralNormalization(layers.Conv3D(4*ch, kernel_size=3, strides=2, padding='same', use_bias=False))
        # self.resR2 = layers.LeakyReLU()
        # self.conv3 = SpectralNormalization(layers.Conv3D(8*ch, kernel_size=3, strides=2, padding='same', use_bias=False))
        # self.resR3 = layers.LeakyReLU()
        self.res0 = ResBlock_discriminator(ch, ksize=3)
        self.res1 = ResBlock_discriminator(ch*2, ksize=3)
        self.res2 = ResBlock_discriminator(ch*4, ksize=3)
        self.res3 = ResBlock_discriminator(ch*8, ksize=3)
        self.res4 = ResBlock_discriminator(ch*8, ksize=3)
        self.res5 = ResBlock_discriminator(ch*16, ksize=3)
       # self.res6 = ResBlock_discriminator(ch*16, ksize=5)
        #self.conv4 = SpectralNormalization(layers.Conv3D(1, kernel_size=3, strides=1, padding='same', use_bias=False))
        #self.GAV3D = layers.GlobalAveragePooling3D()
        #self.out = layers.Dense(1, kernel_initializer=initializers.he_normal())
        
        self.outputD = SpectralNormalization(layers.Dense(1))
        
        #model = keras.Model(inputs = [dataInput], outputs = d)
       # plot_model(model, to_file = "WGAN_Discriminator.png", show_shapes=True)
    def call(self, inputs):
        p1, p2, p3, data = inputs
        x = self.xD0(p1)
        x = self.xR0(x)
        x = self.xD1(x)
        x = self.xR1(x)

        y = self.yD0(p2)
        y = self.yR0(y)
        y = self.yD1(y)
        y = self.yR1(y)
        
        z = self.zD0(p3)
        z = self.zR0(z)
        z = self.zD1(z)
        z = self.zR1(z)

        xyz = layers.concatenate([x, y, z])
        xyz = self.concateD(xyz)
        xyz = self.concateR(xyz)
        # d = self.conv0(inputs)
        # d = self.conv1(d)
        # d = self.resR1(d)
        # d = self.conv2(d)
        # d = self.resR2(d)
        # d = self.conv3(d)
        # d = self.resR3(d)

        #d = self.res0(data)
        d = self.res0(data)
        d = self.res1(d)
        d = self.res2(d)
        d = self.res3(d)
        d = self.res4(d)
        d = self.res5(d)
       # d = self.res6(d)
        #d = self.conv4(d)
        #d = self.resR(d)
        #d = self.resR0(d)
        #d = layers.Flatten()(d)

        #d = self.GAV3D(d)# * (4*4*4)
        #d = tf.nn.avg_pool(input = d, ksize= [1, 4, 4, 4,  1] , strides=[1, 1, 1, 1,  1], padding='VALID') * (4*4*4)
        #d = self.out(d)
        # d = tf.math.reduce_sum(d)
        #d = layers.Flatten()(d)
      
        f1 = layers.Multiply()([d, xyz])
        
        f1 = tf.math.reduce_sum(f1, axis = 1, keepdims=True)
    
        f2 = self.outputD(d)

        r = layers.add([f1, f2])
        

        return r