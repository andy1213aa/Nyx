from resblock import ResBlock_generator
import tensorflow as tf
from tensorflow.keras import layers, initializers
from SpectralNormalization import SpectralNormalization
#from IPython.display import Image

class generator(tf.keras.Model):
    def __init__(self, ch= 8):
        super(generator, self).__init__()
        self.ch = ch
        self.xD0 = layers.Dense(512, name = 'parameter1_layer_1', kernel_initializer=initializers.he_normal())
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        self.xR0 = layers.LeakyReLU()
        self.xD1 = layers.Dense(512, name = 'parameter1_layer_2', kernel_initializer=initializers.he_normal())
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        self.xR1 = layers.LeakyReLU()
        self.xD2 = layers.Dense(512, name = 'parameter1_layer_3', kernel_initializer=initializers.he_normal())
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        self.xR2 = layers.LeakyReLU()
        
        self.yD0 = layers.Dense(512, name = 'parameter2_layer_1', kernel_initializer=initializers.he_normal())
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        self.yR0 = layers.LeakyReLU()
        self.yD1 = layers.Dense(512, name = 'parameter2_layer_2', kernel_initializer=initializers.he_normal())
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        self.yR1 = layers.LeakyReLU()
        self.yD2 = layers.Dense(512, name = 'parameter2_layer_3', kernel_initializer=initializers.he_normal())
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        self.yR2 = layers.LeakyReLU()

        
        self.zD0 = layers.Dense(512, name = 'parameter3_layer_1', kernel_initializer=initializers.he_normal())
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        self.zR0 = layers.LeakyReLU()
        self.zD1 = layers.Dense(512, name = 'parameter3_layer_2', kernel_initializer=initializers.he_normal())
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        self.zR1 = layers.LeakyReLU()
        self.zD2 = layers.Dense(512, name = 'parameter3_layer_3', kernel_initializer=initializers.he_normal())
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        self.zR2 = layers.LeakyReLU()

        
        self.gD0 =layers.Dense(ch*8*4*4*4, kernel_initializer=initializers.he_normal())
        # self.convT0 = layers.Conv3DTranspose(2*ch,  kernel_size=3, strides=2, padding='same', use_bias=False)
        # self.convT1 = layers.Conv3DTranspose(ch,  kernel_size=3, strides=2, padding='same', use_bias=False)
        # self.gR1 = layers.LeakyReLU()
        # g = SpectralNormalization(layers.Conv2DTranspose((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(g)
        #self.gRes0 = ResBlock_generator(16*ch)
       # self.gRes1 = ResBlock_generator(8*ch, ksize=3)
        self.gRes2 = ResBlock_generator(8*ch, ksize=3)
        self.gRes3 = ResBlock_generator(4*ch, ksize=3)
        self.gRes4 = ResBlock_generator(2*ch, ksize=3)
        #self.gR0 = layers.LeakyReLU()
        self.gRes5 = ResBlock_generator(ch, ksize=3)
        #g = layers.Conv3DTranspose((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False)(g)
        

        #g = SpectralNormalization(layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', use_bias=False))(g)
        self.gConv0 = layers.Conv3D(1, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=initializers.he_normal())
        
        #g = layers.Activation(tf.nn.tanh)(g)
        #plot_model(model, to_file='WGAN_generator.png', show_shapes=True)
    
    def call(self, inputs):

        p1, p2, p3 = inputs
        x = self.xD0(p1)
        x = self.xR0(x)
        x = self.xD1(x)
        x = self.xR1(x)
        x = self.xD2(x)
        x = self.xR2(x)

        y = self.yD0(p2)
        y = self.yR0(y)
        y = self.yD1(y)
        y = self.yR1(y)
        y = self.yD2(y)
        y = self.yR2(y)

        z = self.zD0(p3)
        z = self.zR0(z)
        z = self.zD1(z)
        z = self.zR1(z)
        z = self.zD2(z)
        z = self.zR2(z)

        xyz = layers.concatenate([x, y, z])
        xyz = self.gD0(xyz)
        xyz = layers.Reshape((4, 4, 4, 8*self.ch))(xyz)

        # g = self.convT0(xyz)
        # g = self.gR0(g)
        # g = self.convT1(g)
        # g = self.gR1(g)
        #g = self.gRes0(xyz)
        #g = self.gRes1(xyz)
        g = self.gRes2(xyz)
        g = self.gRes3(g)
        g = self.gRes4(g)
        g = self.gRes5(g)
        #g = self.gR0(g)
        g = self.gConv0(g)

        return g
