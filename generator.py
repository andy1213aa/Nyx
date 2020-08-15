from resblock import ResBlock_generator
import tensorflow as tf
from tensorflow.keras import layers
#from IPython.display import Image
class generator(tf.keras.Model):
    def __init__(self, ch=64):
        super(generator, self).__init__()
        self.ch = ch
        self.xD0 = layers.Dense(512, name = 'parameter1_layer_1')
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        self.xR0 = layers.ReLU()
        self.xD1 = layers.Dense(512, name = 'parameter1_layer_2')
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        self.xR1 = layers.ReLU()
        
        self.yD0 = layers.Dense(512, name = 'parameter2_layer_1')
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        self.yR0 = layers.ReLU()
        self.yD1 = layers.Dense(512, name = 'parameter2_layer_2')
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        self.yR1 = layers.ReLU()

        
        self.zD0 = layers.Dense(512, name = 'parameter3_layer_1')
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        self.zR0 = layers.ReLU()
        self.zD1 = layers.Dense(512, name = 'parameter3_layer_2')
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        self.zR1 = layers.ReLU()

        
        self.gD0 =layers.Dense(ch*16*4*4*4)
        
        # g = SpectralNormalization(layers.Conv2DTranspose((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(g)
        self.gRes0 = ResBlock_generator(ch*2)
        self.gRes1 = ResBlock_generator(ch)
        #g = layers.Conv3DTranspose((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False)(g)
        self.gR0 = layers.ReLU()
        

        #g = SpectralNormalization(layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', use_bias=False))(g)
        self.gConv0 = layers.Conv3D(1, kernel_size=3, strides=1, padding='same', use_bias=False)
        
        #g = layers.Activation(tf.nn.tanh)(g)
        #plot_model(model, to_file='WGAN_generator.png', show_shapes=True)
    def call(self, inputs):
        p1, p2, p3 = inputs
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
        xyz = self.gD0(xyz)
        xyz = layers.Reshape((4, 4, 4, 16*self.ch))(xyz)

        g = self.gRes0(xyz)
        g = self.gRes1(g)
        g = self.gR0(g)
        g = self.gConv0(g)

        return g








