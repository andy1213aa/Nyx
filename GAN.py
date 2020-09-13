import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from functools import partial
from math import log
from resblock import ResBlock_generator, ResBlock_discriminator
import tensorflow as tf
import os

class GAN():
    def __init__(self, length, width, height):
        self.length = length
        self.width = width
        self.height = height

        self.filterNumber =16
        self.L2_coefficient =0.5#1/(length*width*height)
        self.dis = self.discriminator()
        #self.disOptimizer = keras.optimizers.RMSprop(lr = 0.0002, clipvalue = 1.0, decay = 1e-8)
        self.disOptimizer = keras.optimizers.Adam(lr = 0.0004,beta_1=0.9, beta_2 = 0.999)
        self.gen = self.generator()
        #self.genOptimizer = keras.optimizers.RMSprop(lr = 0.00005, clipvalue = 1.0, decay = 1e-8)
        self.genOptimizer = keras.optimizers.Adam(lr = 0.0001,beta_1=0.9, beta_2 = 0.999)                                            
        self.gradient_penality_width = 10.0
    

    def generator(self):
        parameter1_input = keras.Input(shape = (1), name = 'parameter1')
        parameter2_input = keras.Input(shape = (1), name = 'parameter2')
        parameter3_input = keras.Input(shape = (1), name = 'parameter3')

        x = layers.Dense(512, name = 'parameter1_layer_1')(parameter1_input)
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        x = layers.PReLU()(x)
        x = layers.Dense(512, name = 'parameter1_layer_2')(x)
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        x = layers.PReLU()(x)

        
        y = layers.Dense(512, name = 'parameter2_layer_1')(parameter2_input)
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        y = layers.PReLU()(y)
        y = layers.Dense(512, name = 'parameter2_layer_2')(y)
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        y = layers.PReLU()(y)

        
        z = layers.Dense(512, name = 'parameter3_layer_1')(parameter3_input)
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        z = layers.PReLU()(z)
        z = layers.Dense(512, name = 'parameter3_layer_2')(z)
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        z = layers.PReLU()(z)

        concatenate = layers.concatenate(inputs = [x, y, z])
        
        g =layers.Dense(4*4*4*int(self.width/4)*self.filterNumber)(concatenate)
        g = layers.Reshape((4, 4, 4, int(self.width/4)*self.filterNumber))(g)
        
        for i in range(int(log(self.width/4, 2))-1, -1, -1):
           # g = SpectralNormalization(layers.Conv2DTranspose((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(g)
            g = ResBlock_generator((2**i)*self.filterNumber)(g)
            #g = layers.Conv3DTranspose((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False)(g)
            g = layers.PReLU()(g)
        

        #g = SpectralNormalization(layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', use_bias=False))(g)
        g = layers.Conv3DTranspose(1, kernel_size=3, strides=1, padding='same', use_bias=False)(g)
        
        #g = layers.Activation(tf.nn.tanh)(g)
        
        model = keras.Model(inputs = [parameter1_input, parameter2_input, parameter3_input], outputs = g)
        plot_model(model, to_file='WGAN_generator.png', show_shapes=True)
        return model
    
    def discriminator(self):
        parameter1_input = keras.Input(shape = (1), name = 'parameter1')
        parameter2_input = keras.Input(shape = (1), name = 'parameter2')
        parameter3_input = keras.Input(shape = (1), name = 'parameter3')
        dataInput = keras.Input(shape = (self.length,self.width, self.height, 1), name = 'groundTruth/fake')
    
        x = layers.Dense(512, name = 'parameter1_layer_1')(parameter1_input)
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        x = layers.PReLU()(x)
        
        x = layers.Dense(512, name = 'parameter1_layer_2')(x)
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        x = layers.PReLU()(x)
        
        y = layers.Dense(512, name = 'parameter2_layer_1')(parameter2_input)
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        y = layers.PReLU()(y)

        y = layers.Dense(512, name = 'parameter2_layer_2')(y)
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        y = layers.PReLU()(y)
        
        z = layers.Dense(512, name = 'parameter3_layer_1')(parameter3_input)
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        z = layers.PReLU()(z)

        z = layers.Dense(512, name = 'parameter3_layer_2')(z)
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        z = layers.PReLU()(z)

        concatenate = layers.concatenate(inputs = [x, y, z])
        xyz = layers.Dense(2**((int(log(self.width/8, 2))))*self.filterNumber)(concatenate)    #Depends on how many level of conv2D you have
        xyz = layers.PReLU()(xyz)
        

        #d = SpectralNormalization(layers.Conv2D(self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(dataInput)
        #d = SpectralNormalization(layers.Conv3D(self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(dataInput)
        d = ResBlock_discriminator(self.filterNumber)(dataInput)
        d = layers.PReLU()(d)
        for i in range(1, int(log(self.width/2, 2))-1):
            d = ResBlock_discriminator((2**i)*self.filterNumber)(d)
            #d = SpectralNormalization(layers.Conv3D((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(d)
            #d = SpectralNormalization(layers.Conv2D((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(d)#
            
            d = layers.PReLU()(d)
        


        d = tf.nn.avg_pool(input = d, ksize= [1, 4, 4, 4,  1] , strides=[1, 1, 1, 1,  1], padding='VALID')*(self.height*self.width*self.length)
        
        
        #d = layers.Flatten()(d)
        f1 = tf.multiply(d, xyz)
        f2 = layers.Dense(1)(d)
        r = layers.Add()([f1, f2])

        model = keras.Model(inputs = [parameter1_input, parameter2_input, parameter3_input, dataInput], outputs = r)
        #model = keras.Model(inputs = [dataInput], outputs = d)
        plot_model(model, to_file = "WGAN_Discriminator.png", show_shapes=True)
        return model

