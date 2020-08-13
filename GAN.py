import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from functools import partial
from resblock import ResBlock_generator, ResBlock_discriminator
import tensorflow as tf
from SpectralNormalization import SpectralNormalization
import os

class GAN():
    def __init__(self, length, width, height, batchSize):
        self.length = length
        self.width = width
        self.height = height
        self.batchSize = batchSize
        self.filterNumber = 64
        self.L2_coefficient = 10.0  
        self.feature_cofficient = 10.0 
        self.dis = self.discriminator()
        #self.disOptimizer = keras.optimizers.RMSprop(lr = 0.0002, clipvalue = 1.0, decay = 1e-8)
        self.disOptimizer = keras.optimizers.Adam(lr = 0.0002)
        self.gen = self.generator()
        #self.genOptimizer = keras.optimizers.RMSprop(lr = 0.00005, clipvalue = 1.0, decay = 1e-8)
        self.genOptimizer = keras.optimizers.Adam(lr = 0.00005)                                            
        self.gradient_penality_width = 10.0
       

    def generator(self):
        parameter1_input = keras.Input(shape = (1), name = 'parameter1')
        parameter2_input = keras.Input(shape = (1), name = 'parameter2')
        parameter3_input = keras.Input(shape = (1), name = 'parameter3')

        x = layers.Dense(512, name = 'parameter1_layer_1')(parameter1_input)
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dense(512, name = 'parameter1_layer_2')(x)
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        
        y = layers.Dense(512, name = 'parameter2_layer_1')(parameter2_input)
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        y = layers.Dense(512, name = 'parameter2_layer_2')(y)
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)

        
        z = layers.Dense(512, name = 'parameter3_layer_1')(parameter3_input)
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        z = layers.ReLU()(z)
        z = layers.Dense(512, name = 'parameter3_layer_2')(z)
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        z = layers.ReLU()(z)

        concatenate = layers.concatenate(inputs = [x, y, z])
        
        g =layers.Dense(4*4*4*2*self.filterNumber)(concatenate)
        g = layers.Reshape((4, 4, 4,  2*self.filterNumber))(g)
           # g = SpectralNormalization(layers.Conv3DTranspose((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(g)
        g = ResBlock_generator(2*self.filterNumber)(g)
        g = ResBlock_generator(self.filterNumber)(g)
            #g = layers.Conv3DTranspose((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False)(g)
        g = layers.ReLU()(g)
        

        #g = SpectralNormalization(layers.Conv3DTranspose(1, kernel_size=3, strides=1, padding='same', use_bias=False))(g)
        g = layers.Conv3D(1, kernel_size=3, strides=1, padding='same', use_bias=False)(g)
        
        #g = layers.Activation(tf.nn.tanh)(g)
        
        model = keras.Model(inputs = [parameter1_input, parameter2_input, parameter3_input], outputs = g)
        plot_model(model, to_file='WGAN_generator.png', show_shapes=True)
        return model
    
    def discriminator(self):
        disConvOutput = []
        parameter1_input = keras.Input(shape = (1), name = 'parameter1')
        parameter2_input = keras.Input(shape = (1), name = 'parameter2')
        parameter3_input = keras.Input(shape = (1), name = 'parameter3')
        dataInput = keras.Input(shape = (self.length,self.width, self.height, 1), name = 'groundTruth/fake')
    
        x = layers.Dense(512, name = 'parameter1_layer_1')(parameter1_input)
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Dense(512, name = 'parameter1_layer_2')(x)
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        y = layers.Dense(512, name = 'parameter2_layer_1')(parameter2_input)
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)

        y = layers.Dense(512, name = 'parameter2_layer_2')(y)
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        
        z = layers.Dense(512, name = 'parameter3_layer_1')(parameter3_input)
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        z = layers.ReLU()(z)

        z = layers.Dense(512, name = 'parameter3_layer_2')(z)
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        z = layers.ReLU()(z)

        concatenate = layers.concatenate(inputs = [x, y, z])
        xyz = layers.Dense(2*self.filterNumber)(concatenate)    #Depends on how many level of conv3D you have
        xyz = layers.ReLU()(xyz)
        

        #d = SpectralNormalization(layers.Conv3D(self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(dataInput)
        #d = SpectralNormalization(layers.Conv3D(self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(dataInput)
        dRes0 = ResBlock_discriminator(self.filterNumber)(dataInput)
       # d = layers.ReLU()(d)
        dRes1 = ResBlock_discriminator(2*self.filterNumber)(dRes0)
       
            #d = SpectralNormalization(layers.Conv3D((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(d)
            #d = SpectralNormalization(layers.Conv3D((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(d)#
            
        d = layers.ReLU()(dRes1)
        


        d = layers.GlobalAveragePooling3D()(d) 
        d = tf.math.scalar_mul((self.height*self.width*self.length), d)
        f1 = layers.Dot(axes = 1)([d, xyz])
     
        f2 = layers.Dense(1)(d)
        r = layers.Add()([f1, f2])
        
        model = keras.Model(inputs = [parameter1_input, parameter2_input, parameter3_input, dataInput], outputs = [r, dRes0, dRes1])
        #model = keras.Model(inputs = [dataInput], outputs = d)
        plot_model(model, to_file = "WGAN_Discriminator.png", show_shapes=True)
        return model
    
    
    def generator_loss(self, real_logit, fake_logit, real_data, fake_data_by_real_parameter):
        #volumetric loss
        l2_norm = tf.norm(tensor = (fake_data_by_real_parameter-real_data[1]), ord='euclidean') / (self.length*self.width*self.height)
        #Feature loss
      
        feature_loss = tf.norm(tensor = (fake_logit[2]-real_logit[2]), ord='euclidean') / (2*self.filterNumber)
       


        #Adversarial loss
        g_loss = - tf.reduce_mean(fake_logit[0])
        return g_loss, l2_norm, feature_loss

    def discriminator_loss(self, real_logit, fake_logit):
        real_loss = -tf.reduce_mean(real_logit)
        fake_loss = tf.reduce_mean(fake_logit)
        return real_loss, fake_loss

    def gradient_penality(self, dis, real_data, fake_data ):
        def _interpolate(a, b):
            shape = [tf.shape(a)[0]]+[1]*(a.shape.ndims - 1)
            alpha = tf.random.uniform(shape = shape, minval = 0, maxval = 1.)
            inter = (alpha * a) + ((1-alpha)*b)
            inter.set_shape(a.shape)
            return inter
        x_img = _interpolate(real_data[1], fake_data)
        with tf.GradientTape() as tape:
            tape.watch(x_img)
            pred_logit = dis([real_data[0][0], real_data[0][1], real_data[0][2], x_img])
            #pred_logit = dis([x_img])
        grad = tape.gradient(pred_logit[0], x_img)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis = 1)
        gp_loss = tf.reduce_mean((norm-1.)**2)
        return gp_loss    
    
    @tf.function
    def train_generator(self, real_data):
        print('GAN SIDE EFFECT')
        with tf.GradientTape() as tape:
            random_vector1 = tf.random.uniform(shape = (self.batchSize, 1), minval=0.12, maxval=0.16)
            random_vector2 = tf.random.uniform(shape = (self.batchSize, 1), minval=0.021, maxval=0.024)
            random_vector3 = tf.random.uniform(shape = (self.batchSize, 1), minval=0.55, maxval=0.9)

            fake_data_by_random_parameter = self.gen([random_vector1, random_vector2, random_vector3],training = True)  #generate by random parameter
            fake_data_by_real_parameter = self.gen([real_data[0][0], real_data[0][1], real_data[0][2]],training = True) #generate by real parameter

            fake_logit = self.dis([random_vector1, random_vector2, random_vector3, fake_data_by_random_parameter], training = False)
            real_logit = self.dis([real_data[0][0], real_data[0][1], real_data[0][2], real_data[1]], training = False)
            #fake_logit = self.dis([fake_data_by_random_parameter], training = False)
            fake_loss, l2_norm, feature_loss= self.generator_loss(real_logit, fake_logit, real_data, fake_data_by_real_parameter)
            gLoss = fake_loss + self.L2_coefficient * l2_norm + self.feature_cofficient * feature_loss
        gradients = tape.gradient(gLoss, self.gen.trainable_variables)
        self.genOptimizer.apply_gradients(zip(gradients, self.gen.trainable_variables))
        return gLoss
    
    @tf.function
    def train_discriminator(self, real_data):
        print("DIS SIDE EFFECT")
        with tf.GradientTape() as t:
            random_vector1 = tf.random.uniform(shape = (self.batchSize, 1), minval=0.12, maxval=0.16)
            random_vector2 = tf.random.uniform(shape = (self.batchSize, 1), minval=0.021, maxval=0.024)
            random_vector3 = tf.random.uniform(shape = (self.batchSize, 1), minval=0.55, maxval=0.9)

            fake_data = self.gen([random_vector1, random_vector2, random_vector3],training = False)
            real_logit = self.dis([real_data[0][0], real_data[0][1], real_data[0][2], real_data[1]] , training = True)
            #real_logit = self.dis([real_data[1]] , training = True)
            fake_logit = self.dis([random_vector1, random_vector2, random_vector3, fake_data], training = True)
            #fake_logit = self.dis([fake_data], training = True)
            real_loss, fake_loss = self.discriminator_loss(real_logit[0], fake_logit[0])
            gp_loss = self.gradient_penality(partial(self.dis, training = True), real_data, fake_data)
            dLoss = (real_loss + fake_loss) + gp_loss*self.gradient_penality_width

        D_grad = t.gradient(dLoss, self.dis.trainable_variables)
        self.disOptimizer.apply_gradients(zip(D_grad, self.dis.trainable_variables))
        return real_loss + fake_loss, gp_loss
 
