import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from loadRawData import loadData
from functools import partial
import os
class GAN():
    def __init__(self, length, weight, heigth, batchsize):
        self.length = length
        self.weight = weight
        self.heigth = heigth
        self.batchsize = batchsize
        self.n_dis = 5
        
        
        self.dis = self.discriminator()
        self.disrOptimizer = keras.optimizers.RMSprop(lr = 0.0001, 
                                                          clipvalue = 1.0, 
                                                          decay = 1e-8)
        self.gen = self.generator()
        self.genOptimizer = keras.optimizers.RMSprop(lr = 0.0001, 
                                                    clipvalue = 1.0, 
                                                    decay = 1e-8)
        self.gradient_penality_weight = 10.0


        # self.ganInput1 =  keras.Input(shape = (1), name = 'ganInput1')
        # self.ganInput2 =  keras.Input(shape = (1), name = 'ganInput2')
        # self.ganInput3 =  keras.Input(shape = (1), name = 'ganInput3')

        # self.ganOutput = self.dis(self.gen([self.ganInput1, self.ganInput2, self.ganInput3]))
        # self.gan = keras.Model(inputs = [self.ganInput1, self.ganInput2, self.ganInput3], outputs = self.ganOutput)

        # plot_model(self.gan, to_file = 'gan.png', show_shapes = True)
        #self.gan.compile(optimizer = self.ganOptimizer, loss = 'binary_crossentropy')
    def generator(self):
        parameter1_input = keras.Input(shape = (1), name = 'parameter1')
        parameter2_input = keras.Input(shape = (1), name = 'parameter2')
        parameter3_input = keras.Input(shape = (1), name = 'parameter3')

        h1_1 = layers.Dense(512, name = 'parameter1_layer_1', activation = 'relu')(parameter1_input)
        h1_2 = layers.Dense(512, name = 'parameter1_layer_2', activation = 'relu')(h1_1)
        h1_3 = layers.Dense(512, name = 'parameter1_layer_3', activation = 'relu')(h1_2)
        
        h2_1 = layers.Dense(512, name = 'parameter2_layer_1', activation = 'relu')(parameter2_input)
        h2_2 = layers.Dense(512, name = 'parameter2_layer_2', activation = 'relu')(h2_1)
        h2_3 = layers.Dense(512, name = 'parameter2_layer_3', activation = 'relu')(h2_2)
        
        h3_1 = layers.Dense(512, name = 'parameter3_layer_1', activation = 'relu')(parameter3_input)
        h3_2 = layers.Dense(512, name = 'parameter3_layer_2', activation = 'relu')(h3_1)
        h3_3 = layers.Dense(512, name = 'parameter3_layer_3', activation = 'relu')(h3_2)

        concatenate = layers.concatenate(inputs = [h1_3, h2_3, h3_3])

        f1_1 = layers.Dense(512, activation = 'relu')(concatenate)
        f1_2 = layers.Dense(256, activation = 'relu')(f1_1)
        f1_3 = layers.Reshape((16, 16))(f1_2)


        model = keras.Model(inputs = [parameter1_input, parameter2_input, parameter3_input], outputs = f1_3)
        plot_model(model, to_file='generator.png', show_shapes=True)
        return model
    
    def discriminator(self):

        dataInput = keras.Input(shape = (16,16), name = 'groundTruth/fake')
        flatten = layers.Flatten()(dataInput)
        d1_1 = layers.Dense(128, activation = 'relu')(flatten)
        d1_2 = layers.Dense( 64, activation = 'relu')(d1_1)
        d1_3 = layers.Dense( 8, activation = 'relu')(d1_2)
        d1_4 = layers.Dense( 1, activation = 'sigmoid')(d1_3)
        
        model = keras.Model(inputs = dataInput, outputs = d1_4)
        plot_model(model, to_file = "Discriminator.png", show_shapes=True)
        return model


    def generator_loss(self, fake_logit):
        g_loss = - tf.reduce_mean(fake_logit)
        return g_loss
    def discriminator_loss(self, real_logit, fake_logit):
        real_loss = -tf.reduce_mean(real_logit)
        fake_loss = tf.reducemean(fake_logit)
        return real_loss, fake_loss
    def gradient_penality(self, dis, real_img, fake_img):
        def _interpolate(a, b):
            shape = [tf.shape(a)[0]]+[1]*(a.shape.ndims - 1)
            alpha = tf.random.uniform(sahpe = shape, minval = 0, maxval = 1.)
            inter = (alpha * a) + ((1-alpha)*b)
            inter.set_shape(a.shape)
            return inter
        x_img = _interpolate(real_img, fake_img)
        with tf.GradientTape() as tape:
            tape.watch(x_img)
            pred_logit = dis(x_img)
        grad = tape.gradient(pred_logit, x_img)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis = 1)
        gp_loss = tf.reduce_mean((norm-1.)**2)
        return gp_loss    
    
    @tf.function
    def train_generator(self):
        with tf.GradientTape() as tape:
            random_vector = tf.random.normal(shape = (self.batchsize, 3))
            fake_img = self.gen([random_vector[:, 0], random_vector[:, 1], random_vector[:, 2]],training = True)
            fake_logit = self.dis(fake_img, training = True)
            gLoss = self.generator_loss(fake_logit)
        gradients = tape.gradient(gLoss, self.gen.trainable_variables)
        self.genOptimizer.apply_gradients(zip(gradients, self.gen.trainable_variables))
        return gLoss
    
    @tf.function
    def train_discriminator(self, real_img):
        with tf.GradientTape() as t:
            random_vector = tf.random.normal(shape = (self.batchsize, 3))
            fake_img = self.gen([random_vector[:, 0], random_vector[:, 1], random_vector[:, 2]],training = True)
            real_logit = self.dis(real_img, training = True)
            fake_logit = self.dis(fake_logit, training = True)
            real_loss, fake_loss = self.discriminator_loss(real_logit, fake_logit)
            gp_loss = self.gradient_penality(partial(self.dis, training = True), real_img, fake_img)
            dLoss = (real_loss + fake_loss) + gp_loss*self.gradient_penality_weight
        D_grad = t.gradient(dLoss, self.dis.trainable_variables)
        self.disrOptimizer.apply_gradients(zip(dLoss, self.dis.trainable_variables))
        return real_loss + fake_loss, gp_loss
    def train_wgan(self):
        rawData = loadData(r'E:\NTNU 1-2\Nyx\NyxDataSet', self.length, self.weight, self.heigth)
        log_dirs = 'logs_wgan'
        model_dir = log_dirs + '\\models\\'
        os.makedirs(model_dir, exist_ok = True)
        summary_writer = tf.summary.create_file_writer(log_dirs)
        sample_random_vector = tf.random.normal((100, 3, 1, 1))
        for epoch in range(25):
            for step, real_img in enumerate(rawData):
                d_loss, gp = self.train_discriminator(real_img)
                with summary_writer.as_default():
                    tf.summary.scalar('discriminator_loss', d_loss, self.disrOptimizer.iterations)
                    tf.summary.scalar('gradient_penalty', gp, self.disrOptimizer.iterations)
                if self.disrOptimizer.iterations.numpy() % self.n_dis == 0:
                    g_loss = train_generator()
                    with summary_writer.as_default():
                        tf.summary.scalar('generator_loss', g_loss, self.genOptimizer.iterations)
                        print('G Loss:  {:.2f}\tD loss: {:.2f}\tGP Loss {:.2f}'.format(g_loss, d_loss, gp))
                        # if  self.genOptimizer.iterations.numpy() % 100 == 0:
                        #     x_fake = self.gen(sample_random_vector, training = False)
                            
        if epoch != 0:
            self.gen.save_weight(model_dir + f"generator-epoch-{epoch}.h5")



GAN = GAN(16, 16, 16, 64)
GAN.train_wgan()
