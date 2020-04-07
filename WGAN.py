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

        h1_1 = layers.Dense(512, name = 'parameter1_layer_1')(parameter1_input)
        b1_1 = layers.BatchNormalization()(h1_1)
        l1_1 = layers.LeakyReLU()(b1_1)
        h1_2 = layers.Dense(512, name = 'parameter1_layer_2')(l1_1)
        b1_2 = layers.BatchNormalization()(h1_2)
        l1_2 = layers.LeakyReLU()(b1_2)
        h1_3 = layers.Dense(512, name = 'parameter1_layer_3')(l1_2)
        b1_3 = layers.BatchNormalization()(h1_3)
        l1_3 = layers.LeakyReLU()(b1_3)
        
        h2_1 = layers.Dense(512, name = 'parameter2_layer_1')(parameter2_input)
        b2_1 = layers.BatchNormalization()(h2_1)
        l2_1 = layers.LeakyReLU()(b2_1)
        h2_2 = layers.Dense(512, name = 'parameter2_layer_2')(l2_1)
        b2_2 = layers.BatchNormalization()(h2_2)
        l2_2 = layers.LeakyReLU()(b2_2)
        h2_3 = layers.Dense(512, name = 'parameter2_layer_3')(l2_2)
        b2_3 = layers.BatchNormalization()(h2_3)
        l2_3 = layers.LeakyReLU()(b2_3)
        
        h3_1 = layers.Dense(512, name = 'parameter3_layer_1')(parameter3_input)
        b3_1 = layers.BatchNormalization()(h3_1)
        l3_1 = layers.LeakyReLU()(b3_1)
        h3_2 = layers.Dense(512, name = 'parameter3_layer_2')(l3_1)
        b3_2 = layers.BatchNormalization()(h3_2)
        l3_2 = layers.LeakyReLU()(b3_2)
        h3_3 = layers.Dense(512, name = 'parameter3_layer_3')(l3_2)
        b3_3 = layers.BatchNormalization()(h3_3)
        l3_3 = layers.LeakyReLU()(b3_3)

        concatenate = layers.concatenate(inputs = [l1_3, l2_3, l3_3])

        h4_1 = layers.Dense(512)(concatenate)
        b4_1 = layers.BatchNormalization()(h4_1)
        l4_1 = layers.LeakyReLU()(b4_1)
        h4_2 = layers.Dense(1024)(l4_1)
        b4_2 = layers.BatchNormalization()(h4_2)
        l4_2 = layers.LeakyReLU()(b4_2)
        h4_3 = layers.Dense(2048)(l4_2)
        b4_3 = layers.BatchNormalization()(h4_3)
        l4_3 = layers.LeakyReLU()(b4_3)
        h4_4 = layers.Dense(4096)(l4_3)
        b4_4 = layers.BatchNormalization()(h4_4)
        t4_4 = layers.Activation(tf.nn.tanh)(b4_4)
        
        h4_5 = layers.Reshape((16, 16, 16))(t4_4)


        model = keras.Model(inputs = [parameter1_input, parameter2_input, parameter3_input], outputs = h4_5)
        plot_model(model, to_file='WGAN_generator.png', show_shapes=True)
        return model
    
    def discriminator(self):

        dataInput = keras.Input(shape = (16,16,16), name = 'groundTruth/fake')
        flatten = layers.Flatten()(dataInput)
        d1_0 = layers.Dense(1024)(flatten)
        b1_0 = layers.BatchNormalization()(d1_0)
        l1_0 = layers.LeakyReLU()(b1_0)
        d1_1 = layers.Dense(128)(l1_0)
        b1_1 = layers.BatchNormalization()(d1_1)
        l1_1 = layers.LeakyReLU()(b1_1)
        d1_2 = layers.Dense( 64)(l1_1)
        b1_2 = layers.BatchNormalization()(d1_2)
        l1_2 = layers.LeakyReLU()(b1_2)
        d1_3 = layers.Dense( 8)(l1_2)
        b1_3 = layers.BatchNormalization()(d1_3)
        l1_3 = layers.LeakyReLU()(b1_3)
        d1_4 = layers.Dense( 1)(l1_3)
        b1_4 = layers.BatchNormalization()(d1_4)
        l1_4 = layers.LeakyReLU()(b1_4)
        
        model = keras.Model(inputs = dataInput, outputs = l1_4)
        plot_model(model, to_file = "WGAN_Discriminator.png", show_shapes=True)
        return model


    def generator_loss(self, fake_logit):
        g_loss = - tf.reduce_mean(fake_logit)
        return g_loss
    def discriminator_loss(self, real_logit, fake_logit):
        real_loss = -tf.reduce_mean(real_logit)
        fake_loss = tf.reduce_mean(fake_logit)
        return real_loss, fake_loss
    def gradient_penality(self, dis, real_img, fake_img):
        def _interpolate(a, b):
            
            shape = [tf.shape(a)[0]]+[1]*(a.shape.ndims - 1)
            alpha = tf.random.uniform(shape = shape, minval = 0, maxval = 1.)
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
            random_vector1 = tf.random.normal(shape = (self.batchsize, 1))
            random_vector2 = tf.random.normal(shape = (self.batchsize, 1))
            random_vector3 = tf.random.normal(shape = (self.batchsize, 1))
            fake_img = self.gen([random_vector1, random_vector2, random_vector3],training = True)
            fake_logit = self.dis(fake_img, training = True)
            gLoss = self.generator_loss(fake_logit)
        gradients = tape.gradient(gLoss, self.gen.trainable_variables)
        self.genOptimizer.apply_gradients(zip(gradients, self.gen.trainable_variables))
        return gLoss
    
    @tf.function
    def train_discriminator(self, real_img):
        with tf.GradientTape() as t:
            random_vector1 = tf.random.normal(shape = (self.batchsize, 1))
            random_vector2 = tf.random.normal(shape = (self.batchsize, 1))
            random_vector3 = tf.random.normal(shape = (self.batchsize, 1))
            fake_img = self.gen([random_vector1, random_vector2, random_vector3],training = True)
            real_logit = self.dis(real_img, training = True)
            fake_logit = self.dis(fake_img, training = True)
            real_loss, fake_loss = self.discriminator_loss(real_logit, fake_logit)
            gp_loss = self.gradient_penality(partial(self.dis, training = True), real_img, fake_img)
            dLoss = (real_loss + fake_loss) + gp_loss*self.gradient_penality_weight
        D_grad = t.gradient(dLoss, self.dis.trainable_variables)
        self.disrOptimizer.apply_gradients(zip(D_grad, self.dis.trainable_variables))
        return real_loss + fake_loss, gp_loss

    def train_wgan(self):
        rawData = loadData(r'E:\NTNU 1-2\Nyx\NyxDataSet', self.length, self.weight, self.heigth)
        train_data = tf.data.Dataset.from_tensor_slices(rawData)
        train_data = train_data.shuffle(10)
        train_data = train_data.map(lambda x: tf.cast(x, tf.float32))
        train_data = train_data.batch(self.batchsize, drop_remainder = True)
        #train_data = train_data.prefetch(buffer_size = AUTOTUNE) 
        log_dirs = 'logs_wgan'
        model_dir = log_dirs + '\\models\\'
        os.makedirs(model_dir, exist_ok = True)
        summary_writer = tf.summary.create_file_writer(log_dirs)
        sample_random_vector = tf.random.normal((100, 3, 1, 1))
        for epoch in range(5000):
            for step, real_img in enumerate(train_data):
                d_loss, gp = self.train_discriminator(real_img)
                with summary_writer.as_default():
                    tf.summary.scalar('discriminator_loss', d_loss, self.disrOptimizer.iterations)
                    tf.summary.scalar('gradient_penalty', gp, self.disrOptimizer.iterations)
                if self.disrOptimizer.iterations.numpy() % self.n_dis == 0:
                    g_loss = self.train_generator()
                    with summary_writer.as_default():
                        tf.summary.scalar('generator_loss', g_loss, self.genOptimizer.iterations)
                        print('G Loss:  {:.2f}\tD loss: {:.2f}\tGP Loss {:.2f}'.format(g_loss, d_loss, gp))
                        # if  self.genOptimizer.iterations.numpy() % 100 == 0:
                        #     x_fake = self.gen(sample_random_vector, training = False)
                            
        if epoch != 0:
            self.gen.save(model_dir + f"wgan-epoch-{epoch}.h5")



GAN = GAN(16, 16, 16, 64)
GAN.train_wgan()
