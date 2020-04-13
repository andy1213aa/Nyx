import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from functools import partial
#from loadRawData import loadData
import os

class GAN():
    def __init__(self, length, weight, height, batchsize):
        self.length = length
        self.weight = weight
        self.height = height
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

        x = layers.Dense(128, name = 'parameter1_layer_1')(parameter1_input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.Dense(256, name = 'parameter1_layer_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, name = 'parameter1_layer_3')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        y = layers.Dense(128, name = 'parameter2_layer_1')(parameter2_input)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        y = layers.Dense(256, name = 'parameter2_layer_2')(y)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        y = layers.Dense(512, name = 'parameter2_layer_3')(y)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        
        z = layers.Dense(128, name = 'parameter3_layer_1')(parameter3_input)
        z = layers.BatchNormalization()(z)
        z = layers.LeakyReLU()(z)
        z = layers.Dense(256, name = 'parameter3_layer_2')(z)
        z = layers.BatchNormalization()(z)
        z = layers.LeakyReLU()(z)
        z = layers.Dense(512, name = 'parameter3_layer_3')(z)
        z = layers.BatchNormalization()(z)
        z = layers.LeakyReLU()(z)

        concatenate = layers.concatenate(inputs = [x, y, z])
        g = layers.Dense(256)(concatenate)
        g = layers.Reshape((4, 4, 16))(g)
        
        g = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', use_bias=False)(g)
        g = layers.BatchNormalization()(g)
        g = layers.LeakyReLU()(g)
        
        g = layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', use_bias=False)(g)
        g = layers.BatchNormalization()(g)
        
        g = layers.Activation(tf.nn.tanh)(g)
        
        model = keras.Model(inputs = [parameter1_input, parameter2_input, parameter3_input], outputs = g)
        plot_model(model, to_file='WGAN_generator.png', show_shapes=True)
        return model
    
    def discriminator(self):

        dataInput = keras.Input(shape = (16,16, 1), name = 'groundTruth/fake')
        
        d = layers.Conv2D(16, kernel_size=3, strides=2, padding='same', use_bias=False)(dataInput)
        d = layers.BatchNormalization()(d)
        d = layers.LeakyReLU()(d)
        
        d = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(d)
        d = layers.BatchNormalization()(d)
        d = layers.LeakyReLU()(d)
        
        d = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', use_bias=False)(d)
        d = layers.BatchNormalization()(d)
        d = layers.LeakyReLU()(d)

        d = layers.Conv2D(1, kernel_size=2, strides=1, padding='valid')(d)
        
        
        model = keras.Model(inputs = dataInput, outputs = d)
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
            random_vector1 = tf.random.uniform(shape = (self.batchsize, 1), minval=0, maxval=1)
            random_vector2 = tf.random.uniform(shape = (self.batchsize, 1), minval=0, maxval=1)
            random_vector3 = tf.random.uniform(shape = (self.batchsize, 1), minval=0, maxval=1)
            fake_img = self.gen([random_vector1, random_vector2, random_vector3],training = True)
            fake_logit = self.dis(fake_img, training = False)
            gLoss = self.generator_loss(fake_logit)
        gradients = tape.gradient(gLoss, self.gen.trainable_variables)
        self.genOptimizer.apply_gradients(zip(gradients, self.gen.trainable_variables))
        return gLoss
    
    @tf.function
    def train_discriminator(self, real_img):
        with tf.GradientTape() as t:
            random_vector1 = tf.random.uniform(shape = (self.batchsize, 1), minval=0, maxval=1)
            random_vector2 = tf.random.uniform(shape = (self.batchsize, 1), minval=0, maxval=1)
            random_vector3 = tf.random.uniform(shape = (self.batchsize, 1), minval=0, maxval=1)
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
        #rawData = loadData(r'C:\Users\Andy\Desktop\Nyx\NyxDataSet', self.length, self.weight, self.height)
        #train_data = tf.data.Dataset.from_tensor_slices(rawData)

        filename = os.listdir(r'C:\Users\Andy\Desktop\Nyx\NyxDataSet16_16')
        filename_list = [os.path.join(r"C:\Users\Andy\Desktop\Nyx\NyxDataSet16_16", file) for file in filename]
        file_queue = tf.data.Dataset.from_tensor_slices(filename_list)
        train_data = tf.data.FixedLengthRecordDataset(file_queue, 1024)
        def process_input_data(ds):
            ds = tf.io.decode_raw(ds, tf.float32)
            ds = tf.reshape(ds, [16, 16, 1])
            #ds = tf.slice(ds, [120, 120, 128], [self.length, self.weight, self.height])
            
            mean = tf.reduce_mean(ds)
            std = tf.math.reduce_std(ds)          
            return (ds-mean)/std
            
            # _max = tf.reduce_max(ds)
            # _min = tf.reduce_min(ds)
            # return (ds-_min)/(_max-_min)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train_data = train_data.map(process_input_data, num_parallel_calls=AUTOTUNE)
        print(train_data)
        train_data = train_data.shuffle(10)
        train_data = train_data.batch(self.batchsize, drop_remainder = True)
        train_data = train_data.prefetch(buffer_size = AUTOTUNE)

        #train_data = train_data.prefetch(buffer_size = AUTOTUNE) 
        log_dirs = 'logs_wgan'
        model_dir = log_dirs + '\\models\\'
        os.makedirs(model_dir, exist_ok = True)
        summary_writer = tf.summary.create_file_writer(log_dirs)
        #sample_random_vector = tf.random.normal((100, 3, 1, 1))
        print('Start training...')
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
            print(f'Step: {epoch} G Loss:  {g_loss}\tD loss: {d_loss}\tGP Loss {gp}')
                        # if  self.genOptimizer.iterations.numpy() % 100 == 0:
                        #     x = self.gen(sample_random_vector, training = False)
                            
        if epoch != 0:
            self.gen.save(model_dir + f"wgan-epoch-{epoch}-0413.h5")



GAN = GAN(16, 16, 1, 64)
GAN.train_wgan()



