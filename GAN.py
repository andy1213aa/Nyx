import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from functools import partial
from math import log
from SaveModel import SaveModel
import tensorflow as tf
from SpectralNormalization import SpectralNormalization
import os

class GAN():
    def __init__(self, length, width, height, batchSize, epochs, dataSetDir,hparams, logdir):
        self.length = length
        self.width = width
        self.height = height
        self.batchSize = batchSize
        self.epochs = epochs
        self.dataSetDir = dataSetDir 
        self.hparams = hparams
        self.recordepoch = 100
        self.logdir = logdir
        self.datamax = 0
        self.datamin = 0

        self.trainSize = 699
        self.filterNumber = 16
        self.L2_coefficient =1/(length*width*height)
        self.dis = self.discriminator()
        #self.disOptimizer = keras.optimizers.RMSprop(lr = 0.0002, clipvalue = 1.0, decay = 1e-8)
        self.disOptimizer = keras.optimizers.Adam(lr = 0.0002, clipvalue = 1.0, decay = 1e-8)
        self.gen = self.generator()
        #self.genOptimizer = keras.optimizers.RMSprop(lr = 0.00005, clipvalue = 1.0, decay = 1e-8)
        self.genOptimizer = keras.optimizers.Adam(lr = 0.00005, clipvalue = 1.0, decay = 1e-8)                                            
        self.gradient_penality_width = 10.0


    def generator(self):
        parameter1_input = keras.Input(shape = (1), name = 'parameter1')
        parameter2_input = keras.Input(shape = (1), name = 'parameter2')
        parameter3_input = keras.Input(shape = (1), name = 'parameter3')

        x = layers.Dense(512, name = 'parameter1_layer_1')(parameter1_input)
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, name = 'parameter1_layer_2')(x)
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        
        y = layers.Dense(512, name = 'parameter2_layer_1')(parameter2_input)
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        y = layers.Dense(512, name = 'parameter2_layer_2')(y)
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        
        z = layers.Dense(512, name = 'parameter3_layer_1')(parameter3_input)
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        z = layers.LeakyReLU()(z)
        z = layers.Dense(512, name = 'parameter3_layer_2')(z)
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        z = layers.LeakyReLU()(z)

        concatenate = layers.concatenate(inputs = [x, y, z])
        
        g =layers.Dense(4*4*4*int(self.width/4)*self.filterNumber)(concatenate)
        g = layers.Reshape((4, 4, 4, int(self.width/4)*self.filterNumber))(g)
        
        for i in range(int(log(self.width/4, 2))-1, -1, -1):
           # g = SpectralNormalization(layers.Conv2DTranspose((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(g)
            g = layers.Conv3DTranspose((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False)(g)
            g = layers.LeakyReLU()(g)
        

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
        x = layers.LeakyReLU()(x)
        
        x = layers.Dense(512, name = 'parameter1_layer_2')(x)
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        y = layers.Dense(512, name = 'parameter2_layer_1')(parameter2_input)
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        y = layers.Dense(512, name = 'parameter2_layer_2')(y)
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        
        z = layers.Dense(512, name = 'parameter3_layer_1')(parameter3_input)
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        z = layers.LeakyReLU()(z)

        z = layers.Dense(512, name = 'parameter3_layer_2')(z)
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        z = layers.LeakyReLU()(z)

        concatenate = layers.concatenate(inputs = [x, y, z])
        xyz = layers.Dense((int(log(self.width/8, 2))+1)*self.filterNumber)(concatenate)    #Depends on how many level of conv2D you have
        xyz = layers.LeakyReLU()(xyz)
        

        #d = SpectralNormalization(layers.Conv2D(self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(dataInput)
        d = SpectralNormalization(layers.Conv3D(self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(dataInput)
        
        d = layers.LeakyReLU()(d)
        for i in range(1, int(log(self.width/2, 2))-1):
            d = SpectralNormalization(layers.Conv3D((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(d)
            #d = SpectralNormalization(layers.Conv2D((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(d)#
            
            d = layers.LeakyReLU()(d)
        


        d = tf.nn.avg_pool(input = d, ksize= [1, 4, 4, 4,  1] , strides=[1, 1, 1, 1,  1], padding='VALID')*(self.height*self.width*self.length)
        
        
        d = layers.Flatten()(d)
        f1 = tf.multiply(d, xyz)
        f2 = layers.Dense(1)(d)
        r = layers.Add()([f1, f2])
        
        #r = layers.Activation(tf.nn.tanh)(r)


        model = keras.Model(inputs = [parameter1_input, parameter2_input, parameter3_input, dataInput], outputs = r)
        #model = keras.Model(inputs = [dataInput], outputs = d)
        plot_model(model, to_file = "WGAN_Discriminator.png", show_shapes=True)
        return model
    
    
    def generator_loss(self, fake_logit, real_data, fake_data_by_real_parameter):
        # l1Loss = 0
        # for i in range(self.batchSize):
        #     l1Loss += tf.norm((fake_data_by_real_parameter[i] - real_data[1][i]), ord=2)**2
        # l1Loss/=self.batchSize
        l2_norm = tf.norm(tensor = (fake_data_by_real_parameter-real_data[1]), ord='euclidean')
        #rmse = tf.sqrt(tf.reduce_mean((real_data[1][1] - fake_data_by_real_parameter)**2)) / (tf.reduce_max(real_data[1][1]) - tf.reduce_min(real_data[1][1]))
        #l2_norm = 0
        #l1_norm = tf.reduce_mean(tf.norm(tensor = ((fake_data_by_real_parameter-real_data[1])**2), ord='euclidean'))
        g_loss = - tf.reduce_mean(fake_logit)
        return g_loss, l2_norm
    def discriminator_loss(self, real_logit, fake_logit):
        real_loss = -tf.reduce_mean(real_logit)
        fake_loss = tf.reduce_mean(fake_logit)
        return real_loss, fake_loss

    def gradient_penality(self, dis, real_data, fake_data, ):
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
        grad = tape.gradient(pred_logit, x_img)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis = 1)
        gp_loss = tf.reduce_mean((norm-1.)**2)
        return gp_loss    
    
    @tf.function
    def train_generator(self, real_data):
        with tf.GradientTape() as tape:
            random_vector1 = tf.random.uniform(shape = (self.batchSize, 1), minval=0.12, maxval=0.16)
            random_vector2 = tf.random.uniform(shape = (self.batchSize, 1), minval=0.021, maxval=0.024)
            random_vector3 = tf.random.uniform(shape = (self.batchSize, 1), minval=0.55, maxval=0.9)
            
            # mean = tf.reshape(real_data[1][2], [self.batchSize, 1, 1, 1])
            # std = tf.reshape(real_data[1][3], [self.batchSize, 1, 1, 1])
            
            fake_data_by_random_parameter = self.gen([random_vector1, random_vector2, random_vector3],training = True)  #generate by random parameter
            fake_data_by_real_parameter = self.gen([real_data[0][0], real_data[0][1], real_data[0][2]],training = True) #generate by real parameter

            fake_logit = self.dis([random_vector1, random_vector2, random_vector3, fake_data_by_random_parameter], training = False)
            #fake_logit = self.dis([fake_data_by_random_parameter], training = False)
            fake_loss, l2_norm = self.generator_loss(fake_logit, real_data, fake_data_by_real_parameter)
            gLoss = fake_loss+self.L2_coefficient*l2_norm
        gradients = tape.gradient(gLoss, self.gen.trainable_variables)
        self.genOptimizer.apply_gradients(zip(gradients, self.gen.trainable_variables))
        return gLoss
    
    @tf.function
    def train_discriminator(self, real_data):
        with tf.GradientTape() as t:
            random_vector1 = tf.random.uniform(shape = (self.batchSize, 1), minval=0.12, maxval=0.16)
            random_vector2 = tf.random.uniform(shape = (self.batchSize, 1), minval=0.021, maxval=0.024)
            random_vector3 = tf.random.uniform(shape = (self.batchSize, 1), minval=0.55, maxval=0.9)

            # mean = tf.reshape(real_data[1][2], [self.batchSize, 1, 1, 1])
            # std = tf.reshape(real_data[1][3], [self.batchSize, 1, 1, 1])

            fake_data = self.gen([random_vector1, random_vector2, random_vector3],training = True)
            real_logit = self.dis([real_data[0][0], real_data[0][1], real_data[0][2], real_data[1]] , training = True)
            #real_logit = self.dis([real_data[1]] , training = True)
            fake_logit = self.dis([random_vector1, random_vector2, random_vector3, fake_data], training = True)
            #fake_logit = self.dis([fake_data], training = True)
            real_loss, fake_loss = self.discriminator_loss(real_logit, fake_logit)
            gp_loss = self.gradient_penality(partial(self.dis, training = True), real_data, fake_data)
            dLoss = (real_loss + fake_loss) + gp_loss*self.gradient_penality_width

        D_grad = t.gradient(dLoss, self.dis.trainable_variables)
        self.disOptimizer.apply_gradients(zip(D_grad, self.dis.trainable_variables))
        return real_loss + fake_loss, gp_loss
 
    def train_wgan(self):

        filename = os.listdir(self.dataSetDir)
                                            
        parameter1 = tf.data.Dataset.from_tensor_slices(np.array([np.float(file[5:12]) for file in filename], dtype=np.float32).reshape((len(filename), 1)))
        parameter2 = tf.data.Dataset.from_tensor_slices(np.array([np.float(file[13:20]) for file in filename], dtype=np.float32).reshape((len(filename), 1)))
        parameter3 = tf.data.Dataset.from_tensor_slices(np.array([np.float(file[21:28]) for file in filename], dtype=np.float32).reshape((len(filename), 1)))
        parameter123 = tf.data.Dataset.zip((parameter1, parameter2, parameter3))

        filename_list = [os.path.join(self.dataSetDir, file) for file in filename]

        file_queue = tf.data.Dataset.from_tensor_slices(filename_list)
        data = tf.data.FixedLengthRecordDataset(file_queue, self.width*self.length*self.height*4)
        def process_input_data(ds):
            ds = tf.io.decode_raw(ds, tf.float32)
            ds = tf.reshape(ds, [self.width, self.length, self.height, 1])
            
            # mean = tf.reduce_mean(ds)
            # std = tf.math.reduce_std(ds) 
            #M_S = tf.stack([tf.math.log(mean), tf.math.log(std)])         
            return  ds #(ds-mean)/std #M_S
            
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data = data.map(process_input_data, num_parallel_calls=AUTOTUNE)
        data = tf.data.Dataset.zip((parameter123, data))
        #data = data.shuffle(800)
        train_data = data.take(self.trainSize)
        training = train_data.take(self.trainSize)

        test_data = data.skip(self.trainSize)
        
        training_batch = training.batch(self.batchSize, drop_remainder = True)
       

        test_data_batch = test_data.batch(100, drop_remainder = True)
        training_batch = training_batch.prefetch(buffer_size = AUTOTUNE)
       

        summary_writer = tf.summary.create_file_writer(self.logdir)
        # tf.summary.trace_on(graph=True, profiler=True)
        saveModel = SaveModel(self.gen, self.logdir, mode = 'min', save_weights_only=False)   #建立一個訓練規則

        epoch = 1
        self.data_max = tf.reduce_max(list(training_batch.as_numpy_iterator())[0][1])
        self.data_min = tf.reduce_min(list(training_batch.as_numpy_iterator())[0][1])

        while saveModel.training:
            for step, real_data in enumerate(training_batch):
                # real_data 中 real_data[0] 代表三input parameter 也就是 real_data[0][0] real_data[0][1] 和 real_data[0][2], real_data[1] 代表 groundtruth
                d_loss, gp = self.train_discriminator(real_data)
                g_loss= self.train_generator(real_data)
                predi_data = self.gen([real_data[0][0], real_data[0][1], real_data[0][2]])      
            
                RMSE =  (tf.sqrt(tf.reduce_mean((real_data[1] - predi_data)**2)) / (self.data_max - self.data_min))
            
                l2 = tf.norm(tensor = real_data[1]-predi_data) / (self.data_max - self.data_min)
                
            with summary_writer.as_default():
                    #hp.hparams(hparams)
                tf.summary.scalar('RMSE', RMSE, epoch)
                tf.summary.scalar('discriminator_loss', d_loss, epoch)
                tf.summary.scalar('generator_loss', g_loss, epoch)
                tf.summary.scalar('gradient_penalty', gp, epoch)
            print(f'Epoch: {epoch:6} G Loss: {g_loss:15.2f} D loss: {d_loss:15.2f} GP Loss {gp:15.2f} L2: {l2:10f} RMSE: {RMSE* 100 :3.5f}%  ')
            saveModel.on_epoch_end(RMSE)
            if epoch%1000 == 0:
                saveModel.save_model()
            epoch += 1