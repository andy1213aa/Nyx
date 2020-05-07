import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from functools import partial
import datetime
from tensorflow.keras.models import load_model
#from loadRawData import loadData
import os

class GAN():
    def __init__(self, length, width, height, batchSize, epochs, dataSetDir):
        self.length = length
        self.width = width
        self.height = height
        self.batchSize = batchSize
        self.epochs = epochs
        self.dataSetDir = dataSetDir 

        self.now = datetime.datetime.now()
        self.nowDate = f'{self.now.year}-{self.now.month}-{self.now.day}_{self.now.hour}-{self.now.minute}'
        self.trainSize = 699
        self.filterNumber = 16
        self.L1_coefficient = 1/(length*width*height)

        self.testa = 0.14903
        self.testb = 0.02182
        self.testc = 0.83355
        self.testinputs1 = tf.constant([self.testa])
        self.testinputs2 = tf.constant([self.testb])
        self.testinputs3 = tf.constant([self.testc])
        
        
        self.dis = self.discriminator()
        self.disrOptimizer = keras.optimizers.RMSprop(lr = 0.0002, 
                                                          clipvalue = 1.0, 
                                                          decay = 1e-8)
        self.gen = self.generator()
        self.genOptimizer = keras.optimizers.RMSprop(lr = 0.00005, 
                                                    clipvalue = 1.0, 
                                                    decay = 1e-8)
        self.gradient_penality_width = 10.0


    def generator(self):
        parameter1_input = keras.Input(shape = (1), name = 'parameter1')
        parameter2_input = keras.Input(shape = (1), name = 'parameter2')
        parameter3_input = keras.Input(shape = (1), name = 'parameter3')

        x = layers.Dense(128, name = 'parameter1_layer_1')(parameter1_input)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(256, name = 'parameter1_layer_2')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512, name = 'parameter1_layer_3')(x)
        x = layers.LeakyReLU()(x)
        
        y = layers.Dense(128, name = 'parameter2_layer_1')(parameter2_input)
        y = layers.LeakyReLU()(y)
        y = layers.Dense(256, name = 'parameter2_layer_2')(y)
        y = layers.LeakyReLU()(y)
        y = layers.Dense(512, name = 'parameter2_layer_3')(y)
        y = layers.LeakyReLU()(y)
        
        z = layers.Dense(128, name = 'parameter3_layer_1')(parameter3_input)
        z = layers.LeakyReLU()(z)
        z = layers.Dense(256, name = 'parameter3_layer_2')(z)
        z = layers.LeakyReLU()(z)
        z = layers.Dense(512, name = 'parameter3_layer_3')(z)
        z = layers.LeakyReLU()(z)

        concatenate = layers.concatenate(inputs = [x, y, z])
        
        g = layers.Dense(4*4*16*self.filterNumber)(concatenate)
        g = layers.Reshape((4, 4, 16*self.filterNumber))(g)
        
        # g = layers.Conv2DTranspose(4*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False)(g)
        # g = layers.BatchNormalization()(g)
        # g = layers.LeakyReLU()(g)
        
        # g = layers.Conv2DTranspose(3*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False)(g)
        # g = layers.BatchNormalization()(g)
        # g = layers.LeakyReLU()(g)

        g = layers.Conv2DTranspose(2*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False)(g)
        g = layers.BatchNormalization()(g)  
        g = layers.LeakyReLU()(g)
        
        g = layers.Conv2DTranspose(1*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False)(g)
        g = layers.BatchNormalization()(g)
        g = layers.LeakyReLU()(g)

        g = layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', use_bias=False)(g)
        #g = layers.Activation(tf.nn.tanh)(g)
        
        model = keras.Model(inputs = [parameter1_input, parameter2_input, parameter3_input], outputs = g)
        plot_model(model, to_file='WGAN_generator.png', show_shapes=True)
        return model
    
    def discriminator(self):
        parameter1_input = keras.Input(shape = (1), name = 'parameter1')
        parameter2_input = keras.Input(shape = (1), name = 'parameter2')
        parameter3_input = keras.Input(shape = (1), name = 'parameter3')
        dataInput = keras.Input(shape = (self.length,self.width, self.height), name = 'groundTruth/fake')

        x = layers.Dense(128, name = 'parameter1_layer_1')(parameter1_input)
        x = layers.LeakyReLU()(x)
        
        x = layers.Dense(256, name = 'parameter1_layer_2')(x)
        x = layers.LeakyReLU()(x)

        x = layers.Dense(512, name = 'parameter1_layer_3')(x)
        x = layers.LeakyReLU()(x)
        
        y = layers.Dense(128, name = 'parameter2_layer_1')(parameter2_input)
        y = layers.LeakyReLU()(y)

        y = layers.Dense(256, name = 'parameter2_layer_2')(y)
        y = layers.LeakyReLU()(y)

        y = layers.Dense(512, name = 'parameter2_layer_3')(y)
        y = layers.LeakyReLU()(y)
        
        z = layers.Dense(128, name = 'parameter3_layer_1')(parameter3_input)
        z = layers.LeakyReLU()(z)

        z = layers.Dense(256, name = 'parameter3_layer_2')(z)
        z = layers.LeakyReLU()(z)

        z = layers.Dense(512, name = 'parameter3_layer_3')(z)
        z = layers.LeakyReLU()(z)

        concatenate = layers.concatenate(inputs = [x, y, z])
        xyz = layers.Dense(2*self.filterNumber)(concatenate)    #Depends on how many level of conv2D you have
        xyz = layers.LeakyReLU()(xyz)
        

        
        d = layers.Conv2D(self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False)(dataInput)
        d = layers.LeakyReLU()(d)

        d = layers.Conv2D(2*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False)(d)
        d = layers.LeakyReLU()(d)
        
        # d = layers.Conv2D(3*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False)(d)
        # d = layers.LeakyReLU()(d)
        
        # d = layers.Conv2D(4*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False)(d)
        # d = layers.LeakyReLU()(d)

        
        d = tf.nn.avg_pool(input = d, ksize= [1, 4, 4,  1] , strides=[1,1, 1, 1], padding='VALID')*(self.height*self.width)
        d = layers.Flatten()(d)

        f1 = tf.multiply(d, xyz)
        f2 = layers.Dense(1)(d)
        r = layers.Add()([f1, f2])



        model = keras.Model(inputs = [parameter1_input, parameter2_input, parameter3_input, dataInput], outputs = r)
        plot_model(model, to_file = "WGAN_Discriminator.png", show_shapes=True)
        return model


    def generator_loss(self, fake_logit, real_data, fake_data_by_real_parameter):
        l1_norm = tf.reduce_mean(tf.norm(tensor = ((fake_data_by_real_parameter-real_data[1])**2), ord='euclidean'))
        g_loss = - tf.reduce_mean(fake_logit)
        return g_loss, l1_norm
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
            
            fake_data_by_random_parameter = self.gen([random_vector1, random_vector2, random_vector3],training = True)  #generate by random parameter
            fake_data_by_real_parameter = self.gen([real_data[0][0], real_data[0][1], real_data[0][2]],training = True) #generate by real parameter


            fake_logit = self.dis([random_vector1, random_vector2, random_vector3, fake_data_by_random_parameter], training = False)
            fake_loss , l1_loss= self.generator_loss(fake_logit, real_data, fake_data_by_real_parameter)
            gLoss = fake_loss + self.L1_coefficient*l1_loss
        gradients = tape.gradient(gLoss, self.gen.trainable_variables)
        self.genOptimizer.apply_gradients(zip(gradients, self.gen.trainable_variables))
        return gLoss
    
    @tf.function
    def train_discriminator(self, real_data):
        with tf.GradientTape() as t:
            random_vector1 = tf.random.uniform(shape = (self.batchSize, 1), minval=0.12, maxval=0.16)
            random_vector2 = tf.random.uniform(shape = (self.batchSize, 1), minval=0.021, maxval=0.024)
            random_vector3 = tf.random.uniform(shape = (self.batchSize, 1), minval=0.55, maxval=0.9)


            fake_data = self.gen([random_vector1, random_vector2, random_vector3],training = True)
            real_logit = self.dis([real_data[0][0],real_data[0][1],real_data[0][2],real_data[1]] , training = True)
            fake_logit = self.dis([random_vector1, random_vector2, random_vector3,fake_data], training = True)
            real_loss, fake_loss = self.discriminator_loss(real_logit, fake_logit)
            gp_loss = self.gradient_penality(partial(self.dis, training = True), real_data, fake_data)
            dLoss = (real_loss + fake_loss) + gp_loss*self.gradient_penality_width
        D_grad = t.gradient(dLoss, self.dis.trainable_variables)
        self.disrOptimizer.apply_gradients(zip(D_grad, self.dis.trainable_variables))
        return real_loss + fake_loss, gp_loss

    def train_wgan(self):
        #rawData = loadData(r'C:\Users\Andy\Desktop\Nyx\NyxDataSet', self.length, self.width, self.height)
        #train_data = tf.data.Dataset.from_tensor_slices(rawData)
        
        filename = os.listdir(self.dataSetDir)
    
        parameter1 = tf.data.Dataset.from_tensor_slices(np.array([np.float(file[5:12]) for file in filename], dtype=np.float32).reshape((len(filename), 1)))
        parameter2 = tf.data.Dataset.from_tensor_slices(np.array([np.float(file[13:20]) for file in filename], dtype=np.float32).reshape((len(filename), 1)))
        parameter3 = tf.data.Dataset.from_tensor_slices(np.array([np.float(file[21:28]) for file in filename], dtype=np.float32).reshape((len(filename), 1)))
        parameter123 = tf.data.Dataset.zip((parameter1, parameter2, parameter3))

        filename_list = [os.path.join(self.dataSetDir, file) for file in filename]

        file_queue = tf.data.Dataset.from_tensor_slices(filename_list)
        data = tf.data.FixedLengthRecordDataset(file_queue, self.width*self.length*4)
        def process_input_data(ds):
            ds = tf.io.decode_raw(ds, tf.float32)
            ds = tf.reshape(ds, [self.width, self.length, 1])
           
            
            mean = tf.reduce_mean(ds)
            std = tf.math.reduce_std(ds)          
            return (ds-mean)/std
            
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data = data.map(process_input_data, num_parallel_calls=AUTOTUNE)
        data = tf.data.Dataset.zip((parameter123, data))
        train_data = data.take(self.trainSize)
        test_data = data.skip(self.trainSize)
        #train_data = train_data.shuffle(10)
        train_data = train_data.batch(self.batchSize, drop_remainder = True)
        train_data = train_data.prefetch(buffer_size = AUTOTUNE)

        wgan_dirs = 'wgan_result\\'

        date_dirs = f'predice_result_{self.nowDate}'
        result_dir = wgan_dirs + date_dirs + f'\\{self.testa}-{self.testb}-{self.testc}\\'
        os.makedirs(result_dir, exist_ok = True)
        
        summary_writer = tf.summary.create_file_writer(wgan_dirs + date_dirs)

        print('Start training...')
        for epoch in range(1, self.epochs+1):
            for step, real_data in enumerate(train_data):

                d_loss, gp = self.train_discriminator(real_data)
                g_loss = self.train_generator(real_data)
             
                with summary_writer.as_default():
                    tf.summary.scalar('discriminator_loss', d_loss, self.disrOptimizer.iterations)
                    tf.summary.scalar('generator_loss', g_loss, self.genOptimizer.iterations)
                    tf.summary.scalar('gradient_penalty', gp, self.disrOptimizer.iterations)
            print(f'Step: {epoch} G Loss:  {g_loss}\tD loss: {d_loss}\tGP Loss {gp}')
                        # if  self.genOptimizer.iterations.numpy() % 100 == 0:
                        #     x = self.gen(sample_random_vector, training = False)
                            
            if epoch % 100 == 0:
                predictResult = self.gen.predict([self.testinputs1, self.testinputs2, self.testinputs3])
                predictResult.tofile(result_dir + f'Nyx-{self.width}-{self.length}-{epoch}' )
        modelName = f'\\wgan-epoch-{self.epochs}-batch-{self.batchSize}_{self.nowDate}.h5'
        self.gen.save(wgan_dirs + date_dirs + modelName)
        

        
        

GAN = GAN(length = 64, width = 64, height = 1, batchSize = 64, epochs = 3000, dataSetDir = r'E:\NTNU1-2\Nyx\NyxDataSet64_64')
GAN.train_wgan()




