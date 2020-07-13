import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from functools import partial
import datetime
from tensorflow.keras.models import load_model
from SaveModel import SaveModel
#from loadRawData import loadData
import os
import matplotlib.pyplot as plt
from tensorboard.plugins.hparams import api as hp
from math import log

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
class SpectralNormalization(layers.Wrapper):
    """
    Attributes:
       layer: tensorflow keras layers (with kernel attribute)
    """

    def __init__(self, layer, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        """Build `Layer`"""

        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, 'kernel'):
                raise ValueError(
                    '`SpectralNormalization` must wrap a layer that'
                    ' contains a `kernel` for weights')

            self.w = self.layer.kernel
            self.w_shape = self.w.shape.as_list()
            self.u = self.add_variable(
                shape=tuple([1, self.w_shape[-1]]),
                initializer=initializers.TruncatedNormal(stddev=0.02),
                name='sn_u',
                trainable=False,
                dtype=dtypes.float32)

        super(SpectralNormalization, self).build()

    @def_function.function
    def call(self, inputs, training=None):
        """Call `Layer`"""
        if training==None:
            training = K.learning_phase()
        
        if training==True:
            # Recompute weights for each forward pass
            self._compute_weights()
        
        output = self.layer(inputs)
        return output

    def _compute_weights(self):
        """Generate normalized weights.
        This method will update the value of self.layer.kernel with the
        normalized value, so that the layer is ready for call().
        """
        w_reshaped = array_ops.reshape(self.w, [-1, self.w_shape[-1]])
        eps = 1e-12
        _u = array_ops.identity(self.u)
        _v = math_ops.matmul(_u, array_ops.transpose(w_reshaped))
        _v = _v / math_ops.maximum(math_ops.reduce_sum(_v**2)**0.5, eps)
        _u = math_ops.matmul(_v, w_reshaped)
        _u = _u / math_ops.maximum(math_ops.reduce_sum(_u**2)**0.5, eps)

        self.u.assign(_u)
        sigma = math_ops.matmul(math_ops.matmul(_v, w_reshaped), array_ops.transpose(_u))

        self.layer.kernel.assign(self.w / sigma)

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())
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

        self.trainSize = 699
        self.filterNumber = 16
        self.L2_coefficient =100# 1/(length*width*height)

  
        
        
        self.dis = self.discriminator()
        self.disrOptimizer = keras.optimizers.RMSprop(lr = 0.002, 
                                                          clipvalue = 1.0, 
                                                          decay = 1e-4)
        # self.disrOptimizer = keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5, beta_2 = 0.9)
        self.gen = self.generator()
        self.genOptimizer = keras.optimizers.RMSprop(lr = 0.0005, 
                                                    clipvalue = 10.0, 
                                                    decay = 1e-4)
        # self.genOptimizer = keras.optimizers.Adam(lr = 0.00005, beta_1 = 0.5, beta_2 = 0.9)                                            
        self.decod = self.decoder()
        self.decoderOptimizer = keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5, beta_2 = 0.9)
        self.gradient_penality_width = 10.0


    def generator(self):
        parameter1_input = keras.Input(shape = (1), name = 'parameter1')
        parameter2_input = keras.Input(shape = (1), name = 'parameter2')
        parameter3_input = keras.Input(shape = (1), name = 'parameter3')

        x = layers.Dense(hparams[HP_NUM_UNITS], name = 'parameter1_layer_1')(parameter1_input)
        if hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(hparams[HP_NUM_UNITS], name = 'parameter1_layer_2')(x)
        if hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(hparams[HP_NUM_UNITS], name = 'parameter1_layer_3')(x)
        if hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        y = layers.Dense(hparams[HP_NUM_UNITS], name = 'parameter2_layer_1')(parameter2_input)
        if hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        y = layers.Dense(hparams[HP_NUM_UNITS], name = 'parameter2_layer_2')(y)
        if hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        y = layers.Dense(hparams[HP_NUM_UNITS], name = 'parameter2_layer_3')(y)
        if hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        
        z = layers.Dense(hparams[HP_NUM_UNITS], name = 'parameter3_layer_1')(parameter3_input)
        if hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        z = layers.LeakyReLU()(z)
        z = layers.Dense(hparams[HP_NUM_UNITS], name = 'parameter3_layer_2')(z)
        if hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        z = layers.LeakyReLU()(z)
        z = layers.Dense(hparams[HP_NUM_UNITS], name = 'parameter3_layer_3')(z)
        if hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        z = layers.LeakyReLU()(z)

        concatenate = layers.concatenate(inputs = [x, y, z])
        
        g =layers.Dense(4*4*2*self.filterNumber)(concatenate)
        g = layers.Reshape((4, 4, 2*self.filterNumber))(g)
        
        for i in range(int(log(self.width/4, 2))-1, -1, -1):
            g = SpectralNormalization(layers.Conv2DTranspose((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(g)
            
            g = layers.LeakyReLU()(g)
        

        g = SpectralNormalization(layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', use_bias=False))(g)
        
        #g = layers.Activation(tf.nn.tanh)(g)
        
        model = keras.Model(inputs = [parameter1_input, parameter2_input, parameter3_input], outputs = g)
        plot_model(model, to_file='WGAN_generator.png', show_shapes=True)
        return model
    
    def discriminator(self):
        parameter1_input = keras.Input(shape = (1), name = 'parameter1')
        parameter2_input = keras.Input(shape = (1), name = 'parameter2')
        parameter3_input = keras.Input(shape = (1), name = 'parameter3')
        dataInput = keras.Input(shape = (self.length,self.width, self.height), name = 'groundTruth/fake')

        x = layers.Dense(hparams[HP_NUM_UNITS], name = 'parameter1_layer_1')(parameter1_input)
        if hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.Dense(hparams[HP_NUM_UNITS], name = 'parameter1_layer_2')(x)
        if hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Dense(hparams[HP_NUM_UNITS], name = 'parameter1_layer_3')(x)
        if hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        y = layers.Dense(hparams[HP_NUM_UNITS], name = 'parameter2_layer_1')(parameter2_input)
        if hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        y = layers.Dense(hparams[HP_NUM_UNITS], name = 'parameter2_layer_2')(y)
        if hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        y = layers.Dense(hparams[HP_NUM_UNITS], name = 'parameter2_layer_3')(y)
        if hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        
        z = layers.Dense(hparams[HP_NUM_UNITS], name = 'parameter3_layer_1')(parameter3_input)
        if hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        z = layers.LeakyReLU()(z)

        z = layers.Dense(hparams[HP_NUM_UNITS], name = 'parameter3_layer_2')(z)
        if hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        z = layers.LeakyReLU()(z)

        z = layers.Dense(hparams[HP_NUM_UNITS], name = 'parameter3_layer_3')(z)
        if hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        z = layers.LeakyReLU()(z)

        concatenate = layers.concatenate(inputs = [x, y, z])
        xyz = layers.Dense((int(log(self.width/8, 2))+1)*self.filterNumber)(concatenate)    #Depends on how many level of conv2D you have
        xyz = layers.LeakyReLU()(xyz)
        

        d = SpectralNormalization(layers.Conv2D(self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(dataInput)
        
        d = layers.LeakyReLU()(d)
        for i in range(1, int(log(self.width/8, 2))+1):
            d = SpectralNormalization(layers.Conv2D((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False))(d)
            
            d = layers.LeakyReLU()(d)
        


        d = tf.nn.avg_pool(input = d, ksize= [1, 4, 4,  1] , strides=[1, 1, 1, 1], padding='VALID')*(self.height*self.width)
        d = layers.Flatten()(d)

        f1 = tf.multiply(d, xyz)
        f2 = layers.Dense(1)(d)
        r = layers.Add()([f1, f2])
        
        #r = layers.Activation(tf.nn.tanh)(r)


        model = keras.Model(inputs = [parameter1_input, parameter2_input, parameter3_input, dataInput], outputs = d)
        plot_model(model, to_file = "WGAN_Discriminator.png", show_shapes=True)
        return model
    
    def decoder(self):
        parameter1_input = keras.Input(shape = (1), name = 'decoderInput1')
        parameter2_input = keras.Input(shape = (1), name = 'decoderInput2')
        parameter3_input = keras.Input(shape = (1), name = 'decoderInput3')
        concatenate = layers.concatenate(inputs = [parameter1_input, parameter2_input, parameter3_input])
        m = layers.Dense(128)(concatenate)
        m = layers.Dense(128)(m)
        m = layers.Dense(256)(m)
        m = layers.Dense(256)(m)
        m = layers.Dense(512)(m)
        m = layers.Dense(256)(m)
        m = layers.Dense(128)(m)
        m = layers.Dense(2)(m)
        
        model = keras.Model(inputs = [parameter1_input, parameter2_input, parameter3_input], outputs = m)
        return model
    def decoder_loss(self, real,  predict):
        de_loss = tf.norm(tensor = (real-predict), ord='euclidean')
        
        return tf.clip_by_value(de_loss, clip_value_min=-1, clip_value_max=1)
    def generator_loss(self, fake_logit, real_data, fake_data_by_real_parameter):
        # l1Loss = 0
        # for i in range(self.batchSize):
        #     l1Loss += tf.norm((fake_data_by_real_parameter[i] - real_data[1][i]), ord=2)**2
        # l1Loss/=self.batchSize
        l2_norm = tf.norm(tensor = (fake_data_by_real_parameter-real_data[1][1]), ord='euclidean')
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
        x_img = _interpolate(real_data[1][0], fake_data)
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
            
            # mean = tf.reshape(real_data[1][2], [self.batchSize, 1, 1, 1])
            # std = tf.reshape(real_data[1][3], [self.batchSize, 1, 1, 1])
            
            fake_data_by_random_parameter = self.gen([random_vector1, random_vector2, random_vector3],training = True)  #generate by random parameter
            fake_data_by_real_parameter = self.gen([real_data[0][0], real_data[0][1], real_data[0][2]],training = True) #generate by real parameter

            fake_logit = self.dis([random_vector1, random_vector2, random_vector3, fake_data_by_random_parameter], training = False)
            fake_loss, l2_loss = self.generator_loss(fake_logit, real_data, fake_data_by_real_parameter)
            gLoss = fake_loss+self.L2_coefficient*l2_loss
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
            real_logit = self.dis([real_data[0][0], real_data[0][1], real_data[0][2], real_data[1][1]] , training = True)
            fake_logit = self.dis([random_vector1, random_vector2, random_vector3, fake_data], training = True)
            real_loss, fake_loss = self.discriminator_loss(real_logit, fake_logit)
            gp_loss = self.gradient_penality(partial(self.dis, training = True), real_data, fake_data)
            dLoss = (real_loss + fake_loss) + gp_loss*self.gradient_penality_width

        D_grad = t.gradient(dLoss, self.dis.trainable_variables)
        self.disrOptimizer.apply_gradients(zip(D_grad, self.dis.trainable_variables))
        return real_loss + fake_loss, gp_loss
    
    @tf.function
    def train_decoder(self, real_data):
        with tf.GradientTape() as t:
            predictM_S = self.decod([real_data[0][0], real_data[0][1], real_data[0][2]])
            de_loss = self.decoder_loss(real_data[1][2], predictM_S)
        de_grad = t.gradient(de_loss, self.decod.trainable_variables)
        self.decoderOptimizer.apply_gradients(zip(de_grad, self.decod.trainable_variables))
        return de_loss
    
    def train_wgan(self):
        #rawData = loadData(r'C:\Users\Andy\Desktop\Nyx\NyxDataSet', self.length, self.width, self.height)
        #train_data = tf.data.Dataset.from_tensor_slices(rawData)
        
        filename = os.listdir(self.dataSetDir)
        # parameter = []
        # for file in filename:
        #     parameter.append([file[5:12], file[13:20], file[21:28]])
        # parameter = tf.data.Dataset.from_tensor_slices(np.array(parameter, dtype=np.float32).reshape((len(filename), 3)))
                                                       
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
            #M_S = tf.stack([tf.math.log(mean), tf.math.log(std)])         
            return (ds-mean)/std, ds #M_S
            
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data = data.map(process_input_data, num_parallel_calls=AUTOTUNE)
        data = tf.data.Dataset.zip((parameter123, data))
        #data = data.shuffle(800)
        train_data = data.take(self.trainSize)
        test_data = data.skip(self.trainSize)
        
        train_data = train_data.batch(self.batchSize, drop_remainder = True)
        test_data = test_data.batch(100, drop_remainder = True)
        train_data = train_data.prefetch(buffer_size = AUTOTUNE)
        
        summary_writer = tf.summary.create_file_writer(self.logdir)
        # tf.summary.trace_on(graph=True, profiler=True)
        saveModel = SaveModel(self.gen, self.logdir, mode = 'min', save_weights_only=False)
        epoch = 1
        while saveModel.training:
            for step, real_data in enumerate(train_data):
                
                
                # real_data 中 real_data[0] 代表三input parameter 也就是 real_data[0][0] real_data[0][1] 和 real_data[0][2]
                # real_data 中 real_data[1] 分別是 real_data[1][0]: 壓縮後， real_data[1][1]:壓縮前， real_data[1][2]: 平均數， real_data[1][3]: 標準差 
                d_loss, gp = self.train_discriminator(real_data)
                g_loss = self.train_generator(real_data)
                # de_loss = self.train_decoder(real_data)
                
                # predi_mean = tf.reshape(self.decod([real_data[0][0], real_data[0][1], real_data[0][2]])[:, 0], [self.batchSize, 1, 1, 1])
                # predi_std = tf.reshape(self.decod([real_data[0][0], real_data[0][1], real_data[0][2]])[:, 1], [self.batchSize, 1, 1, 1])
                predi_data = self.gen([real_data[0][0], real_data[0][1], real_data[0][2]])
                
                
                # # print("data:", predi_data)
                # # print("test:", predi_data*tf.math.exp(predi_std) + tf.math.exp(predi_mean))
                
                # MSE = tf.reduce_mean(tf.keras.losses.MSE((predi_data*tf.math.exp(predi_std) + tf.math.exp(predi_mean)), real_data[1][1]))
                MSE = tf.reduce_mean(tf.keras.losses.MSE(real_data[1][1] ,  predi_data))
                l2 = tf.norm(tensor = real_data[1][1]-predi_data)
            with summary_writer.as_default():
                hp.hparams(hparams)
                tf.summary.scalar('Mean square error', MSE, epoch)
                tf.summary.scalar('discriminator_loss', d_loss, epoch)
                tf.summary.scalar('generator_loss', g_loss, epoch)
                tf.summary.scalar('gradient_penalty', gp, epoch)
                tf.summary.scalar('L2 loss', l2, epoch)
            print(f'Epoch: {epoch:6} G Loss: {g_loss:15.2f} D loss: {d_loss:15.2f} GP Loss {gp:15.10f} L2: {l2:20}')
            saveModel.on_epoch_end(l2)
            if epoch%1000 == 0:
                saveModel.save_model()
            epoch += 1
                            # if  self.genOptimizer.iterations.numpy() % 100 == 0:
                            #     x = self.gen(sample_random_vector, training = False)
                # print("Real", real_data[1][1])
                # print("Predict" ,predi_data)
                #if epoch % 100 == 0:
                
            #predictTrainResult[0].tofile(result_dir + f'NyxTrain-{self.width}-{self.length}-{real_data[0][0][0]}-{real_data[0][1][0]}-{real_data[0][2][0]}' )
            # with summary_writer.as_default():
            #      tf.summary.trace_export(name="my_func_trace", step=0, profiler_outdir=self.logdir)
            
            
      
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([512]))
HP_BN_UNITS = hp.HParam('BatchNormalization', hp.Discrete([False]))
#HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

now = datetime.datetime.now()
nowDate = f'{now.year}-{now.month}-{now.day}'+'_'+f'{now.hour}-{now.minute}'
dirs = 'wgan\\' + nowDate + '\\'

with tf.summary.create_file_writer(dirs).as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_BN_UNITS],
    metrics=[hp.Metric('discriminator_loss', display_name='discriminator_loss'), hp.Metric('generator_loss', display_name='generator_loss'), hp.Metric('Mean square error', display_name='Mean square error')]
  )        
num = 0


for num_units in HP_NUM_UNITS.domain.values:
    for bn_unit in HP_BN_UNITS.domain.values:
        session_num = f'run-{num}'
        hparams = {
            HP_NUM_UNITS: num_units,
            HP_BN_UNITS: bn_unit
        }
        GANs = GAN(length = 16, width = 16, height = 1, batchSize = 64, epochs = 5000, dataSetDir = r'E:\NTNU1-2\Nyx\NyxDataSet16_16', hparams = hparams, logdir = dirs+'\\'+session_num)
        GANs.train_wgan()
        num+=1





