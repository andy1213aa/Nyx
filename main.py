
import tensorflow as tf
from Nyx_Reconstruction.utlis.loadData import generateData
from Nyx_Reconstruction.utlis import config
from Nyx_Reconstruction.utlis.SaveModel import SaveModel
from Nyx_Reconstruction.model.generator import generator
from Nyx_Reconstruction.model.discriminator import discriminator
from Nyx_Reconstruction.utlis.loss_function import generator_loss, gradient_penality, discriminator_loss
#from functools import partial
import numpy as np

#import os
# def main():
#     HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([512]))
#     HP_BN_UNITS = hp.HParam('BatchNormalization', hp.Discrete([False]))
#     #HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

#     now = datetime.datetime.now()
#     nowDate = f'{now.year}-{now.month}-{now.day}'+'_'+f'{now.hour}-{now.minute}'
#     dirs = 'E:\\NTNU1-2\\Nyx\\NyxDataSet\\wgan\\' + nowDate + '\\'

#     with tf.summary.create_file_writer(dirs).as_default():
#         hp.hparams_config(
#             hparams=[HP_NUM_UNITS, HP_BN_UNITS],
#             metrics=[hp.Metric('discriminator_loss', display_name='discriminator_loss'), hp.Metric('generator_loss', display_name='generator_loss'), hp.Metric('Mean square error', display_name='Mean square error')]
#         )        
#     num = 0
#     #@

#     for num_units in HP_NUM_UNITS.domain.values:
#         for bn_unit in HP_BN_UNITS.domain.values:
#             session_num = f'run-{num}'
#             hparams = {
#                 HP_NUM_UNITS: num_units,
#                 HP_BN_UNITS: bn_unit
#             }
#             GANs = GAN(length = 16, width = 16, height = 16, batchSize = 64, epochs = 5000, dataSetDir = r'E:\NTNU1-2\Nyx\NyxDataSet\NyxDataSet16_16_16', hparams = hparams, logdir = dirs+'\\'+session_num)
#             GANs.train_wgan()
#             num+=1


        
def main():
    
    #os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    # physical_devices = tf.config.list_physical_devices('GPU')
    # try:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # except:
    #     print('Invalid device or cannot modify virtual devices once initialized.')
    #     return 0
    @tf.function
    def train_generator(real_data):
        print('GEN GRAPH SIDE EFFECT')
        with tf.GradientTape() as tape:
            random_vector1 = tf.random.uniform(shape = (dataSetConfig['batchSize'], 1), minval=0.12, maxval=0.16)
            random_vector2 = tf.random.uniform(shape = (dataSetConfig['batchSize'], 1), minval=0.021, maxval=0.024)
            random_vector3 = tf.random.uniform(shape = (dataSetConfig['batchSize'], 1), minval=0.55, maxval=0.9)

            fake_data_by_random_parameter = gen([random_vector1, random_vector2, random_vector3],training = True)  #generate by random parameter
            fake_data_by_real_parameter = gen([real_data[0], real_data[1], real_data[2]],training = True) #generate by real parameter

            #fake_logit = dis([random_vector1, random_vector2, random_vector3, fake_data_by_random_parameter], training = False)
            fake_logit = dis([random_vector1, random_vector2, random_vector3, fake_data_by_random_parameter], training = False)
            fake_loss, l2_norm = generator_loss(fake_logit, real_data, fake_data_by_real_parameter)
            gLoss = fake_loss+L2_coefficient*l2_norm
        gradients = tape.gradient(gLoss, gen.trainable_variables)
        genOptimizer.apply_gradients(zip(gradients, gen.trainable_variables))
        return fake_loss, l2_norm

    @tf.function
    def train_discriminator(real_data):
        print('DIS GRAPH SIDE EFFECT')
        with tf.GradientTape() as t:
            random_vector1 = tf.random.uniform(shape = (dataSetConfig['batchSize'], 1), minval=0.12, maxval=0.16)
            random_vector2 = tf.random.uniform(shape = (dataSetConfig['batchSize'], 1), minval=0.021, maxval=0.024)
            random_vector3 = tf.random.uniform(shape = (dataSetConfig['batchSize'], 1), minval=0.55, maxval=0.9)

            fake_data = gen([random_vector1, random_vector2, random_vector3],training = True)
            real_logit = dis([real_data[0], real_data[1], real_data[2], real_data[3]], training = True)
            # real_logit = dis(real_data[1] , training = True)
            #fake_logit = dis([random_vector1, random_vector2, random_vector3, fake_data], training = True)
            fake_logit = dis([random_vector1, random_vector2, random_vector3, fake_data], training = True)
            real_loss, fake_loss = discriminator_loss(real_logit, fake_logit)
            #gp_loss = gradient_penality(partial(dis, training = True), real_data, fake_data)
            dLoss = (real_loss + fake_loss)# + gp_loss*gradient_penality_width

        D_grad = t.gradient(dLoss, dis.trainable_variables)
        disOptimizer.apply_gradients(zip(D_grad, dis.trainable_variables))
        return real_loss , fake_loss#, gp_loss
    
    
    
    dataSetConfig = config.dataSet['Nyx'] #What data you want to load
    #model = GAN(length = dataSetConfig['length'], width = dataSetConfig['width'], height = dataSetConfig['height'])
    gen = generator()

    dis = discriminator()


    L2_coefficient = 10# 1/(dataSetConfig['length'] * dataSetConfig['width'] * dataSetConfig['height'])
    
    #disOptimizer = tf.keras.optimizers.RMSprop(lr = 0.0001, clipvalue = 1.0, decay = 1e-8)
    disOptimizer = tf.keras.optimizers.Adam(lr = 0.0002,beta_1=0.9, beta_2 = 0.999)#,  clipvalue = 1.0, decay = 1e-8)
    
    #genOptimizer = tf.keras.optimizers.RMSprop(lr = 0.00005, clipvalue = 1.0, decay = 1e-8)
    genOptimizer = tf.keras.optimizers.Adam(lr = 0.00005,beta_1=0.9, beta_2 = 0.999)#,  clipvalue = 1.0, decay = 1e-8)                                            
    gradient_penality_width = 10.0

    training_batch, validating_batch = generateData(dataSetConfig)
    
    summary_writer = tf.summary.create_file_writer(dataSetConfig['logDir'])
    # tf.summary.trace_on(graph=True, profiler=True)
    saveModel = SaveModel(gen,dis, dataSetConfig, mode = 'min', save_weights_only=True)   #建立一個訓練規則
    datarange = np.zeros((dataSetConfig['validationSize']))
    for i in range(dataSetConfig['validationSize']):
        datarange[i] = np.max(list(validating_batch.as_numpy_iterator())[0][3][i]) - np.min(list(validating_batch.as_numpy_iterator())[0][3][i])
    # data_max = tf.reduce_max(list(validating_batch.as_numpy_iterator())[0][1])
    # data_min = tf.reduce_min(list(validating_batch.as_numpy_iterator())[0][1])
    # dataRange = data_max - data_min
    while saveModel.training:
        Average_percentage = 0
        for step, real_data in enumerate(training_batch):
            # real_data 中 real_data[0] 代表三input parameter 也就是 real_data[0][0] real_data[0][1] 和 real_data[0][2], real_data[1] 代表 groundtruth
            
            dRealLoss, dFakeLoss = train_discriminator(real_data)
            gFakeLoss, gL2Loss= train_generator(real_data)
            #l2 = tf.norm(tensor = list(validating_batch.as_numpy_iterator())[0][1]-predi_data)/ (data_max - data_min)      
               

           # print(f'Epoch: {saveModel.epoch:6} Step: {step:3} dLoss: {d_loss} gLoss: {g_loss} ')    
        predi_data = gen([list(validating_batch.as_numpy_iterator())[0][0], list(validating_batch.as_numpy_iterator())[0][1], list(validating_batch.as_numpy_iterator())[0][2]])
        #l2 = (tf.norm(tensor = list(validating_batch.as_numpy_iterator())[0][1]-predi_data)/dataSetConfig['validationSize'])/  (data_max - data_min) *100
        l2 = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(((list(validating_batch.as_numpy_iterator())[0][3]-predi_data)**2), axis = (1, 2, 3)))/ datarange) * 100
        RMSE_percentage = np.mean(np.sqrt(np.mean(((list(validating_batch.as_numpy_iterator())[0][3]-predi_data))**2, axis = (1, 2, 3))) / datarange) * 100
        #RMSE_percentage =  (tf.sqrt(tf.reduce_mean((list(validating_batch.as_numpy_iterator())[0][1] - predi_data)**2)) / (data_max - data_min)) *100   
        #RSquard = 1- tf.reduce_mean((list(validating_batch.as_numpy_iterator())[0][1]-predi_data)**2)/ tf.math.reduce_variance(list(validating_batch.as_numpy_iterator())[0][1])
        with summary_writer.as_default():
            tf.summary.scalar('RMSE-percentage', RMSE_percentage, saveModel.epoch)
                 #hp.hparams(hparams)
                #tf.summary.scalar('RMSE-percentage', l2, step)
            tf.summary.scalar('discriminator_loss', dRealLoss+dFakeLoss, saveModel.epoch)
            tf.summary.scalar('generator_loss', gFakeLoss, saveModel.epoch)
            tf.summary.histogram(f'predict data', predi_data[0], saveModel.epoch)
            tf.summary.histogram(f'raw data', list(validating_batch.as_numpy_iterator())[0][3][0], saveModel.epoch)
        #    dataRange = tf.reduce_max(real_data[1]) - tf.reduce_min(real_data[1])
        #RMSE_percentage =  (tf.sqrt(tf.reduce_mean((real_data[1] - predi_data)**2)) / dataRange)*100
           # Average_percentage += RMSE_percentage
        #Average_percentage /= (dataSetConfig['trainSize']//dataSetConfig['batchSize'])
        
        #l2 = tf.norm(tensor = real_data[1]-predi_data)
        print(f'Epoch: {saveModel.epoch:6} Step: {step:3} L2: {l2:3.3f}% RMSE: {RMSE_percentage :3.5f}%') #RSquard: {RSquard : 1.5f}')    
            

        saveModel.on_epoch_end(l2)
        if saveModel.epoch%20 == 0:
            saveModel.save_model()
            saveModel.save_config(RMSE_percentage)
        if saveModel.epoch%100 ==0:
            for i in range(predi_data.shape[0]):
                predi_data[i].numpy().tofile(dataSetConfig["logDir"] + f'Nyx{dataSetConfig["width"]}Predict-{str(i).zfill(4)}.bin')
main()