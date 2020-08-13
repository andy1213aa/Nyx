from GAN import GAN
import datetime
import tensorflow as tf
from loadData import generateData
import config
from SaveModel import SaveModel
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
    dataSetConfig = config.dataSet['Nyx'] #What data you want to load



    model = GAN(length = dataSetConfig['length'], width = dataSetConfig['width'], height = dataSetConfig['height'], batchSize = dataSetConfig['batchSize'])
    training_batch, testing_batch = generateData(dataSetConfig)
    
    summary_writer = tf.summary.create_file_writer(dataSetConfig['logDir'])
    # tf.summary.trace_on(graph=True, profiler=True)
    saveModel = SaveModel(model.gen,dataSetConfig, mode = 'min', save_weights_only=False)   #建立一個訓練規則

    data_max = tf.reduce_max(list(training_batch.as_numpy_iterator())[0][1])
    data_min = tf.reduce_min(list(training_batch.as_numpy_iterator())[0][1])
    dataRange = data_max - data_min
    while saveModel.training:
        for _, real_data in enumerate(training_batch):
            # real_data 中 real_data[0] 代表三input parameter 也就是 real_data[0][0] real_data[0][1] 和 real_data[0][2], real_data[1] 代表 groundtruth
            
            d_loss, gp = model.train_discriminator(real_data)
            g_loss= model.train_generator(real_data)
        predi_data = model.gen([real_data[0][0], real_data[0][1], real_data[0][2]])      
    
        RMSE =  tf.sqrt(tf.reduce_mean((real_data[1] - predi_data)**2)) / dataRange
    
        l2 = tf.norm(tensor = real_data[1]-predi_data)
            
        with summary_writer.as_default():
                #hp.hparams(hparams)
            tf.summary.scalar('RMSE', RMSE, saveModel.epoch)
            tf.summary.scalar('discriminator_loss', d_loss, saveModel.epoch)
            tf.summary.scalar('generator_loss', g_loss, saveModel.epoch)
            tf.summary.scalar('gradient_penalty', gp, saveModel.epoch)

        print(f'Epoch: {saveModel.epoch:6} G Loss: {g_loss:15.2f} D loss: {d_loss:15.2f} GP Loss {gp:15.2f} L2: {l2:10f} RMSE: {RMSE* 100 :3.5f}% ')
        saveModel.on_epoch_end(RMSE)
        if saveModel.epoch%100 == 0:
            saveModel.save_model()
            saveModel.save_config(RMSE)
      
main()