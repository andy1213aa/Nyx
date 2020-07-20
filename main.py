import os
from tensorboard.plugins.hparams import api as hp
from GAN_version16_16 import GAN
import datetime
import tensorflow as tf

def main():
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
            GANs = GAN(length = 16, width = 16, height = 1, batchSize = 64, epochs = 5000, dataSetDir = r'E:\NTNU1-2\NyxDataSet\NyxDataSet16_16', hparams = hparams, logdir = dirs+'\\'+session_num)
            GANs.train_wgan()
            num+=1

main()