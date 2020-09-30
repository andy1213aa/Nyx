from ..utlis import config
import numpy as np 
import os
import tensorflow as tf
from shutil import copyfile
def generateData(dataSetConfig):
    
    dataType = {'float': [4, tf.float32]}
    
    def _parse_function(example_proto):
        features = tf.io.parse_single_example(
            example_proto,
            features={
                "Parameter1": tf.io.FixedLenFeature([], tf.float32),
                "Parameter2": tf.io.FixedLenFeature([], tf.float32),
                "Parameter3": tf.io.FixedLenFeature([], tf.float32),
                'data_raw': tf.io.FixedLenFeature([], tf.string)
            }
        )
     
        P1 = features['Parameter1']
        P2 = features['Parameter2']
        P3 = features['Parameter3']
        data = features['data_raw']
        data = tf.io.decode_raw(data, tf.float32)
        # mean = tf.reduce_mean(data)
        # std = tf.math.reduce_std(data)
        # data = (data-mean)/std
        data = tf.reshape(data, [dataSetConfig['height'], dataSetConfig['width'], dataSetConfig['length'], 1])
        P1 = tf.reshape(P1, [1])
        P2 = tf.reshape(P2, [1])
        P3 = tf.reshape(P3, [1])
        return P1, P2, P3, data
        
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    data = tf.data.TFRecordDataset(dataSetConfig['dataSetDir'])
    data = data.map(_parse_function, num_parallel_calls=AUTOTUNE)

    #data = data.shuffle(800, reshuffle_each_iteration=True)
    train_data = data.take(dataSetConfig['trainSize'])
    training = train_data.take(dataSetConfig['trainSize'])
    training = training.shuffle(dataSetConfig['trainSize'], reshuffle_each_iteration=True)
    validating = train_data.skip(dataSetConfig['trainSize'] - dataSetConfig['validationSize'])
    #test_data = data.skip(dataSetConfig['trainSize'])
    

    training_batch = training.batch(dataSetConfig['batchSize'], drop_remainder = True)
    validating_batch = validating.batch(dataSetConfig['validationSize'], drop_remainder = True)
    validating_batch = validating_batch.prefetch(buffer_size = AUTOTUNE)
    #test_data_batch = test_data.batch(100, drop_remainder = True)
    training_batch = training_batch.prefetch(buffer_size = AUTOTUNE)
    return training_batch,validating_batch#, test_data_batch