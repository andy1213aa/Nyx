import config
import numpy as np 
import os
import tensorflow as tf

def generateData(dataSetConfig):
    
    dataType = {'float': [4, tf.float32]}
    filename = os.listdir(dataSetConfig['dataSetDir'])



    parameter1 = tf.data.Dataset.from_tensor_slices(np.array([np.float(file[5:12]) for file in filename], dtype=np.float32).reshape((len(filename), 1)))
    parameter2 = tf.data.Dataset.from_tensor_slices(np.array([np.float(file[13:20]) for file in filename], dtype=np.float32).reshape((len(filename), 1)))
    parameter3 = tf.data.Dataset.from_tensor_slices(np.array([np.float(file[21:28]) for file in filename], dtype=np.float32).reshape((len(filename), 1)))
    parameter123 = tf.data.Dataset.zip((parameter1, parameter2, parameter3))

    filename_list = [os.path.join(dataSetConfig['dataSetDir'], file) for file in filename]
    file_queue = tf.data.Dataset.from_tensor_slices(filename_list)
    data = tf.data.FixedLengthRecordDataset(file_queue, dataSetConfig['width']*dataSetConfig['length']*dataSetConfig['height']*dataType[dataSetConfig['dataType']][0])
    
    def process_input_data(ds):
        ds = tf.io.decode_raw(ds, dataType[dataSetConfig['dataType']][1])
        ds = tf.reshape(ds, [dataSetConfig['width'], dataSetConfig['length'], dataSetConfig['height'], 1]) 
        #mean = tf.reduce_mean(ds)
        #std = tf.math.reduce_std(ds)      
        return ds#(ds-mean)/std
        
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    data = data.map(process_input_data, num_parallel_calls=AUTOTUNE)
    data = tf.data.Dataset.zip((parameter123, data))
    data = data.shuffle(800, reshuffle_each_iteration=True)
    train_data = data.take(dataSetConfig['trainSize'])
    training = train_data.take(dataSetConfig['trainSize'])

    test_data = data.skip(dataSetConfig['trainSize'])
    
    training_batch = training.batch(dataSetConfig['batchSize'], drop_remainder = True)
    test_data_batch = test_data.batch(100, drop_remainder = True)
    training_batch = training_batch.prefetch(buffer_size = AUTOTUNE)
    return training_batch, test_data_batch