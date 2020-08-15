import tensorflow as tf
import numpy as np
import datetime
class SaveModel(tf.keras.callbacks.Callback):
    def __init__(self, model, dataSetConfig, mode = 'min', save_weights_only = True):
        super(SaveModel, self).__init__()
        self.model = model
        # setting directory of saving weight
        self.dataSetConfig = dataSetConfig

        # biggest better or lowest better
        self.mode = mode
        # save type
        self.save_weights_only = save_weights_only
        if mode == 'min':
            self.best = np.inf
        else:
            self.best = -np.inf
        self.counter = 0
        self.training = True
        self.epoch = 1
    def save_model(self):
        if self.save_weights_only:
            self.model.save_weights(self.dataSetConfig['logDir'] + "trained_ckpt")
        else:
            self.model.save(self.dataSetConfig['logDir'])
    def save_config(self, monitor_value):
        saveLogTxt = f"""
    Parameter Setting
    =======================================================
    DataSet: { self.dataSetConfig['dataSet']}
    DataShape: ({ self.dataSetConfig['length']}, { self.dataSetConfig['width']}, {self.dataSetConfig['height']})
    DataSize: {self.dataSetConfig['datasize']}
    TrainingSize: { self.dataSetConfig['trainSize']}
    TestingSize: { self.dataSetConfig['testSize']}
    BatchSize: { self.dataSetConfig['batchSize']}
    =======================================================

    Training log
    =======================================================
    Training start: { self.dataSetConfig['startingTime']}
    Training stop: {datetime.datetime.now()}
    Training epoch: {self.epoch}
    Root Mean Square Error: {monitor_value}%
    =======================================================
    """
        with open( self.dataSetConfig['logDir']+'config.txt', 'w') as f:
            f.write(saveLogTxt) 
    def on_epoch_end(self, monitor_value, logs = None):
        # read monitor value from logs
        # monitor_value = logs.get(self.monitor)
        # Create the saving rule
        
        if self.mode == 'min' and monitor_value < self.best:
            
            self.best = monitor_value
            self.counter = 0
        elif self.mode == 'max' and monitor_value > self.best:
            
            self.best = monitor_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.dataSetConfig['stopConsecutiveEpoch']:
                self.save_model()
                self.save_config(monitor_value)
                self.training = False
        self.epoch += 1