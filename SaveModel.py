import tensorflow as tf
import numpy as np

class SaveModel(tf.keras.callbacks.Callback):
    def __init__(self, model, weights_file, mode = 'min', save_weights_only = False):
        super(SaveModel, self).__init__()
        self.model = model
        # setting directory of saving weight
        self.weights_file = weights_file

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
    def save_model(self):
        if self.save_weights_only:
            self.save_weights(self.weights_file)
        else:
            self.model.save(self.weights_file)
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
            if self.counter >= 100:
                self.save_model()
                self.training = False