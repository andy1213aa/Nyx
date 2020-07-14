import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import os

g = os.walk(r"E:\NTNU1-2\Nyx\NyxDataSet16_16")  
training_set = []
for path,dir_list,file_list in g:  
    for file_name in file_list:  
        tmp_data = np.fromfile(path+'\\'+file_name, dtype = 'float32').reshape((16, 16, 1))
        training_set.append(tmp_data)
x_train = np.array(training_set)[:699]
x_test= np.array(training_set)[699:]
print(x_train.shape)
print(x_test.shape)
y_train = np.ones((699,1, 1,  1))
y_test = np.zeros((100,1, 1,  1))

model = Sequential()
model.add(layers.Conv2D(16, kernel_size=3, strides=1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(32, kernel_size=3, strides=1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(64, kernel_size=3, strides=1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(1, kernel_size=3, strides=1, padding='same'))
model.add(layers.Activation('relu'))
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=64,
          nb_epoch=25,
          verbose=1,
          validation_data=(x_test, y_test))    
#outputs = [layer.output for layer in model.layers]    
print(model.layers[1].output)
