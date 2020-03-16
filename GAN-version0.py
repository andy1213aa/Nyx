import tensorflow
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from loadRawData import loadData

class GAN():
    def __init__(self, height, weight):
        self.height = height
        self.weight = weight

        self.gen = self.generator()
        self.dis = self.discriminator()
        self.discriminatorOptimizer = keras.optimizers.RMSprop(lr = 0.0008, 
                                                          clipvalue = 1.0, 
                                                          decay = 1e-8)
        self.dis.compile(optimizer = self.discriminatorOptimizer, loss = 'binary_crossentropy')
        self.dis.trainable = False


        self.ganInput1 =  keras.Input(shape = (1), name = 'ganInput1')
        self.ganInput2 =  keras.Input(shape = (1), name = 'ganInput2')
        self.ganInput3 =  keras.Input(shape = (1), name = 'ganInput3')

        self.ganOutput = self.dis(self.gen([self.ganInput1, self.ganInput2, self.ganInput3]))
        self.gan = keras.Model(inputs = [self.ganInput1, self.ganInput2, self.ganInput3], outputs = self.ganOutput)

        plot_model(self.gan, to_file = 'gan.png', show_shapes = True)
        self.ganOptimizer = keras.optimizers.RMSprop(lr = 0.0004, 
                                                    clipvalue = 1.0, 
                                                    decay = 1e-8)
        self.gan.compile(optimizer = self.ganOptimizer, loss = 'binary_crossentropy')
    def generator(self):
        parameter1_input = keras.Input(shape = (1), name = 'parameter1')
        parameter2_input = keras.Input(shape = (1), name = 'parameter2')
        parameter3_input = keras.Input(shape = (1), name = 'parameter3')

        h1_1 = layers.Dense(512, name = 'parameter1_layer_1', activation = 'relu')(parameter1_input)
        h1_2 = layers.Dense(512, name = 'parameter1_layer_2', activation = 'relu')(h1_1)
        h1_3 = layers.Dense(512, name = 'parameter1_layer_3', activation = 'relu')(h1_2)
        
        h2_1 = layers.Dense(512, name = 'parameter2_layer_1', activation = 'relu')(parameter2_input)
        h2_2 = layers.Dense(512, name = 'parameter2_layer_2', activation = 'relu')(h2_1)
        h2_3 = layers.Dense(512, name = 'parameter2_layer_3', activation = 'relu')(h2_2)
        
        h3_1 = layers.Dense(512, name = 'parameter3_layer_1', activation = 'relu')(parameter3_input)
        h3_2 = layers.Dense(512, name = 'parameter3_layer_2', activation = 'relu')(h3_1)
        h3_3 = layers.Dense(512, name = 'parameter3_layer_3', activation = 'relu')(h3_2)

        concatenate = layers.concatenate(inputs = [h1_3, h2_3, h3_3])

        f1_1 = layers.Dense(512, activation = 'relu')(concatenate)
        f1_2 = layers.Dense(256, activation = 'relu')(f1_1)
        f1_3 = layers.Reshape((16, 16))(f1_2)


        model = keras.Model(inputs = [parameter1_input, parameter2_input, parameter3_input], outputs = f1_3)
        plot_model(model, to_file='generator.png', show_shapes=True)
        return model
    
    def discriminator(self):

        dataInput = keras.Input(shape = (16,16), name = 'groundTruth/fake')
        flatten = layers.Flatten()(dataInput)
        d1_1 = layers.Dense(128, activation = 'relu')(flatten)
        d1_2 = layers.Dense( 64, activation = 'relu')(d1_1)
        d1_3 = layers.Dense( 8, activation = 'relu')(d1_2)
        d1_4 = layers.Dense( 1, activation = 'sigmoid')(d1_3)
        
        model = keras.Model(inputs = dataInput, outputs = d1_4)
        plot_model(model, to_file = "Discriminator.png", show_shapes=True)
        return model
    def training(self, epochs, batchsize):
        rawData = loadData(r'C:\Users\User\Desktop\NTNU 1-2\Nyx\NyxDataSet', self.height, self.weight)
        for epoch in range(epochs):
            start = 0
            training = True
            while training:
                noise1 = np.random.normal(size = (batchsize, 1))
                noise2 = np.random.normal(size = (batchsize, 1))
                noise3 = np.random.normal(size = (batchsize, 1))
                batch = rawData[start:start + batchsize]
                
                # Feature scalling
                for i in range(16):         # number of 16 Depends on dimension of raw data we load.
                    mean = np.mean(batch[:, i, :])
                    std = np.std(batch[:, i, :])
                    batch[:, i, :] = (batch[:, i, :] - mean) / std
                
                fakeData = self.gen.predict([noise1, noise2, noise3])
                combineData = np.concatenate([batch, fakeData])
                combineLabel = np.concatenate([np.ones((batchsize, 1)), np.zeros((batchsize, 1))])
                dLoss = self.dis.train_on_batch(combineData, combineLabel)
                gLoss = self.gan.train_on_batch([noise1, noise2, noise3], np.ones((batchsize, 1)))

                print("======================================================================")
                print(f"Epoch: {epoch}, Batch: {start/batchsize}, Discriminator loss: {dLoss}")    
                print(f"Epoch: {epoch}, Batch: {start/batchsize}, GAN loss: {gLoss}")
                print("======================================================================")
                
                start += batchsize
                if start + batchsize >= rawData.shape[0]:
                    training = False



GAN = GAN(16, 16)
GAN.training(10000, 30)
