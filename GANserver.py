import keras
from keras import layers
import numpy as np
from loadRawData import loadData
class GAN():
    def __init__(self, latent_dim, height, width, channel):
        self.latent_dim = latent_dim
        self.height = height
        self.width = width
        self.channel = channel  
        self.generator = self.buildGenerator()
        
        self.discriminator = self.buildDiscriminator()
        self.discriminatorOptimizer = keras.optimizers.RMSprop(lr = 0.0008, 
                                                          clipvalue = 1.0, 
                                                          decay = 1e-8)
        self.discriminator.compile(optimizer = self.discriminatorOptimizer, loss = 'binary_crossentropy')
        self.discriminator.trainable = False


        self.ganInput = keras.Input(shape=(latent_dim, ))
        self.ganOutput = self.discriminator(self.generator(self.ganInput))
        self.gan = keras.models.Model(self.ganInput, self.ganOutput)
        self.gan.summary()
        self.ganOptimizer = keras.optimizers.RMSprop(lr = 0.0004, 
                                                    clipvalue = 1.0, 
                                                    decay = 1e-8)
        self.gan.compile(optimizer = self.ganOptimizer, loss = 'binary_crossentropy')





    def buildGenerator(self):
        
        generatorInput = keras.Input(shape = (self.latent_dim, ))
        
        x = layers.Dense(128*16*16)(generatorInput)
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((16, 16, 128))(x)
        print(x.shape)

        x = layers.Conv2D(256, 5, padding = 'same')(x)
        x = layers.LeakyReLU()(x)
        print(x.shape)

        x = layers.Conv2D(256, 5, padding = 'same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(256, 5, padding = 'same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(self.channel, 7, activation = 'tanh', padding = 'same')(x)
        generator = keras.models.Model(generatorInput, x)
        generator.summary()
        return generator
    def buildDiscriminator(self):
        discriminatorInput = keras.Input(shape = (self.height, self.width, self.channel))

        x = layers.Conv2D(128, 3)(discriminatorInput)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(128, 4)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(128, 4)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(128, 4)(x)     
        x = layers.LeakyReLU()(x)
        print(x.shape)

        x = layers.Flatten()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(1, activation = 'sigmoid')(x)
        print(x.shape) 

        discriminator = keras.models.Model(discriminatorInput, x)
        discriminator.summary()
        return discriminator
    def training(self, iteration, batchSize):
        trainingData = loadData(r'C:\Users\Andy\Desktop\Nyx\NyxDataSet', 16, 16, 1)
        
        start = 0
        for step in range(iteration):
            stop = start + batchSize
            
            noise = np.random.normal(size = (batchSize, self.latent_dim))
            fakeImage = self.generator.predict(noise)
            
            print('fakeImageShape: ', fakeImage.shape)
            print('TrainingDataShape: ', trainingData.shape)

            combindImages = np.concatenate([trainingData[start:stop], fakeImage])
            combindLabel = np.concatenate([np.ones((batchSize, 1)), np.zeros((batchSize, 1))])

            dLoss = self.discriminator.train_on_batch(combindImages, combindLabel)

            noise = np.random.normal(size = (batchSize, self.latent_dim))
            misLeadingLabel = np.ones((batchSize, 1))
            
            gLoss = self.gan.train_on_batch(noise, misLeadingLabel)
            print(' ')
            print('--------------------------------------------')
            print(f'discriminator loss at step {step} : {dLoss}')
            print(f'adversarial loss at step {step} : {gLoss}')
            print('--------------------------------------------')
            print(' ')
            if start > trainingData.shape[0] - batchSize:
                start = 0
          #  if step % 2 == 0:
          #      self.gan.save_weights('gan.h5')
def main():
       GANs = GAN(16, 16, 16, 1)
       GANs.training(1000, 30)


#範本
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        pass
    def build(self, inputshape):
        #建立權重的地方
        pass
    def call(self, inputs):
        #定義網路前向傳遞
        pass
    def get_config(self):
        pass


main()



        #x_train = x_train.reshape((x_train[0], ) + (self.height, self.width, self.channel)).astype('float32') / 255.

        # iteration = 10000
        # batchSize = 20
        # saveDir = r'C:\Users\Andy\Desktop'
        
        # # Start training

        # start = 0
        # for step in range(iteration):
        #     randomLatentVectors = np.random.normal(size = (batchSize, self.latent_dim))
        #     generatorImages = self.generator.predict(randomLatentVectors)
        #     stop = start + batchSize
        #     realImages = x_train[start:stop]
        #     combindImages = np.concatenate([generatorImages, realImages])
        #     labels = np.concatenate([np.ones((batchSize, 1)), 
        #                                     np.zeros((batchSize, 1))])
        #     labels += 0.05* np.random.random(labels.shape)
        #     dLoss = self.discriminator.train_on_batch(combindImages, labels)
        #     randomLatentVectors = np.random.normal(size = (batchSize, self.latent_dim))
        #     misleadingTargets = np.zeros((batchSize, 1))

        #     aLoss = self.gan.train_on_batch(randomLatentVectors, misleadingTargets)

        #     start += batchSize
        #     if start > len(x_train) - batchSize:
        #         start = 0

        #     if step % 100 == 0:
        #         self.gan.save_weights('gan.h5')

        #         print(f'discriminator loss at step {step} : {dLoss}')
        #         print(f'adversarial loss at step {step} : {aLoss}')
        
            









