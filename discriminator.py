from resblock import ResBlock_discriminator
import tensorflow as tf
from tensorflow.keras import layers
#from IPython.display import Image
class discriminator(tf.keras.Model):
    def __init__(self, ch=64):
        super(discriminator, self).__init__()

        self.xD0 = layers.Dense(512, name = 'parameter1_layer_1')
        self.xR0 = layers.PReLU()
        self.xD1 = layers.Dense(512, name = 'parameter1_layer_2')
        self.xR1 = layers.PReLU()

        self.yD0 = layers.Dense(512, name = 'parameter2_layer_1')
        self.yR0 = layers.PReLU()
        self.yD1 = layers.Dense(512, name = 'parameter2_layer_2')
        self.yR1 = layers.PReLU()

        self.zD0 = layers.Dense(512, name = 'parameter3_layer_1')
        self.zR0 = layers.PReLU()
        self.zD1 = layers.Dense(512, name = 'parameter3_layer_2')
        self.zR1 = layers.PReLU()

        self.concateD = layers.Dense(ch*2)
        self.concateR = layers.PReLU()


        self.res0 = ResBlock_discriminator(ch)
        self.res1 = ResBlock_discriminator(ch*2)
        self.resR = layers.PReLU()
        self.GAV3D = layers.GlobalAveragePooling3D()
        
        #d = layers.Flatten()(d)
        self.outputD = layers.Dense(1)
        
        #model = keras.Model(inputs = [dataInput], outputs = d)
       # plot_model(model, to_file = "WGAN_Discriminator.png", show_shapes=True)
    def call(self, p1, p2, p3, inputs):
        x = self.xD0(p1)
        x = self.xR0(x)
        x = self.xD1(x)
        x = self.xR1(x)

        y = self.yD0(p2)
        y = self.yR0(y)
        y = self.yD1(y)
        y = self.yR1(y)
        
        z = self.zD0(p3)
        z = self.zR0(z)
        z = self.zD1(z)
        z = self.zR1(z)

        xyz = layers.concatenate([x, y, z])
        xyz = self.concateD(xyz)
        xyz = self.concateR(xyz)

        d = self.res0(inputs)
        d = self.res1(d)
        d = self.resR(d)
        print(d)
        d = self.GAV3D(d) * (16*16*16)
        print(d)
        f1 = layers.Multiply()([d, xyz])
        f2 = self.outputD(d)
        r = layers.add([f1, f2])

        return r





