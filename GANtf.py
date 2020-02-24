class CustomConv2D(tf.keras.layers.Layer):
    def __init__(self, filter, kernel_size, strides=(1,1), padding = 'VALID', **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = (1, *strides, 1)
        self.padding = padding


    def build(self, input_shape):
        #建立權重的地方
        kernel_h, kernel_w = self.kernel_size
        input_dim = input_shape][-1]
        self.w = self.add_weight(name = 'kernal', 
                                 shape = (kernel_h, kernel_w, input_dim, self.filters), 
                                 initializer = 'glorot_uniform', 
                                 trainable = True)
        self.b = self.add_weight(name = 'bias', 
                                 shape = (self.filters, ),
                                 initializer = 'zeros', 
                                 trainable = True)
        pass
    def call(self, inputs):
        #定義網路前向傳遞
        x = tf.nn.conv2d(inputs, self.w, self.strides, padding=self.padding)
        
        pass
    def get_config(self):
        #如果你要支援序列化要在這定義
        pass