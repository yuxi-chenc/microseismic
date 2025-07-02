import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Activation, GRU, GlobalAveragePooling2D, Reshape, Multiply

class LocalAttentionModule(Model):
    def __init__(self, filters):
        super(LocalAttentionModule, self).__init__()
        self.global_avg_pool = GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(filters // 4, activation='relu', use_bias=False)
        self.fc2 = tf.keras.layers.Dense(filters, activation='sigmoid', use_bias=False)
        self.reshape = Reshape((1, 1, filters))
    
    def call(self, skip, main):
        attention_weights = self.global_avg_pool(skip)
        attention_weights = self.fc1(attention_weights)
        attention_weights = self.fc2(attention_weights)
        attention_weights = self.reshape(attention_weights)
        
        attended_skip = Multiply()([skip, attention_weights])
        
        return attended_skip + main

class down_layer_block(Model):
    def __init__(self, filters, strides, kernel_high, kernel_weigh):
        super(down_layer_block, self).__init__()
        self.down_conv = tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=(kernel_high, kernel_weigh),
                                                strides=strides,
                                                padding='same')
        self.down_BN = BatchNormalization()
        self.down_ACT = Activation('relu')

    def call(self, inputs):
        y = self.down_conv(inputs)
        y = self.down_BN(y)
        y = self.down_ACT(y)
        return y

class up_layer_block(Model):
    def __init__(self, filters, strides, kernel_high, kernel_weigh):
        super(up_layer_block, self).__init__()
        self.up_conv = tf.keras.layers.Conv2DTranspose(filters=filters,
                                                       kernel_size=(kernel_high, kernel_weigh),
                                                       strides=strides,
                                                       padding='same')
        self.up_BN = BatchNormalization()
        self.up_ACT = Activation('relu')

    def call(self, inputs):
        y = self.up_conv(inputs)
        y = self.up_BN(y)
        y = self.up_ACT(y)
        return y

class BAMModule(Model):
    def __init__(self, filters):
        super(BAMModule, self).__init__()
        self.global_pool = GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(filters // 4, activation='relu')
        self.fc2 = tf.keras.layers.Dense(filters, activation='sigmoid')
        self.reshape = Reshape((1, 1, filters))

    def call(self, inputs):
        y = self.global_pool(inputs)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.reshape(y)
        y = Multiply()([inputs, y])
        return y
    
class base_model(Model):
    def __init__(self):
        super(base_model, self).__init__()
        self.int_layer = tf.keras.layers.Conv2D(filters=8, kernel_size=(9, 1), strides=(1, 1), padding='same')
        self.int_BN = BatchNormalization()
        self.int_ACT = Activation('relu')

        self.layer_two = down_layer_block(8, (5, 1), 9, 1)
        self.layer_three = down_layer_block(16, (1, 1), 7, 1)
        self.layer_four = down_layer_block(16, (5, 1), 7, 1)
        self.layer_five = down_layer_block(24, (1, 1), 5, 1)
        self.layer_six = down_layer_block(32, (5, 1), 3, 1)
        
        self.attention_module_1 = LocalAttentionModule(filters=32)
        self.attention_module_2 = LocalAttentionModule(filters=24)
        self.attention_module_3 = LocalAttentionModule(filters=16)
        
        self.reshape_to_3D = Reshape((-1, 32))
        self.gru = GRU(32, return_sequences=True)
        self.reshape_back_to_4D = Reshape((-1, 1, 32))
        
        self.bam_block = BAMModule(filters=32)
        
        self.layer_seven = down_layer_block(32, (1, 1), 3, 1)
        self.layer_eight = up_layer_block(24, (5, 1), 5, 1)
        self.layer_nine = up_layer_block(16, (1, 1), 7, 1)
        self.layer_ten = up_layer_block(16, (5, 1), 7, 1)
        self.layer_eleven = up_layer_block(8, (1, 1), 9, 1)
        self.layer_twelve = up_layer_block(8, (5, 1), 9, 1)

        self.out_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=(9, 1), strides=(1, 1), padding='same')
        self.out_BN = BatchNormalization()
        self.out_ACT = Activation('softmax')

    def call(self, x):
        y1 = self.int_layer(x)
        y1 = self.int_BN(y1)
        y1 = self.int_ACT(y1)
        
        y2 = self.layer_two(y1)
        y3 = self.layer_three(y2)
        y4 = self.layer_four(y3)
        y5 = self.layer_five(y4)
        y6 = self.layer_six(y5)

        y6_reshaped = self.reshape_to_3D(y6)
        y_gru = self.gru(y6_reshaped)
        y_gru_reshaped = self.reshape_back_to_4D(y_gru)
        y_bam = self.bam_block(y_gru_reshaped)
        
        y7 = self.layer_seven(y_bam)
        y8 = self.layer_eight(y7)
        y9 = self.layer_nine(self.attention_module_2(y5, y8))
        y10 = self.layer_ten(self.attention_module_3(y4, y9))
        y11 = self.layer_eleven(self.attention_module_3(y3, y10))
        y12 = self.layer_twelve(y11)

        y_out = self.out_layer(y12)
        y_out = self.out_BN(y_out)
        y_out = self.out_ACT(y_out)
        return y_out