from tensorflow.keras import Model,layers,Sequential
import tensorflow as tf

class VGGNet16(Model):
    def __init__(self):
        super(VGGNet16,self).__init__()
        # 第一段由 2 个卷积层和 1 个最大池化层构成
        self.c1 = layers.Conv2D(filters=64,kernel_size=(3,3),padding='same')
        self.b1 = layers.BatchNormalization()
        self.a1 = layers.Activation('relu')
        self.c2 = layers.Conv2D(filters=64,kernel_size=(3,3),padding='same')
        self.b2 = layers.BatchNormalization()
        self.a2 = layers.Activation('relu')
        self.p1 = layers.MaxPooling2D(pool_size=(2,2),strides=2,padding='same')# (224 - 2 +1) / 1 + 1 = 112
        # 第二段由 2 个卷积层和 1 个最大池化层构成
        self.c3 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b3 = layers.BatchNormalization()
        self.a3 = layers.Activation('relu')
        self.c4 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b4 = layers.BatchNormalization()
        self.a4 = layers.Activation('relu')
        self.p2 = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')# (112 - 2 + 1) / 2 + 1 = 56
        # 第三段由 3个卷积层和 1 个最大池化层构成
        self.c5 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b5 = layers.BatchNormalization()
        self.a5 = layers.Activation('relu')
        self.c6 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b6 = layers.BatchNormalization()
        self.a6 = layers.Activation('relu')
        self.c7 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b7 = layers.BatchNormalization()
        self.a7 = layers.Activation('relu')
        self.p3 = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')# (56 - 2 + 1) / 2 + 1 = 28
        # 第四段由 3个卷积层和 1 个最大池化层构成
        self.c8 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b8 = layers.BatchNormalization()
        self.a8 = layers.Activation('relu')
        self.c9 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b9 = layers.BatchNormalization()
        self.a9 = layers.Activation('relu')
        self.c10 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b10 = layers.BatchNormalization()
        self.a10 = layers.Activation('relu')
        self.p4 = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')# (28 - 2 + 1) / 2 + 1 = 14
        # 第五段由 3个卷积层和 1 个最大池化层构成
        self.c11 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b11 = layers.BatchNormalization()
        self.a11 = layers.Activation('relu')
        self.c12 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b12 = layers.BatchNormalization()
        self.a12 = layers.Activation('relu')
        self.c13 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b13 = layers.BatchNormalization()
        self.a13 = layers.Activation('relu')
        self.p5 = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')# (14 - 2 + 1) / 2 + 1 = 7

        self.flatten=layers.Flatten()
        self.d1=layers.Dense(4096,activation='relu')
        self.drop1=layers.Dropout(0.5)
        self.d2=layers.Dense(4096,activation='relu')
        self.drop2=layers.Dropout(0.5)
        self.d3 = layers.Dense(1000, activation='softmax')

