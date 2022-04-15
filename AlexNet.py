from tensorflow.keras import Model,layers,Sequential
import tensorflow as tf

class AlexNet(Model):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.c1 = layers.Conv2D(filters=96,kernel_size=(3,3),strides=1,padding='valid')
        self.b1 = layers.BatchNormalization()
        self.a1 = layers.Activation('relu')
        self.p1 = layers.MaxPooling2D(pool_size=(3,3),strides=2)

        self.c2 = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='valid')
        self.b2 = layers.BatchNormalization()
        self.a2 = layers.Activation('relu')
        self.p2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)

        self.c3 = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same',
                                activation='relu')
        self.c4 = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same',
                                activation='relu')
        self.c5 = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same',
                                activation='relu')
        self.p3 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)

        self.flatten = layers.Flatten()
        self.f1 = layers.Dense(2048,activation='relu')
        self.d1 = layers.Dropout(0.5)
        self.f2 = layers.Dense(2048,activation='relu')
        self.d2 = layers.Dropout(0.5)
        self.f3 = layers.Dense(1000,activation='softmax')

    def call(self,x):
        x=self.c1(x)
        x=self.b1(x)
        x=self.a1(x)
        x=self.p1(x)

        x=self.c2(x)
        x=self.b2(x)
        x=self.a2(x)
        x=self.p2(x)

        x=self.c3(x)

        x=self.c4(x)

        x=self.c5(x)
        x=self.p3(x)

        x=self.flatten(x)
        x=self.f1(x)
        x=self.d1(x)
        x=self.f2(x)
        x=self.d2(x)
        y=self.f3(x)

        return y




if __name__ == '__main__':
    model=AlexNet()
    x=tf.random.normal([1,32,32,3])
    y=model(x)
    print(y)