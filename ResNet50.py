import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model,layers,Sequential
import numpy as np
from tensorflow.keras.utils import plot_model

class BasicBlock(Model):
    expansion = 4
    def __init__(self,filter_num,strides=1,identity=True):
        super(BasicBlock,self).__init__()
        self.c1=Conv2D(filter_num,(1,1),strides=1,padding='same')
        self.b1=BatchNormalization()
        self.a1=Activation('relu')

        self.c2=Conv2D(filter_num,(3,3),strides=strides,padding='same')
        self.b2=BatchNormalization()
        self.a2=Activation('relu')

        self.c3=Conv2D(filter_num*self.expansion,(1,1),strides=1,padding='same')
        self.b3=BatchNormalization()

        if identity:
            self.map=lambda x:x
        else:
            self.map = Sequential()
            self.map.add((Conv2D(filter_num*self.expansion, (1, 1), strides=strides, padding='same')))
            self.map.add((BatchNormalization()))
        self.a=Activation('relu')

    def call(self,inputs,**kwargs):
        x=self.c1(inputs)
        x=self.b1(x)
        x=self.a1(x)

        x=self.c2(x)
        x=self.b2(x)
        x=self.a2(x)

        x=self.c3(x)
        x=self.b3(x)
        identity=self.map(inputs)
        out=layers.add([x,identity])
        y=self.a(out)
        return y

def ResNet50(input_shape=(224,224,3),num_classes=1000):
    input_tensor=Input(shape=(224,224,3))
    x=ZeroPadding2D((3,3))(input_tensor)
    x=Conv2D(64,(7,7),strides=2,padding='valid')(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(x)
    # (None, 56, 56, 64)

    res_block=Sequential()
    res_block.add(BasicBlock(64,strides=1,identity=False))
    res_block.add(BasicBlock(64,strides=1,identity=True))
    res_block.add(BasicBlock(64,strides=1,identity=True))


    res_block.add(BasicBlock(128,strides=2,identity=False))
    res_block.add(BasicBlock(128, strides=1, identity=True))
    res_block.add(BasicBlock(128, strides=1, identity=True))
    res_block.add(BasicBlock(128, strides=1, identity=True))

    res_block.add(BasicBlock(256, strides=2, identity=False))
    res_block.add(BasicBlock(256, strides=1, identity=True))
    res_block.add(BasicBlock(256, strides=1, identity=True))
    res_block.add(BasicBlock(256, strides=1, identity=True))
    res_block.add(BasicBlock(256, strides=1, identity=True))
    res_block.add(BasicBlock(256, strides=1, identity=True))

    res_block.add(BasicBlock(512, strides=2, identity=False))
    res_block.add(BasicBlock(512, strides=1, identity=True))
    res_block.add(BasicBlock(512, strides=1, identity=True))

    y=res_block(x)#(None, 4, 4, 2048)

    x = AveragePooling2D((7, 7), name='avg_pool')(y)

    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='fc1000')(x)

    model = Model(input_tensor, x, name='resnet50')

    return model

model = ResNet50()
model.summary()

plot_model(model)


