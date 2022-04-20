from tensorflow.keras import Model,layers,Sequential
from tensorflow.keras.layers import Conv2D,\
    BatchNormalization,Activation,AveragePooling2D,GlobalAveragePooling2D,Dense,MaxPooling2D
import tensorflow as tf

class Conv_BN_ReLU(Model):#将卷积层、批归一化层和激活函数层封装
    def __init__(self,ch,kernel_size=3,strides=1,padding='same'):
        super(Conv_BN_ReLU,self).__init__()
        self.model=Sequential([
            Conv2D(ch,kernel_size=kernel_size,strides=strides,padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])


    def call(self,x):
        x=self.model(x)
        return x

class InceptionBlock(Model):
    def __init__(self,ch,strides=1):
        super(InceptionBlock,self).__init__()
        self.ch=ch
        self.strides=strides
        self.c1=Conv_BN_ReLU(ch,kernel_size=1,strides=strides)
        self.c2_1=Conv_BN_ReLU(ch,kernel_size=1,strides=strides)
        self.c2_2=Conv_BN_ReLU(ch,kernel_size=3,strides=1)
        self.c3_1=Conv_BN_ReLU(ch,kernel_size=1, strides=strides)
        self.c3_2=Conv_BN_ReLU(ch,kernel_size=5,strides=1)
        self.p4_1=MaxPooling2D(ch,pool_size=3,strides=1)
        self.c4_2=Conv_BN_ReLU(ch,kernel_size=3,strides=strides)


    def call(self,x):
        x=self.c1(x)
        x2_1=self.c2_1(x)
        x2_2=self.c2_2(x2_1)
        x3_1=self.c3_1(x)
        x3_2=self.c3_2(x3_1)
        x4_1=self.c4_1(x)
        x4_2=self.c4_2(x4_1)
       #concat along axis=channel
        x=tf.concat([x,x2_2,x3_2,x4_2],axis=3)
        return x

#---------------------------------------------------------
'''
四个Inception结构块顺序相连，每两个Inception结构块组成
一个block，每个block中的第一个Inception结构块，卷积步长s=2，
输出特征图尺寸减半，第二个Inception结构块，卷积步长s=1
'''
#---------------------------------------------------------
class InceptionNet(Model):
    def __init__(self,num_blocks,num_classes,init_ch=16,**kwargs):
        super(InceptionNet,self).__init__(**kwargs)
        self.num_blocks=num_blocks
        self.in_channels=init_ch
        self.out_channels=init_ch
        self.init_ch=init_ch
        self.c1=Conv_BN_ReLU(init_ch)
        self.blocks=Sequential()
        for block_id in range(num_blocks):
            for layer_id in range (2):
                if layer_id==0:
                    block=InceptionBlock(self.out_channels,strides=2)
                else:
                    block=InceptionBlock(self.out_channels,strides=1)
                self.blocks.add(block)
            self.out_channels*=2
        self.p1=GlobalAveragePooling2D()
        self.d1=Dense(num_classes,activation='softmax')

    def call(self,x):
        x=self.c1(x)
        x=self.blocks(x)
        x=self.p1(x)
        x=self.d1(x)
        return x

model=InceptionNet(num_blocks=2,num_classes=10)

