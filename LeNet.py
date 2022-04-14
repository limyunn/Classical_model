from tensorflow.keras import Model,layers,Sequential
import tensorflow as tf

class LeNet(Model):
    def __init__(self):
        super(LeNet,self).__init__()
        self.c1=layers.Conv2D(filters=6,kernel_size=(5,5),padding='valid',strides=1,activation='sigmoid')
        self.p1=layers.MaxPooling2D(pool_size=(2,2),strides=2)
        self.c2=layers.Conv2D(filters=16,kernel_size=(5,5),padding='valid',activation='sigmoid')
        self.p2=layers.MaxPooling2D(pool_size=(2,2),strides=2)

        self.flatten=layers.Flatten()
        self.d1=layers.Dense(120,activation='sigmoid')
        self.d2=layers.Dense(84,activation='sigmoid')
        self.d3=layers.Dense(10,activation='softmax')


    def call(self,x):
        x=self.c1(x)
        x=self.p1(x)
        x=self.c2(x)
        x=self.p2(x)

        x=self.flatten(x)
        x=self.d1(x)
        x=self.d2(x)
        y=self.d3(x)
        return y

###############################################    model   ###############################################
network = Sequential([ # 网络容器
          layers.Conv2D(6,kernel_size=3,strides=1), # 第一个卷积层, 6个3x3卷积核
          layers.MaxPooling2D(pool_size=2,strides=2), # 高宽各减半的池化层
          layers.ReLU(), # 激活函数
          layers.Conv2D(16,kernel_size=3,strides=1), # 第二个卷积层, 16个3x3卷积核
          layers.MaxPooling2D(pool_size=2,strides=2), # 高宽各减半的池化层
          layers.ReLU(), # 激活函数
          layers.Flatten(), # 拉平层，方便全连接层处理
          layers.Dense(120, activation='relu'), # 全连接层，120个节点
          layers.Dense(84, activation='relu'), # 全连接层，84节点
          layers.Dense(10,activation='softmax') ])# 全连接层，10个节点


if __name__ == '__main__':
    model=LeNet()
    x=tf.random.normal([1,32,32,3])
    y=model(x)
    print(y)