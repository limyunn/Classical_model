from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import tensorflow as tf
import tensorflow.keras.backend as K

base_model = ResNet50(weights='imagenet', include_top=True)
#x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)
model = Model(inputs=base_model.input, outputs=base_model.output)

img_path='36.JPEG'
img = image.load_img(img_path, target_size=(224, 224))#Returns: A PIL Image instance
x = image.img_to_array(img)#Converts a PIL Image instance to a Numpy array
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)#[1, 1000]
# 将结果解码为元组列表 (class, description, probability)
# (一个列表代表批次中的一个样本）
print('Predicted:', decode_predictions(preds, top=3)[0])#Returns:A list of lists of top class prediction tuples
    ##One list of tuples per sample in batch input.
    #这些地方所加的0皆是batch中第个sample的意思

tf.print(tf.argmax(preds[0]))
np.set_printoptions(threshold=np.inf)
#-------------------Grad CAM----------------------#
# 首先，我们创建一个模型，将输入图像映射到最后一个conv层的激活以及输出预测
last_conv_layer=model.get_layer('conv4_block2_3_conv')#shape=(None, 7, 7, 512)
heatmap_model =Model([base_model.inputs], [last_conv_layer.output, base_model.output])
##然后，我们为输入图像计算top预测类关于最后一个conv层的激活的梯度
with tf.GradientTape() as tape:
    conv_output, Predictions = heatmap_model(x)
    prob = Predictions[:, np.argmax(Predictions[0])]  # 最大可能性类别的预测概率的索引
    #Predictions[0]返回一维数组
    grads = tape.gradient(prob, conv_output)# 这是输出神经元(预测概率最高的或者选定的那个)对最后一个卷积层输出特征图的梯度
    #grads.shape->(1, 7, 7, 512)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))  # 特征层梯度的全局平均代表每个特征层权重
    #pooled_grads.shape->(512,)
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output[0]), axis=-1)  # 权重与特征层相乘，512层求和平均
#heatmap.shape->(7,7)
heatmap = np.maximum(heatmap, 0)
max_heat = np.max(heatmap)
if max_heat == 0:
    max_heat = 1e-10
heatmap /= max_heat#将热力图标准化到0~1范围内
plt.matshow(heatmap,cmap='jet')
plt.show()

#-------------------将原始图像叠加在刚刚得到的热力图上----------------------#
import cv2
original_img=cv2.imread('36.JPEG') #用cv2加载原始图像
print(original_img.shape)
heatmap1 = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0])) #将热力图的大小调整为与原始图像相同
heatmap1 = np.uint8(255*heatmap1) #将热力图转换为RGB格式
heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET) #将热力图应用于原始图像
superimposed_img=heatmap1*0.5+original_img#这里的0.45是热力图强度因子

#激活图结果上的文字显示
cls_index='Persian_cat'
text = '%s %.2f%%' % (cls_index, preds[:,np.argmax(preds[0])]*100)
score=preds[:,np.argmax(preds[0])]#[0.97232866]
# text= '{} {:.2%}'.format(cls_index, score[0])
cv2.putText(superimposed_img, text, (270, 40), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.9,
            color=(123, 222, 238), thickness=2, lineType=cv2.LINE_AA)

cv2.imwrite('Persian_cat.jpg', superimposed_img)

