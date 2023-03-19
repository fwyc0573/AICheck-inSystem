from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.applications.imagenet_utils import preprocess_input
import os
import numpy as np
import cv2

HEIGHT = 160
WIDTH = 160
NUM_CLASSES = 2
# def get_class():
#         classes_path = os.path.expanduser(r"D:\PyCharm2017\PROJECT\mask-recognize-master\model_data\classes.txt")
#         with open(classes_path) as f:
#             class_names = f.readlines()
#         class_names = [c.strip() for c in class_names]
#         return class_names

# def preprocess_input(x):
#     x /= 255.
#     x -= 0.5
#     x *= 2.
#     return x
def CNN(input_shape=[160,160,3],classes=2):
    model = Sequential()  # 建立顺序模型
    # 进行各种层的添加
    model.add(Conv2D(kernel_size=(5, 5),nb_filter=32, border_mode='same', input_shape=(160, 160, 3)))  # 第一次卷积
    model.add(BatchNormalization())  # 利用BN层规范化
    model.add(Activation("relu"))  # 激活层
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))  # 最大化池连接
    model.add(Dropout(0.2))  # Dropout层防止过拟合

    model.add(Conv2D(kernel_size=(5, 5),nb_filter=64,  padding='same'))  # 第二次卷积
    model.add(BatchNormalization())  # 利用BN层规范化
    model.add(Activation("relu"))  # 激活层
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))  # 最大化池连接
    model.add(Dropout(0.2))  # Dropout层防止过拟合

    # 全连接层
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    return model



def mask_predict(model,src_img):
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    src_img = cv2.resize(src_img, (HEIGHT, WIDTH))
    new_img = preprocess_input(np.reshape(np.array(src_img, np.float64), [1,HEIGHT, WIDTH, 3]))
    class_names = ["mask", "nomask"]
    classes = class_names[np.argmax(model.predict(new_img)[0])]

    print("model.predict(img)[0]:",model.predict(new_img)[0])#预测结果
    print("np.argmax(model.predict(img)[0]):",np.argmax(model.predict(new_img)[0]))#取出最大值对应序号
    print(classes)


if __name__ == '__main__':
    model = CNN(input_shape=(160, 160, 3),classes=2)
    model.summary()
    # img_path = '22.jpg'
    # model.load_weights(r"D:\PyCharm2017\PROJECT\mask-recognize-master\logs\CNN_model.h5")
    #
    # #opencv读取图片的方式
    # src_img = cv2.imread(img_path)
    # # src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    # src_img = cv2.resize(src_img, (HEIGHT, WIDTH))
    # cv2.imshow('Video', src_img)
    # cv2.waitKey()
    # new_img = preprocess_input(np.reshape(np.array(src_img, np.float64), [1,HEIGHT, WIDTH, 3]))
    # # class_names = get_class()#读取txt划分出的多个种类
    # class_names = ["mask","nomask"]
    # classes = class_names[np.argmax(model.predict(new_img)[0])]
    #
    # # PIL读取图片的方式
    # # img = image.load_img(img_path, target_size=(160, 160))
    # # img = image.img_to_array(img)#转为浮点型，提高准确率
    # # img = np.expand_dims(img, axis=0)
    # # new_img = preprocess_input(np.reshape(np.array(img, np.float64), [1, HEIGHT, WIDTH, 3]))
    # # class_names = get_class()#读取txt划分出的多个种类
    # # classes = class_names[np.argmax(model.predict(new_img)[0])]
    #
    # print("model.predict(img)[0]:",model.predict(new_img)[0])#预测结果
    # print("np.argmax(model.predict(img)[0]):",np.argmax(model.predict(new_img)[0]))#取出最大值对应序号
    # print(classes)
