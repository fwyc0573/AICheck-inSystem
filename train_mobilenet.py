from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import np_utils, get_file
from keras.optimizers import Adam
from keras import backend as K
from utilss.utils import get_random_data
from net.mobileNet import MobileNet
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

HEIGHT = 160
WIDTH = 160
NUM_CLASSES = 2

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0, 0, 0))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image

def generate_arrays_from_file(lines):
    # 获取总长度
    n = len(lines)
    print(n)
    X_train = []
    Y_train = []
    for b in range(n):
        name = lines[b].split(';')[0]
        # 从文件中读取图像
        img = Image.open(r".\dataset\data\image\train" + '/' + name)
        img = np.array(letterbox_image(img, [HEIGHT, WIDTH]), dtype=np.float64)
        X_train.append(img)
        Y_train.append(lines[b].split(';')[1])
        # print(b)
    # 处理图像
    X_train = preprocess_input(np.array(X_train).reshape(-1, HEIGHT, WIDTH, 3))
    Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=NUM_CLASSES)
    # print(i)
    return X_train, Y_train

def training_vis(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['acc']  # new version => hist.history['accuracy']
    val_acc = hist.history['val_acc']  # => hist.history['val_accuracy']

    # make a figure
    fig = plt.figure(figsize=(8, 4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss, label='train_loss')
    ax1.plot(val_loss, label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc, label='train_acc')
    ax2.plot(val_acc, label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    with open(r".\dataset\data\train.txt","r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    # 一次的训练集大小
    batch_size = 40
    # 90%用于训练，10%用于估计。
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val
    X_train, Y_train = generate_arrays_from_file(lines[:num_train])
    X_test, Y_test = generate_arrays_from_file(lines[num_train:])

    # 建立MobileNet模型
    model = MobileNet(input_shape=[HEIGHT, WIDTH, 3], classes=NUM_CLASSES)
    model.load_weights(".\models\mobilenet_1_0_224_tf_no_top.h5",by_name=True)

    # 学习率下降的方式，acc3次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1
    )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1
    )
    # 编译模型
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-3),metrics=['accuracy'])

    # 训练
    hist = model.fit(X_train, Y_train, nb_epoch=50, batch_size=50, validation_data=(X_test, Y_test),shuffle=True,
                     callbacks=[early_stopping, reduce_lr])
    # 测试集正确率
    cost, accuracy = model.evaluate(X_test, Y_test)
    print("accuracy:", accuracy)
  #  保存权重模型
  #   if(accuracy>0.85):
  #       model.save_weights(log_dir + 'MOBILENET_model.h5')
  #       print("文件已存入logs\n")
    training_vis(hist)