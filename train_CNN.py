from keras.utils import np_utils
from keras.optimizers import Adam
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import keras
from keras.callbacks import  EarlyStopping, ReduceLROnPlateau,TensorBoard
from net.cnn import *

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
        #读取正方形图像
        img = np.array(letterbox_image(img, [HEIGHT, WIDTH]), dtype=np.float64)
        X_train.append(img)
        Y_train.append(lines[b].split(';')[1])
        # print(b)
    # 处理图像
    X_train = preprocess_input(np.array(X_train).reshape(-1, HEIGHT, WIDTH, 3))
    Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=NUM_CLASSES)
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

# 模型保存的位置
log_dir = "./logs/"
# 打开数据集的txt
with open(r".\dataset\data\train.txt","r") as f:
    lines = f.readlines()

# 打乱txt行
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
# 一次的训练集大小
batch_size = 100

# 90%用于训练，10%用于测试。
num_val = int(len(lines) * 0.1)
num_train = len(lines) - num_val

X_train, Y_train= generate_arrays_from_file(lines[:num_train])
X_test, Y_test = generate_arrays_from_file(lines[num_train:])
print(len(X_train),len(Y_train))
print(len(X_test),len(Y_test))

#构建CNN模型
model = CNN(input_shape=[160,160,3],classes=2)

# 早停设置，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    verbose=1
)

# 学习率下降的方式，acc5次不下降就下降学习率继续训练
reduce_lr = ReduceLROnPlateau(
    monitor='acc',
    factor=0.5,
    patience=5,
    verbose=1
)

# Tensorboard= TensorBoard(log_dir="./logs", histogram_freq=1,write_grads=True, write_images=True)#开启可视化记录,

#编译模型
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
#训练
hist  = model.fit(X_train, Y_train, nb_epoch=50, batch_size=50,shuffle=True,validation_data=(X_test,Y_test),callbacks=[early_stopping, reduce_lr],)
#测试集正确率
cost, accuracy = model.evaluate(X_test, Y_test)
print("accuracy:", accuracy)
#保存权重模型
# if(accuracy>0.85):
#     model.save_weights(log_dir + 'CNN_model.h5')
#     print("文件已存入logs\n")
#绘制acc-loss曲线
training_vis(hist)