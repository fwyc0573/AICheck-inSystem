import numpy as np
import cv2
import os
import glob
from net.mtcnn import mtcnn
import facenet
# from keras.applications.imagenet_utils import preprocess_input
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

resize_width = 160
resize_height = 160

class facenetEmbedding:
    def __init__(self,model_path):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        #加载模型
        facenet.load_model(model_path)
        # Get input and output tensors
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.tf_embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    def get_embedding(self,images):
        feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
        embedding = self.sess.run(self.tf_embeddings, feed_dict=feed_dict)
        print("embedding完成.")
        return embedding

    def free(self):
        self.sess.close()


def get_images_list(image_dir,postfix=['*.jpg']):
    images_list=[]
    for format in postfix:
        image_format=os.path.join(image_dir,format)
        image_list=glob.glob(image_format)
        if not image_list==[]:
            images_list+=image_list
    images_list=sorted(images_list)
    return images_list

def getFilePathList(file_dir):
    #获取file_dir目录下，所有文本路径，包括子目录文件
    filePath_list = []
    for walk in os.walk(file_dir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list

def get_files_list(file_dir, postfix=None):
    file_list = []
    filePath_list = getFilePathList(file_dir)
    if postfix is None:
        file_list = filePath_list
    else:
        postfix = [p.split('.')[-1] for p in postfix]
        for file in filePath_list:
            basename = os.path.basename(file)  # 获得路径下的文件名
            postfix_name = basename.split('.')[-1]
            if postfix_name in postfix:
                file_list.append(file)
    file_list.sort()
    return file_list

def gen_files_labels(files_dir,postfix=None):
    # 获取files_dir路径下所有文件路径，以及labels
    filePath_list=get_files_list(files_dir, postfix=postfix)
    print("files nums:{}".format(len(filePath_list)))
    # 获取所有样本标签
    label_list = []
    for filePath in filePath_list:
        label = filePath.split(os.sep)[-2]
        label_list.append(label)
    labels_set = list(set(label_list))
    print("labels:{}".format(labels_set))
    return filePath_list, label_list


def create_face_embedding(model_path,dataset_path,out_emb_path,out_filename):
    threshold = [0.5, 0.6, 0.8]  # mtcnn对应阈值
    embeddings=[] # 用于保存人脸特征数据库
    label_list=[] # 人脸名称，与embeddings对应
    image_list,names_list = gen_files_labels(dataset_path, postfix=['*.jpg', '*.png'])

    # 初始化facenet
    face_net = facenetEmbedding(model_path)

    for image_path in image_list:
        basename = os.path.basename(image_path)
        names = basename.split('_')[0]
        names_list.append(names)
    print("names_list:",names_list)
    print("image_list:", image_list)

    model = mtcnn()  # 人脸检测
    for image_path, name in zip(image_list, names_list):
        draw = cv2.imread(image_path)
        print("image_path:",image_path)
        height,width,_ = np.shape(draw)
        draw_rgb = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)
        # cv2.imshow("66",draw_rgb)
        # cv2.waitKey()
        # 检测人脸
        rectangles = model.detectFace(draw_rgb,threshold)
        if len(rectangles)==0:
            print(len(rectangles))
        # print(rectangles)
        rectangles = np.array(rectangles, dtype=np.int32)
        print(rectangles)
        w = rectangles[0][2] - rectangles[0][0]#宽
        h = rectangles[0][3] - rectangles[0][1]#高
        #裁剪为正方形，保留较大的边
        if(w>h):
            rectangles[0][3] += w-h#先扩充，再整体上移
            rectangles[0][1] -= int((w-h)/2)
            rectangles[0][3] -= int((w - h) / 2)
            if(rectangles[0][1]<0):
                rectangles[0][1]=0
            if(rectangles[0][3]>height):
                rectangles[0][3]=height
        else:
            rectangles[0][2] += h-w#先扩充，再整体左移
            rectangles[0][0] -= int((h-w)/2)
            rectangles[0][2] -=int((h-w)/2)
            if(rectangles[0][0]<0):
                rectangles[0][0]=0
            if(rectangles[0][2]>width):
                rectangles[0][2]=width
        cut_pic = draw_rgb[rectangles[0][1]:rectangles[0][3],rectangles[0][0]:rectangles[0][2]]
        src_img = cv2.resize(cut_pic, (resize_width, resize_height))
        new_img = preprocess_input(np.reshape(np.array(src_img, np.float64), [1, resize_width, resize_height, 3]))
        # 获得人脸特征并保存
        pred_emb = face_net.get_embedding(new_img)
        embeddings.append(pred_emb)
        label_list.append(name)

    #进行数据的保存
    embeddings = np.asarray(embeddings)
    #faceEmbedding.npy
    np.save(out_emb_path, embeddings)
    #写入name.txt
    with open(out_filename, mode='w', encoding='utf-8') as f:
        for line in label_list:
            # 将list转为string
            f.write(str(line)+"\n")


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ == '__main__':
    model_path = r'models/20180408-102900'
    dataset_path=r'dataset/images'
    out_emb_path = r'dataset/emb/faceEmbedding.npy'
    out_filename = r'dataset/emb/name.txt'
    create_face_embedding(model_path, dataset_path,out_emb_path, out_filename)