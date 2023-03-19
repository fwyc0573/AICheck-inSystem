from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
from scipy import misc
import tensorflow as tf
import numpy as np
import os
from utilss import file_processing,image_processing
import face_recognition
resize_width = 160
resize_height = 160

def face_net(model_path):
    face_net = face_recognition.facenetEmbedding(model_path)
    return face_net

def face_recognition_image(dataset_emb,names_list,face_net,image_path):
    #处理图像
    image = image_processing.read_image_gbk(image_path)
    image = image_processing.resize_image(image,resize_width,resize_height)
    tempPic = []
    tempPic.append(image)
    face_images = image_processing.get_prewhiten_images(tempPic)
    #进行编码
    pred_emb=face_net.get_embedding(face_images)
    #与npy文件中数据进行比对，返回欧式距离最小的
    pred_name,pred_score=compare_embadding(pred_emb, dataset_emb, names_list)#与库中数据进行比较
    # 在图像上绘制识别的结果
    show_info=[ n+':'+str(s)[:5] for n,s in zip(pred_name,pred_score)]
    image = cv2.putText(image, str(show_info), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 将BGR转为RGB
    cv2.imshow("result", image)
    cv2.waitKey(0)
    if pred_name[0] == "unknow":
        return -1
    else:
        return pred_name



def load_dataset(dataset_path,filename):
    embeddings=np.load(dataset_path)
    names_list=file_processing.read_data(filename,split=None,convertNum=False)
    print("加载人脸数据库成功")
    return embeddings,names_list


def compare_embadding(pred_emb, dataset_emb, names_list,threshold=0.65):
    # 为bounding_box 匹配标签
    pred_num = len(pred_emb)
    dataset_num = len(dataset_emb)
    pred_name = []
    pred_score=[]
    for i in range(pred_num):
        dist_list = []
        for j in range(dataset_num):
            dist = np.sqrt(np.sum(np.square(np.subtract(pred_emb[i, :], dataset_emb[j, :]))))
            dist_list.append(dist)
        min_value = min(dist_list)
        pred_score.append(min_value)
        if (min_value > threshold):
            pred_name.append('未知身份')
            print("未知身份")
        else:
            pred_name.append(names_list[dist_list.index(min_value)])
            print(names_list[dist_list.index(min_value)])
    return pred_name,pred_score

def main(embeddings,names_list,face_net):
    print("执行调用predict")
    image_path = r'D:\PyCharm2017\PROJECT\Pyqt\CAM\fyc.jpg'
    return face_recognition_image(embeddings, names_list, face_net,image_path)



if __name__=='__main__':
    print("执行调用predict")
    model_path=r'D:\PyCharm2017\PROJECT\Pyqt\models\20180408-102900'
    dataset_path=r'D:\PyCharm2017\PROJECT\Pyqt\dataset\emb\faceEmbedding.npy'
    filename=r'D:\PyCharm2017\PROJECT\Pyqt\dataset\emb\name.txt'
    image_path = r'D:\PyCharm2017\PROJECT\Pyqt\CAM\fyc.jpg'
    face_recognition_image(model_path, dataset_path, filename,image_path)