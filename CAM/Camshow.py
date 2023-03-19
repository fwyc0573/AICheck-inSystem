import datetime
import sys
import time

import qimage2ndarray
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from keras.applications.imagenet_utils import preprocess_input

import CAM.SQLLINK
import predict
from CAM.OboardCamDisp import Ui_MainWindow
from net.cnn import *
from net.mobileNet import MobileNet
from net.mtcnn import mtcnn

face_xml = cv2.CascadeClassifier('C:\\Users\\FYC\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
face_xml.load('C:\\Users\\FYC\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
eye_xml = cv2.CascadeClassifier('C:\\Users\\FYC\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml')
eye_xml.load('C:\\Users\\FYC\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml')

model_path = r'D:\PyCharm2017\PROJECT\Pyqt\models\20180408-102900'
dataset_path = r'D:\PyCharm2017\PROJECT\Pyqt\dataset\emb\faceEmbedding.npy'
filename = r'D:\PyCharm2017\PROJECT\Pyqt\dataset\emb\name.txt'

class CamShow(QMainWindow,Ui_MainWindow):
    def __del__(self):
        try:
            self.camera.release()  # 释放资源
        except:
            return
    def __init__(self,parent=None):
        super(CamShow,self).__init__(parent)
        self.has_stop = False
        self.is_life = 0
        self.can_mctnn_detct = False

        #模型数据导入
        self.embeddings,self.names_list = predict.load_dataset(dataset_path,filename)
        self.face_net = predict.face_net(model_path)#facenet身份识别
        self.cnn = CNN(input_shape=[160,160,3],classes=2)#cnn口罩分类
        self.cnn.load_weights(r"D:\PyCharm2017\PROJECT\Pyqt\logs\CNN_model.h5")#cnn网络权重导入
        self.mask_model = MobileNet(input_shape=[160,160,3],classes=2)
        self.mask_model.load_weights(r"D:\PyCharm2017\PROJECT\Pyqt\logs\MOBILENET_model.h5")
        self.mtcnn_model = mtcnn()#人脸检测
        self.threshold = [0.5, 0.6, 0.8]#mtcnn杜对应阈值
        self.class_names = get_class()#读取txt划分出的多个种类



        self.dict = {}
        self.setupUi(self)
        self.PrepSliders()
        self.PrepWidgets()
        self.PrepParameters()
        self.CallBackFunctions()
        self.Timer=QTimer()
        self.Timer.timeout.connect(self.TimerOutFun)

    def PrepSliders(self):
        self.RedColorSld.valueChanged.connect(self.RedColorSpB.setValue)
        self.RedColorSpB.valueChanged.connect(self.RedColorSld.setValue)
        self.GreenColorSld.valueChanged.connect(self.GreenColorSpB.setValue)
        self.GreenColorSpB.valueChanged.connect(self.GreenColorSld.setValue)
        self.BlueColorSld.valueChanged.connect(self.BlueColorSpB.setValue)
        self.BlueColorSpB.valueChanged.connect(self.BlueColorSld.setValue)
        self.ExpTimeSld.valueChanged.connect(self.ExpTimeSpB.setValue)
        self.ExpTimeSpB.valueChanged.connect(self.ExpTimeSld.setValue)
        self.GainSld.valueChanged.connect(self.GainSpB.setValue)
        self.GainSpB.valueChanged.connect(self.GainSld.setValue)
        self.BrightSld.valueChanged.connect(self.BrightSpB.setValue)
        self.BrightSpB.valueChanged.connect(self.BrightSld.setValue)
        self.ContrastSld.valueChanged.connect(self.ContrastSpB.setValue)
        self.ContrastSpB.valueChanged.connect(self.ContrastSld.setValue)
    def PrepWidgets(self):
        self.PrepCamera()
        self.StopBt.setEnabled(False)
        self.ImportBt.setEnabled(False)
        self.RecordBt.setEnabled(False)
        self.GrayImgCkB.setEnabled(False)
        self.RedColorSld.setEnabled(False)
        self.RedColorSpB.setEnabled(False)
        self.GreenColorSld.setEnabled(False)
        self.GreenColorSpB.setEnabled(False)
        self.BlueColorSld.setEnabled(False)
        self.BlueColorSpB.setEnabled(False)
        self.ExpTimeSld.setEnabled(False)
        self.ExpTimeSpB.setEnabled(False)
        self.GainSld.setEnabled(False)
        self.GainSpB.setEnabled(False)
        self.BrightSld.setEnabled(False)
        self.BrightSpB.setEnabled(False)
        self.ContrastSld.setEnabled(False)
        self.ContrastSpB.setEnabled(False)
        self.FaceBt.setEnabled(False)
        self.InfoBt.setEnabled(False)
    def PrepCamera(self):
        try:
            self.camera=cv2.VideoCapture(0)
            self.MsgTE.clear()
            self.MsgTE.append('Oboard camera connected.')
            self.MsgTE.setPlainText()

        except Exception as e:
            self.MsgTE.clear()
            self.MsgTE.append(str(e))
    def PrepParameters(self):
        self.usr = ""
        self.RecordFlag = 0
        self.can_counteye = False
        self.RecordPath = 'H:\\'
        self.FilePathLE.setText(self.RecordPath)
        self.Image_num=0
        self.R=1
        self.G=1
        self.B=1
        self.can_detect = False
        self.ExpTimeSld.setValue(self.camera.get(15))
        self.SetExposure()
        self.GainSld.setValue(self.camera.get(14))
        self.SetGain()
        self.BrightSld.setValue(self.camera.get(10))
        self.SetBrightness()
        self.ContrastSld.setValue(self.camera.get(11))
        self.SetContrast()
        self.MsgTE.clear()
    def CallBackFunctions(self):
        self.FilePathBt.clicked.connect(self.SetFilePath)
        self.ShowBt.clicked.connect(self.StartCamera)#开始
        #self.LifeBt.clicked.connect(self.LifeCheck)#活体检测
        self.StopBt.clicked.connect(self.StopCamera)#暂停
        self.RecordBt.clicked.connect(self.RecordCamera)#录像
        self.ExitBt.clicked.connect(self.ExitApp)#退出
        self.FaceBt.clicked.connect(self.FaceDetect)#人脸识别
        self.InfoBt.clicked.connect(self.LifeCheck)#打卡签到，首先会进行活体检测，满足条件会调用打卡记录函数
        self.ImportBt.clicked.connect(self.mask_check)  #口罩检测
        self.GrayImgCkB.stateChanged.connect(self.SetGray)
        self.ExpTimeSld.valueChanged.connect(self.SetExposure)
        self.GainSld.valueChanged.connect(self.SetGain)#灰度化
        self.BrightSld.valueChanged.connect(self.SetBrightness)
        self.ContrastSld.valueChanged.connect(self.SetContrast)
        self.RedColorSld.valueChanged.connect(self.SetR)
        self.GreenColorSld.valueChanged.connect(self.SetG)
        self.BlueColorSld.valueChanged.connect(self.SetB)
    def SetR(self):
        R=self.RedColorSld.value()
        self.R=R/255
    def SetG(self):
        G=self.GreenColorSld.value()
        self.G=G/255
    def SetB(self):
        B=self.BlueColorSld.value()
        self.B=B/255
    def SetContrast(self):
        contrast_toset=self.ContrastSld.value()
        try:
            self.camera.set(11,contrast_toset)
            self.MsgTE.setPlainText('The contrast is set to ' + str(self.camera.get(11)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))
    def SetBrightness(self):
        brightness_toset=self.BrightSld.value()
        try:
            self.camera.set(10,brightness_toset)
            self.MsgTE.setPlainText('The brightness is set to ' + str(self.camera.get(10)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))
    def SetGain(self):
        gain_toset=self.GainSld.value()
        try:
            self.camera.set(14,gain_toset)
            self.MsgTE.setPlainText('The gain is set to '+str(self.camera.get(14)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))
    def SetExposure(self):
        try:
            exposure_time_toset=self.ExpTimeSld.value()
            self.camera.set(15,exposure_time_toset)
            self.MsgTE.setPlainText('The exposure time is set to '+str(self.camera.get(15)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))
    def SetGray(self):
        if self.GrayImgCkB.isChecked():
            self.RedColorSld.setEnabled(False)
            self.RedColorSpB.setEnabled(False)
            self.GreenColorSld.setEnabled(False)
            self.GreenColorSpB.setEnabled(False)
            self.BlueColorSld.setEnabled(False)
            self.BlueColorSpB.setEnabled(False)
        else:
            self.RedColorSld.setEnabled(True)
            self.RedColorSpB.setEnabled(True)
            self.GreenColorSld.setEnabled(True)
            self.GreenColorSpB.setEnabled(True)
            self.BlueColorSld.setEnabled(True)
            self.BlueColorSpB.setEnabled(True)
    def StartCamera(self):
        self.ShowBt.setEnabled(False)
        self.ImportBt.setEnabled(True)
        self.FaceBt.setEnabled(True)
        self.StopBt.setEnabled(True)
        self.RecordBt.setEnabled(True)
        self.GrayImgCkB.setEnabled(True)
        if self.GrayImgCkB.isChecked()==0:
            self.RedColorSld.setEnabled(True)
            self.RedColorSpB.setEnabled(True)
            self.GreenColorSld.setEnabled(True)
            self.GreenColorSpB.setEnabled(True)
            self.BlueColorSld.setEnabled(True)
            self.BlueColorSpB.setEnabled(True)
        self.ExpTimeSld.setEnabled(True)
        self.ExpTimeSpB.setEnabled(True)
        self.GainSld.setEnabled(True)
        self.GainSpB.setEnabled(True)
        self.BrightSld.setEnabled(True)
        self.BrightSpB.setEnabled(True)
        self.ContrastSld.setEnabled(True)
        self.ContrastSpB.setEnabled(True)
        self.RecordBt.setText('录像')

        self.Timer.start(1)
        self.timelb=time.clock()
    def SetFilePath(self):
        dirname = QFileDialog.getExistingDirectory(self, "浏览", '.')
        if dirname:
            self.FilePathLE.setText(dirname)
            self.RecordPath=dirname+'/'
    def mask_check(self):
        if(self.can_mctnn_detct == False):
            self.can_mctnn_detct = True
            self.can_detect = False
            self.FaceBt.setEnabled(True)
            self.ImportBt.setEnabled(False)
            self.MsgTE.setPlainText('系统开启口罩检测功能。 ')
        else:
            self.can_mctnn_detct = False
            self.can_detect = True
            self.FaceBt.setEnabled(False)
            self.ImportBt.setEnabled(True)
            self.MsgTE.setPlainText('系统关闭口罩检测功能。 ')

    def TimerOutFun(self):
        success,img=self.camera.read()
        if success:
            self.Image = self.ColorAdjust(img)
            #harr级联器检测
            if self.can_detect == True:
                self.gray = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
                self.Image_copy = self.Image.copy()
                self.faces = face_xml.detectMultiScale(self.gray, 1.3, 5)
                #人脸、人眼基本检测
                for (x, y, w, h) in self.faces:
                     cv2.rectangle(self.Image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                     self.cut_pic = self.Image_copy[y:y + w, x:x + h ]
                     self.roi_gray = self.gray[y:y + h, x:x + w]
                     self.eye = eye_xml.detectMultiScale(self.roi_gray, 1.8, 5)
                    #活体检测部分
                     if(self.can_counteye):
                       dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                       current_time = int(time.mktime(time.strptime(dt, "%Y-%m-%d %H:%M:%S")))
                       if(current_time - self.record_time <=4):
                         if (len(self.eye) == 0):#统计睁闭眼
                             self.sum_closeeyes += 1
                         else:
                             self.sum_openeyes += 1
                         if(len(self.faces)>0):#统计脸出现的次数，因为只有脸被检测到眼睛才会被检测到；防止用户摇晃
                             self.sum_faces += 1
                       else:
                           self.can_counteye = False
                           self.is_life = 1
                           if (self.sum_closeeyes > 5 and self.sum_openeyes > 5 and self.sum_faces > 53): #通过脸部数量来防止照片晃动情况
                               QMessageBox.warning(self,
                                                   "提示",
                                                   "眨眼检测成功！",
                                                   QMessageBox.Yes)
                               self.StopCamera()
                               cv2.imwrite('fyc.jpg', self.cut_pic)
                               # print("图片保存成功")
                               self.DailyAttendance()
                           else:
                               self.is_life = -1
                               QMessageBox.warning(self,
                                                   "警告",
                                                   "眨眼检测失败，请重新尝试！",
                                                   QMessageBox.Yes)
                     roi_color = self.Image[y:y + h, x:x + w]
                     for (ex, ey, ew, eh) in self.eye:
                          cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            #mctnn检测
            elif self.can_mctnn_detct == True:
                # 检测口罩
                face = self.mctnn_rec(self.Image)
            self.DispImg()
            self.Image_num+=1
            if self.RecordFlag:
                self.video_writer.write(img)
            if self.Image_num%10==9:
                frame_rate=10/(time.clock()-self.timelb)
                self.FmRateLCD.display(frame_rate)
                self.timelb=time.clock()
                #获取视频显示画面的长宽尺寸
                self.ImgWidthLCD.display(self.camera.get(3))
                self.ImgHeightLCD.display(self.camera.get(4))
        else:
            self.MsgTE.clear()
            self.MsgTE.setPlainText('Image obtaining failed.')

    def ColorAdjust(self,img):
        try:
            B=img[:,:,0]
            G=img[:,:,1]
            R=img[:,:,2]
            B=B*self.B
            G=G*self.G
            R=R*self.R
            img1=img
            img1[:,:,0]=B
            img1[:,:,1]=G
            img1[:,:,2]=R
            return img1
        except Exception as e:
            self.MsgTE.setPlainText(str(e))
    def DispImg(self):
        if self.GrayImgCkB.isChecked():
            img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        qimg = qimage2ndarray.array2qimage(img)
        self.DispLb.setPixmap(QPixmap(qimg))
        self.DispLb.show()
    def LifeCheck(self):
     if self.can_detect:
       dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
       current_time = int(time.mktime(time.strptime(dt, "%Y-%m-%d %H:%M:%S")))
       self.record_time = current_time
       self.sum_closeeyes = 0
       self.sum_openeyes = 0
       self.sum_faces = 0
       self.can_counteye = True
       QMessageBox.warning(self,
                           "提示",
                           "开启眨眼检测，请保持原位眨动眼睛2-3秒",
                           QMessageBox.Yes)
    def StopCamera(self):
        if self.StopBt.text()=='暂停':
            self.has_stop = True
            self.MsgTE.clear()
            self.FaceBt.setEnabled(False)
            self.ImportBt.setEnabled(False)
            self.StopBt.setText('继续')
            self.RecordBt.setText('保存')
            self.Timer.stop()
        elif self.StopBt.text()=='继续':
            self.has_stop = False
            self.MsgTE.clear()
            self.FaceBt.setEnabled(True)
            self.ImportBt.setEnabled(True)
            self.StopBt.setText('暂停')
            self.RecordBt.setText('录像')
            self.Timer.start(1)
    def RecordCamera(self):
        tag=self.RecordBt.text()
        if tag=='保存':
            try:
                image_name=self.RecordPath+'image'+time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))+'.jpg'
                print(image_name)
                cv2.imwrite(image_name, self.Image)
                self.MsgTE.clear()
                self.MsgTE.setPlainText('Image saved.')
            except Exception as e:
                self.MsgTE.clear()
                self.MsgTE.setPlainText(str(e))
        elif tag=='录像':
            self.RecordBt.setText('停止')
            video_name = self.RecordPath + 'video' + time.strftime('%Y%m%d%H%M%S',time.localtime(time.time())) + '.avi'
            fps = self.FmRateLCD.value()
            size = (self.Image.shape[1],self.Image.shape[0])
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self.video_writer = cv2.VideoWriter(video_name, fourcc,self.camera.get(5), size)
            self.RecordFlag=1
            self.MsgTE.setPlainText('Video recording...')
            self.StopBt.setEnabled(False)
            self.ExitBt.setEnabled(False)
        elif tag == '停止':
            self.RecordBt.setText('录像')
            self.video_writer.release()
            self.RecordFlag = 0
            self.MsgTE.setPlainText('Video saved.')
            self.StopBt.setEnabled(True)
            self.ExitBt.setEnabled(True)
    def ExitApp(self):
        self.Timer.Stop()
        self.camera.release()
        self.MsgTE.setPlainText('Exiting the application..')
        QCoreApplication.quit()
    def msg(self):
        # 使用infomation信息框
        reply = QMessageBox.information(self, "提示", "未搜索到相关信息，请检查是否录入", QMessageBox.Yes)
    def FaceDetect(self):
        if self.can_detect == True:
            self.can_detect = False
            self.can_mctnn_detct = True
            self.FaceBt.setEnabled(True)
            self.InfoBt.setEnabled(False)
            self.ImportBt.setEnabled(False)
            self.MsgTE.setPlainText('系统关闭检测人脸功能。 ')
        else:
            self.can_detect = True
            self.can_mctnn_detct = False
            self.FaceBt.setEnabled(False)
            self.InfoBt.setEnabled(True)
            self.ImportBt.setEnabled(True)
            self.MsgTE.setPlainText('系统开启检测人脸功能。')

    def DailyAttendance(self):
     print("进入DailyAttendance")
     if self.has_stop == True and self.can_detect == True and len(self.faces) == 1:#当同时开启人脸识别和暂停功能并且屏幕中检测到一张人脸时才进行下一步判断
       # does_find =  CAM.checkTry.main(self.cut_pic)
       pred_name = predict.main(self.embeddings, self.names_list, self.face_net)
       if pred_name != -1:
           curr_time = datetime.datetime.now()
           time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')  # 2019-07-06 15:50:12
           # MYSQL部分
           CAM.SQLLINK.WriteIntoSQL(time_str, pred_name)
           QMessageBox.warning(self,
                               "提示",
                               "签到成功!",
                               QMessageBox.Yes)
           self.StartCamera()
           self.InfoBt.setEnabled(False)
       else:#与库中图像差距都比较大
           QMessageBox.warning(self,
                               "警告",
                               "身份验证失败，未知身份！",
                               QMessageBox.Yes)
     elif self.has_stop == False:#暂停未开启
         QMessageBox.warning(self,
                             "警告",
                             "请暂停确认画面",
                             QMessageBox.Yes)
     elif self.can_detect == False:#人脸识别未开启
         QMessageBox.warning(self,
                             "警告",
                             "请先开启人脸识别",
                             QMessageBox.Yes)
     elif self.has_stop == True and self.can_detect == True and len(self.faces) == 0 :#人脸数目不符合要求
         QMessageBox.warning(self,
                             "警告",
                             "检测到的人脸数目不符合要求，请调整位置!",
                             QMessageBox.Yes)

    def mctnn_rec(self,draw):
        height,width,_ = np.shape(draw)
        draw_rgb = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)

        # 检测人脸
        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)
        if len(rectangles)==0:
            return
        rectangles = np.array(rectangles, dtype=np.int32)
        print(rectangles)
        w = rectangles[0][2] - rectangles[0][0]#宽
        h = rectangles[0][3] - rectangles[0][1]#高
        #裁剪为正方形，保留较大的边
        if(w>h):
            rectangles[0][3] += w-h#先扩充，再整体上移
            rectangles[0][1] -= int((w-h)/2)
            rectangles[0][3] -= int((w - h) / 2)
        else:
            rectangles[0][2] += h-w#先扩充，再整体左移
            rectangles[0][0] -= int((h-w)/2)
            rectangles[0][2] -=int((h-w)/2)

        cv2.rectangle(self.Image, (rectangles[0][0], rectangles[0][1]), (rectangles[0][2], rectangles[0][3]), (0, 0, 255), 2)
        cut_pic = self.Image[rectangles[0][1]:rectangles[0][3],rectangles[0][0]:rectangles[0][2]]
        src_img = cv2.cvtColor(cut_pic, cv2.COLOR_BGR2RGB)
        src_img = cv2.resize(src_img, (HEIGHT, WIDTH))
        cv2.imshow('Video2', src_img)
        new_img = preprocess_input(np.reshape(np.array(src_img, np.float64), [1, HEIGHT, WIDTH, 3]))#归一化
        classes = self.class_names[np.argmax(self.cnn.predict(new_img)[0])]#CNN预测
        #classes = self.class_names[np.argmax(self.mask_model.predict(new_img)[0])]#mobilenet预测
        cv2.putText(self.Image, classes, (rectangles[0][0], rectangles[0][3] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        # print("cnn|model.predict(img)[0]:", self.cnn.predict(new_img)[0])  # 预测结果
        print("mobilenet|model.predict(img)[0]:", self.mask_model.predict(new_img)[0])  # 预测结果
        print(classes)


def get_class():
    classes_path = os.path.expanduser(r"D:\PyCharm2017\PROJECT\Pyqt\models\classes.txt")
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def main():
    app = QApplication(sys.argv)
    ui=CamShow()
    ui.show()
    sys.exit(app.exec_())