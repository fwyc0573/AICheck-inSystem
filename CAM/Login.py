import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDockWidget, QListWidget
from PyQt5.QtGui import *
from CAM.OboardCamDisp import Ui_MainWindow
from CAM.Camshow import CamShow
from PyQt5.QtWidgets import *
import sys
import CAM.SQLLINK

class LoginDlg(QDialog):
    def __init__(self, parent=None):
        super(LoginDlg, self).__init__(parent)
        usr = QLabel("用户：")
        pwd = QLabel("密码：")
        self.usrLineEdit = QLineEdit()
        self.pwdLineEdit = QLineEdit()
        self.pwdLineEdit.setEchoMode(QLineEdit.Password)

        gridLayout = QGridLayout()
        gridLayout.addWidget(usr, 0, 0, 1, 1)
        gridLayout.addWidget(pwd, 1, 0, 1, 1)
        gridLayout.addWidget(self.usrLineEdit, 0, 1, 1, 3);
        gridLayout.addWidget(self.pwdLineEdit, 1, 1, 1, 3);

        okBtn = QPushButton("确定")
        registerBtn = QPushButton("注册")
        btnLayout = QHBoxLayout()

        btnLayout.setSpacing(60)
        btnLayout.addWidget(okBtn)
        btnLayout.addWidget(registerBtn)

        dlgLayout = QVBoxLayout()
        dlgLayout.setContentsMargins(40, 40, 40, 40)
        dlgLayout.addLayout(gridLayout)
        dlgLayout.addStretch(40)
        dlgLayout.addLayout(btnLayout)

        self.setLayout(dlgLayout)
        okBtn.clicked.connect(self.accept)
        registerBtn.clicked.connect(self.register)
        self.setWindowTitle("人脸识别系签到系统登录")
        self.resize(700, 200)

    def accept(self):
        has_found = False
        self.account = self.usrLineEdit.text().strip()
        password = self.pwdLineEdit.text().strip()
        if(CAM.SQLLINK.checkAccount(self.account,password)==1):
            super(LoginDlg, self).accept()
            has_find = True
            return 1
        if has_found == False:
                QMessageBox.warning(self,
                                    "提示",
                                    "登录失败，请核对账号信息",
                                    QMessageBox.Yes)
                self.usrLineEdit.setFocus()

    def register(self):
        QMessageBox.warning(self,
                            "提示",
                            "确认注册？",
                            QMessageBox.Yes)
        has_found = False
        self.account = self.usrLineEdit.text().strip()
        password = self.pwdLineEdit.text().strip()
        if(CAM.SQLLINK.checkAccount(self.account,password)==1):
            has_find = True
            QMessageBox.warning(self,
                                "提示",
                                "该账户已被注册，请更换名称.",
                                QMessageBox.Yes)
            return 1
            self.usrLineEdit.setFocus()

        if has_found == False:
            # MYSQL部分
            CAM.SQLLINK.WriteRegisterIntoSQL(self.account, password)
            QMessageBox.warning(self,
                                "提示",
                                "注册成功!",
                                QMessageBox.Yes)


if __name__ == '__main__':  # 如果这个文件是主程序。
    app = QtWidgets.QApplication(sys.argv)  # QApplication相当于main函数，也就是整个程序（很多文件）的主入口函数。对于GUI程序必须至少有一个这样的实例来让程序运行。
    window = LoginDlg()  # 生成一个实例（对象）
    window.show()  # 有了实例，就得让它显示。这里的show()是QWidget的方法，用来显示窗口。
    if window.exec_() == QDialog.Accepted:
        the_window = CamShow()
        the_window.usr = window.account
        print(the_window.usr)
        the_window.show()
        sys.exit(app.exec_())