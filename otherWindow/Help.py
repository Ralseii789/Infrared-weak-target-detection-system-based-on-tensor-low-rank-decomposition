import sys

from PyQt6 import QtCore, QtGui, QtWidgets

from PyQt6.QtWidgets import QApplication, QWidget

#帮助窗口
class Ui_Help(QWidget):
    def __init__(self):
        super(Ui_Help, self).__init__()
        self.setupUi(self)
    def setupUi(self, Help):
        Help.setObjectName("Help")
        Help.resize(583, 790)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(".\\icon/帮助.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        Help.setWindowIcon(icon)
        self.frame = QtWidgets.QFrame(parent=Help)
        self.frame.setGeometry(QtCore.QRect(-1, -1, 581, 801))
        self.frame.setStyleSheet("QFrame{\n"
"    background-color : rgb(50, 50, 50)\n"
"}\n"
"")
        self.frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame.setObjectName("frame")
        self.OK = QtWidgets.QPushButton(parent=self.frame)
        self.OK.setGeometry(QtCore.QRect(230, 750, 112, 34))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.OK.sizePolicy().hasHeightForWidth())
        self.OK.setSizePolicy(sizePolicy)
        self.OK.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(".\\icon/确定.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.OK.setIcon(icon1)
        self.OK.setObjectName("OK")
        self.help_details = QtWidgets.QTextEdit(parent=self.frame)
        self.help_details.setGeometry(QtCore.QRect(1, 1, 581, 741))
        self.help_details.setObjectName("help_details")
        self.help_details.setReadOnly(True)
        self.retranslateUi(Help)
        QtCore.QMetaObject.connectSlotsByName(Help)
        # 信号与槽函数
        self.OK.clicked.connect(self.close)
    def retranslateUi(self, Help):
        _translate = QtCore.QCoreApplication.translate
        Help.setWindowTitle(_translate("Help", "帮助"))
        self.OK.setText(_translate("Help", "确定"))
        self.help_details.setHtml(_translate("Help", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">红外弱小目标检测系统</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">版本：2.0</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">作者：2022217204沈德俊</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">基于Pytorch和PyQt6开发</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">使用说明：</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">1. “打开图片”-&gt;添加单/多张图片到图片列表</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">2. “打开文件夹”-&gt;添加一个文件夹的图片到图片列表</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">3. “控制面板”-&gt;调节参数</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">4. “历史记录”-&gt;用于重新加载过去用于推理的图片(双击)</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">5. “保存”-&gt;用于保存当前结果（另存为）</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">6. 结果展示在主界面的四个标签中</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">7. 图像支持滚轮缩放和鼠标拖动</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">8. “重置”->重置图片大小</span></p>\n"                                                     
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">9. “上一张”，“下一张”按钮用于切换次此推理的不同图片</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">10. 点击图片列表的项也能切换相应的图片</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\"> </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">2.0:</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">1. 点击菜单栏上的“图像”，“视频”和“摄像头”可以切换图像来源</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">2. “图片转视频”->选择整个文件夹的图片序列转成视频</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\"> </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">3.0: 优化代码结构</span></p></body></html>"))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    Ui = Ui_Help()
    Ui.show()
    sys.exit(app.exec())