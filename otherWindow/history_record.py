import os
import sys
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QApplication, QWidget,QMessageBox
from PyQt6.QtCore import QFile, QTextStream, QIODevice, QUrl,pyqtSignal

#历史记录窗口
class Ui_history_record(QWidget):
    double_click_item = pyqtSignal(str)
    def __init__(self):
        super(Ui_history_record, self).__init__()
        self.setupUi(self)
        self.read_history()
    def setupUi(self, history_record):
        history_record.setObjectName("history_record")
        history_record.resize(526, 817)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(".\\icon/历史.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        history_record.setWindowIcon(icon)
        self.frame = QtWidgets.QFrame(parent=history_record)
        self.frame.setGeometry(QtCore.QRect(0, 0, 731, 971))
        self.frame.setStyleSheet("QFrame{\n"
                                 "    background-color : rgb(50, 50, 50)\n"
                                 "}")
        self.frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame.setObjectName("frame")
        self.layoutWidget = QtWidgets.QWidget(parent=self.frame)
        self.layoutWidget.setGeometry(QtCore.QRect(90, 760, 341, 37))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.clear_history = QtWidgets.QPushButton(parent=self.layoutWidget)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(".\\icon/41清空.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.clear_history.setIcon(icon1)
        self.clear_history.setObjectName("clear_history")
        self.horizontalLayout.addWidget(self.clear_history)
        self.history_file = QtWidgets.QPushButton(parent=self.layoutWidget)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(".\\icon/文件夹.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.history_file.setIcon(icon2)
        self.history_file.setObjectName("history_file")
        self.horizontalLayout.addWidget(self.history_file)
        self.layoutWidget1 = QtWidgets.QWidget(parent=self.frame)
        self.layoutWidget1.setGeometry(QtCore.QRect(20, 40, 491, 691))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(parent=self.layoutWidget1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.history_list = QtWidgets.QListWidget(parent=self.layoutWidget1)
        self.history_list.setObjectName("history_list")
        self.verticalLayout.addWidget(self.history_list)

        self.retranslateUi(history_record)
        QtCore.QMetaObject.connectSlotsByName(history_record)
        # 信号与槽的连接
        self.clear_history.clicked.connect(self.history_clear)
        self.history_file.clicked.connect(self.open_dir)
        self.history_list.itemDoubleClicked.connect(self.double_click_emit)

    def double_click_emit(self):
        self.double_click_item.emit(self.history_list.currentItem().text())
        self.close()

    def history_clear(self):
        reply = QMessageBox().question(
            self,
            '清空',
            '确定要清空历史记录吗',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.history_list.clear()
            file = QFile("history_list.txt")
            file.open(QIODevice.OpenModeFlag.WriteOnly | QIODevice.OpenModeFlag.Truncate)
            file.close()

    def open_dir(self):
        current_dir = os.getcwd()
        url = QUrl.fromLocalFile(current_dir)
        QDesktopServices.openUrl(url)

    def read_history(self):
        file = QFile("history_list.txt")
        file.open(QIODevice.OpenModeFlag.ReadOnly | QIODevice.OpenModeFlag.Text)
        stream = QTextStream(file)
        contents = stream.readAll()
        self.history_list.clear()
        lines = contents.split("\n")
        self.history_list.addItems(lines)
        file.close()
    def retranslateUi(self, history_record):
        _translate = QtCore.QCoreApplication.translate
        history_record.setWindowTitle(_translate("history_record", "history_record"))
        self.clear_history.setText(_translate("history_record", "清空历史"))
        self.history_file.setText(_translate("history_record", "打开所在文件夹"))
        self.label.setText(_translate("history_record", "历史记录"))

