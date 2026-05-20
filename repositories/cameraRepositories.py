import cv2
from PyQt6.QtCore import QObject
from PyQt6.QtGui import QPixmap
from PyQt6 import QtGui

#摄像头资源管理类
class cameraRepositories(QObject):
    def __init__(self):
        super().__init__()

    # 更新摄像头显示
    def update(self,main):
        ret,frame = main.camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = QtGui.QImage(frame, frame.shape[1],frame.shape[0],QtGui.QImage.Format.Format_Grayscale8)
        pixmap = QtGui.QPixmap(img).fromImage(img)
        main.original.set_image(pixmap)

    # 关闭摄像头
    def close(self,main):
        main.timer.stop()
        main.original.set_image(QPixmap())
        main.camera.release()
        main.open_camera.setEnabled(True)
        main.close_camera.setEnabled(False)
        main.image_detection.setEnabled(True)
        main.video_detection.setEnabled(True)

    # 开启摄像头
    def open(self,main):
        main.camera = cv2.VideoCapture(0)
        main.timer.start(33)
        main.open_camera.setEnabled(False)
        main.close_camera.setEnabled(True)
        main.image_detection.setEnabled(False)
        main.video_detection.setEnabled(False)
