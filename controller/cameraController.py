from PyQt6.QtCore import QObject
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QMainWindow, QMessageBox

from utils.interence_thread import InferenceThread
from view.cameraView import cameraView
from repositories.cameraRepositories import cameraRepositories
from PyQt6 import QtGui
import time
#摄像头控制类
class cameraController(QObject):
    def __init__(self,mainWindow:QMainWindow):
        super().__init__()
        self.cV = cameraView()
        self.mW = mainWindow
        self.cR = cameraRepositories()

    #更新摄像头显示
    def update(self):
        self.cR.update(self.mW)

    #关闭摄像头
    def close(self):
        self.cR.close(self.mW)

    #开启摄像头
    def open(self):
        self.cR.open(self.mW)

    #展示摄像头视图
    def set_camera_detection(self):
        self.cV.set_camera_detection(self.mW)

    #摄像头图像回调展示
    def show_camera(self,return_image):
        pixmap = QtGui.QPixmap(return_image).fromImage(return_image)
        self.mW.original.set_image(pixmap)

    #摄像头推理启动方法
    def inference_start(self):
        self.mW.stat_label_FPS.setText("FPS:                          ")
        self.mW.mC.set_enable_False()
        self.mW.play_video.setEnabled(False)
        self.mW.timer.stop()
        self.mW.original.set_image(QPixmap())
        self.mW.camera.release()
        self.mW.open_camera.setEnabled(False)
        self.mW.close_camera.setEnabled(True)
        self.mW.image_detection.setEnabled(False)
        self.mW.video_detection.setEnabled(False)
        self.mW.camera_detection.setEnabled(False)
        self.mW.interence_thread = InferenceThread(None,self.mW.model_name,self.mW.pth_dir,run_type=self.mW.run_type)
        self.mW.interence_thread.finished_signal.connect(self.interence_finish)
        self.mW.interence_thread.progress_signal.connect(self.mW.mC.update_status)
        self.mW.interence_thread.error_report.connect(self.mW.mC.error_show)
        self.mW.close_camera.clicked.connect(self.mW.interence_thread.stop_running)
        self.mW.interence_thread.image_return.connect(self.mW.cC.show_camera)
        self.mW.interence_thread.start()
        self.mW.start = time.time()

    #摄像头结束回调函数
    def interence_finish(self):
        self.mW.end = time.time()
        self.mW.mC.set_enable_True()
        self.mW.open_camera.setEnabled(True)
        self.mW.original.set_image(QPixmap())
        self.mW.play_video.setEnabled(True)
        self.mW.image_detection.setEnabled(True)
        self.mW.video_detection.setEnabled(True)
        self.mW.camera_detection.setEnabled(True)
        QMessageBox.information(self.mW,"提示","推理完成")



