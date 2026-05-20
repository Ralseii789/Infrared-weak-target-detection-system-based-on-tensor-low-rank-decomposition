from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QMainWindow, QMessageBox
from otherWindow import history_record
from repositories.photoRepositories import photoRepositories
from utils.interence_thread import InferenceThread
from view.photoView import photoViewer
import time

#图像控制类
class photoController(QObject):
    def __init__(self,mainWindow:QMainWindow):
        super().__init__()
        self.mW = mainWindow
        self.pR = photoRepositories()
        self.pV = photoViewer()

    #图像列表项选中展示回调函数
    def selected_photo_show(self):
        self.pR.selected_photo_show(self.mW)

    #展示图像视图
    def set_photo_detection(self):
        self.pV.set_photo_detection(self.mW)

    #从文件夹中选取图片到文件列表
    def open_photos(self):
        self.pR.open_photos(self.mW)

    #选取整个文件夹的图片到文件列表
    def open_photo_dir(self):
        self.pR.open_photo_dir(self.mW)

    #保存图片到本地
    def save_as_resutls(self):
        self.pR.save_as_resutls(self.mW)

    #打开历史记录窗口
    def open_history(self):
        self.open_history = history_record.Ui_history_record()
        self.open_history.double_click_item.connect(self.add_history)
        self.open_history.show()

    #添加历史记录到图像列表的回调
    def add_history(self,history_item):
        self.mW.photo_list.addItem(history_item)

    #从图像列表中删除选中项
    def delete_from_index(self):
        selected_index = self.mW.photo_list.currentRow()
        self.mW.photo_list.takeItem(selected_index)

    #图像推理函数
    def inference_start(self):
        items_list = []
        if self.mW.photo_list.count() > 0 :
            self.mW.stat_label_FPS.setText("FPS:                          ")
            self.mW.mC.set_enable_False()
            for i in range(self.mW.photo_list.count()):
                items_list.append(self.mW.photo_list.item(i).text())
            self.mW.play_video.setEnabled(False)
            self.mW.image_detection.setEnabled(False)
            self.mW.video_detection.setEnabled(False)
            self.mW.camera_detection.setEnabled(False)
            self.interence_thread = InferenceThread(items_list, self.mW.model_name, self.mW.pth_dir,
                                                    run_type=self.mW.run_type)
            self.interence_thread.finished_signal.connect(self.interence_finish)
            self.interence_thread.progress_signal.connect(self.mW.mC.update_status)
            self.interence_thread.error_report.connect(self.mW.mC.error_show)
            self.interence_thread.start()
            self.mW.start = time.time()
        else:
            QMessageBox.warning(self.mW, "警告", "无可推理图片")
            return

    #图像推理结束回调
    def interence_finish(self):
        self.mW.end = time.time()
        self.mW.mC.set_enable_True()
        self.mW.open_camera.setEnabled(True)
        self.mW.play_video.setEnabled(True)
        self.mW.image_detection.setEnabled(True)
        self.mW.video_detection.setEnabled(True)
        self.mW.camera_detection.setEnabled(True)
        time_cost = self.mW.end - self.mW.start
        self.mW.stat_label_FPS.setText(f"FPS:{self.mW.photo_list.count() / time_cost:.2f}                            ")
        self.pR.save_history(self.mW)
        QMessageBox.information(self.mW, "提示", "推理完成")
