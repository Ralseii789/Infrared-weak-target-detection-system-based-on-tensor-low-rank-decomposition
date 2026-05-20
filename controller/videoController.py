import os
from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QMainWindow, QMessageBox
from repositories.videoRepositories import videoRepositories
from utils.interence_thread import InferenceThread
from view.videoView import videoView
import time
#视频控制类
class videoController(QObject):
    def __init__(self,main:QMainWindow):
        super().__init__()
        self.vR = videoRepositories()
        self.vV = videoView()
        self.mW = main

    #视频图像展示函数
    def update_video(self):
        self.vR.update_video(self.mW)

    #文件夹图像转视频函数
    def open_img2video(self):
        self.vR.open_img2video(self.mW)

    #选中视频的封面展示回调函数
    def selected_video_surface_show(self):
        self.vR.selected_video_surface_show(self.mW)

    #展示视频视图
    def set_video_detection(self):
        self.vV.set_video_detection(self.mW)

    #选取整个文件夹的视频到视频列表
    def _open_video_folder(self):
        self.vR._open_video_folder(self.mW)

    #选取视频到视频列表
    def open_video(self):
        self.vR.open_video(self.mW)

    #获取视频总帧数
    def set_now_video_fps_count(self, count):
        self.mW.video_fps_count = count

    #从视频列表中删除选定视频
    def delete_from_index_video(self):
        selected_index = self.mW.photo_list_video.currentRow()
        self.mW.photo_list_video.takeItem(selected_index)

    #视频播放
    def video_play(self):
        if self.mW.run_type == 1:
            items = self.mW.photo_list_video.selectedItems()
            if not items:
                QMessageBox.warning(self.mW, "警告", "无待播放视频")
                return
            path = items[0].text()
            item_name = items[0].text().split('/')[-1].split('.')[0]
            item_postfixs = items[0].text().split('/')[-1].split('.')[1]
            bbox_path = os.path.join("./result_video", item_name, f'bbox.{item_postfixs}')
            self.vR.video_play(path,bbox_path,self.mW)

    #视频停止
    def video_stop(self):
        if self.mW.run_type == 1:
            self.mW.play_video.setEnabled(True)
            self.mW.pause_video.setEnabled(False)
            self.mW.stop_video.setEnabled(False)
            self.mW.continue_video.setEnabled(False)
            self.mW.video_timer.blockSignals(False)
            self.mW.video_timer.stop()
            self.mW.video.release()
            if self.mW.result_video is not None:
                self.mW.result_video.release()
            self.mW.inference.setEnabled(True)
            QMessageBox.information(self.mW,"完成","视频播放完毕")
            self.mW.mC.set_enable_True()

    #视频暂停
    def video_pause(self):
        if self.mW.run_type == 1:
            self.mW.play_video.setEnabled(False)
            self.mW.pause_video.setEnabled(False)
            self.mW.stop_video.setEnabled(True)
            self.mW.continue_video.setEnabled(True)
            self.mW.video_timer.blockSignals(True)
            self.mW.mC.set_enable_True()

    #视频继续播放
    def video_continue(self):
        self.mW.video_timer.blockSignals(False)
        self.mW.play_video.setEnabled(False)
        self.mW.pause_video.setEnabled(True)
        self.mW.stop_video.setEnabled(True)
        self.mW.continue_video.setEnabled(False)

    #视频推理函数
    def inference_start(self):
        items_list = []
        if self.mW.photo_list_video.count() > 0 :
            self.mW.stat_label_FPS.setText("FPS:                          ")
            self.mW.mC.set_enable_False()
            for i in range(self.mW.photo_list_video.count()):
                items_list.append(self.mW.photo_list_video.item(i).text())
            self.mW.play_video.setEnabled(False)
            self.mW.image_detection.setEnabled(False)
            self.mW.video_detection.setEnabled(False)
            self.mW.camera_detection.setEnabled(False)
            self.interence_thread = InferenceThread(items_list,self.mW.model_name,self.mW.pth_dir,run_type=self.mW.run_type)
            self.interence_thread.finished_signal.connect(self.interence_finish)
            self.interence_thread.progress_signal.connect(self.mW.mC.update_status)
            self.interence_thread.error_report.connect(self.mW.mC.error_show)
            self.interence_thread.fps_count_signal.connect(self.mW.vC.set_now_video_fps_count)
            self.interence_thread.start()
            self.mW.start = time.time()
        else:
            QMessageBox.warning(self.mW,"警告","无可推理视频")
            return

    #视频推理结束回调
    def interence_finish(self):
        self.mW.end = time.time()
        self.mW.mC.set_enable_True()
        self.mW.open_camera.setEnabled(True)
        self.mW.play_video.setEnabled(True)
        self.mW.image_detection.setEnabled(True)
        self.mW.video_detection.setEnabled(True)
        self.mW.camera_detection.setEnabled(True)
        time_cost = self.mW.end - self.mW.start
        self.mW.stat_label_FPS.setText(f"FPS:{self.mW.video_fps_count / time_cost:.2f}                            ")
        QMessageBox.information(self.mW, "提示", "推理完成")