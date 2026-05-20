import os
import cv2
from PyQt6.QtCore import QObject
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from utils.image2video import Image2Video

#视频资源管理类
class videoRepositories(QObject):
    def __init__(self):
        super().__init__()

    # 视频图像展示函数
    def update_video(self,main):
        if main.run_type == 1:
            ret, frame = main.video.read()
            if not ret:
                main.vC.video_stop()
                return
            h, w, c = frame.shape
            frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image = QImage(frame_grey.data, w, h, QImage.Format.Format_Grayscale8)
            main.pixmap3 = QPixmap.fromImage(image)
            main.original.set_image(main.pixmap3)
            if main.result_video is not None:
                ret_result,frame_result = main.result_video.read()
                if not ret_result:
                    main.vC.video_stop()
                    return
                h,w,c = frame_result.shape
                frame_rgb = cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB)
                result_image = QImage(frame_rgb.data, w, h,w*3, QImage.Format.Format_RGB888)
                main.pixmap4 = QPixmap.fromImage(result_image)
                main.superposition.set_image(main.pixmap4)

    # 视频播放
    def video_play(self,path,bbox_path,main):
        main.video = cv2.VideoCapture(path)
        if os.path.exists(bbox_path):
            main.result_video = cv2.VideoCapture(bbox_path)
            if not main.result_video.isOpened():
                QMessageBox.critical(main,"错误","视频打开错误")
                return
        if main.video.isOpened() :
            main.play_video.setEnabled(False)
            main.pause_video.setEnabled(True)
            main.stop_video.setEnabled(True)
            main.continue_video.setEnabled(False)
            main.video_timer.start(33)
        main.inference.setEnabled(False)
        main.mC.set_enable_False()

    # 文件夹图像转视频函数
    def open_img2video(self,main):
        selected_dir = QFileDialog.getExistingDirectory(
            main,
            "选择图片序列文件夹",
            "../video"
        )
        if selected_dir:
            image_postifixs = ('png','jpg','bmp','jepg')
            count = 0
            for f in os.listdir(selected_dir):
                if not f.endswith(image_postifixs):
                   count += 1
                   if count == len(os.listdir(selected_dir)):
                       QMessageBox.warning(main,"警告","该文件夹下没有图片")
                else:
                    self.image2video_instance = Image2Video(selected_dir)
                    self.image2video_instance.image2video()
                    QMessageBox.information(main,"结束","图像转视频完成")
                    break

    # 选中视频的封面展示回调函数
    def selected_video_surface_show(self,main):
        items = main.photo_list_video.selectedItems()
        if not items:
            return
        path = items[0].text()
        item_name = items[0].text().split('/')[-1].split('.')[0]
        item_postfixs = items[0].text().split('/')[-1].split('.')[1]
        video = cv2.VideoCapture(path)
        ret,frame = video.read()
        h,w,c = frame.shape
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = QImage(frame_grey.data,w,h,QImage.Format.Format_Grayscale8)
        main.pixmap3 = QPixmap.fromImage(image)
        main.original.set_image(main.pixmap3)
        video.release()
        bbxo_path = os.path.join("./result_video",item_name, f'bbox.{item_postfixs}')
        if os.path.exists(bbxo_path):
            video = cv2.VideoCapture(bbxo_path)
            if video.isOpened() :
                ret, frame = video.read()
                h, w, c = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QImage(frame_rgb.data, w, h,w*3, QImage.Format.Format_RGB888)
                main.pixmap4 = QPixmap.fromImage(image)
                main.superposition.set_image(main.pixmap4)
            video.release()
        else:
            main.superposition.set_image(QPixmap())

    # 选取整个文件夹的视频到视频列表
    def _open_video_folder(self,main):
        selected_dir = QFileDialog.getExistingDirectory(
            main,
            "选择文件夹",
            "./video"
        )
        if selected_dir:
            video_postifixs = ('mp4' ,'avi' )
            for path,_,videos in os.walk(selected_dir):
                for video in videos:
                    if video.lower().endswith(video_postifixs):
                        video_names = os.path.join(path,video).replace('\\','/')
                        main.photo_list_video.addItem(video_names)

    # 选取视频到视频列表
    def open_video(self,main):
        video_names,_ = QFileDialog.getOpenFileNames(
            main,
            "选择单/多个视频",
            "./video",
            "videos (*.mp4 *.avi)"
        )
        if video_names:
            main.photo_list_video.addItems(video_names)