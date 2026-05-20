from PyQt6.QtCore import QObject
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QMainWindow

#视频视图管理类
class videoView(QObject):
    def __init__(self):
        super().__init__()

    # 关闭其他视图
    def close_other_detection(self,main):
        main.delete_photo.setVisible(False)
        main.get_photo.setVisible(False)
        main.open_folder.setVisible(False)

        main.open_camera.setVisible(False)
        main.close_camera.setVisible(False)

    # 展示视频视图
    def set_video_detection(self,main):
        main.clear.setVisible(True)
        main.delete_video.setVisible(True)
        main.get_video.setVisible(True)
        main.open_video_folder.setVisible(True)
        main.photo_list_video.setVisible(True)
        main.photo_list.setVisible(False)
        main.save.setEnabled(False)
        self.close_other_detection(main)
        self.two_layout(main)
        main.run_type = 1

    # 设置视频视图结构
    def two_layout(self,main):
        main.image_grid_layout.removeWidget(main.BackGround)
        main.image_grid_layout.removeWidget(main.target)
        main.image_grid_layout.removeWidget(main.superposition)
        main.image_grid_layout.removeWidget(main.original)
        main.image_grid_layout.addWidget(main.superposition, 0, 0, 2, 1)
        main.image_grid_layout.addWidget(main.original, 0, 1, 1, 1)
        main.original.setVisible(True)
        main.superposition.setVisible(True)
        main.BackGround.setVisible(False)
        main.target.setVisible(False)
        main.image_grid_layout.setColumnStretch(0, 3)
        main.image_grid_layout.setColumnStretch(1, 1)
