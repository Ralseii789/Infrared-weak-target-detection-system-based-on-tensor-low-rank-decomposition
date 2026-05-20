from PyQt6.QtCore import QObject
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QMainWindow

#图像视图管理类
class photoViewer(QObject):
    def __init__(self):
        super().__init__()

    # 关闭其他视图
    def close_other_detection(self,main):
        main.open_camera.setVisible(False)
        main.close_camera.setVisible(False)

        main.delete_video.setVisible(False)
        main.get_video.setVisible(False)
        main.open_video_folder.setVisible(False)

    # 展示图像视图
    def set_photo_detection(self,main):
        if main.video_timer.isActive():
            main.vC.video_stop()
        main.clear.setVisible(True)
        main.delete_photo.setVisible(True)
        main.get_photo.setVisible(True)
        main.open_folder.setVisible(True)
        main.photo_list_video.setVisible(False)
        main.photo_list.setVisible(True)
        main.save.setEnabled(True)
        self.close_other_detection(main)
        self.grid_layout(main)
        main.run_type = 0

    # 设置图像视图结构
    def grid_layout(self,main):
        main.original.set_image(QPixmap())
        main.BackGround.set_image(QPixmap())
        main.target.set_image(QPixmap())
        main.superposition.set_image(QPixmap())
        main.image_grid_layout.removeWidget(main.BackGround)
        main.image_grid_layout.removeWidget(main.target)
        main.image_grid_layout.removeWidget(main.superposition)
        main.image_grid_layout.removeWidget(main.original)
        main.image_grid_layout.addWidget(main.original, 1, 0)
        main.image_grid_layout.addWidget(main.BackGround, 0, 0)
        main.image_grid_layout.addWidget(main.target, 0, 1)
        main.image_grid_layout.addWidget(main.superposition, 1, 1)
        for label in [main.BackGround, main.target, main.original, main.superposition]:
            label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        main.frame_v_layout.addLayout(main.image_grid_layout, stretch=1)
        main.image_grid_layout.setRowStretch(0, 1)
        main.image_grid_layout.setRowStretch(1, 1)
        main.image_grid_layout.setColumnStretch(0, 1)
        main.image_grid_layout.setColumnStretch(1, 1)
        main.BackGround.setVisible(True)
        main.target.setVisible(True)
        main.superposition.setVisible(True)