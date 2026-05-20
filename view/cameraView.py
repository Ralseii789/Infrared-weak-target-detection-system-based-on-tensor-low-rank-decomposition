from PyQt6.QtCore import QObject
from PyQt6.QtGui import QPixmap

#摄像头视图管理类
class cameraView(QObject):
    def __init__(self):
        super().__init__()

    #关闭其他视图
    def close_other_detection(self,main):
        main.delete_video.setVisible(False)
        main.get_video.setVisible(False)
        main.open_video_folder.setVisible(False)
        main.delete_photo.setVisible(False)
        main.get_photo.setVisible(False)
        main.open_folder.setVisible(False)

    #展示摄像头视图
    def set_camera_detection(self,main):
        if main.video_timer.isActive():
            main.vC.video_stop()
        main.open_camera.setVisible(True)
        main.close_camera.setVisible(True)
        main.clear.setVisible(False)
        main.photo_list_video.setVisible(False)
        main.photo_list.setVisible(False)
        main.save.setEnabled(False)
        self.close_other_detection(main)
        self.single_layout(main)
        main.run_type = 2

    #设置摄像头视图结构
    def single_layout(self,main):
        main.original.set_image(QPixmap())
        main.BackGround.set_image(QPixmap())
        main.target.set_image(QPixmap())
        main.superposition.set_image(QPixmap())
        main.image_grid_layout.removeWidget(main.BackGround)
        main.image_grid_layout.removeWidget(main.target)
        main.image_grid_layout.removeWidget(main.superposition)
        main.image_grid_layout.removeWidget(main.original)
        main.image_grid_layout.addWidget(main.original,0,0,2,2)
        main.BackGround.setVisible(False)
        main.target.setVisible(False)
        main.superposition.setVisible(False)