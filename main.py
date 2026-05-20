import os
from controller.cameraController import cameraController
from controller.mainController import mainController
from controller.photoController import photoController
from controller.videoController import videoController
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import sys
sys.path.insert(0, "D:\\ana3\\envs\\pyrotch\\Lib\\site-packages")
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, \
    QLabel, QListWidget
from otherWindow.ImageViewer import ImageViewer

#主类
class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setWindowTitle("红外弱小目标检测系统")
        self.resize(1565, 1147)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./otherWindow/icon/红外检测.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.setWindowIcon(icon)
        self.pC = photoController(self)
        self.vC = videoController(self)
        self.cC = cameraController(self)
        self.mC = mainController(self)
        self.timer = QtCore.QTimer()
        self.video_timer = QtCore.QTimer()
        self.video_timer.timeout.connect(self.vC.update_video)
        self.timer.timeout.connect(self.cC.update)
        self.setupUi()
        self.showMaximized()
        self.model_name = 'NUDT'
        self.threshold = 50
        self.pth_dir = './best_weight/LTLNet_NUDT_best.pth.tar'
        self.start = 0
        self.end = 0
        self.camera = cv2.VideoCapture(0)
        self.is_playing = False
        self.run_type = 0
        self.video = None
        self.result_video = None
        self.video_fps_count = 0

    def setupUi(self):
        self.centralwidget = QWidget()
        self.setCentralWidget(self.centralwidget)
        self.main_h_layout = QHBoxLayout(self.centralwidget)
        self.main_h_layout.setContentsMargins(0, 0, 0, 0)
        self.main_h_layout.setSpacing(0)
        self.frame = QWidget()
        self.main_h_layout.addWidget(self.frame)
        self.frame_v_layout = QVBoxLayout(self.frame)
        self.frame_v_layout.setContentsMargins(20, 10, 20, 10)

        self.top_label_layout = QHBoxLayout()
        self.label_5 = QLabel("背景")
        self.label_6 = QLabel("目标")
        self.top_label_layout.addStretch()
        self.top_label_layout.addWidget(self.label_5)
        self.top_label_layout.addStretch(8)
        self.top_label_layout.addWidget(self.label_6)
        self.top_label_layout.addStretch()
        self.frame_v_layout.addLayout(self.top_label_layout)

        self.image_grid_layout = QGridLayout()
        self.image_grid_layout.setSpacing(0)
        self.BackGround = ImageViewer()
        self.BackGround.setStyleSheet("border:1px solid #cccccc;")
        self.target = ImageViewer()
        self.target.setStyleSheet("border:1px solid #cccccc;")
        self.original = ImageViewer()
        self.original.setStyleSheet("border:1px solid #cccccc;")
        self.superposition = ImageViewer()
        self.superposition.setStyleSheet("border:1px solid #cccccc;")
        self.image_grid_layout.addWidget(self.BackGround, 0, 0)
        self.image_grid_layout.addWidget(self.target, 0, 1)
        self.image_grid_layout.addWidget(self.original, 1, 0)
        self.image_grid_layout.addWidget(self.superposition, 1, 1)
        for label in [self.BackGround, self.target, self.original, self.superposition]:
            label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.frame_v_layout.addLayout(self.image_grid_layout, stretch=1)
        self.image_grid_layout.setRowStretch(0, 1)
        self.image_grid_layout.setRowStretch(1, 1)
        self.image_grid_layout.setColumnStretch(0, 1)
        self.image_grid_layout.setColumnStretch(1, 1)
        self.bottom_label_layout = QHBoxLayout()
        self.label_7 = QLabel("原图")
        self.label_8 = QLabel("检测")
        self.bottom_label_layout.addStretch()
        self.bottom_label_layout.addWidget(self.label_7)
        self.bottom_label_layout.addStretch(8)
        self.bottom_label_layout.addWidget(self.label_8)
        self.bottom_label_layout.addStretch()
        self.frame_v_layout.addLayout(self.bottom_label_layout)

        self.btn_h_layout = QHBoxLayout()
        self.inference = QPushButton("推理")
        self.save = QPushButton("保存")
        self.prior = QPushButton("上一张")
        self.next_btn = QPushButton("下一张")
        self.reset_btn = QPushButton("重置")
        self.image2video = QPushButton("图片转视频")
        self.play_video = QPushButton("播放")
        self.continue_video = QPushButton("继续")
        self.pause_video = QPushButton("暂停")
        self.stop_video = QPushButton("停止")
        self.inference.setIcon(QtGui.QIcon("./otherWindow/icon/推理.png"))
        self.save.setIcon(QtGui.QIcon("./otherWindow/icon/保存.png"))
        self.prior.setIcon(QtGui.QIcon("./otherWindow/icon/上一张.png"))
        self.next_btn.setIcon(QtGui.QIcon("./otherWindow/icon/下一张.png"))
        self.reset_btn.setIcon(QtGui.QIcon("./otherWindow/icon/重置.png"))
        self.image2video.setIcon(QtGui.QIcon("./otherWindow/icon/图片转视频.png"))
        self.play_video.setIcon(QtGui.QIcon("./otherWindow/icon/播放.png"))
        self.continue_video.setIcon(QtGui.QIcon("./otherWindow/icon/继续播放.png"))
        self.pause_video.setIcon(QtGui.QIcon("./otherWindow/icon/暂停.png"))
        self.stop_video.setIcon(QtGui.QIcon("./otherWindow/icon/停止.png"))
        self.btn_h_layout.addWidget(self.inference)
        self.btn_h_layout.addWidget(self.save)
        self.btn_h_layout.addWidget(self.prior)
        self.btn_h_layout.addWidget(self.next_btn)
        self.btn_h_layout.addWidget(self.reset_btn)
        self.btn_h_layout.addWidget(self.image2video)
        self.btn_h_layout.addWidget(self.play_video)
        self.btn_h_layout.addWidget(self.continue_video)
        self.btn_h_layout.addWidget(self.pause_video)
        self.btn_h_layout.addWidget(self.stop_video)
        self.btn_h_layout.addStretch()

        self.pause_video.setEnabled(False)
        self.stop_video.setEnabled(False)
        self.frame_v_layout.addLayout(self.btn_h_layout)
        self.frame_2 = QWidget()
        self.frame_2.setMaximumWidth(300)
        self.frame.setStyleSheet("""
                    QWidget {
                        background-color: rgb(50,50, 50);
                    }
                    """)
        self.frame_2.setStyleSheet("""
            QWidget {
                background-color: rgb(100, 100, 100);
            }
            QListWidget {
                background-color: rgb(150, 150, 150);
            }
        """)
        self.main_h_layout.addWidget(self.frame_2)
        self.right_v_layout = QVBoxLayout(self.frame_2)
        self.right_v_layout.setContentsMargins(10, 10, 10, 10)
        self.right_v_layout.setSpacing(10)
        self.photo_list = QListWidget()
        self.right_v_layout.addWidget(self.photo_list, stretch=1)
        self.photo_list_video = QListWidget()
        self.right_v_layout.addWidget(self.photo_list_video, stretch=1)

        self.delete_photo = QPushButton("删除图片")
        self.get_photo = QPushButton("打开图片")
        self.open_folder = QPushButton("打开文件夹")
        self.clear = QPushButton("清空")

        self.delete_video = QPushButton("删除视频")
        self.get_video = QPushButton("打开视频")
        self.open_video_folder = QPushButton("打开文件夹")

        self.open_camera = QPushButton("打开摄像头")
        self.close_camera = QPushButton("关闭摄像头")

        self.right_v_layout.addWidget(self.clear)
        self.right_v_layout.addWidget(self.delete_photo)
        self.right_v_layout.addWidget(self.get_photo)
        self.right_v_layout.addWidget(self.open_folder)
        self.right_v_layout.addWidget(self.delete_video)
        self.right_v_layout.addWidget(self.get_video)
        self.right_v_layout.addWidget(self.open_video_folder)
        self.right_v_layout.addWidget(self.open_camera)
        self.right_v_layout.addWidget(self.close_camera)

        self.menubar = self.menuBar()
        self.statusbar = self.statusBar()
        self.image_detection = QAction("图像", self)
        self.menubar.addAction(self.image_detection)
        self.video_detection = QAction("视频", self)
        self.menubar.addAction(self.video_detection)
        self.camera_detection = QAction("摄像头", self)
        self.menubar.addAction(self.camera_detection)
        self.control_action = QAction("控制面板", self)
        self.menubar.addAction(self.control_action)
        self.history_action = QAction("历史记录", self)
        self.menubar.addAction(self.history_action)
        self.help_action = QAction("帮助", self)
        self.menubar.addAction(self.help_action)
        self.stat_label_FPS = QLabel()
        self.stat_label_FPS.setText("FPS:                          ")
        self.stat_label_progress = QLabel()
        self.stat_label_progress.setText(f"进度：{self.photo_list.count()}/{self.photo_list.count()}                          ")
        self.stat_label_model = QLabel()
        self.stat_label_model.setText("权重:NUDT                          ")
        self.statusbar.addWidget(self.stat_label_model)
        self.statusbar.addWidget(self.stat_label_progress)
        self.statusbar.addWidget(self.stat_label_FPS)
        self.close_camera.setEnabled(False)
        self.continue_video.setEnabled(False)

        #信号与槽的连接
        self.control_action.triggered.connect(self.mC.open_control)
        self.help_action.triggered.connect(self.mC.open_help)
        self.clear.clicked.connect(self.mC.photo_clear)
        self.reset_btn.clicked.connect(self.mC.reset_size)
        self.prior.clicked.connect(self.mC.last_photo)
        self.next_btn.clicked.connect(self.mC.next_photo)
        self.inference.clicked.connect(self.mC.inference_start)

        self.history_action.triggered.connect(self.pC.open_history)
        self.delete_photo.clicked.connect(self.pC.delete_from_index)
        self.get_photo.clicked.connect(self.pC.open_photos)
        self.open_folder.clicked.connect(self.pC.open_photo_dir)
        self.photo_list.itemSelectionChanged.connect(self.pC.selected_photo_show)
        self.save.clicked.connect(self.pC.save_as_resutls)
        self.image_detection.triggered.connect(self.pC.set_photo_detection)

        self.video_detection.triggered.connect(self.vC.set_video_detection)
        self.image2video.clicked.connect(self.vC.open_img2video)
        self.delete_video.clicked.connect(self.vC.delete_from_index_video)
        self.get_video.clicked.connect(self.vC.open_video)
        self.open_video_folder.clicked.connect(self.vC._open_video_folder)
        self.photo_list_video.itemSelectionChanged.connect(self.vC.selected_video_surface_show)
        self.play_video.clicked.connect(self.vC.video_play)
        self.continue_video.clicked.connect(self.vC.video_continue)
        self.stop_video.clicked.connect(self.vC.video_stop)
        self.pause_video.clicked.connect(self.vC.video_pause)

        self.camera_detection.triggered.connect(self.cC.set_camera_detection)
        self.open_camera.clicked.connect(self.cC.open)
        self.close_camera.clicked.connect(self.cC.close)

        self.pC.set_photo_detection()

    def closeEvent(self, event):
        QApplication.closeAllWindows()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    Ui = Ui_MainWindow()
    Ui.show()
    sys.exit(app.exec())