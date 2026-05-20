from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QMainWindow,  QMessageBox
from otherWindow import control
from otherWindow import Help
from view.mainView import mainView

#通用管理类
class mainController(QObject):
    def __init__(self,mainWindow:QMainWindow):
        super().__init__()
        self.mW = mainWindow
        self.mV = mainView()

    #非推理时使能设置
    def set_enable_True(self):
        self.mV.set_enable_True(self.mW)

    #推理时使能设置
    def set_enable_False(self):
        self.mV.set_enable_False(self.mW)

    #重置视图大小
    def reset_size(self):
        self.mV.reset_size(self.mW)

    #上一张图像/视频切换
    def last_photo(self):
        self.mV.last_photo(self.mW)

    #下一张图像/视频切换
    def next_photo(self):
        self.mV.next_photo(self.mW)

    #展示推理进度的方法
    def update_status(self,current_idx):
        if self.mW.run_type == 0:
            self.mW.stat_label_progress.setText(f"进度：{current_idx}/{self.mW.photo_list.count()}                ")
        else:
            self.mW.stat_label_progress.setText(f"进度：{current_idx}/{self.mW.video_fps_count}                ")

    #打开帮助窗口
    def open_help(self):
        self.open_help = Help.Ui_Help()
        self.open_help.show()

    #打开控制面板窗口
    def open_control(self):
        self.open_control = control.Ui_Form()
        self.open_control.parames_signal.connect(self.rec_params)
        self.open_control.show()

    #控制面板窗口用于接收控制参数的回调函数
    def rec_params(self,model_name,threshold):
        self.mW.model_name = model_name
        self.mW.threshold = threshold
        if model_name == 'NUDT':
            self.mW.pth_dir = './best_weight/LTLNet_NUDT_best.pth.tar'
        elif model_name == 'NUAA':
            self.mW.pth_dir = './best_weight/LTLNet_NUAA_best.pth.tar'
        else:
            self.mW.pth_dir = './best_weight/LTLNet_IRSTD_best.pth.tar'
        self.mW.stat_label_model.setText(f"权重:{model_name}                          ")

    #待推理列表清空方法
    def photo_clear(self):
        reply = QMessageBox().question(
            self.mW,
            '清空',
            '确定要图片列表吗',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            if self.mW.run_type == 0:
                self.mW.photo_list.clear()
            elif self.mW.run_type == 1:
                self.mW.photo_list_video.clear()

    #错误信息展示回调函数
    def error_show(self,message):
        QMessageBox.critical(self,"错误",message)

    #推理调用接口
    def inference_start(self):
        if self.mW.run_type == 0:
            self.mW.pC.inference_start()
        elif self.mW.run_type == 1:
            self.mW.vC.inference_start()
        elif self.mW.run_type == 2:
            self.mW.cC.inference_start()
