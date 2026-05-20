from PyQt6.QtCore import QObject

#通用视图管理类
class mainView(QObject):
    def __init__(self):
        super().__init__()

    #重置视图大小
    def reset_size(self,main):
        main.BackGround.reset_zoom()
        main.target.reset_zoom()
        main.original.reset_zoom()
        main.superposition.reset_zoom()

    # 上一张图像/视频切换
    def last_photo(self,main):
        if main.run_type == 0:
            selected_index = main.photo_list.currentRow()
            if(selected_index != 0):
                main.photo_list.setCurrentRow(selected_index-1)
        elif main.run_type == 1:
            selected_index = main.photo_list_video.currentRow()
            if (selected_index != 0):
                main.photo_list_video.setCurrentRow(selected_index - 1)

    # 下一张图像/视频切换
    def next_photo(self,main):
        if main.run_type == 0:
            selected_index = main.photo_list.currentRow()
            if (selected_index != main.photo_list.count ()-1):
                main.photo_list.setCurrentRow(selected_index + 1)
        elif main.run_type == 1:
            selected_index = main.photo_list_video.currentRow()
            if (selected_index != main.photo_list_video.count() - 1):
                main.photo_list_video.setCurrentRow(selected_index + 1)

    # 推理时使能设置
    def set_enable_False(self,main):
        main.inference.setEnabled(False)
        main.open_folder.setEnabled(False)
        main.get_photo.setEnabled(False)
        main.control_action.setEnabled(False)
        main.history_action.setEnabled(False)
        main.save.setEnabled(False)
        main.clear.setEnabled(False)
        main.delete_photo.setEnabled(False)

    # 非推理时使能设置
    def set_enable_True(self,main):
        main.inference.setEnabled(True)
        main.open_folder.setEnabled(True)
        main.get_photo.setEnabled(True)
        main.control_action.setEnabled(True)
        main.history_action.setEnabled(True)
        main.save.setEnabled(True)
        main.clear.setEnabled(True)
        main.delete_photo.setEnabled(True)