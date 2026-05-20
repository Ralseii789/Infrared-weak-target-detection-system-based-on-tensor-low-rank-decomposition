import os
import shutil
from PyQt6.QtCore import QObject
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QMessageBox, QFileDialog
from PyQt6.QtCore import QFile, QIODevice

#图像资源管理类
class photoRepositories(QObject):
    def __init__(self):
        super().__init__()

    #推理后保存当前图像列表到历史记录文件
    def save_history(self,main):
        file = QFile("history_list.txt")
        file.open(QIODevice.OpenModeFlag.Append | QIODevice.OpenModeFlag.Text)
        for i in range(main.photo_list.count()):
            file.write(main.photo_list.item(i).text().encode("utf-8"))
            file.write('\n'.encode("utf-8"))
        file.close()

    #选定图像展示函数
    def selected_photo_show(self,main):
        items = main.photo_list.selectedItems()
        if not items:
            return
        path = items[0].text()
        image_postfixs = ('png', 'jpg', 'bmp', 'jpeg')
        item_name = items[0].text().split('/')[-1].split('.')[0]
        item_postfixs = items[0].text().split('/')[-1].split('.')[1]
        selected_Item = os.path.splitext(path)[1].lower()
        if selected_Item.endswith(image_postfixs):
            main.pixmap3 = QPixmap(path)
            main.original.set_image(main.pixmap3)
        result_dir = os.path.join('./result',item_name)
        pred_path = os.path.join(result_dir,f'pred.{item_postfixs}')
        L_path = os.path.join(result_dir,f'L.{item_postfixs}')
        bbxo_path = os.path.join(result_dir, f'bbox.{item_postfixs}')
        if os.path.exists(bbxo_path) :
            main.superposition.setVisible(True)
            main.pixmap4 = QPixmap(bbxo_path)
            main.superposition.set_image(main.pixmap4)
        else:
            main.superposition.setVisible(False)
        if os.path.exists(L_path) :
            main.BackGround.setVisible(True)
            main.pixmap2 = QPixmap(L_path)
            main.BackGround.set_image(main.pixmap2)
        else:
            main.BackGround.setVisible(False)
        if os.path.exists(pred_path) :
            main.target.setVisible(True)
            main.pixmap1 = QPixmap(pred_path)
            main.target.set_image(main.pixmap1)
        else:
            main.target.setVisible(False)

    #从文件夹中选中图像到图像列表
    def open_photos(self,main):
        photos_names,_ = QFileDialog.getOpenFileNames(
            main,
            "选择单/多张图片",
            "./datasets",
            "photos (*.png *.jpg *.bmp *.jpeg)"
        )
        if photos_names:
            main.photo_list.addItems(photos_names)

    #选中一整个文件夹的图像到图像列表
    def open_photo_dir(self,main):
        selected_dir = QFileDialog.getExistingDirectory(
            main,
            "选择文件夹",
            "./datasets"
        )
        if selected_dir:
            image_postfixs = ('png', 'jpg', 'bmp', 'jpeg')
            for path,_,photos in os.walk(selected_dir):
                for photo in photos:
                    if photo.lower().endswith(image_postfixs):
                        photo_names = os.path.join(path, photo).replace('\\','/')
                        main.photo_list.addItem(photo_names)

    #保存图像推理结果到本地
    def save_as_resutls(self,main):
        selected_dir = QFileDialog.getExistingDirectory(
            main,
            "选择结果保存的位置",
            ""
        )
        if selected_dir:
            items = main.photo_list.selectedItems()
            print(items)
            if not items:
                QMessageBox.warning(main,"警告","非法项")
                return
            item_name = items[0].text().split('/')[-1].split('.')[0]
            path = os.path.join(selected_dir,item_name)
            print(path)
            if os.path.exists(path):
                reply = QMessageBox.question(
                    main,
                    "覆盖",
                    f"{item_name}的结果在当前目录下已存在，是否替换",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
                try:
                    shutil.rmtree(path)
                except Exception as e:
                    QMessageBox.critical(main,"错误",f"删除原有文件夹失败:{e}")
                    return
            try:
                os.mkdir(path)
            except Exception as e:
                QMessageBox.critical(main,"失败",f"创建文件夹失败:{e}")
            item_postfixs = items[0].text().split('/')[-1].split('.')[1]
            save_name = ('S','L',item_name,'pred')
            pixmap_list = []
            pixmap_list.append(main.pixmap1)
            pixmap_list.append(main.pixmap2)
            pixmap_list.append(main.pixmap3)
            pixmap_list.append(main.pixmap4)
            for i in range(4):
                pixmap_list[i].save(path+"/"+save_name[i]+"."+item_postfixs)
            QMessageBox.information(main, "成功", "保存成功")