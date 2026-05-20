from PyQt6.QtCore import  QRectF, QPoint, Qt
from PyQt6.QtGui import QMouseEvent, QPixmap, QWheelEvent, QPainter, QResizeEvent
from PyQt6.QtWidgets import  QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

#主窗口图像展示组件
class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.graphicsScene = QGraphicsScene(self)
        self.setScene(self.graphicsScene)

        self.pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap_item)

        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)

        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.last_mouse_pos = QPoint()#记录上一次鼠标位置
        self.is_panning = False

        self.zoom_factor = 1.1

    def set_image(self,pixmap: QPixmap):
        self.pixmap_item.setPixmap(pixmap)
        self.pixmap_item.setOffset(0,0)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.pixmap_item,Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self,event:QWheelEvent):
        zoom_in = event.angleDelta().y() > 0
        if zoom_in:
            self.scale(self.zoom_factor, self.zoom_factor)
        else:
            self.scale(1/self.zoom_factor, 1/self.zoom_factor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = True
            self.last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.is_panning:
            delta = event.position() - self.last_mouse_pos
            self.last_mouse_pos = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.is_panning:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def reset_zoom(self):
        pixmap = self.pixmap_item.pixmap()
        if pixmap and not pixmap.isNull():
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
