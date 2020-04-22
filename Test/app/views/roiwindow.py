import cv2
from PyQt5.QtCore import QSize, Qt, QRect
from PyQt5.QtGui import QColor, QIcon, QImage
from PyQt5.QtWidgets import QApplication, QDialog, QGridLayout, QPushButton, QSpacerItem, QSizePolicy, QWidget, \
	QFormLayout, QLineEdit, QLabel, QDesktopWidget
from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPixmap, QPen
from PyQt5.QtWidgets import QGraphicsView, QGraphicsPixmapItem, QGraphicsScene, QGraphicsItem
import app.icons.resource
from app.config import ROIS_DIR


class LabelNameDialog(QDialog):

	def __init__(self, parent=None):
		super().__init__(parent=parent)
		self.init_ui()
		self.save_button.clicked.connect(parent.saveimg)

	def init_ui(self):
		self.setModal(True)
		self.setWindowIcon(QIcon(":icons/set_roi.png"))
		self.setWindowTitle("设置地标")
		self.form_layout = QFormLayout()
		test_label = QLabel("标签名称：")
		self.label_name_edit = QLineEdit()
		self.label_name_edit.setReadOnly(False)
		self.save_button = QPushButton('保存')
		self.form_layout.addRow(test_label, self.label_name_edit)
		self.form_layout.addWidget(self.save_button)
		self.setLayout(self.form_layout)


class GraphicsView(QGraphicsView):
	save_signal = pyqtSignal(bool)

	def __init__(self, picture, parent=None):
		super(GraphicsView, self).__init__(parent)
		# 设置放大缩小时跟随鼠标
		self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
		self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

		self.scene = QGraphicsScene()
		self.setScene(self.scene)

		if isinstance(picture, str):
			self.image_item = GraphicsPixmapItem(QPixmap(picture))
		else:
			show = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
			showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
			self.image_item = GraphicsPixmapItem(QPixmap.fromImage(showImage))

		self.image_item.setFlag(QGraphicsItem.ItemIsMovable)
		self.scene.addItem(self.image_item)

		size = self.image_item.pixmap().size()
		# 调整图片在中间
		self.image_item.setPos(-size.width() / 2, -size.height() / 2)
		self.scale(0.6, 0.8)

	def wheelEvent(self, event):
		'''滚轮事件'''
		zoomInFactor = 1.25
		zoomOutFactor = 1 / zoomInFactor

		if event.angleDelta().y() > 0:
			zoomFactor = zoomInFactor
		else:
			zoomFactor = zoomOutFactor

		self.scale(zoomFactor, zoomFactor)

	def mouseReleaseEvent(self, event):
		'''鼠标释放事件'''
		# print(self.image_item.is_finish_cut, self.image_item.is_start_cut)
		if self.image_item.is_finish_cut and self.image_item.is_start_cut:
			self.save_signal.emit(True)
		else:
			self.save_signal.emit(False)


class GraphicsPixmapItem(QGraphicsPixmapItem):
	save_signal = pyqtSignal(bool)

	def __init__(self, picture, parent=None):
		super(GraphicsPixmapItem, self).__init__(parent)

		self.setPixmap(picture)
		self.is_start_cut = False
		self.current_point = None
		self.is_finish_cut = False

	def mouseMoveEvent(self, event):
		'''鼠标移动事件'''
		self.current_point = event.pos()
		if not self.is_start_cut or self.is_midbutton:
			self.moveBy(self.current_point.x() - self.start_point.x(),
			            self.current_point.y() - self.start_point.y())
			self.is_finish_cut = False
		self.update()

	def mousePressEvent(self, event):
		'''鼠标按压事件'''
		super(GraphicsPixmapItem, self).mousePressEvent(event)
		self.start_point = event.pos()
		self.current_point = None
		self.is_finish_cut = False
		if event.button() == Qt.MidButton:
			self.is_midbutton = True
			self.update()
		else:
			self.is_midbutton = False
			self.update()

	def paint(self, painter, QStyleOptionGraphicsItem, QWidget):
		super(GraphicsPixmapItem, self).paint(painter, QStyleOptionGraphicsItem, QWidget)
		if self.is_start_cut and not self.is_midbutton:
			# print(self.start_point, self.current_point)
			pen = QPen(Qt.DashLine)
			pen.setColor(QColor(0, 150, 0, 70))
			pen.setWidth(3)
			painter.setPen(pen)
			painter.setBrush(QColor(0, 0, 255, 70))
			if not self.current_point:
				return
			painter.drawRect(QRectF(self.start_point, self.current_point))
			self.end_point = self.current_point
			self.is_finish_cut = True


class SetRoiWidget(QWidget):
	update_listmodel_signal=pyqtSignal(bool)
	def __init__(self, img=r'D:/2020-04-10-15-26-22test.bmp'):
		super(SetRoiWidget, self).__init__()
		self.setWindowIcon(QIcon(":icons/set_roi.png"))
		self.setWindowTitle("选择ROI")

		self.picture = img
		self.init_ui()
		self.label_save_dialog = LabelNameDialog(parent=self)
		# 视图背景颜色
		self.graphicsView.setBackgroundBrush(QColor(0, 160, 0))
		self.graphicsView.save_signal.connect(self.pushButton_save.setEnabled)
		self.pushButton_cut.clicked.connect(self.pushButton_cut_clicked)
		self.pushButton_save.clicked.connect(self.pushButton_save_clicked)

	def init_ui(self):
		# self.showMaximized()
		self.gridLayout = QGridLayout(self)
		self.pushButton_cut = QPushButton('裁剪', self)
		self.pushButton_cut.setCheckable(True)
		self.pushButton_cut.setMaximumSize(QSize(100, 16777215))
		self.gridLayout.addWidget(self.pushButton_cut, 0, 0, 1, 2)
		self.pushButton_save = QPushButton('保存', self)
		self.pushButton_save.setEnabled(False)
		self.gridLayout.addWidget(self.pushButton_save, 1, 0, 1, 1)
		spacerItem = QSpacerItem(50, 549, QSizePolicy.Minimum, QSizePolicy.Expanding)
		self.gridLayout.addItem(spacerItem, 2, 0, 1, 1)
		self.graphicsView = GraphicsView(picture=self.picture, parent=self)
		self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		self.gridLayout.addWidget(self.graphicsView, 0, 1, 3, 1)

		# screen = QDesktopWidget().screenGeometry()
		# size = self.geometry()
		# self.move((screen.width() - size.width()) / 2,
		#           (screen.height() - size.height()) / 2)


	def pushButton_cut_clicked(self):
		if self.graphicsView.image_item.is_start_cut:
			self.graphicsView.image_item.is_start_cut = False
			self.graphicsView.image_item.setCursor(Qt.ArrowCursor)  # 箭头光标
		else:
			self.graphicsView.image_item.is_start_cut = True
			self.graphicsView.image_item.setCursor(Qt.CrossCursor)  # 十字光标

	def saveimg(self):
		rect = QRect(self.graphicsView.image_item.start_point.toPoint(),
		             self.graphicsView.image_item.end_point.toPoint())
		new_pixmap = self.graphicsView.image_item.pixmap().copy(rect)
		imgname = self.label_save_dialog.label_name_edit.text()
		new_pixmap.save(r'{dir}/{imgname}.png'.format(dir=ROIS_DIR,imgname=imgname))
		self.label_save_dialog.hide()
		self.pushButton_save.setEnabled(False)
		self.update_listmodel_signal.emit(True)

	def pushButton_save_clicked(self):
		self.label_save_dialog.show()


# if __name__ == '__main__':
# 	import sys
#
# 	app = QApplication(sys.argv)
# 	form = SetRoiWidget(img="d:/2020-04-10-15-26-22test.bmp")
# 	form.show()
# 	app.exec_()
