import cv2
from PyQt5.QtGui import QImage, QPixmap

from app.config import IMG_HEIGHT, IMG_WIDTH
from app.core.autowork.intelligentthread import IntelligentThread
from app.core.autowork.plcthread import PlcThread
from app.core.location.landmarklocation import start_location_landmark
from app.core.plc.plchandle import PlcHandle
from app.core.location.locationservice import PointLocationService
from app.log.logtool import mylog_debug
from app.status import HockStatus


class IntelligentProcess(object):
	def __init__(self, IMGHANDLE, img_play, plc_status_show, bag_num_show):
		self.plchandle = PlcHandle()
		self._IMGHANDLE = IMGHANDLE
		self.init_imgplay(img_play)
		self.plc_status_show = plc_status_show
		self.bag_num_show = bag_num_show
		self.init_plc_thread()
		self.init_imgdetector_thread()
		self.check_plc_status()

	@property
	def IMGHANDLE(self):
		return self._IMGHANDLE

	@IMGHANDLE.setter
	def IMGHANDLE(self, value):
		self._IMGHANDLE = value
		self.intelligentthread.IMAGE_HANDLE = value

	def init_imgplay(self, img_play):
		self.img_play = img_play
		self.img_play.left_button_release_signal.connect(self.set_roi)

	def check_plc_status(self):
		'''检测plc状态'''
		self.plc_status_show.setText('plc连接成功' if self.plchandle.status else "很抱歉，连接失败")

	def init_plc_thread(self):
		'''初始化plc线程'''
		self.plcthread = PlcThread(plchandle=self.plchandle)
		self.plcthread.askforSingnal.connect(self.askposition_plcsignal_process)
		self.plcthread.moveSignal.connect(self.notneedposition_plcsignal_process)

	def init_imgdetector_thread(self):
		'''初始化图像处理线程'''
		self.intelligentthread = IntelligentThread(IMGHANDLE=self.IMGHANDLE, positionservice=PointLocationService(),
		                                           video_player=self.img_play)
		self.intelligentthread.hockstatus = HockStatus.POSITION_NEARESTBAG
		self.intelligentthread.positionSignal.connect(self.writetoplc_imgsignal_process)  # 发送移动位置
		self.intelligentthread.dropHockSignal.connect(self.drophock_imgsignal_process)  # 命令放下钩子
		self.intelligentthread.pullHockSignal.connect(self.pullhock_imgsignal_process)  # 命令拉起钩子
		self.intelligentthread.findConveyerBeltSignal.connect(self.findconveyerbelt_imgsignal_process)  # 命令移动袋子到传送带
		self.intelligentthread.dropBagSignal.connect(self.drop_bag_imgsignal_process)
		self.intelligentthread.rebackSignal.connect(self.reback_imgsignal_process)
		self.intelligentthread.finishSignal.connect(self.finish_imgsignal_process)
		self.intelligentthread.foundbagSignal.connect(self.editbagnum_imgsignal_process)

	def askposition_plcsignal_process(self, info: str):
		print(info)
		self.intelligentthread.work = True
		self.plcthread.work = True

	def notneedposition_plcsignal_process(self, info: str):
		print(info)
		self.intelligentthread.work = True
		self.plcthread.work = True

	def editbagnum_imgsignal_process(self, bagnum):
		self.bag_num_show.setText(str(bagnum))
		print("bag num is {}".format(bagnum))

	def writetoplc_imgsignal_process(self, position):
		'''接收反馈信号'''
		x, y, z = position
		# print("X轴移动：{}，Y轴移动{},z轴移动{}".format(*position))
		# print(position)
		self.plchandle.write_position(position)
		self.plcthread.work = True
		self.intelligentthread.work = True

	def drophock_imgsignal_process(self, position):
		_x, _y, z = position
		print("已经定位袋子，正在放下钩子,z:{}米".format(z))
		self.plchandle.write_position(position)
		self.intelligentthread.hockstatus = HockStatus.DROP_HOCK
		self.currentstatus_edit.setText("已经定位袋子，正在放下钩子")

	def pullhock_imgsignal_process(self, position):
		_x, _y, z = position
		self.currentstatus_edit.setText("正在拉起袋子,向上拉取{}米".format(z))
		self.plchandle.write_position(position)
		self.intelligentthread.hockstatus = HockStatus.PULL_HOCK

	def findconveyerbelt_imgsignal_process(self, position):
		self.plchandle.write_position(position)
		self.currentstatus_edit.setText("袋子要到传送带，先向x,y,z各自移动{},{},{}米".format(*position))
		self.intelligentthread.hockstatus = HockStatus.FIND_CONVEYERBELT

	def drop_bag_imgsignal_process(self, position):
		self.plchandle.write_position(position)
		self.currentstatus_edit.setText("移动袋子抵达传送带，向下移动钩子{}米,放下袋子".format(position[2]))
		self.intelligentthread.hockstatus = HockStatus.DROP_BAG

	def reback_imgsignal_process(self, position):
		self.plchandle.write_position(position)
		print("放下钩子")
		self.currentstatus_edit.setText("复位")
		self.intelligentthread.hockstatus = HockStatus.POSITION_NEARESTBAG

	def finish_imgsignal_process(self, info):
		# print("工作完成")
		# print(info)
		self.plchandle.ugent_stop()
		self.plcthread.work = False

	def afterreback(self, info):
		'''处理重置'''
		print(info)
		self.plcthread.work = True
		self.intelligentthread.work = False

	def landmark_location(self, image=None):
		if image:
			img = cv2.imread(image)
		else:
			img = cv2.imread('d:/2020-04-10-15-26-22test.bmp')
		# with PointLocationService(img=img) as  a:
		# 	locationinfo = a.find_nearest_bag()
		# 	if locationinfo is not None:
		# 		nearest_bag_position, hockposition = locationinfo
		# 		img_distance, real_distance, real_x_distance, real_y_distance = a.compute_distance(
		# 			nearest_bag_position, hockposition)
		# 		mylog_debug("最近的袋子距离钩子:{}公分".format(real_distance))
		# # img = a.move()
		dest = start_location_landmark(img=img)

		# img = cv2.resize(a.img, (900, 700))
		show = cv2.cvtColor(dest, cv2.COLOR_BGR2RGB)
		showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
		self.img_play.setPixmap(QPixmap.fromImage(showImage))
		self.img_play.setScaledContents(True)

	def set_roi(self,points):
		print("ok")
		print(points)
		point1, point2 = points

		# print('first:({}{})'.format(point1[0], point1[1]))
		# print('second:({}{})'.format(point2[0], point2[1]))

		def img_roi_position(label_x, label_y):
			label_w, label_h = self.img_play.size().width(), self.img_play.size().height()
			raio = min(IMG_WIDTH / label_w, IMG_HEIGHT / label_h)
			img_w = int(raio * label_x) - 50
			img_h = int(raio * label_y) - 50
			# img_w=label_x
			# img_h=label_y
			return img_w, img_h

		# 必须是经过放射变换的
		image = self.img_play.copy_img()
		#
		#
		point1_modify = img_roi_position(point1[0], point1[1])
		point2_modify = img_roi_position(point2[0], point2[1])
		cv2.rectangle(image, point1_modify, point2_modify, (0, 255, 0), 2)
		self.img_play.set_img(image)

		# refPt = []
		# image = img
		#
		# def mouse_click(event, x, y, flags, param):
		# 	# grab references to the global variables
		# 	global refPt
		# 	if event == cv2.EVENT_LBUTTONDOWN:
		# 		refPt = [(x, y)]
		# 		cv2.circle(img=img, center=(x, y), color=(0, 255, 127), radius=3, thickness=2)
		# 	elif event == cv2.EVENT_MOUSEMOVE and event == cv2.EVENT_LBUTTONDOWN:
		# 		cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
		# 	elif event == cv2.EVENT_LBUTTONUP:
		# 		refPt.append((x, y))
		# 		# drawing == False
		# 		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		# 		cv2.circle(img=image, center=(refPt[0][0], refPt[1][1]), color=(0, 255, 127), radius=3, thickness=2)
		# 		cv2.circle(img=image, center=(refPt[1][0], refPt[1][1]), color=(0, 255, 127), radius=3, thickness=2)
		# 		cv2.circle(img=image, center=(refPt[1][0], refPt[0][1]), color=(0, 255, 127), radius=3, thickness=2)
		# 		cv2.imshow("image", image)
		#
		# def set_roi(image):
		# 	clone = image.copy()
		# 	cv2.namedWindow("image", 0)
		# 	cv2.setMouseCallback("image", mouse_click)
		# 	# keep looping until the 'q' key is pressed
		# 	cv2.imshow("image", image)
		# 	# if there are two reference points, then crop the region of interest
		# 	# from teh image and display it
		# 	if len(refPt) > 1:
		# 		roi = clone[refPt[0][1]:refPt[1][1], refPt[-1][0]:refPt[-1][0]]
		# 		cv2.imshow("ROI", roi)
		#
		# 	cv2.waitKey(0)
		# 	cv2.destroyAllWindows()
		#
		# set_roi(image)


