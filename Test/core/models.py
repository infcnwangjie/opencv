import cv2


class DigitLocation:
	def __init__(self, digitvalue, locationpoint, bagcenterpoint, boxid):
		self.boxid = boxid
		self.digitvalue = digitvalue
		self.boxcenterpoint_x, self.boxcenterpoint_y = bagcenterpoint
		self.locationpoint_x, self.locationpoint_y = locationpoint

	def __str__(self):
		return "boxid:{boxid}-digitvalue:{digitvalue}-x:{x}-y:{y}-box_x:{box_x}-box_y:{box_y}".format(
			boxid=self.boxid,
			digitvalue=self.digitvalue,
			x=self.locationpoint_x,
			y=self.locationpoint_y,
			box_x=self.boxcenterpoint_x,
			box_y=self.boxcenterpoint_y)


class Box:
	def __init__(self, contour, img, id=1, digitdetector=None):
		if img is None:
			raise Exception("box img must not none")
		self.id, self.img = id, img
		self.contour = contour
		self.box = cv2.boundingRect(contour)
		self.x, self.y, self.w, self.h = self.box
		self.boxcenterpoint = (self.x + round(self.w * 0.5), self.y + round(self.h * 0.5))
		self.roi_contours, self.thresh = None, None
		self.digitLocations = []
		self.box_content = ""
		self.has_compute_contours = False
		# 数字检测对象
		self.digitdetector = digitdetector
		self.compute_iner_contours()

	def __str__(self):
		return 'id:{id}-box_center_x:{box_center_x}-box_center_y:{box_center_y}'.format(id=self.id, box_center_x=
		self.boxcenterpoint[0],
		                                                                                box_center_y=
		                                                                                self.boxcenterpoint[1])

	# 内部使用,通过轮廓面积过滤轮廓
	def __contours_area_filter(self, c, minarea=50, maxarea=3000):
		[x1, y1, w1, h1] = cv2.boundingRect(c)
		area = cv2.contourArea(c)
		# and h1 > h * 0.50
		return maxarea > area > minarea

	# 内部使用，计算box中的所有数字轮廓
	def compute_iner_contours(self):
		if self.roi_contours is None and self.has_compute_contours == False:
			self.has_compute_contours = True
			x, y, w, h = self.box
			roi_img = self.img[y + 1:y + h, x + 1:x + w]
			if roi_img is None:
				raise Exception("roi_img 拷贝失败！")
			roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
			ret, thresh = cv2.threshold(roi_gray, 127, 255, cv2.THRESH_BINARY_INV)  # 简单阈值
			# 在特征区域中再次寻找轮廓
			roi_contours, hierarchy1 = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
			roi_contours = [contour for contour in roi_contours if self.__contours_area_filter(contour)]
			self.roi_contours = roi_contours
			self.thresh = thresh

	@property
	def inercontours(self):
		if self.has_compute_contours:
			return self.roi_contours
		else:
			self.compute_iner_contours()
		return self.roi_contours

	# 修改目标物的显示内容
	def modify_box_content(self, digitdetector, no_num=True):
		# 如果box内部没有内部轮廓，就直接退出循环
		if no_num:
			# x, y, w, h = self.box
			# self.box_content = "bag_location:{},bag_area:{} ，width:{} ,height:{}".format(
			# 	"->(" + str(self.boxcenterpoint[0]) + "," + str(self.boxcenterpoint[1]) + ")",
			# 	cv2.contourArea(self.contour), w, h)
			self.box_content = "bag_location:" + "->(" + str(self.boxcenterpoint[0]) + "," + str(
				self.boxcenterpoint[1]) + ")"
			return

		# x, y, w, h = self.box
		# self.box_content = "bag_location:{},bag_area:{} ，width:{} ,height:{}".format(
		# 	"->(" + str(self.boxcenterpoint[0]) + "," + str(self.boxcenterpoint[1]) + ")",
		# 	cv2.contourArea(self.contour), w, h)
		for digital_contour in self.inercontours:
			[digit_point_x, digit_point_y, digit_contor_width, digit_contor_height] = cv2.boundingRect(
				digital_contour)
			roi = self.thresh[digit_point_y:digit_point_y + digit_contor_height,
			      digit_point_x:digit_point_x + digit_contor_width]
			results = digitdetector.readnum(roi)
			roi_digitvalue = str(int((results[0][0])))
			cv2.drawContours(roi, digital_contour, -1, (0, 0, 205), 1)
			boxdigitlocation = DigitLocation(digitvalue=roi_digitvalue, boxid=self.id,
			                                 bagcenterpoint=self.boxcenterpoint,
			                                 locationpoint=(
				                                 self.x + digit_point_x, self.y + digit_point_y))
			self.digitLocations.append(boxdigitlocation)
		# modify locations if has locations
		if self.digitLocations is None or len(self.digitLocations) == 0:
			return
		self.digitLocations.sort(key=lambda location: location.locationpoint_x, reverse=False)
		# 用于拼接数字，当然遇到6,8的时候回检测出两个轮廓，用x轴之差决定是否拼接
		last_point_x, box_digitnum = 0, ""
		for location in self.digitLocations:
			current_x = location.locationpoint_x
			if current_x - last_point_x > 10:
				box_digitnum += location.digitvalue
			last_point_x = current_x
		self.box_content = box_digitnum + "->(" + str(self.boxcenterpoint[0]) + "," + str(
			self.boxcenterpoint[1]) + ")"
