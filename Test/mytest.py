# -*- coding: utf-8 -*-
import cv2
import time

# https://baijiahao.baidu.com/s?id=1615404760897105428&wfr=spider&for=pc
from app.core.processers.preprocess import Preprocess


class SimilarSlideWindow:
	def __init__(self, col, row, similarity):
		self.col = col
		self.row = row
		self.similarity = similarity


class LandMarkRoi:
	def __init__(self, img, label):
		self.roi = img
		self.label = label
		self.best_similar_window = None
		self.best_similar_score = 0

	def add_match_slide(self, slide_window: SimilarSlideWindow):
		if slide_window.similarity > self.best_similar_score:
			self.best_similar_score
			self.best_similar_window = slide_window

	@property
	def match_window(self):
		return self.best_similar_window if self.best_similar_window else None


def tjtime(fun):
	def inner(*args, **kwargs):
		start = time.clock()
		result = fun(*args, **kwargs)
		end = time.clock()
		print("{}cost {}秒".format(fun.__name__, end - start))
		return result

	return inner


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		print("x:{},y:{}".format(x, y))


# @tjtime
def color_similar_ratio(image1, image2):
	if image1 is None or image2 is None:
		return 0
	img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
	img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
	hist1 = cv2.calcHist([img1], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
	cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
	# cv2.imshow("hist1",hist1)
	hist2 = cv2.calcHist([img2], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
	cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
	degree = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  # HISTCMP_BHATTACHARYYA    HISTCMP_CORREL

	return degree


def slide():
	img = cv2.imread("D:/2020-04-10-15-26-22test.bmp")
	dest = cv2.resize(img, (1000, 800))
	gray = cv2.cvtColor(dest, cv2.COLOR_BGR2GRAY)
	rows, cols = gray.shape

	for row in range(0, rows):

		for col in range(156, 213):
			# print("-" * 1000)
			yield (col, row, dest[row:row + 30, col:col + 30])
		for col in range(850, 890):
			# print("-" * 1000)
			yield (col, row, img[row:row + 30, col:col + 30])


def my_testslide():
	landmark_rois = []
	landmark_rois.append(LandMarkRoi(img=cv2.imread("D:/red.png"), label='red'))
	landmark_rois.append(LandMarkRoi(img=cv2.imread("D:/greenyellow.png"), label='greenyellow'))
	landmark_rois.append(LandMarkRoi(img=cv2.imread("D:/yellow_red.png"), label='yellow_red'))
	landmark_rois.append(LandMarkRoi(img=cv2.imread("D:/red_green.png"), label='red_green'))
	# loc = cv2.setMouseCallback("landmark", on_EVENT_LBUTTONDOWN)
	srcimg = cv2.imread("D:/2020-04-10-15-26-22test.bmp")
	dest = cv2.resize(srcimg, (1000, 800))
	gray = cv2.cvtColor(dest, cv2.COLOR_BGR2GRAY)
	rows, cols = gray.shape

	for row in range(0, rows):
		for left_col in range(156, 213):
			show_img = dest.copy()
			cv2.rectangle(show_img, (left_col, row), (left_col + 30, row + 30), color=(255, 255, 0), thickness=2)
			# cv2.imshow("img", show_img)
			print("左侧滑窗位置 x:{},y:{}".format(left_col, row))
			# index += 1
			for landmark_roi in landmark_rois:
				roi = cv2.resize(landmark_roi.roi, (30, 30))
				img = dest[row:row + 30, left_col:left_col + 30]
				similar = color_similar_ratio(roi, img)
				print("正在匹配roi:{} simailar:{}".format(landmark_roi.label, similar))

				if similar > 0.6:
					print("find {} roi similar is {}".format(landmark_roi.label, similar))

					landmark_roi.add_match_slide(SimilarSlideWindow(col=left_col, row=row, similarity=similar))
					cv2.rectangle(show_img, (left_col, row), (left_col + 30, row + 30), color=(255, 255, 0), thickness=2)
					cv2.putText(show_img, "find {} roi similar is {}".format(landmark_roi.label, similar),
					            (left_col + 15, row + 15),
					            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)
					cv2.namedWindow("target")
					cv2.imshow("target", show_img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		for right_col in range(860, 894):
			show_img = dest.copy()
			cv2.rectangle(show_img, (right_col, row), (right_col + 30, row + 30), color=(255, 255, 0), thickness=2)
			# cv2.imshow("img", show_img)
			print("右侧滑窗位置 x:{},y:{}".format(right_col, row))
			# index += 1
			for landmark_roi in landmark_rois:
				roi = cv2.resize(landmark_roi.roi, (30, 30))
				img = dest[row:row + 30, right_col:right_col + 30]
				similar = color_similar_ratio(roi, img)

				print("正在匹配roi:{} simailar:{}".format(landmark_roi.label, similar))

				if similar > 0.6:
					print("find {} roi similar is {}".format(landmark_roi.label, similar))
					landmark_roi.add_match_slide(SimilarSlideWindow(col=right_col, row=row, similarity=similar))
					cv2.rectangle(show_img, (right_col, row), (right_col + 30, row + 30), color=(255, 255, 0), thickness=2)
					cv2.putText(show_img, "find {} roi similar is {}".format(landmark_roi.label, similar),
					            (right_col + 15, row + 15),
					            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)
					cv2.namedWindow("target")
					cv2.imshow("target", show_img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	for landmarkroi in landmark_rois:
		final_img = dest.copy()
		if landmarkroi.match_window is not None:
			cv2.rectangle(final_img, (landmarkroi.match_window.col, landmarkroi.match_window.row),
			              (landmarkroi.match_window.col + 30, landmarkroi.match_window.row + 30),
			              color=(255, 255, 0), thickness=2)
		cv2.namedWindow("final")
		cv2.imshow("final", final_img)
	# cv2.rectangle(final_img, (col, row), (col + 30, row + 30), color=(255, 255, 0), thickness=2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# for col, row, img in slide():
# print("+"*100)
# print("rows:{},cols:{}".format(row,col))


if __name__ == '__main__':
	# image1 = cv2.imread("D:/roi1.png")
	# image2 = cv2.imread("D:/target_gy.png")
	# i = color_similar_ratio(image1, image2)
	# print("color,相似度为:{}".format(i))
	my_testslide()
