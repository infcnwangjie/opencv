import cv2


class Shape:
	TRIANGLE = 3
	RECTANGLE = 4
	SQUARE = 5
	PENTAGON = 6
	CIRCLE = 7

class ShapeDetector:
	def detect(self, c, shape_flag=7):
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04 * peri, True)

		print(len(approx))
		# if the shape is a triangle, it will have 3 vertices
		shape=0
		if len(approx) == 3:
			shape = Shape.TRIANGLE

		# if the shape has 4 vertices, it is either a square or
		# a rectangle
		elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)

			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
			shape = Shape.SQUARE if ar >= 0.95 and ar <= 1.05 else Shape.RECTANGLE

		# if the shape is a pentagon, it will have 5 vertices
		elif len(approx) == 5:
			shape = Shape.PENTAGON
		# otherwise, we assume the shape is a circle
		else :
			shape = Shape.CIRCLE
		# return the name of the shape
		return shape == shape_flag
