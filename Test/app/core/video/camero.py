import cv2

class CameroService(object):
	def __init__(self,video_file=None):
		self.open = True
		if video_file:
			self.cap = cv2.VideoCapture(video_file)
		else:
			self.cap = cv2.VideoCapture(0)

	def run(self):
		while self.open:
			ret, frame = self.cap.read()
			cv2.imshow("cap", frame)
			if cv2.waitKey(100) & 0xff == ord('q'):
				break
		self.cap.release()
		cv2.destroyAllWindows()


if __name__ == '__main__':
	CameroService().run()