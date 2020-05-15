#绘制网格线
def draw_grid_lines(img):
	H_rows, W_cols = img.shape[:2]
	for row in range(0,H_rows):
		if row % 50==0:
			cv2.line(img,(0,row),(W_cols,row),color=(191,62,255),thickness=1,lineType=cv2.LINE_8)
	for col in range(0,W_cols):
		if col %50 ==0:
			cv2.line(img,(col,0),(col,H_rows),color=(191,62,255),thickness=1,lineType=cv2.LINE_8)