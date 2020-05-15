
# 透视变化
def perspective_transform(src, position_dic):
	H_rows, W_cols = src.shape[:2]
	print(H_rows, W_cols)

	with open(os.path.join(PROGRAM_DATA_DIR, 'coordinate_data.txt'), 'rb') as coordinate:
		real_positions = pickle.load(coordinate)
	real_position_dic = {key: [int(float(value.split(',')[0])), int(float(value.split(',')[1]))] for key, value in
	                     real_positions.items()}
	print(real_position_dic)

	img_left_2, img_right_2, img_left_4, img_right_4 = position_dic.get('NO2_L'), position_dic.get(
		'NO2_R'), position_dic.get('NO4_L'), position_dic.get('NO4_R')

	real_left_2, real_right_2, real_left_4, real_right_4 = real_position_dic.get('NO2_L'), real_position_dic.get(
		'NO2_R'), real_position_dic.get('NO4_L'), real_position_dic.get('NO4_R')

	# 原图中四个角点(左上、右上、左下、右下),与变换后矩阵位置
	# pts1 = np.float32([[161, 80], [449, 12], [1, 430], [480, 394]])
	pts1 = np.float32([img_left_2, img_right_2, img_left_4, img_right_4])
	pts2 = np.float32([real_left_2, real_right_2, real_left_4, real_right_4])
	# pts2 = np.float32([[0, 0], [W_cols, 0], [0, H_rows], [H_rows, W_cols] ])

	# 生成透视变换矩阵；进行透视变换
	M = cv2.getPerspectiveTransform(pts1, pts2)

	dst = cv2.warpPerspective(src, M, (W_cols, H_rows))

	return dst, real_position_dic