file_r=open("D:\\行车项目数据分析报告\改进前\\2020-08-25.log",'rt')

for line in file_r:
	if "laster is" in line or "bag contour" in line:
		continue
	print(line)