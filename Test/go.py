from functools import cmp_to_key

land_mark, (x, y) = ('NO1_L', (151, 0))


# print(x,y)
#
#
# print(land_mark)


def get_next_no(landmark_name):
	import re
	result = re.match(r'''NO(?P<NO>[0-9]*)_(?P<direct>[A-Z]{1})''', "NO1_R")

	if result is None:
		return land_mark

	current_no = int(result.group(1))

	if current_no > 6:
		next_no = current_no - 1
	else:
		next_no = current_no + 1

	next_landmark = "NO{NO}_{D}".format(NO=next_no, D=result.group(2))
	return next_landmark


print(get_next_no("NO1_R"))


def custom_sort(x,y):
	if x>y:
		return -1
	if x<y:
		return 1
	return 0

a=sorted([2,4,5,7,3],key=cmp_to_key(custom_sort),reverse=True)
print(a)