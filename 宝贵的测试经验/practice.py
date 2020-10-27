def cal_sequence(n):
	total_sequence = list(range(1, n + 1))
	start = 0
	while start < n :
		for end in range(start+1,n):
			total=sum(total_sequence[start:end])
			if total==n:
				print(total_sequence[start:end])
		start+=1



def str_split(str_info):
	if str_info is None:
		return ""
	if len(str_info)==1:
		return str_info
	elif len(str_info)==2:
		a,b=str_info
		return (a,b),(b,a)
	else:
		return str_split(str_info[:len(str_info)/2]),str_split(str_info[len(str_info)/2+1:])


result=str_split("abc")
print(result)
# print("abc"[2:])