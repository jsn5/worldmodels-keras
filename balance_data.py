import numpy as np
import os

count = 0
for root,subdirs,files in os.walk("data/"):
	count = len(files)


left = []
right = []
accel = []
no_accel = []
for i in range(count):
	data = np.load("data/training_data_{}.npy".format((i+1)*1000))
	for row in data:
		if row[1][0] == 1:
			no_accel.append(row)
		if row[1][1] == 1:
			accel.append(row)
		if row[1][2] == 1:
			left.append(row)
		if row[1][3] == 1:
			right.append(row)


lengths = np.array([len(no_accel),len(accel),len(left),len(right)])

data = [no_accel,accel,left,right]



final_data = []

size = len(data[np.argmin(lengths)])

for i in data:
	np.random.shuffle(i)
	final_data+=i[:size]

print(size,len(final_data))
np.save("final_data.npy",final_data)
