import matplotlib.pyplot as plt
import csv
import numpy as np


data = []
with open('C:\\Users\\SawadaLab\\PycharmProjects\\kubo\\kubo\\data\\1.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)
datas = np.array(data,dtype="float")
[rows, cols] = datas.shape
for i in range(rows):
    for j in range(cols):
        if(datas[i,j] != 0):
            datas[i,j]=1
k=0
datas_c = datas
for i in range(rows):
    for j in range(cols):
        if(i<(rows-1) and datas[i,j]==1 and datas_c[i+1,j]==1):
            datas[i,j] =0

# for i in range(rows):
#     for j in range(cols):
#         if(datas[i,j] ==1):
#             k = k+1
# print(k)
for i in range(rows):
    for j in range(cols):
        if(datas[i,j] ==1):
            k = k+1
print(k)
print(datas.shape)


plt.imshow(datas)
plt.show()
