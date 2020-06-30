import matplotlib.pyplot as plt
import csv
import numpy as np


data = []
with open('C:\\Users\\ytjun\\Desktop\\kubo\\kubo\\data\\data0.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)
datas = np.array(data,dtype="float")
print(datas.shape)
datas1 = np.zeros([160,160],dtype="float")
print(datas1)
[rows,cols] = datas.shape

for i in range(rows):
    for j in range(cols):
        if(datas[i,j]>0):
            datas[i,j] =255
            datas1[i,j] = datas[i,j]
# x = np.empty([rows,16], dtype = "float")
# #print(datas)
# for i in range(49):
#     if(i+ % 3 ==0 ):
#         for j in range(15):
#             x[:,j] = datas[:,i-1]
# print(x.shape)
# np.savetxt('new1.csv', x, delimiter = ',')
print(datas.shape)
print(datas1.shape)
plt.imshow(datas1)
plt.show()

import scipy.misc
scipy.misc.imsave('C:\\Users\\ytjun\\Desktop\\kubo\\kubo\\test\\t2\\1.jpg', datas1)
img = plt.imread("C:\\Users\\ytjun\\Desktop\\kubo\\kubo\\test\\t2\\1.jpg")
plt.imshow(img)
plt.show()


