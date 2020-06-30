import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import serial
import numpy as np
import csv
import time
import os


def arrangeTo2D(arr):
    step=3
    arr=[arr[i:i+step] for i in range(0,len(arr),step)]

    return arr

# delete all number which absolute value smaller than 100
def subtractAverageRemoveNoiseGetSum(arr):
    sum = 0
    for i in range(len(arr)):
        # we found out that output generally go above 100 while it's working
        if abs(arr[i]-ave_arr[i]) < 100:
            arr[i] = 0
        else:
            arr[i] = arr[i]-ave_arr[i]
            sum += abs(arr[i])

    return arr,sum

# convert array to Int
def convertToInt(arr):
    for i in range(len(arr)):
        if arr[i].isnumeric():
            arr[i] = int(arr[i])
        else:
            return [0]
    return arr

def loadtraindata():
    path = r"C:\\Users\\ytjun\\Desktop\\kubo\\kubo\\train"                                         # 路径
    trainset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((160, 160)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小

                                                    transforms.CenterCrop(160),
                                                    transforms.ToTensor()])
                                                )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=True, num_workers=2)
    return trainloader


class Net(nn.Module):  # 定义网络，继承torch.nn.Module
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 卷积层
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.conv2 = nn.Conv2d(6, 16, 5)  # 卷积层
        self.fc1 = nn.Linear(21904, 120)  # 全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3
                             )  # 10个输出

    def forward(self, x):  # 前向传播

        x = self.pool(F.relu(self.conv1(x)))  # F就是torch.nn.functional
        x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)  # .view( )是一个tensor的方法，使得tensor改变size但是元素的总数是不变的。
        x = x.view(-1, 21904)  # .view( )是一个tensor的方法，使得tensor改变size但是元素的总数是不变的。
        # 从卷基层到全连接层的维度转换

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
classes = ('0','1')

def loadtestdata():
    ser = serial.Serial('COM4', 921600, timeout=None,
                        bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)
    # init a zerofilled array to record average array
    ave_arr = [0] * 48

    count = 0
    file_count = 0
    char = 'w'
    path_name = 'test1'
    # path_name = char + str(int(time.time()))
    if not os.path.exists('data/' + path_name):
        os.mkdir('data/' + path_name)
    temp_data = []
    for i in range(2):
        for line in ser.read():
            raw_data = ser.readline()
            try:
                raw_data = str(raw_data, 'utf-8')
            except UnicodeDecodeError:
                pass
            else:
                raw_data = raw_data.strip('AB\n')
                arr = raw_data.split(',')
                arr = convertToInt(arr)

                if len(arr) == 48:
                    if count < 100:
                        ave_arr = list(np.array(ave_arr) + np.array(arr))
                    elif count == 100:
                        ave_arr = list(np.array(ave_arr) / count)
                        print('start')
                    else:
                        # since it output really fast, I concerned to arrange them by looping only once
                        arr, sum = subtractAverageRemoveNoiseGetSum(arr)
                        # arr = arrangeTo2D(arr)

                        if sum > 200:
                            temp_data.append(arr)
                            print(len(temp_data))
                        # I tested the code and found out that even a gentle touch would generate more than 100 rows
                        elif len(temp_data) < 80:
                            if len(temp_data) > 0:
                                print('\n\n\n\n\n\n\n\nhold on, your hand blured! Start over\n\n\n\n\n\n\n\n')
                                time.sleep(2)
                                print('OK GO')
                            temp_data = []
                        else:
                            f = open('data/' + path_name + '/' + str(file_count) + '.csv', 'w')
                            f_csv = csv.writer(f)
                            f_csv.writerows(temp_data)
                            f.close()
                            temp_data = []
                            print('\n\n\n\n\n\n\n\n' + str(file_count) + '.csv\n\n\n\n\n\n\n\n')

                            file_count += 1
                            time.sleep(1)
                            print('OK GO')

                    count += 1
    ser.close()

    path = r"C:\\Users\\ytjun\\Desktop\\kubo\\kubo\\test1"
    testset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((160, 160)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.ToTensor()])
                                                )
    testloader = torch.utils.data.DataLoader(testset, batch_size=25,
                                             shuffle=True, num_workers=2)
    return testloader

# def loadtestdata():
#     path = r"C:\\Users\\ytjun\\Desktop\\kubo\\kubo\\test"
#     testset = torchvision.datasets.ImageFolder(path,
#                                                 transform=transforms.Compose([
#                                                     transforms.Resize((160, 160)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
#                                                     transforms.ToTensor()])
#                                                 )
#     testloader = torch.utils.data.DataLoader(testset, batch_size=25,
#                                              shuffle=True, num_workers=2)
#     return testloader


def trainandsave():
    trainloader = loadtraindata()
    # 神经网络结构
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 学习率为0.001
    criterion = nn.CrossEntropyLoss()  # 损失函数也可以自己定义，我们这里用的交叉熵损失函数
    # 训练部分
    for epoch in range(5):  # 训练的数据量为5个epoch，每个epoch为一个循环
        # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
        running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
        for i, data in enumerate(trainloader, 0):  # 这里我们遇到了第一步中出现的trailoader，代码传入数据
            # enumerate是python的内置函数，既获得索引也获得数据
            # get the inputs
            inputs, labels = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)  # 转换数据格式用Variable

            optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度

            # forward + backward + optimize
            outputs = net(inputs)  # 把数据输进CNN网络net
            loss = criterion(outputs, labels)  # 计算损失值
            loss.backward()  # loss反向传播
            optimizer.step()  # 反向传播后参数更新
            running_loss += loss.item()  # loss累加
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))  # 然后再除以200，就得到这两百次的平均损失值
                running_loss = 0.0  # 这一个200次结束后，就把running_loss归零，下一个200次继续使用

    print('Finished Training')
    # 保存神经网络
    torch.save(net, 'net.pkl')  # 保存整个神经网络的结构和模型参数
    torch.save(net.state_dict(), 'net_params.pkl')  # 只保存神经网络的模型参数

def reload_net():
    trainednet = torch.load('net.pkl')
    return trainednet

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def test():
    testloader = loadtestdata()
    net = reload_net()
    dataiter = iter(testloader)
    images, labels = dataiter.next()  #
    imshow(torchvision.utils.make_grid(images, nrow=5))  # nrow是每行显示的图片数量，缺省值为8
    print('GroundTruth: '
          , " ".join('%5s' % classes[labels[j]] for j in range(25)))  # 打印前25个GT（test集里图片的标签）
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)

    print('Predicted: ', " ".join('%5s' % classes[predicted[j]] for j in range(25)))
    # 打印前25个预测值
    return predicted

if __name__ == '__main__':




    classes = ('0', '1','2','3','4','5','6','7','8','9')
    #trainandsave()
    test()
    # if classes[a[0]] == '1':
    #     print("1")
    # if classes[a[0]] == '0':
    #     print("0")