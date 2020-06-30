import serial
import numpy as np
import csv
import time
import os

#right sensor
ser = serial.Serial('COM4', 921600, timeout=None,
                    bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)
#init a zerofilled array to record average array
ave_arr = [0]*48


# arrange data into 3x16
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


if __name__ == '__main__':

    count = 0
    file_count = 0
    char = 'w'
    path_name = char + str(int(time.time()))
    if not os.path.exists('data/'+path_name):
        os.mkdir('data/'+path_name)
    temp_data = []
    
    while True:
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
                        ave_arr = list(np.array(ave_arr)/count)
                        print('start')
                    else:
                        # since it output really fast, I concerned to arrange them by looping only once
                        arr,sum = subtractAverageRemoveNoiseGetSum(arr)
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
                            f = open('data/'+path_name+'/'+str(file_count)+'.csv','w')
                            f_csv = csv.writer(f)
                            f_csv.writerows(temp_data)
                            f.close()
                            temp_data =[]
                            print('\n\n\n\n\n\n\n\n'+str(file_count)+'.csv\n\n\n\n\n\n\n\n')

                            file_count += 1
                            time.sleep(1)
                            print('OK GO')


                    count += 1

    else:
        ser.close()


