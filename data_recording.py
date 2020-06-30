import serial
import numpy as np
import csv
import os
import msvcrt

## Initial parameters
del_num = 0
done = False

## Recording a Baseline
for j in range(0, 100):
    if os.path.exists('data%s.csv' % del_num):
        del_num = del_num + 1
    else:
        break

ser = serial.Serial('COM3', 921600, timeout=0.5,
                    bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)

print('Recording a base line:')
base_list = []
base_axi = np.zeros([48], dtype=float)
for i in range(0, 150):
    try:
        for line in ser.read():
            base_a = ser.readline()
            base_a = str(base_a, 'utf-8')
            without_space = base_a.strip()
            base_x = without_space.strip('ABA')
            base_x = without_space.replace('A', '')
            base_x = base_x.replace('B', '')
            per_num = base_x.split(',')
            base_b = np.array(per_num)
            base_b = base_b.astype(np.float)
            if len(base_b) == 48:
                base_list.append(base_b)
    except ValueError:
        continue

base_array = np.array(base_list)

for n in range(0, 48):
    for k in range(0, 100):
        base_axi[n] = base_axi[n] + base_array[k][n]
    base_axi[n] = base_axi[n] / 100.0

print('Finished')
print("============= Notation =============")
print('1.Press "n" for a next record')
print('2.Press "x" to exit')
print("====================================")

while not done:
    judge = input('Enter "s" to start recording: ')
    if judge == 's':
        done = True
    else:
        print(judge)

## Creat CSV file
csvfile = open('data%s.csv' % del_num, 'w', newline='')
filewrite = csv.writer(csvfile)
filewrite.writerow(
    ['B1S1X', 'B1S1Y', 'B1S1Z', 'B1S2X', 'B1S2Y', 'B1S2Z', 'B1S3X', 'B1S3Y', 'B1S3Z', 'B1S4X', 'B1S4Y', 'B1S4Z',
     'B2S1X', 'B2S1Y', 'B2S1Z', 'B2S2X', 'B2S2Y', 'B2S2Z', 'B2S3X', 'B2S3Y', 'B2S3Z', 'B2S4X', 'B2S4Y', 'B2S4Z',
     'B3S1X', 'B3S1Y', 'B3S1Z', 'B3S2X', 'B3S2Y', 'B3S2Z', 'B3S3X', 'B3S3Y', 'B3S3Z', 'B3S4X', 'B3S4Y', 'B3S4Z',
     'B4S1X', 'B4S1Y', 'B4S1Z', 'B4S2X', 'B4S2Y', 'B4S2Z', 'B4S3X', 'B4S3Y', 'B4S3Z', 'B4S4X', 'B4S4Y', 'B4S4Z'])
print("Csv File 'data%s' Created" % del_num)
print('Recording data...')

while True:
    try:
        for line in ser.read():
            sum_b = 0.0
            base_a = ser.readline()
            base_a = str(base_a, 'utf-8')
            without_space = base_a.strip()
            base_x = without_space.strip('ABA')
            base_x = without_space.replace('A', '')
            base_x = base_x.replace('B', '')
            per_num = base_x.split(',')
            b = np.array(per_num)
            b = b.astype(np.float)
            if len(b) == 48:
                for i in range(0, 48):
                    b[i] = b[i] - base_axi[i]
                    sum_b = sum_b + abs(b[i])
                if sum_b >= 1500:
                    filewrite.writerow(b)



    except ValueError:
        continue

    ## Type n for new recording, Type x to exit
    if msvcrt.kbhit():
        presskey = msvcrt.getwch()
        if presskey == 'n':
            print("============================================================================")
            print("New round of recording")
            del_num = del_num + 1
            csvfile.close()
            csvfile = open('data%s.csv' % del_num, 'w', newline='')
            filewrite = csv.writer(csvfile)
            filewrite.writerow(
                ['B1S1X', 'B1S1Y', 'B1S1Z', 'B1S2X', 'B1S2Y', 'B1S2Z', 'B1S3X', 'B1S3Y', 'B1S3Z', 'B1S4X', 'B1S4Y',
                 'B1S4Z',
                 'B2S1X', 'B2S1Y', 'B2S1Z', 'B2S2X', 'B2S2Y', 'B2S2Z', 'B2S3X', 'B2S3Y', 'B2S3Z', 'B2S4X', 'B2S4Y',
                 'B2S4Z',
                 'B3S1X', 'B3S1Y', 'B3S1Z', 'B3S2X', 'B3S2Y', 'B3S2Z', 'B3S3X', 'B3S3Y', 'B3S3Z', 'B3S4X', 'B3S4Y',
                 'B3S4Z',
                 'B4S1X', 'B4S1Y', 'B4S1Z', 'B4S2X', 'B4S2Y', 'B4S2Z', 'B4S3X', 'B4S3Y', 'B4S3Z', 'B4S4X', 'B4S4Y',
                 'B4S4Z'])
            print("Csv File 'data%s' Created" % del_num)
            print('Recording data...')
        if presskey == 'x':
            sys.exit()
        msvcrt.heapmin()
    presskey = 'N\A'

csvfile.close()
ser.close()


