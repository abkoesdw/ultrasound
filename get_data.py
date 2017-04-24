import os
import numpy as np
import glob
import pandas as pd
from operator import itemgetter, attrgetter

path_data = "/home/abkoesdw/Documents/cognitech/ultrasound-project/ultrasound-sensor-data/"
path_feature = "/home/abkoesdw/Documents/cognitech/ultrasound-project/ultrasound-sensor-data/"

if not os.path.exists(path_feature):
    os.makedirs(path_feature)

files = glob.glob(path_feature + "*.npz")

for file_ in files:
    os.remove(file_)

if os.path.exists(path_feature + "timestamp.txt"):
    os.remove(path_feature + "timestamp.txt")

files = os.listdir(path_data)
print(files[0])
timestamp_cmplt = pd.date_range('2016-11-21 00:00:00', periods=86401, freq='s')

with open(path_data + files[0], 'r') as f:
    lines = f.readlines()
    rows = [line.split()[0:] for line in lines[1:]]
    data_sorted = np.sort(rows, axis=0)
    # print(data_sorted)
    timestamp_temp = data_sorted[:, 0]
    timestamp = []
    count = 1
    for string_ in timestamp_temp:
        a = string_[0:10]
        b = ' '
        c = string_[11:]
        d = a + b + c
        timestamp = np.append(timestamp, d)
        print(count)
        count += 1

np.savetxt(path_feature + 'timestamp.txt', timestamp, fmt='%s')
np.save('timestamp.npy', timestamp)
k = 0
for file in files:
    print(file)
    with open(path_data + file, 'r') as f:
        lines = f.readlines()
        rows = [line.split()[0:] for line in lines[1:]]

    data_sorted = np.sort(rows, axis=0)

    input_ = np.float64(data_sorted[:, 2])
    output_ = np.int16(data_sorted[:, 1])
    np.savez(path_feature + file,
             timestamp=timestamp,
             input_=input_,
             output_=output_)
    k += 1
    # if k >= 1:
    #     break

print(timestamp)
