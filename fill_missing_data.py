import os
import numpy as np
import glob
import pandas as pd

path_feature = "/home/abkoesdw/Documents/cognitech/ultrasound-project/ultrasound-sensor-data/"

timestamp_cmplt_temp = pd.date_range('2016-11-21 00:00:00', periods=86401, freq='s')



x = timestamp_cmplt_temp[0]
y = timestamp_cmplt_temp[1]
z = timestamp_cmplt_temp[2]
print(str(x.date()), str(x.time()))

# np.save('timestamp_cmplt.npy', timestamp_cmplt)
# print(timestamp_cmplt)
files = glob.glob(path_feature + "*.npz")

for file_ in files:
    data = np.load(file_)
    timestamp = data['timestamp']
    input_ = data['input_']
    output_ = data['output_']

