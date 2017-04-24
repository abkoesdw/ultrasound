import os
import numpy as np
import glob
import pandas as pd
from operator import itemgetter, attrgetter

path_data = "/home/abkoesdw/Documents/cognitech/ultrasound-project/ultrasound-sensor-data/"
path_txt = "/home/abkoesdw/Documents/cognitech/keyword-project/mfctxt-files-arief/"
path_feature = "/home/abkoesdw/Documents/cognitech/ultrasound-project/ultrasound-sensor-data/"
path_feature_cmplt = "/home/abkoesdw/Documents/cognitech/ultrasound-project/ultrasound-sensor-data/cmplt/"

if not os.path.exists(path_feature):
    os.makedirs(path_feature)

files = glob.glob(path_feature + "*.npz")
timestamp = np.load(path_feature + 'timestamp.npy')

# for file_ in files:
#     os.remove(file_)

# if os.path.exists(path_feature + "timestamp.txt"):
#     os.remove(path_feature + "timestamp.txt")

# files = os.listdir(path_data)
#
k = 0
for file_ in files:
    data = np.load(file_)
    filename = file_[77:-8]
    print(filename)
    timestamp = data['timestamp']
    input_ = data['input_'].reshape(len(timestamp), 1)
    output_ = data['output_'].reshape(len(timestamp), 1)
    ts_cmplt = pd.date_range('2016-11-21 00:00:00', periods=86401, freq='s')
    ts = pd.Series(timestamp)

    df = pd.DataFrame(np.append(input_, output_, axis=1), index=ts, columns=['input', 'output'])
    df.index = pd.to_datetime(df.index)
    df2 = df.reindex(ts_cmplt)
    inds = pd.isnull(df2).any(1).nonzero()[0]
    # print(inds)
    df2 = df2.fillna(method='pad')
    # print(df)
    # print(df2)

    df2.to_csv(path_feature + 'df2.txt', header=None, index=True, sep=' ', mode='w')
    df.to_csv(path_feature + 'df1.txt', header=None, index=True, sep=' ', mode='w')
    # print(df2.index)
    np.savez(path_feature_cmplt + filename + '_complete.npz', timestamp=df2.index,
             input=df2['input'], output=df2['output'])
    k += 1
    # if k >= 1:
    #     break
#
#
#
