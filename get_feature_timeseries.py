import os
import numpy as np
import glob
import pandas as pd
from python_speech_features import logfbank, mfcc
import matplotlib.pyplot as plt

path_feature_cmplt = "/home/abkoesdw/Documents/cognitech/ultrasound-project/ultrasound-sensor-data/cmplt/"
path_feature = "/home/abkoesdw/Documents/cognitech/ultrasound-project/feature/"
lag = 5
k = 0
for filename in os.listdir(path_feature_cmplt):
    if filename.endswith(".npz"):
        file_ = os.path.splitext(filename)[0]
        print("file :", file_)
    else:
        continue

    data = np.load(path_feature_cmplt + file_ + '.npz')
    input_ = data['input']
    input_ = np.append(np.zeros(lag), input_)
    output_ = data['output']
    i = 0
    feature = np.empty((0, lag + 1))
    while (i + lag) < len(input_):
        input_temp = input_[i:i+lag+1]
        input_temp = input_temp[::-1]
        input_temp = np.reshape(input_temp, (1, lag + 1))
        feature = np.append(feature, input_temp, axis=0)
        i += 1

    target = output_
    print(feature.shape, target.shape)
    np.savez(path_feature + file_ + '_feature.npz', feature=feature,
             target=output_)

    k += 1
    if k >= 1:
        break


