import os
import numpy as np
import glob
import pandas as pd
from python_speech_features import logfbank, mfcc
import matplotlib.pyplot as plt
import matplotlib.dates as md
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
import dateutil
path_feature_cmplt = "/home/abkoesdw/Documents/cognitech/ultrasound-project/ultrasound-sensor-data/cmplt/"
path_feature = "/home/abkoesdw/Documents/cognitech/ultrasound-project/feature/"
winlen = 64
k = 0
timestamp = pd.date_range('2016-11-21 00:00:00', periods=86401, freq='s')

for filename in os.listdir(path_feature_cmplt):
    if filename.endswith(".npz"):
        file__ = os.path.splitext(filename)[0]

    else:
        continue
    if file__[3] == 'p':
        file_ = file__
        print("file :", file_)
        k += 1
    else:
        continue
    data = np.load(path_feature_cmplt + file_ + '.npz')
    input_ = data['input']

    plt.figure(1)
    plt.plot(input_, label='ut_pv_' + str(k))
    plt.xlabel('time')
    plt.ylabel('ut_pv')
    plt.legend()
    input_ = np.append(np.zeros(winlen-1), input_)
    feature_temp_1 = logfbank(input_, 1, winlen=winlen, winstep=1)
    feature_temp_2 = data['output'].reshape(len(feature_temp_1), 1)
    feature_temp_3 = (data['output'].reshape(len(feature_temp_1), 1) * -1) + 1

    # print(feature_temp1.shape, feature_temp2.shape)
    feature_temp1 = np.concatenate((feature_temp_1, feature_temp_2), axis=1)
    feature_temp2 = np.concatenate((feature_temp_1, feature_temp_3), axis=1)
    feature = np.concatenate((feature_temp1, feature_temp2), axis=0)

    plt.figure(2)
    plt.plot(45 * feature_temp_2, label='status_' + str(k))
    plt.xlabel('time')
    plt.ylabel('status')

    target_temp1 = np.zeros((len(feature_temp1), 1))
    target_temp2 = np.ones((len(feature_temp2), 1))

    target = np.concatenate((target_temp1, target_temp2), axis=0)
    idx = np.arange(len(target))
    np.random.shuffle(idx)
    feature = feature[idx, :]
    target = target[idx, :]
    idx_0 = np.where(target == 0)[0]
    idx_1 = np.where(target == 1)[0]
    idx_train = np.concatenate((idx_0[0:int(len(idx_0) * 5/10)], idx_1[0:int(len(idx_1) * 1 / 20)]))
    idx_test = np.concatenate((idx_0[int(len(idx_0) * 5/10):],
                               idx_1[int(len(idx_1) * 1 / 20):int(len(idx_1) * 1 / 20)+int(len(idx_1) * 1 / 20)]))

    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = feature[idx_train, :]
    x_train = min_max_scaler.fit_transform(x_train)
    y_train = target[idx_train, :]
    print(x_train.shape, y_train.shape)
    y_train = to_categorical(np.int16(y_train[:, 0]), nb_classes=2)

    x_test = feature[idx_test, :]
    x_test = min_max_scaler.transform(x_test)
    y_test = target[idx_test, :]
    print(x_test.shape, y_test.shape)
    y_test = to_categorical(np.int16(y_test[:, 0]), nb_classes=2)

    if k >= 2:
        break
    np.savez(path_feature + file_ + '_fbank.npz', x_train=x_train, y_train=y_train, x_test=x_test,
             y_test=y_test)


#
# plt.figure(2)
# plt.plot(input_)
# plt.plot(45*feature_temp_2, 'r')
plt.legend()
plt.show()
