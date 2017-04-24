import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as md
import pandas as pd
import datetime as dt
with open('batch_10241_edited.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    column = {}
    for h in headers:
        column[h] = []

    for row in reader:
        for h, v in zip(headers, row):
            column[h].append(v)

time_temp = [dt.datetime(2017, 3, 8, 12, 27, 44) + dt.timedelta(seconds=i) for i in range(23179)]

cycle_phase = np.asarray(column['cycle_phase']).astype(np.int).reshape(len(time_temp), 1)
cycle_step = np.asarray(column['cycle_step']).astype(np.int).reshape(len(time_temp), 1)
valve_pv220 = np.asarray(column['chamber_vacuum_valve_pv220']).astype(np.int).reshape(len(time_temp), 1)
ut_pv220 = np.asarray(column['ut_pv220']).astype(np.float).reshape(len(time_temp), 1)

feature_temp = np.concatenate((cycle_phase, cycle_step, ut_pv220), axis=1)
target_temp = np.asarray(valve_pv220).astype(np.int)

list_duration = [4500]
x_train = dict()
y_train = dict()
x_test = dict()
y_test = dict()
time_train = dict()
time_test = dict()
idx_train = dict()
idx_test = dict()
for duration in list_duration:
    feature = feature_temp[2400:duration, :]
    target = target_temp[2400:duration]
    time = np.array(time_temp[2400:duration])

    indices = np.random.permutation(feature.shape[0])
    train_percentage = int(np.floor(0.8 * len(indices)))
    idx_train[duration], idx_test[duration] = indices[:train_percentage], indices[train_percentage:]

    x_train[duration] = feature[idx_train[duration], :]
    y_train[duration] = target[idx_train[duration]]
    time_train[duration] = time[idx_train[duration]]

    x_test[duration] = feature[idx_test[duration], :]
    y_test[duration] = target[idx_test[duration]]
    time_test[duration] = time[idx_test[duration]]

# plt.figure(1)
# ax = plt.gca()
# xfmt = md.DateFormatter('%H:%M:%S')
# ax.xaxis.set_major_formatter(xfmt)
# plt.plot(time_test[2400], x_test[2400][:, -1], '.')
# plt.show()

np.savez('dataset3.npz', x_train=x_train, y_train=y_train,
         x_test=x_test, y_test=y_test, time_train=time_train, time_test=time_test)
