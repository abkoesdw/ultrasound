import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
import time
import matplotlib.dates as md
import itertools
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
import matplotlib.pyplot as plt
unused_var = os.system('clear')
today_ = time.strftime("%Y-%m-%d")

duration = 10800
path_model = "/home/abkoesdw/Documents/cognitech/ultrasound-project/model/"
data = np.load("dataset3.npz")
data = {key: data[key].item() for key in data}
time_train = data['time_train'][duration]
time_test = data['time_test'][duration]

title_string = '--- data'
x_train = data['x_train'][duration]
y_train = data['y_train'][duration]
x_test = data['x_test'][duration]
y_test = data['y_test'][duration]

idx_sort_test = np.argsort(time_test)
idx_sort_train = np.argsort(time_train)
new_time_test = time_test[idx_sort_test]
new_time_train = time_train[idx_sort_train]

# window_len = 20
new_x_test = x_test[idx_sort_test, :]
new_y_test = y_test[idx_sort_test, :]
new_x_train = x_train[idx_sort_train, :]
new_y_train = y_train[idx_sort_train, :]

_, num_feat = np.shape(x_train)
init = 'normal'
neurons = 256
drop_out = 0.2
model = Sequential()

model.add(Dense(neurons, input_dim=num_feat, kernel_initializer=init, activation='relu'))
model.add(Dropout(drop_out))

num_hid_layer = 3

for i in range(num_hid_layer-1):
    model.add(Dense(neurons, kernel_initializer=init, activation='relu'))
    model.add(Dropout(drop_out))

model.add(Dense(2, kernel_initializer=init, activation='relu'))

# compile the model
sgd = SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
# model.summary()
model.compile(loss='mse', optimizer=sgd, metrics=['mse'])

# train the model
nb_epoch = 1200
batch_size = 16
callbacks = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3, verbose=0, mode='auto')
filepath = (path_model + "model_valve_DNN.best.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(new_x_train, new_y_train, validation_data=(new_x_test, new_y_test),
          epochs=nb_epoch, batch_size=batch_size, shuffle=True,
          verbose=2, callbacks=[callbacks, checkpoint])

model.load_weights(filepath)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.save(path_model + 'model_valve_DNN_' + str(neurons) + 'x' + str(num_hid_layer) + '.h5')


model_json = model.to_json()
with open(path_model + "model_" + str(neurons) + 'x' + str(num_hid_layer) + ".json", "w") as json_file:
    json_file.write(model_json)
model_yaml = model.to_yaml()
with open(path_model + "model_" + str(neurons) + 'x' + str(num_hid_layer) + ".yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)

# evaluate the model
y_est_test = model.predict(new_x_test, batch_size=batch_size)

new_y_est = y_est_test  # y_est_test[idx_sort_test]
plt.figure(1)
plt.subplot(3, 1, 1)
plt.title(title_string)
ax = plt.gca()
xfmt = md.DateFormatter('%H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
plt.plot(new_time_test, new_x_test[:, 0], 'b', label='cycle phase')
plt.ylabel('cycle phase')
plt.xlabel('time')
plt.grid()
plt.legend()
plt.subplot(3, 1, 2)
ax = plt.gca()
xfmt = md.DateFormatter('%H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
plt.plot(new_time_test, new_x_test[:, 1], 'b', label='cycle step')
plt.ylabel('cycle step')
plt.xlabel('time')
plt.grid()
plt.legend()
plt.subplot(3, 1, 3)
ax = plt.gca()
xfmt = md.DateFormatter('%H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
plt.plot(new_time_test, new_x_test[:, 2], 'b', label='open/close')
plt.ylabel('open/close')
plt.xlabel('time')
plt.grid()
plt.legend()

plt.figure(2)
plt.title(title_string)
ax = plt.gca()
xfmt = md.DateFormatter('%H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
plt.subplot(2, 1, 1)
plt.plot(new_time_test, new_y_test[:, 0], 'b', label='ground_truth-mean')
plt.plot(new_time_test, new_y_est[:, 0], 'r', label='predicted-mean')
plt.ylabel('ut_pv220')
plt.xlabel('time')
plt.grid()
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(new_time_test, new_y_test[:, 1], 'b', label='ground_truth-stdev')
plt.plot(new_time_test, new_y_est[:, 1], 'r', label='predicted-stdev')
plt.ylabel('ut_pv220')
plt.xlabel('time')
plt.grid()
plt.legend()
plt.show()
