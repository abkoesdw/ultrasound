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

path_model = "/home/abkoesdw/Documents/cognitech/ultrasound-project/model/"
data = np.load("dataset3.npz")
data = {key: data[key].item() for key in data}

duration = 2400
title_string = '40 minutes data'
x_train = data['x_train'][duration]
y_train = data['y_train'][duration]
# y_train = to_categorical(y_train, nb_classes=2)
# sort_idx = np.sort(time[data['idx_train']])
sort_idx = np.sort(data['time_train'][duration])
time_train = data['time_train']

x_test = data['x_test'][duration]
y_test = data['y_test'][duration]
# y_test = to_categorical(y_test, nb_classes=2)
time_test = data['time_test'][duration]

_, num_feat = np.shape(x_train)
init = 'normal'
neurons = 256
drop_out = 0.2
model = Sequential()

model.add(Dense(neurons, input_dim=num_feat, init=init, activation='relu'))
model.add(Dropout(drop_out))

num_hid_layer = 3

for i in range(num_hid_layer-1):
    model.add(Dense(neurons, init=init, activation='relu'))
    model.add(Dropout(drop_out))

model.add(Dense(1, init=init, activation='softmax'))

# compile the model
sgd = SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True)
# model.summary()
model.compile(loss='mse', optimizer=sgd, metrics=['mse'])

# train the model
nb_epoch = 200
batch_size = 32
callbacks = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3, verbose=0, mode='auto')
filepath = (path_model + "model_valve_DNN.best.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=nb_epoch, batch_size=batch_size, shuffle=True,
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

y_est_train = model.predict(x_train, batch_size=batch_size)
y_est_train = np.argmax(y_est_train, axis=1)
y_train = np.argmax(y_train, axis=1)
accuracy_train = float(len(np.where(y_est_train == y_train)[0]))/len(y_train) * 100

print("\n Training accuracy:", accuracy_train, "%")
print("Classification report - Training")
print("-----------------------------------------------------")
print(classification_report(y_train, y_est_train, target_names=['0', '1']))
print("-----------------------------------------------------")
print("Confusion matrix - Training")
print("--------------------------")
print(confusion_matrix(y_train, y_est_train, labels=[0, 1]))
print("--------------------------")

# evaluate the model
y_est_test = model.predict(x_test, batch_size=batch_size)
y_est_test = np.argmax(y_est_test, axis=1)
y_test = np.argmax(y_test, axis=1)
accuracy_test = float(len(np.where(y_est_test == y_test)[0]))/len(y_test) * 100
print("\n Testing accuracy:", accuracy_test, "%")

print("Classification report - Testing")
print("-----------------------------------------------------")
print(classification_report(y_test, y_est_test, target_names=['0', '1']))
print("-----------------------------------------------------")
print("Confusion matrix - Testing")
print("--------------------------")
print(confusion_matrix(y_test, y_est_test, labels=[0, 1]))
print("--------------------------")
print(x_test.shape, time_test.shape)

idx_sort = np.argsort(time_test)
new_time = time_test[idx_sort]
new_x = x_test[idx_sort, :]
print(len(new_x))
plt.figure(1)
ax = plt.gca()
xfmt = md.DateFormatter('%H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
plt.subplot(3, 1, 1)
plt.title(title_string)
plt.plot(new_time, new_x[:, 0], 'b', label='cycle phase')
plt.ylabel('cycle phase')
plt.xlabel('time')
plt.grid()
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(new_time, new_x[:, 1], 'b', label='cycle step')
plt.ylabel('cycle step')
plt.xlabel('time')
plt.grid()
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(new_time, new_x[:, 2], 'b', label='ut_pv220')
plt.ylabel('ut_pv220')
plt.xlabel('time')
plt.grid()
plt.legend()
plt.figure(2)
plt.title(title_string)
ax = plt.gca()
xfmt = md.DateFormatter('%H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
plt.plot(time_test, y_test, '.b', label='ground_truth')
plt.plot(time_test, y_est_test + 0.01, '.r', label='predicted')
plt.ylabel('open/close')
plt.xlabel('time')
plt.grid()
plt.legend()
plt.show()
