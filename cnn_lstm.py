import PIL
from PIL import Image
from numpy import asarray
import numpy as np
from scapy.all import *
import os
import binascii
import pandas as pd

#### TENSOR
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras import models, datasets, layers, optimizers
from tensorflow.keras import backend as k
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def pre_process(pcap_file, csv_file, qty):
    #Process PCAP
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(THIS_FOLDER, pcap_file)
    pcap_data=rdpcap(my_file)

    T = np.empty((qty, 16, 16))

    for pkt in range(qty):
        pcap_raw = pcap_data[pkt]
        pcap_array = bytearray(bytes(pcap_raw))
        pcap_np_array = asarray(pcap_array)
        pcap_length = len(pcap_np_array)
        if pcap_length > 256:
            pcap_np_array_256 = pcap_np_array[0:256]
            pcap_np_16x16 = np.reshape(pcap_np_array_256, (16, 16)) / 255
            for row in range(16):
                for col in range(16):
                    T[pkt][row][col] = pcap_np_16x16[row][col]
        else:
            fill_cnt = 256 - pcap_length
            fill_array = np.zeros((fill_cnt,), dtype=int)
            pcap_np_array_256 = np.append(pcap_np_array, fill_array)
            pcap_np_16x16 = np.reshape(pcap_np_array_256, (16, 16)) / 255
            for row in range(16):
                for col in range(16):
                    T[pkt][row][col] = pcap_np_16x16[row][col]

    #Process CSV get label
    df = pd.read_csv(csv_file)
    df['Tag'] = pd.to_numeric(df['Tag'])
    label = df['Tag']

    return T, label

x1,y1 = pre_process('iec104-only.pcap', r'iec104.csv', 64)
x2,y2 = pre_process('s62-eth2-Training.pcap', r'packet_training_mod.csv', 32020)

# Combine all data
data = np.append(x1,x2, axis=0)
label = np.append(y1,y2)

x_train = data.reshape((-1,16,16,1))
num_category = 9
print(label.shape)
y_train = tf.keras.utils.to_categorical(label, num_category)

x_train_shuffled, y_train_shuffled = shuffle(x_train,y_train, random_state=0)

a,b = np.split(x_train_shuffled,2)
c,d = np.split(y_train_shuffled,2)

#Build Model
IMG_SIZE = (16, 16, 1)
input_img = layers.Input(shape=IMG_SIZE)

model = layers.Conv2D(32, (3, 3), padding='same')(input_img)
model = layers.Activation('relu')(model)
model = layers.Conv2D(32, (3, 3), padding='same', strides=(2, 2))(model)
model = layers.Activation('relu')(model)

model = layers.Conv2D(32, (3, 3), padding='same')(input_img)
model = layers.Activation('relu')(model)
model = layers.Conv2D(32, (3, 3), padding='same', strides=(2, 2))(model)
model = layers.Activation('relu')(model)

model = layers.Conv2D(64, (3, 3), padding='same')(model)
model = layers.Activation('relu')(model)
model = layers.Conv2D(64, (3, 3), padding='same', strides=(2, 2))(model)
model = layers.Activation('relu')(model)
"""
model = layers.Conv2D(64, (3, 3), padding='same')(model)
model = layers.Activation('relu')(model)
model = layers.Conv2D(64, (3, 3), padding='same')(model)
model = layers.Activation('relu')(model)
"""

model = layers.GlobalAveragePooling2D()(model)

model = layers.Reshape((-1,64))(model)
model = layers.LSTM(100, dropout=0.2)(model)

model = layers.Dense(32)(model)
model = layers.Activation('relu')(model)
model = layers.Dense(9)(model)

output_img = layers.Activation('softmax')(model)

model = models.Model(input_img, output_img)

model.summary()


adam = optimizers.Adam(lr=0.0001)
model.compile(adam, loss='categorical_crossentropy', metrics=["accuracy"])

history = model.fit(a, c, epochs=20, validation_data=(b, d))


plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8
plt.rcParams['pdf.fonttype'] = 42
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)


width = 6
height = 4
plt.rcParams["figure.figsize"] = (width,height)
plt.title('Learning Accuracy')

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
#plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='best')

test_loss, test_accuracy = model.evaluate(x_train, y_train, verbose=2)
print('\nTest accuracy = {0:.2f}%'.format(test_accuracy*100.0))

plt.grid()
plt.savefig("plotTrainLSTM",dpi=300)

