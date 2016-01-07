#!/usr/bin/env python2
import json
import sys
import time

import numpy as np
np.random.seed(1337)

import h5py
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.regularizers import l2

if len(sys.argv) != 2:
    print 'Usage: %s subset_filepath' % sys.argv[0]
    sys.exit(1)

subset_filepath = sys.argv[1]

# as described in http://cs231n.stanford.edu/reports/zyh_project.pdf
model = Sequential()
model.add(Convolution2D(64, 3, 3, weights=[np.random.normal(0, 0.01, size=(64, 1, 3, 3)), np.zeros(64)],
                        activation='relu', border_mode='same', subsample=(1, 1),
                        input_shape=(1, 64, 64)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Convolution2D(128, 3, 3, weights=[np.random.normal(0, 0.01, size=(128, 64, 3, 3)), np.zeros(128)],
                        activation='relu', border_mode='same', subsample=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Convolution2D(256, 3, 3, weights=[np.random.normal(0, 0.01, size=(256, 128, 3, 3)), np.zeros(256)],
                        activation='relu', border_mode='same', subsample=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())

model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(200, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

timestamp = int(time.time())
with open('%d-model.json' % timestamp, 'w') as f:
    d = json.loads(model.to_json())
    json.dump(d, f, indent=4)

with h5py.File(subset_filepath, 'r') as f:
    model.fit(f['trn/x'], f['trn/y'], validation_data=(f['vld/x'], f['vld/y']),
              nb_epoch=15, batch_size=128, shuffle='batch', verbose=1, show_accuracy=True)

    score = model.evaluate(f['tst/x'], f['tst/y'], verbose=0, show_accuracy=True)
    print 'Test score:', score[0]
    print 'Test accuracy:', score[1]

    model.save_weights('%d-weights-%f.hdf5' % (timestamp, score[1]))
