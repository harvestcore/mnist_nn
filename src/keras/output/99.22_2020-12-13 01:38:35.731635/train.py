from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import numpy as np
import time
import os
from datetime import datetime

from utils import get_predictions, to_csv, to_file, predictions_to_file

(images_train, labels_train), (images_test, labels_test) = mnist.load_data()

images_train = images_train.reshape(images_train.shape[0], 28, 28, 1).astype('float32')
images_test = images_test.reshape(images_test.shape[0], 28, 28, 1).astype('float32')

images_train /= 255
images_test /= 255

labels_train = np_utils.to_categorical(labels_train, 10)
labels_test = np_utils.to_categorical(labels_test, 10)

nn_model = Sequential()

nn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
nn_model.add(MaxPooling2D(pool_size=(2,2)))
nn_model.add(Conv2D(32, (3, 3), activation='relu'))
nn_model.add(BatchNormalization())

nn_model.add(MaxPooling2D(pool_size=(2,2)))
nn_model.add(Dropout(0.25))

nn_model.add(Flatten())
nn_model.add(Dense(128, activation='relu'))
nn_model.add(Dropout(0.5))

nn_model.add(Dense(10, activation='softmax'))

nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()
epochs = 15
history = nn_model.fit(images_train, labels_train, batch_size=64, epochs=epochs, verbose=1)
fit_time = time.time() - start_time

evaluation_score = nn_model.evaluate(images_test, labels_test, verbose=0)
scores = nn_model.predict(images_test)
predictions = get_predictions(scores)

now = datetime.now()
dirpath = 'output/' + str(evaluation_score[1] * 100)[:5] + '_' + str(now)
os.mkdir(dirpath)

history_filename = dirpath + '/history_epochs-' + str(epochs) + '_time-' + str(fit_time) + 'sec_' + str(now) + '.csv'
score_filename = dirpath + '/score_epochs-' + str(epochs) + '_time-' + str(fit_time) + 'sec_' + str(now) + '.csv'
prediction_filename = dirpath + '/predictions_epochs-' + str(epochs) + '_time-' + str(fit_time) + 'sec_' + str(now) + '.csv'

to_csv(history_filename, history.history, epochs)
to_file(score_filename, 'loss: ' + str(evaluation_score[0]) + '\naccuracy: ' + str(evaluation_score[1]))
predictions_to_file(prediction_filename, predictions)
