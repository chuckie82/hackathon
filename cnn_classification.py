'''Train a simple convnet on the MNIST dataset.

Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py

Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

batch_size = 10
nb_classes = 2
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 200, 200
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 11

# the data, shuffled and split between train and test sets
texture = np.load('texture_data_np_array.npy')
standard = np.load('standard_data_np_array.npy')
extra = np.load('extra_standard_data_np_array.npy')

# shuffle them
np.random.shuffle(texture)
np.random.shuffle(standard)
np.random.shuffle(extra)

X_train = np.zeros((200,200,200))
X_train[0:99] = texture[0:99]
X_train[100:124] =standard
X_train[125:199] = extra[0:74]

#X_train[100:199] = 0.7

y_train = np.zeros(200)
y_train[100:199] = 1


X_test = np.zeros((300,200,200))
X_test = texture[100:400]
X_test[200:299] = 0.6
print(X_test.shape)
y_test = np.zeros(300)
print(y_test.shape)
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = (X_train-X_train.max())/(X_train.min()-X_train.max())
X_test = (X_test-X_test.max())/(X_test.min()-X_test.max())
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Convolution2D(nb_filters*2, nb_conv,nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print('Predicting')
predicted_output = model.predict(X_test, batch_size=batch_size)
print('Predcited class',predicted_output)
print('Ploting Results')
plt.subplot(2, 1, 1)
plt.plot(Y_test)
plt.title('Expected')
plt.subplot(2, 1, 2)
plt.plot(predicted_output)
plt.title('Predicted')
plt.show()
