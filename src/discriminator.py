import numpy
# for experiment reproducibility
numpy.random.seed(1337)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD

num_classes = 2
batch_size = 128
epochs = 20

raw_data = numpy.load('/home/maximinus/code/music_samples/X_DATA.npy')
raw_answers = numpy.load('/home/maximinus/code/music_samples/Y_DATA.npy')

x_train = raw_data[:35000]
x_test = raw_data[35000:]
y_train = raw_answers[:35000]
y_test = raw_answers[35000:]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(16, activation='sigmoid', input_shape=(8192,)  ))
model.add(Dropout(0.25))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(),metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
