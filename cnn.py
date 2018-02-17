from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os
import MySQLdb

def load_data():
    x = []
    y = []

    conn = MySQLdb.Connection(
        host='localhost',
        user='root',
        port=3306,
        db='image_classifier',
    )
    conn.query("""SELECT * FROM images""")
    result = conn.store_result()
    for i in range(result.num_rows()):
        row = result.fetch_row()
        image_id = row[0][0]
        rotation = row[0][1]

        path = "data-sanitized/%07d.png" % image_id
        if os.path.exists(path):
            print(path)
            img = cv2.imread(path)
            img = img_to_array(img)
            x.append(img)
            y.append(int(rotation / 90))

    x = np.array(x)
    y = np.array(y)

    # Shuffle data and split 80% 20% for training vs test data
    indices = np.random.permutation(x.shape[0])
    split = int(x.shape[0] * 4 / 5)
    print(split)
    training_idx, test_idx = indices[:split], indices[split:]
    x_train = x[training_idx, :]
    y_train = y[training_idx]
    x_test = x[test_idx, :]
    y_test = y[test_idx]
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    return ((x_train, y_train), (x_test, y_test))

def main():
    batch_size = 32
    num_classes = 4
    epochs = 100
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_orientation_trained_model.h5'

    (x_train, y_train), (x_test, y_test) = load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s' % model_path)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

if __name__ == '__main__':
    main()
