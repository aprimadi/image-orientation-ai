from __future__ import print_function
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import MySQLdb
import os

from data import DataGenerator
from inception_v3 import InceptionV3

def load_data():
    conn = MySQLdb.Connection(
        host='localhost',
        user='root',
        port=3306,
        db='image_classifier',
    )
    conn.query("""SELECT * FROM images""")
    result = conn.store_result()
    data = []
    for i in range(min(result.num_rows(), 100000)):
        row = result.fetch_row()
        image_id = row[0][0]
        rotation = int(row[0][1] / 90)
        data.append((image_id, rotation))

    data = np.array(data)

    # Shuffle data and split 80% 20% for training vs test data
    indices = np.random.permutation(len(data))
    split = int(len(data) * 4 / 5)
    training_idx, test_idx = indices[:split], indices[split:]
    data_train = data[training_idx]
    data_test = data[test_idx]
    return (data_train, data_test)

def main():
    batch_size = 32
    num_classes = 4
    epochs = 100
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'orientation-inception.h5'

    data_train, data_test = load_data()

    # Use Google Inception v3 model
    model = InceptionV3(
        include_top=False,
        weights=None,
        input_shape=(192, 192, 3),
        pooling='softmax',
        classes=4,
    )

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(save_dir, 'checkpoint.h5'),
        verbose=1,
        save_best_only=True,
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    train_generator = DataGenerator(data_train)
    val_generator = DataGenerator(data_test)
    model.fit_generator(
        train_generator.flow(batch_size=batch_size),
        epochs=epochs,
        validation_data=val_generator.flow(batch_size=batch_size),
        shuffle=True,
        callbacks=[checkpointer, early_stopping],
    )

    # Save model and weights
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s' % model_path)

    # Score trained model.
    scores = model.evaluate_generator(val_generator.flow(batch_size=batch_size))
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

if __name__ == '__main__':
    main()
