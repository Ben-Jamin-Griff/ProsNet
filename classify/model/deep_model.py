from model.model import Model

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

class DeepModel(Model):
    def __init__(self):
        super().__init__()

    def show_training(history):
        # summarize history for accuracy
        plt.figure(figsize=(10, 4))
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.figure(figsize=(10, 4))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def train_and_save_model(model_to_train, X_train, y_train, model_name):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
        # Hyper-parameters
        EPOCHS = 50
        model_to_train.compile(loss='categorical_crossentropy',
                    optimizer='adam', 
                    metrics=['accuracy'])

        history = model_to_train.fit(X_train,
                        y_train,
                        epochs=EPOCHS,
                        validation_split=0.2,
                        batch_size=32,
                        #callbacks=[callback],
                        verbose=1)
        show_training(history)
        filename  = './models/' + model_name + '.h5'
        # save the model
        model_to_train.save(filename)