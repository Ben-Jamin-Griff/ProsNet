from model.model_abc import ABCModel

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn.metrics as metrics

class DeepModel(ABCModel):
    def __init__(self):
        self.model = None
        self.dataset = None
        self.predictions = None

    def show_set(self):
        print('Engineering Set')
        print('----------')
        print('Extracted Set')
        print(f"{self.dataset.shape[0]} epochs / events were extracted for classification.")
        print('----------')

    def show_predictions(self):
        print('Prediction Stack')
        print('----------')
        print('Unique class values')
        unique, counts = np.unique(self.predictions, return_counts=True)
        for classy, count in zip(unique, counts):
            print(f"Class {classy} has {count} predictions")
        print('----------')

    def load_model(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title = "Load model")
        self.model = tf.keras.models.load_model(file_path)
        print(f"Loaded model: {file_path}")
        self.model.summary()
        print('----------')

    def get_data(self, set):
        self.dataset = set.dataset[0]
            
    def reassign_classes(classes):
        for count, value in enumerate(classes):
            if classes[count] == 2.1:
                classes[count] = 2
            elif classes[count] == 3.1:
                classes[count] = 3
            elif classes[count] == 3.2:
                classes[count] = 3
            elif classes[count] == 5.0:
                classes[count] = 0
            else:
                continue
        return classes

    def remove_classes(dataset, stack, class_to_remove):
        remove_non_wear_idx = posture_classes != class_to_remove
        stack = stack[remove_non_wear_idx]
        dataset = dataset[remove_non_wear_idx]
        return dataset, stack

    def reshape_set(self, new_shape):
        shaper = self.dataset.shape
        new_shape.insert(0, shaper[0])
        new_shape = tuple(new_shape)
        self.dataset = self.dataset.reshape(new_shape)

    def one_hot_postures(stack):
        unique_classes = np.unique(stack)
        stack = tf.one_hot(stack, len(unique_classes))
        return stack

    def review_class_imbalance(y_train, y_test, labels=None):
        # Find the unique label values...
        unique_classes_train = np.unique(y_train)
        unique_classes_test = np.unique(y_test)
        # Count the unique label values
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        try:
            count_class_values_train = dict(zip(labels, counts_train))
            count_class_values_test = dict(zip(labels, counts_test))
        except:
            count_class_values_train = dict(zip(unique_train, counts_train))
            count_class_values_test = dict(zip(unique_test, counts_test))
        print('Train Classes')
        print(count_class_values_train)
        print('--------------')
        print('Test Classes')
        print(count_class_values_test)
        print('--------------')
        return unique_classes_train, unique_classes_test

    def norm_accel_data(self, x):
        x_minimum = 0
        x_maximum = 255
        x_normalized = ((x - x_minimum) / (x_maximum - x_minimum))
        return x_normalized

    def process_epochs(self, shuffle=False):
        print("Processing epochs...")
        # Normalize the acceleration data
        self.dataset = self.norm_accel_data(self.dataset)
        # Convert data to a tensor
        self.dataset = tf.constant(self.dataset)
        # If the data is a training dataset, we shuffle it
        if shuffle:
            indices = tf.range(start=0, limit=tf.shape(data)[0], dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)
            shuffled_x = tf.gather(self.dataset, shuffled_indices)
            #shuffled_y = tf.gather(y, shuffled_indices)
            return

    def show_confusion_matrix(validations, predictions):
        matrix = metrics.confusion_matrix(validations, predictions, normalize ='true')
        plt.figure(figsize=(6, 4))
        sns.heatmap(matrix,
                    cmap='coolwarm',
                    linecolor='white',
                    linewidths=1,
                    xticklabels=LABELS,
                    yticklabels=LABELS,
                    annot=True)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

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
    
    def show_model_results(model_to_test, data_to_test, validation):
        predictions = model_to_test.predict(data_to_test)
        predictions_max = np.argmax(predictions, axis=1)

        show_confusion_matrix(validation, predictions_max)
        print('------------')
        print(classification_report(validation, predictions_max))

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

    def make_prediction(self):
        predictions_probabilities = self.model.predict(self.dataset)
        predictions = np.argmax(predictions_probabilities, axis=1)
        self.predictions = predictions

    def save_predictions(self, filename):
        # Make a df... That might be a better way of creating a table to save to CSV
        np.savetxt(filename + '_predictions.csv', self.predictions, delimiter=',', fmt='%d' , header='ActivityCodes (0=sedentary 1=standing 2=stepping 3=lying')