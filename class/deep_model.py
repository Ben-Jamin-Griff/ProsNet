from model_abc import ABCModel

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
        self.data_to_analyse = None
        self.predictions = None

    def show_set(self):
        print('Engineering Set')
        print('----------')
        print('Extracted Set')
        print(f"{self.data_to_analyse.shape[0]} epochs were extracted for classification.")
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
        file_path = filedialog.askopenfilename()
        self.model = tf.keras.models.load_model(file_path)
        print(f"Loaded model: {file_path}")
        self.model.summary()
        print('----------')

    def get_data(self, data_holder):
        def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
            """
            Call in a loop to create terminal progress bar
            @params:
                iteration   - Required  : current iteration (Int)
                total       - Required  : total iterations (Int)
                prefix      - Optional  : prefix string (Str)
                suffix      - Optional  : suffix string (Str)
                decimals    - Optional  : positive number of decimals in percent complete (Int)
                length      - Optional  : character length of bar (Int)
                fill        - Optional  : bar fill character (Str)
                printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
            """
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
            # Print New Line on Complete
            if iteration == total: 
                print()

        try:
            self.data_to_analyse = data_holder.dataset
        except:
            filename = data_holder.raw_data
            # Creat empty engineering set
            engineering_set = np.empty((0,295,3), int)
            CHUNKSIZE = 300000
            loaded_chunks = 0
            #max_number_of_chunks = 100000
            #printProgressBar (loaded_chunks, max_number_of_chunks, 'Engineering set progress:')
            print(f"Loaded chunk: {loaded_chunks}")
            for chunk in pd.read_csv(filename, chunksize=CHUNKSIZE):
                loaded_chunks += 1
                chunk = chunk['sep=;'].apply(lambda x: pd.Series(x.split(';')))
                chunk.columns = ["Time", "Index", "X", "Y", "Z"]
                try:
                    chunk = chunk.apply(pd.to_numeric)
                    chunk = chunk.reset_index(drop=True)
                    chunk.Time = pd.to_datetime(chunk.Time, unit='d', origin='1899-12-30')
                except:
                    chunk = chunk.iloc[1:,]
                    chunk = chunk.apply(pd.to_numeric)
                    chunk = chunk.reset_index(drop=True)
                    chunk.Time = pd.to_datetime(chunk.Time, unit='d', origin='1899-12-30')

                epochSize = 15
                numOfEpochs = (CHUNKSIZE / epochSize)
                for i in range(int(numOfEpochs)):
                    current_epoch = chunk.iloc[(15*20)*i:(15*20)*(i+1):,:].copy()
                    current_epoch_accel_data = current_epoch[['X','Y','Z']].to_numpy()
                    # Break if the last epoch is too small
                    if current_epoch_accel_data.shape[0] < 300:
                        break
                    engineering_set = np.append(engineering_set, [current_epoch_accel_data[:295,:]], axis=0)
                #printProgressBar (loaded_chunks, max_number_of_chunks, 'Engineering set progress:')
                print(f"Loaded chunk: {loaded_chunks}")
        self.data_to_analyse = engineering_set

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
        shaper = self.data_to_analyse.shape
        new_shape.insert(0, shaper[0])
        new_shape = tuple(new_shape)
        self.data_to_analyse = self.data_to_analyse.reshape(new_shape)

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
        self.data_to_analyse = self.norm_accel_data(self.data_to_analyse)
        # Convert data to a tensor
        self.data_to_analyse = tf.constant(self.data_to_analyse)
        # If the data is a training dataset, we shuffle it
        if shuffle:
            indices = tf.range(start=0, limit=tf.shape(data)[0], dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)
            shuffled_x = tf.gather(self.data_to_analyse, shuffled_indices)
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
        predictions_probabilities = self.model.predict(self.data_to_analyse)
        predictions = np.argmax(predictions_probabilities, axis=1)
        self.predictions = predictions