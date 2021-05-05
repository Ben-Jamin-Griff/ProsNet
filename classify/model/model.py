from model.model_abc import ABCModel
from plotter import Plotter

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
import tensorflow as tf

class Model(ABCModel, Plotter):
    def __init__(self):
        super().__init__()
        self.model = None
        self.pipeline = None
        self.dataset = None
        self.predictions = None
        self.postures = None
        self.posture_stack_start_time = None
        self.one_hot_postures = None

    def show_set(self):
        print('Dataset')
        print('----------')
        print('Extracted Set')
        print(f"{self.dataset.shape[0]} epochs / events were extracted for classification.")
        print('----------')
        if self.postures is not None:
            print('Extracted Postures')
            print(f"{self.postures.shape[0]} postures were extracted for validation.")
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

    def get_postures(self, set):
        self.postures = set.dataset[1]
        self.posture_stack_start_time = set.posture_stack_start_time
            
    def reassign_classes(self):
        for count, value in enumerate(self.postures):
            if self.postures[count] == 2.1:
                self.postures[count] = 2
            elif self.postures[count] == 3.1:
                self.postures[count] = 3
            elif self.postures[count] == 3.2:
                self.postures[count] = 3
            elif self.postures[count] == 5.0:
                self.postures[count] = 0
            else:
                continue

    def balance_classes(self):
        pass
        #sedData = data.loc[data.EventCode == 0]
        #shapeSed = sedData.shape[0]
        #standData = data.loc[data.EventCode == 1]
        #shapeStand = standData.shape[0]
        #stepData = data.loc[data.EventCode == 2]
        #shapeSteps = stepData.shape[0]
        #smallestClass = min([shapeSed, shapeStand, shapeSteps])
        #newSedData = sedData.sample(smallestClass)
        #newStandData = standData.sample(smallestClass)
        #newStepData = stepData.sample(smallestClass)
        #newDataFrame = pd.concat([newSedData, newStandData, newStepData], axis=0)

    def remove_classes(self, class_to_remove):
        keep_idx = self.postures != class_to_remove
        self.postures = self.postures[keep_idx]
        self.dataset = self.dataset[keep_idx]

    def one_hot_postures(self):
        unique_classes = np.unique(self.postures)
        self.one_hot_postures = tf.one_hot(self.postures, len(unique_classes))

    def review_class_imbalance(self, y_train, y_test, labels=None):
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
        
    def show_confusion_matrix(self):
        matrix = metrics.confusion_matrix(self.postures.astype(int), self.predictions.astype(int), normalize ='true')
        LABELS = ['Sedentary', 'Standing', 'Stepping', 'Lying']
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
        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")
    
    def show_model_results(self):
        print('Model Results')
        print('------------')
        self.show_confusion_matrix()
        print('------------')
        print(classification_report(self.postures.astype(int), self.predictions.astype(int)))

    def make_prediction(self):
        predictions_probabilities = self.model.predict(self.dataset)
        predictions = np.argmax(predictions_probabilities, axis=1)
        self.predictions = predictions.astype(float)

    def save_predictions(self, filename):
        # Make a df... That might be a better way of creating a table to save to CSV
        np.savetxt(filename + '_predictions.csv', self.predictions, delimiter=',', fmt='%d' , header='ActivityCodes (0=sedentary 1=standing 2=stepping 3=lying')