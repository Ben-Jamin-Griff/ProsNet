from ProsNet.model.model_abc import ABCModel
from ProsNet.plotter import Plotter

import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
#import tensorflow as tf
import pickle
import datetime

class Model(ABCModel, Plotter):
    def __init__(self):
        super().__init__()
        self.model = None
        self.scaler = None
        self.pipeline = None
        self.dataset = None
        self.predictions = None
        self.postures = None
        self.posture_stack_start_time = None
        self.timeset = None
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

    def load_model(self, filename = None):
        if filename is not None:
            file_path = filename
        else:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(title = "Load model")
        self.model = pickle.load(open(file_path, 'rb'))
        print(f"Loaded model: {file_path}")
        #self.model.summary()
        print('----------')

    def load_scaler(self, filename = None):
        if filename is not None:
            file_path = filename
        else:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(title = "Load scaler")
        self.scaler = pickle.load(open(file_path, 'rb'))
        print(f"Loaded scaler: {file_path}")
        print('----------')

    def get_data(self, set):
        self.dataset = set.dataset[0]
        self.timeset = set.dataset[3]
        self.posture_stack_start_time = set.posture_stack_start_time

    def get_postures(self, set):
        self.postures = set.dataset[1]
        self.posture_stack_start_time = set.posture_stack_start_time

    def export_predictions(self, filename):
        export_df = pd.DataFrame(list(zip(self.timeset, self.predictions)), columns = ['Time', 'ActivityCode (0=sedentary 1=standing 2=stepping 3=lying)'])
        export_df.index = export_df['Time']
        del export_df['Time']
        export_df = export_df.resample('1S').ffill()
        export_df.to_csv(filename + '_.csv', index=True)
        self.predictions = export_df
            
    def reassign_classes(self):
        for count, value in enumerate(self.postures):
            if self.postures[count] == 2.1:
                self.postures[count] = 2
            elif self.postures[count] == 3.1:
                self.postures[count] = 3
            elif self.postures[count] == 3.2:
                self.postures[count] = 3
            else:
                continue

    def remove_classes(self, class_to_remove):
        keep_idx = self.postures != class_to_remove
        self.postures = self.postures[keep_idx]
        self.dataset = self.dataset[keep_idx]

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

#    def make_prediction(self):
#        if self.pipeline == None:
#            predictions_probabilities = self.model.predict(self.dataset)
#            predictions = np.argmax(predictions_probabilities, axis=1)
#            self.predictions = predictions.astype(float)
#        else:
#            predictions = self.model.predict(self.pipeline.transform(self.dataset))
#            self.predictions = predictions.astype(float)

    def save_predictions(self, filename):
        # Make a df... That might be a better way of creating a table to save to CSV
        np.savetxt(filename + '_predictions.csv', self.predictions, delimiter=',', fmt='%d' , header='ActivityCodes (0=sedentary 1=standing 2=stepping 3=lying')

    def save_object(self, filename):
        filehandler = open('./models/' + filename + '_model.obj', 'wb')
        pickle.dump(self, filehandler)

    def load_object(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title = "Load object")
        filehandler = open(file_path, 'rb')
        loaded_object = pickle.load(filehandler) 
        self.model = loaded_object.model
        self.pipeline = loaded_object.pipeline
        self.dataset = loaded_object.dataset
        self.predictions = loaded_object.predictions
        self.postures = loaded_object.postures
        self.posture_stack_start_time = loaded_object.posture_stack_start_time
        self.one_hot_postures = loaded_object.one_hot_postures
        print(f"Loaded object: {file_path}")
        print('----------')