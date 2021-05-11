from model.model import Model

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import (KNeighborsClassifier)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

class ShallowModel(Model):
    def __init__(self):
        super().__init__()

    def create_model(self, type_of_model, plot_results = False, save_model_results = None):
        if type_of_model == 'knn':
            X_train, X_test, y_train, y_test = train_test_split(self.dataset, self.postures, test_size=0.2, random_state=42)
            LABELS = ['Sedentary', 'Standing', 'Stepping', 'Lying']
            unique_classes_train, unique_classes_test = self.review_class_imbalance(y_train, y_test, LABELS)
            n_neighbors = 10
            random_state = 42

            pipeline = make_pipeline(Normalizer(), LinearDiscriminantAnalysis(n_components=2))
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            pipeline.fit(X_train, y_train)
            knn.fit(pipeline.transform(X_train), y_train)

            acc_knn = knn.score(pipeline.transform(X_test), y_test)

            X_embedded = pipeline.transform(self.dataset)
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=self.postures, s=30, cmap='Set1')
            ax.set_title("{}, {}, KNN (k={})\nTest accuracy = {:.2f}".format('Example Set',
                                                                            'Normalized',
                                                                            n_neighbors,
                                                                            acc_knn));
            # produce a legend with the unique colors from the scatter
            legend1 = ax.legend(*scatter.legend_elements(),
                                loc="upper left", title="Classes")
                    
            ax.add_artist(legend1)

            if plot_results:
                plt.grid(False)
                plt.ion()
                plt.show()
                plt.draw()
                plt.pause(0.001)
                input("Press [enter] to continue.")

            if save_model_results is not None:
                plt.savefig(save_model_results + '_plot.png', bbox_inches='tight')

            disp = plot_confusion_matrix(knn, pipeline.transform(X_test), y_test,
                                        #display_labels=LABELS,
                                        cmap=plt.cm.Blues,
                                        normalize='true');

            if plot_results:
                plt.grid(False)
                plt.ion()
                plt.show()
                plt.draw()
                plt.pause(0.001)
                input("Press [enter] to continue.")

            if save_model_results is not None:
                plt.savefig(save_model_results + '_results.png', bbox_inches='tight')

        self.model = knn
        self.pipeline = pipeline