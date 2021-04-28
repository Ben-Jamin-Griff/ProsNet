from dataset.dataset_abc import ABCDataset
from process import Process

import pandas as pd
import numpy as np
import math

class Dataset(ABCDataset, Process):
    def __init__(self, processing_type='epoch'):
        self.processing_type = processing_type
        self.dataset = None
        self.raw_acceleration_data = None
        self.posture_stack = None
        self.posture_stack_duration = None
        self.posture_stack_epoch_type = None

    def show_set(self):
        print('Engineering Set')
        print('----------')
        if self.posture_stack_epoch_type == 'mixed':
            print('Extracted Set')
            print(f" {len(self.dataset[1])} mixed epochs were extracted from {len(self.posture_stack.index)} total epochs.")
            print('----------')
        elif self.posture_stack_epoch_type == 'pure':
            print('Extracted Set')
            print(f" {len(self.dataset[1])} pure epochs were extracted from {len(self.posture_stack.index)} total epochs.")
            print('----------')
        else:
            print('Extracted Set')
            print(f" {len(self.dataset[0])} pure epochs were extracted.")
            print('----------')

    def get_data(self, activity_monitor):
        self.raw_acceleration_data = activity_monitor.raw_data

    def get_posture_stack(self, posture_stack):
        self.processing_type = posture_stack.processing_type
        self.posture_stack = posture_stack.posture_stack
        self.posture_stack_duration = posture_stack.posture_stack_duration
        self.posture_stack_epoch_type = posture_stack.posture_stack_epoch_type

    def remove_classes(self, classes_to_remove = []):
        # remove the default non pure classes
        classes_to_keep = self.dataset[1] != 99
        for classes in classes_to_remove:
            classes_to_keep = classes_to_keep + (self.dataset[1] != classes)
        self.dataset[1] = self.dataset[1][classes_to_keep]
        self.dataset[0] = self.dataset[0][classes_to_keep]

    def save_set(self, filename):
        print('...saving engineering set')
        np.save(filename + '_engineering_set.npy', self.dataset[0])
        np.save(filename + '_engineering_set_classes.npy', self.dataset[1])

    def load_set(self, filename):
        print('...loading engineering set')
        self.dataset[0] = np.load(filename + '_engineering_set.npy')
        self.dataset[1] = np.load(filename + '_engineering_set_classes.npy')