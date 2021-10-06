from ProsNet.dataset.dataset_abc import ABCDataset
from ProsNet.helper import Helper

import numpy as np

class Dataset(ABCDataset, Helper):
    def __init__(self, processing_type='epoch'):
        self.processing_type = processing_type
        self.dataset = None
        self.raw_acceleration_data = None
        self.posture_stack = None
        self.posture_stack_duration = None
        self.posture_stack_epoch_type = None
        self.posture_stack_start_time = None

    def show_set(self):
        print('Dataset')
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
        self.posture_stack_start_time = posture_stack.posture_stack_start_time

    def remove_classes(self, classes_to_remove = []):
        # remove the default non pure classes
        classes_to_keep = self.dataset[1] != 99
        for classes in classes_to_remove:
            classes_to_keep = classes_to_keep + (self.dataset[1] != classes)
        self.dataset[0] = self.dataset[0][classes_to_keep]
        self.dataset[1] = self.dataset[1][classes_to_keep]
        try: # This is in as a quick fix for the old engineering set code which doesn't currently use R.B's import function or track participants id's
            self.dataset[2] = self.dataset[2][classes_to_keep]
        except:
            pass

    def save_set(self, filename, type_of_set):
        print('...saving set')
        np.save(filename + '_' + type_of_set + '_set.npy', self.dataset[0])
        np.save(filename + '_' + type_of_set + '_set_classes.npy', self.dataset[1])
        np.save(filename + '_' + type_of_set + '_set_ids.npy', self.dataset[2])

    def load_set(self, filename, type_of_set):
        print('...loading set')
        engineering_set = np.load(filename + '_' + type_of_set + '_set.npy')
        posture_set = np.load(filename + '_' + type_of_set + '_set_classes.npy')
        participant_ids = np.load(filename + '_' + type_of_set + '_set_ids.npy')
        self.dataset = [engineering_set, posture_set, participant_ids]

    def combine_sets(self, set):
        try:
            self.dataset[0] = np.concatenate((self.dataset[0], set.dataset[0]), axis=0)
            self.dataset[1] = np.concatenate((self.dataset[1], set.dataset[1]), axis=0)
            self.dataset[2] = np.concatenate((self.dataset[2], set.dataset[2]), axis=0)
        except:
            print("...No current data in set")
            self.dataset = set.dataset
