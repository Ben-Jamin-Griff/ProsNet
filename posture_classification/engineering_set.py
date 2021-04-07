from algorithm_dataset_abc import ABCDataset

#import pandas as pd
#import numpy as np
#import math
#import datetime
#import xlrd

class EngineeringSet(ABCDataset):
    def __init__(self):
        self.dataset = None
        self.raw_acceleration_data = None
        self.posture_stack = None

    def show_dataset(self):
        print('Engineering Set')

    def get_data(self, activity_monitor):
        self.raw_acceleration_data = activity_monitor.raw_data

    def get_posture_stack(self, posture_stack):
        self.posture_stack = posture_stack.posture_stack

    def create_set(self):
        pass

    def save_set(self):
        print("Make me work too!")