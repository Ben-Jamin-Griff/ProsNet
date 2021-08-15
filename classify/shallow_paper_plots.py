from activity_monitor.activpal import Activpal
from stack.epoch_stack import EpochStack
from dataset.engineering_set import EngineeringSet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Running shallow paper plots

This script creates the plots for the shallow paper...
"""

epoch_sizes = [15]

raw_data_paths = [
    "./apc-data/bg2-data/shank-AP472387 202a 19Sep20 1-00pm for 2d 15m-CREA-PA08110254-AccelDataUncompressed.csv",
]

event_data_paths = [
    "./apc-data/bg2-data/thigh-AP870085 202a 19Sep20 1-00pm for 2d 17m-CREA-PA08110254-Events.csv",
    ]

# Load in each participant's data
activPal = Activpal()
activPal.load_raw_data(raw_data_paths[0])
activPal.load_event_data(event_data_paths[0])

posture_stack = EpochStack()
posture_stack.get_data(activPal)
posture_stack.create_stack(stack_type = 'pure', subset_of_data = 100)

engineering_set = EngineeringSet()
engineering_set.get_data(activPal)
engineering_set.get_posture_stack(posture_stack)
engineering_set.create_set()

engineering_set.dataset[0][0]
engineering_set.dataset[1][0]

counter = 0
for data, posture in zip(engineering_set.dataset[0], engineering_set.dataset[1]):
    if posture == 0:
        counter =+ 1
        plt.plot(data)
        if counter == 10:
            plt.title('Sitting')
            plt.ion()
            plt.show()
            plt.savefig('Sitting_example.png')
            plt.close()
            break

counter = 0
for data, posture in zip(engineering_set.dataset[0], engineering_set.dataset[1]):
    if posture == 1:
        counter =+ 1
        plt.plot(data)
        if counter == 10:
            plt.title('Standing')
            plt.ion()
            plt.show()
            plt.savefig('Standing_example.png')
            plt.close()
            break

counter = 0
for data, posture in zip(engineering_set.dataset[0], engineering_set.dataset[1]):
    if posture == 2:
        counter =+ 1
        plt.plot(data)
        if counter == 10:
            plt.title('Stepping')
            plt.ion()
            plt.show()
            plt.savefig('Stepping_example.png')
            plt.close()
            break

counter = 0
for data, posture in zip(engineering_set.dataset[0], engineering_set.dataset[1]):
    if posture == 3:
        counter =+ 1
        plt.plot(data)
        if counter == 10:
            plt.title('Lying')
            plt.ion()
            plt.show()
            plt.savefig('Lying_example.png')
            plt.close()
            break

print('El Finisimo...')
