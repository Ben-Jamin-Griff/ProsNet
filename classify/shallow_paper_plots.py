from activity_monitor.activpal import Activpal
from stack.epoch_stack import EpochStack
from dataset.engineering_set import EngineeringSet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics

"""
Running shallow paper plots

This script creates the plots for the shallow paper...
"""

epoch_sizes = [15]

raw_data_paths = [
    "./apc-data/af-data/AF_Shin-AP971770 202a 28May21 3-28pm for 16d 20h 54m-CREA-PA08110254-AccelDataUncompressed.csv",
]

event_data_paths = [
    "./apc-data/af-data/AF_Thigh-AP971728 202a 28May21 3-24pm for 16d 21h-CREA-PA08110254-Events.csv",
]

# Load in each participant's data
activPal = Activpal()
activPal.load_raw_data(raw_data_paths[0])
activPal.load_event_data(event_data_paths[0])

posture_stack = EpochStack()
posture_stack.get_data(activPal)
posture_stack.create_stack(stack_type = 'pure', subset_of_data = 1000)

engineering_set = EngineeringSet()
engineering_set.get_data(activPal)
engineering_set.get_posture_stack(posture_stack)
engineering_set.create_set()

final_count = 6 # 3 works well for this value but test out some valuse

float_range_array = np.arange(0, 15, 15/295)
#float_range_list = list(float_range_array)

counter = 0
data_ensemble = np.empty([295, 3])
for data, posture in zip(engineering_set.dataset[0], engineering_set.dataset[1]):
    if posture == 0:
        counter += 1
        data_ensemble = data_ensemble + data
        if counter == final_count:
            data_ensemble = data_ensemble / final_count
            data_ensemble = ((data_ensemble/253)-0.5)*4
            plt.plot(np.arange(0, 15, 15/295), data_ensemble)
            plt.ylabel('Acceleration gs')
            plt.xlabel('Seconds')
            #plt.title('Sitting')
            plt.savefig('Sitting_example.png')
            break
plt.close()

counter = 0
data_ensemble = np.empty([295, 3])
for data, posture in zip(engineering_set.dataset[0], engineering_set.dataset[1]):
    if posture == 1:
        counter += 1
        data_ensemble = data_ensemble + data
        if counter == final_count:
            data_ensemble = data_ensemble / final_count
            data_ensemble = ((data_ensemble/253)-0.5)*4
            plt.plot(np.arange(0, 15, 15/295), data_ensemble)
            plt.ylabel('Acceleration gs')
            plt.xlabel('Seconds')
            #plt.title('Standing')
            plt.savefig('Standing_example.png')
            break
plt.close()

counter = 0
data_ensemble = np.empty([295, 3])
for data, posture in zip(engineering_set.dataset[0], engineering_set.dataset[1]):
    if posture == 2:
        counter += 1
        data_ensemble = data_ensemble + data
        if counter == final_count:
            data_ensemble = data_ensemble / final_count
            data_ensemble = ((data_ensemble/253)-0.5)*4
            plt.plot(np.arange(0, 15, 15/295), data_ensemble)
            plt.ylabel('Acceleration gs')
            plt.xlabel('Seconds')
            #plt.title('Stepping')
            plt.savefig('Stepping_example.png')
            break
plt.close()

counter = 0
for data, posture in zip(engineering_set.dataset[0], engineering_set.dataset[1]):
    if posture == 3:
        counter += 1
        data_ensemble = data_ensemble + data
        if counter == final_count:
            data_ensemble = data_ensemble / final_count
            data_ensemble = ((data_ensemble/253)-0.5)*4
            plt.plot(np.arange(0, 15, 15/295), data_ensemble)
            plt.ylabel('Acceleration (g)')
            plt.xlabel('Seconds')
            #plt.title('Lying')
            plt.savefig('Lying_example.png')
            break
plt.close()

print('El Finisimo...')