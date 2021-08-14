import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Running shallow paper plots

This script creates the plots for the shallow paper...
"""

# Collecting the datasets

file_dir = './apc-data/processed-5000-events/'

feature_set = []
posture_set = []
participant_ids = []

EPOCH_SIZES = [5, 15, 30, 60, 120, 180]

for epoch_size in EPOCH_SIZES:
  feature_set.append(np.load(file_dir + 'set_size_' + str(epoch_size) + '_feature_set.npy'))
  posture_set.append(np.load(file_dir + 'set_size_' + str(epoch_size) + '_feature_set_classes.npy'))
  participant_ids.append(np.load(file_dir + 'set_size_' + str(epoch_size) + '_feature_set_ids.npy'))

# Combining similar activity codes

for epoch, dataset in enumerate(posture_set):
  for count, value in enumerate(dataset):
    if dataset[count] == 2.1:
      dataset[count] = 2
    elif dataset[count] == 3.1:
      dataset[count] = 3
    elif dataset[count] == 3.2:
      dataset[count] = 3
    else:
      continue
  posture_set[epoch] = dataset

# Removing specific activity codes from dataset

for count, epoch in enumerate(EPOCH_SIZES):
    for class_to_remove in [4,5]:
        keep_idx = posture_set[count] != class_to_remove
        posture_set[count] = posture_set[count][keep_idx]
        feature_set[count] = feature_set[count][keep_idx]
        participant_ids[count] = participant_ids[count][keep_idx]

# How many participants are in the datasets

print('Number of participants')
print(len(np.unique(participant_ids[0])))

for count, epoch in enumerate(EPOCH_SIZES):

    print('Creating plots with epoch size ' + str(epoch))

    analysis_feature_set = feature_set[count].copy()
    analysis_posture_set = posture_set[count].copy()
    analysis_participant_ids = participant_ids[count].copy()

    breakpoint()

    print('El Finisimo...')
