from ProsNet.dataset.dataset import Dataset
from ProsNet.plotter import Plotter

import pandas as pd
import numpy as np
import math
from scipy import signal, misc, stats
import datetime
import random as r

from uos_activpal.io.raw import load_activpal_data

import warnings
warnings.filterwarnings("ignore")

class FeatureSet(Dataset, Plotter):
    def __init__(self):
        super().__init__()

    def create_set(self):
        if self.processing_type == 'epoch':
            if self.posture_stack is not None:
                feature_set = np.empty((0,100), int) # 4 or 100
                posture_class = []
                meta, signals = load_activpal_data(self.raw_acceleration_data)
                total_time = meta.stop_datetime - meta.start_datetime
                total_samples = int(total_time.total_seconds() * 20)
                arr = np.array([meta.start_datetime + datetime.timedelta(seconds=i*0.05) for i in range(total_samples)])
                x = signals[:total_samples,0]
                y = signals[:total_samples,1]
                z = signals[:total_samples,2]
                chunk = pd.DataFrame({'Time':arr, 'X':x, 'Y':y, 'Z':z})
                #Loop through the posture stack and pull out the accelerometer data
                for index, row in self.posture_stack.iterrows():
                    # If the epoch start time is less than the first value in data then look at next epoch
                    if row.Start_Time < chunk.Time.iloc[0]:
                        continue
                    # If the epoch start time is in the dataset and the end time is in the dataset then extract the data
                    elif row.Start_Time >= chunk.Time.iloc[0] and row.Finish_Time <= chunk.Time.iloc[-1]:
                        current_epoch = chunk[(chunk.Time >= row.Start_Time) & (chunk.Time <= row.Finish_Time)].copy()
                    # If the start time is in the dataset but the end time is not in the dataset then break to loop
                    elif row.Start_Time >= chunk.Time.iloc[0] and row.Finish_Time > chunk.Time.iloc[-1]:
                        break
                    # If the start time is greater than the last value in the dataset then break the loop
                    elif row.Start_Time > chunk.Time.iloc[-1]:
                        break

                    feature_array = self.create_feature_array(current_epoch)
                    # Add this row to the feature set
                    feature_set=np.append(feature_set, [feature_array], axis=0)

                    # Assign the corresponding event code to a posture class list
                    posture_class.append(row.Event_Code)
                    self.print_progress_bar(index, len(self.posture_stack.index), 'Feature set progress:')

                posture_class = np.array(posture_class)
                participant_id = np.full(shape=len(posture_class), fill_value=r.randint(0,9999), dtype=np.int)
                self.dataset = [feature_set, posture_class, participant_id]
                self.remove_classes()
            else: # processing data without a posture stack, only raw data without events
                EPOCH_SIZE = 15
                feature_set = np.empty((0,100), int) # 4 or 100
                meta, signals = load_activpal_data(self.raw_acceleration_data)
                total_time = meta.stop_datetime - meta.start_datetime
                self.posture_stack_start_time = meta.start_datetime
                total_samples = int(total_time.total_seconds() * 20)
                total_windows = math.floor(total_time.total_seconds() / EPOCH_SIZE)
                arr = np.array([meta.start_datetime + datetime.timedelta(seconds=i*0.05) for i in range(total_samples)])
                x = signals[:total_samples,0]
                y = signals[:total_samples,1]
                z = signals[:total_samples,2]
                chunk = pd.DataFrame({'Time':arr, 'X':x, 'Y':y, 'Z':z})
                #Loop through the raw data in epoch sizes and add to a chunk
                time_set = []
                for window in range(0, total_windows):
                    current_epoch_x = x[window*(20*EPOCH_SIZE):(window+1)*(20*EPOCH_SIZE)]
                    current_epoch_y = y[window*(20*EPOCH_SIZE):(window+1)*(20*EPOCH_SIZE)]
                    current_epoch_z = z[window*(20*EPOCH_SIZE):(window+1)*(20*EPOCH_SIZE)]
                    time_set.append(arr[window*(20*EPOCH_SIZE)])
                    current_epoch = pd.DataFrame(list(zip(current_epoch_x, current_epoch_y, current_epoch_z)),columns =['X', 'Y', 'Z'])

                    feature_array = self.create_feature_array(current_epoch)
                    # Add this row to the feature set
                    feature_set=np.append(feature_set, [feature_array], axis=0)

                    self.print_progress_bar(window+1, total_windows, 'Feature set progress:')

                posture_class = []
                participant_id = []
                self.dataset = [feature_set, posture_class, participant_id, time_set]
        else:
            pass # this should be populated for analysing data using an event based approach instead of an epoch based approach.

    def create_feature_array(self, current_epoch):
        xData = current_epoch['X'].to_numpy() #self.plot_signal(xData, 'xData')
        yData = current_epoch['Y'].to_numpy()
        zData = current_epoch['Z'].to_numpy()

        # Filter the data
        fc = 5  # Cut-off frequency of the filter
        w = fc / (20 / 2) # Normalize the frequency
        b, a = signal.butter(5, w, 'low')
        xDataFilt = signal.filtfilt(b, a, xData)
        yDataFilt = signal.filtfilt(b, a, yData)
        zDataFilt = signal.filtfilt(b, a, zData)
        
        # Calculate vector magnitude
        vmData = np.sqrt(xDataFilt**2 + yDataFilt**2 + zDataFilt**2)

        feature_array = []
        # Average value in signal buffer for all acceleration components
        feature_array=np.append(feature_array, np.mean(xDataFilt))
        feature_array=np.append(feature_array, np.mean(yDataFilt))
        feature_array=np.append(feature_array, np.mean(zDataFilt))
        feature_array=np.append(feature_array, np.mean(vmData))
        # Standard deviation
        feature_array=np.append(feature_array, np.std(xDataFilt))
        feature_array=np.append(feature_array, np.std(yDataFilt))
        feature_array=np.append(feature_array, np.std(zDataFilt))
        feature_array=np.append(feature_array, np.std(vmData))
        # Median absolute deviation
        feature_array=np.append(feature_array, stats.median_absolute_deviation(xDataFilt))
        feature_array=np.append(feature_array, stats.median_absolute_deviation(yDataFilt))
        feature_array=np.append(feature_array, stats.median_absolute_deviation(zDataFilt))
        feature_array=np.append(feature_array, stats.median_absolute_deviation(vmData))
        # Maximum sample
        feature_array=np.append(feature_array, np.max(xDataFilt))
        feature_array=np.append(feature_array, np.max(yDataFilt))
        feature_array=np.append(feature_array, np.max(zDataFilt))
        feature_array=np.append(feature_array, np.max(vmData))
        # Minimum sample
        feature_array=np.append(feature_array, np.min(xDataFilt))
        feature_array=np.append(feature_array, np.min(yDataFilt))
        feature_array=np.append(feature_array, np.min(zDataFilt))
        feature_array=np.append(feature_array, np.min(vmData))
        # Signal magnitude area
        feature_array=np.append(feature_array, np.trapz(xDataFilt))
        feature_array=np.append(feature_array, np.trapz(yDataFilt))
        feature_array=np.append(feature_array, np.trapz(zDataFilt))
        feature_array=np.append(feature_array, np.trapz(vmData))
        # Energy measure
        energy = np.sum(xDataFilt**2) / len(xDataFilt)
        feature_array=np.append(feature_array, energy)
        energy = np.sum(yDataFilt**2) / len(yDataFilt)
        feature_array=np.append(feature_array, energy)
        energy = np.sum(zDataFilt**2) / len(zDataFilt)
        feature_array=np.append(feature_array, energy)
        energy = np.sum(vmData**2) / len(vmData)
        feature_array=np.append(feature_array, energy)
        # Inter-quartile range
        feature_array=np.append(feature_array, stats.iqr(xDataFilt, axis=0))
        feature_array=np.append(feature_array, stats.iqr(yDataFilt, axis=0))
        feature_array=np.append(feature_array, stats.iqr(zDataFilt, axis=0))
        feature_array=np.append(feature_array, stats.iqr(vmData, axis=0))
        # Autocorrelation features for all three acceleration components (3 each): height of main peak; height and position of second peak - Not sure this is right?
        autocorrelation = np.correlate(xDataFilt, xDataFilt, mode='full')
        autocorrelation = autocorrelation[len(xDataFilt)-1:][0]
        feature_array=np.append(feature_array, autocorrelation)

        autocorrelation = np.correlate(yDataFilt, yDataFilt, mode='full')
        autocorrelation = autocorrelation[len(yDataFilt)-1:][0]
        feature_array=np.append(feature_array, autocorrelation)

        autocorrelation = np.correlate(zDataFilt, zDataFilt, mode='full')
        autocorrelation = autocorrelation[len(zDataFilt)-1:][0]
        feature_array=np.append(feature_array, autocorrelation)

        autocorrelation = np.correlate(vmData, vmData, mode='full')
        autocorrelation = autocorrelation[len(vmData)-1:][0]
        feature_array=np.append(feature_array, autocorrelation)
        # Spectral peak features (12 each): height and position of first 6 peaks
        f, p = signal.periodogram(xDataFilt, 20e0)
        sort_index = np.argsort(p)
        p_sorted = p[sort_index]
        f_sorted = f[sort_index]
        speak_feats = p_sorted[-6:]
        speak_feats2 = f_sorted[-6:]
        feature_array=np.append(feature_array, speak_feats)
        feature_array=np.append(feature_array, speak_feats2)

        f, p = signal.periodogram(yDataFilt, 20e0)
        sort_index = np.argsort(p)
        p_sorted = p[sort_index]
        f_sorted = f[sort_index]
        speak_feats = p_sorted[-6:]
        speak_feats2 = f_sorted[-6:]
        feature_array=np.append(feature_array, speak_feats)
        feature_array=np.append(feature_array, speak_feats2)

        f, p = signal.periodogram(zDataFilt, 20e0)
        sort_index = np.argsort(p)
        p_sorted = p[sort_index]
        f_sorted = f[sort_index]
        speak_feats = p_sorted[-6:]
        speak_feats2 = f_sorted[-6:]
        feature_array=np.append(feature_array, speak_feats)
        feature_array=np.append(feature_array, speak_feats2)

        f, p = signal.periodogram(vmData, 20e0)
        sort_index = np.argsort(p)
        p_sorted = p[sort_index]
        f_sorted = f[sort_index]
        speak_feats = p_sorted[-6:]
        speak_feats2 = f_sorted[-6:]
        feature_array=np.append(feature_array, speak_feats)
        feature_array=np.append(feature_array, speak_feats2)
        # Spectral power features (4 each): total power in 4 adjacent and pre-defined frequency bands
        edges = [0.5, 1.5, 5, 7.5, 10]
        n_feats = len(edges)-1
        
        spower_feats = []
        f, p = signal.periodogram(xDataFilt, 20e0)
        for i in range(n_feats):
            mask = (f >= edges[i]) & (f <= edges[i+1])
            sum(p[mask])
            spower_feats=np.append(spower_feats, sum(p[mask]))
        feature_array=np.append(feature_array, spower_feats)

        spower_feats = []
        f, p = signal.periodogram(yDataFilt, 20e0)
        for i in range(n_feats):
            mask = (f >= edges[i]) & (f <= edges[i+1])
            sum(p[mask])
            spower_feats=np.append(spower_feats, sum(p[mask]))
        feature_array=np.append(feature_array, spower_feats)

        spower_feats = []
        f, p = signal.periodogram(zDataFilt, 20e0)
        for i in range(n_feats):
            mask = (f >= edges[i]) & (f <= edges[i+1])
            sum(p[mask])
            spower_feats=np.append(spower_feats, sum(p[mask]))
        feature_array=np.append(feature_array, spower_feats)

        spower_feats = []
        f, p = signal.periodogram(vmData, 20e0)
        for i in range(n_feats):
            mask = (f >= edges[i]) & (f <= edges[i+1])
            sum(p[mask])
            spower_feats=np.append(spower_feats, sum(p[mask]))
        feature_array=np.append(feature_array, spower_feats)
        return feature_array