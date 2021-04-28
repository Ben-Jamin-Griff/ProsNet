from dataset.dataset import Dataset

import pandas as pd
import numpy as np
import math
from scipy import signal, misc, stats

class FeatureSet(Dataset):
    def __init__(self):
        super().__init__()

    def create_set(self):
        if self.processing_type == 'epoch':
            if self.posture_stack is not None:

                feature_set = np.empty((0,7), int)
                posture_class = []

                raw_data_file_size = self.posture_stack_duration * 20
                CHUNKSIZE = 300000
                max_number_of_chunks = math.ceil(raw_data_file_size / CHUNKSIZE)
                loaded_chunks = 0
                self.print_progress_bar(loaded_chunks, max_number_of_chunks, 'Feature set progress:')
                for chunk in pd.read_csv(self.raw_acceleration_data, chunksize=CHUNKSIZE):
                    loaded_chunks += 1
                    chunk = chunk['sep=;'].apply(lambda x: pd.Series(x.split(';')))
                    chunk.columns = ["Time", "Index", "X", "Y", "Z"]
                    try:
                        chunk = chunk.apply(pd.to_numeric)
                        chunk = chunk.reset_index(drop=True)
                        chunk.Time = pd.to_datetime(chunk.Time, unit='d', origin='1899-12-30')
                    except:
                        chunk = chunk.iloc[1:,]
                        chunk = chunk.apply(pd.to_numeric)
                        chunk = chunk.reset_index(drop=True)
                        chunk.Time = pd.to_datetime(chunk.Time, unit='d', origin='1899-12-30')
                    #Loop through the posture stack and pull out the accelerometer data
                    for index, row in self.posture_stack.iterrows():
                        # If the epoch start time is less than the first value in data then look at next epoch
                        if row.Start_Time < chunk.Time.iloc[0]:
                            continue
                        # If the epoch start time is in the dataset and the end time is in the dataset then extract the data
                        elif row.Start_Time >= chunk.Time.iloc[0] and row.Finish_Time <= chunk.Time.iloc[-1]:
                            current_epoch = chunk[(chunk.Time >= row.Start_Time) & (chunk.Time <= row.Finish_Time)].copy()
                        # If the start time is in the dataset but the end time is not in the dataset then load in the next dataset
                        elif row.Start_Time >= chunk.Time.iloc[0] and row.Finish_Time > chunk.Time.iloc[-1]:
                            break
                        # If the start time is greater than the last value in the dataset then load in the next dataset 
                        elif row.Start_Time > chunk.Time.iloc[-1]:
                            break

                        # Assign the accelerometer data to a tensor index (in numpy form, convert to tf later)
                        #current_epoch_accel_data = current_epoch[['X','Y','Z']].to_numpy()
                        xData = current_epoch['X'].to_numpy()
                        #self.plot_signal(xData, 'xData')
                        yData = current_epoch['Y'].to_numpy()
                        #self.plot_signal(yData, 'yData')
                        zData = current_epoch['Z'].to_numpy()
                        #self.plot_signal(zData, 'zData')

                        # Filter the data
                        fc = 5  # Cut-off frequency of the filter
                        w = fc / (20 / 2) # Normalize the frequency
                        b, a = signal.butter(5, w, 'low')
                        xDataFilt = signal.filtfilt(b, a, xData)
                        #self.plot_signal(xDataFilt, 'xDataFilt')
                        yDataFilt = signal.filtfilt(b, a, yData)
                        #self.plot_signal(yDataFilt, 'yDataFilt')
                        zDataFilt = signal.filtfilt(b, a, zData)
                        #self.plot_signal(zDataFilt, 'zDataFilt')
                        
                        # Calculate vector magnitude
                        vmData = np.sqrt(xDataFilt**2 + yDataFilt**2 + zDataFilt**2)
                        #self.plot_signal(vmData, 'vmData')
                        
                        # Calculate jerk
                        xJerkData = np.diff(xDataFilt)
                        #self.plot_signal(xJerkData, 'xJerkData')
                        yJerkData = np.diff(yDataFilt)
                        #self.plot_signal(yJerkData, 'yJerkData')
                        zJerkData = np.diff(zDataFilt)
                        #self.plot_signal(zJerkData, 'zJerkData')

                        # Calculate vector magnitude
                        vmJerkData = np.sqrt(xJerkData**2 + yJerkData**2 + zJerkData**2)
                        #self.plot_signal(vmJerkData, 'vmJerkData')

                        feature_array = []

                        # Average value in signal buffer for all acceleration components
                        #feature_array=np.append(feature_array, np.mean(xDataFilt))
                        #feature_array=np.append(feature_array, np.mean(yDataFilt))
                        #feature_array=np.append(feature_array, np.mean(zDataFilt))
                        feature_array=np.append(feature_array, np.mean(vmData))

                        # Standard deviation
                        #feature_array=np.append(feature_array, np.std(xDataFilt))
                        #feature_array=np.append(feature_array, np.std(yDataFilt))
                        #feature_array=np.append(feature_array, np.std(zDataFilt))
                        feature_array=np.append(feature_array, np.std(vmData))

                        # Median absolute deviation
                        #feature_array=np.append(feature_array, stats.median_absolute_deviation(xDataFilt))
                        #feature_array=np.append(feature_array, stats.median_absolute_deviation(yDataFilt))
                        #feature_array=np.append(feature_array, stats.median_absolute_deviation(zDataFilt))
                        feature_array=np.append(feature_array, stats.median_absolute_deviation(vmData))

                        # Maximum sample
                        #feature_array=np.append(feature_array, np.max(xDataFilt))
                        #feature_array=np.append(feature_array, np.max(yDataFilt))
                        #feature_array=np.append(feature_array, np.max(zDataFilt))
                        feature_array=np.append(feature_array, np.max(vmData))

                        # Minimum sample
                        #feature_array=np.append(feature_array, np.min(xDataFilt))
                        #feature_array=np.append(feature_array, np.min(yDataFilt))
                        #feature_array=np.append(feature_array, np.min(zDataFilt))
                        feature_array=np.append(feature_array, np.min(vmData))

                        # Signal magnitude area

                        # Signal magnitude area jerk

                        # Energy measure
                        energy = np.sum(vmData**2) / len(vmData)
                        feature_array=np.append(feature_array, energy)

                        # Inter-quartile range
                        feature_array=np.append(feature_array, stats.iqr(vmData, axis=0))

                        # Autocorrelation features for all three acceleration components (3 each): 
                        # height of main peak; height and position of second peak

                        # Spectral peak features (12 each): height and position of first 6 peaks
                        #f, t, Sxx = signal.spectrogram(vmData, 20)

                        #np.sort(Sxx)[-6:]

                        #np.argmax(Sxx)

                        # Spectral power features (5 each): total power in 5 adjacent
                        # and pre-defined frequency bands


                        # Add this row to the feature set
                        feature_set=np.append(feature_set, [feature_array], axis=0)
                        # Assign the corresponding event code to a posture class list
                        posture_class.append(row.Event_Code)

                    self.print_progress_bar(loaded_chunks, max_number_of_chunks, 'Feature set progress:')
                    # Contitions for early ending chunking for cropped datasets
                    if loaded_chunks == max_number_of_chunks:
                        break

                posture_class = np.array(posture_class)
                self.dataset = [feature_set, posture_class]
                self.remove_classes()
            else:
                pass
        else:
            pass