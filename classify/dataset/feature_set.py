from dataset.dataset import Dataset

import pandas as pd
import numpy as np
import math
from scipy import signal

class FeatureSet(Dataset):
    def __init__(self):
        super().__init__()

    def create_set(self):
        if self.processing_type == 'epoch':
            if self.posture_stack is not None:
                feature_set = np.empty((0,295,3), int)
                posture_class = []

                raw_data_file_size = self.posture_stack_duration * 20
                CHUNKSIZE = 300000
                max_number_of_chunks = math.ceil(raw_data_file_size / CHUNKSIZE)
                loaded_chunks = 0
                self.print_progress_bar(loaded_chunks, max_number_of_chunks, 'Engineering set progress:')
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
                        yData = current_epoch['Y'].to_numpy()
                        zData = current_epoch['Z'].to_numpy()

                        # Filter the data
                        fc = 30  # Cut-off frequency of the filter
                        w = fc / (20 / 2) # Normalize the frequency
                        b, a = signal.butter(5, w, 'low')
                        xDataFilt = signal.filtfilt(b, a, xData)
                        yDataFilt = signal.filtfilt(b, a, yData)
                        zDataFilt = signal.filtfilt(b, a, zData)

                        # Calculate vector magnitude
                        vmData = math.sqrt(xDataFilt**2 + yDataFilt**2 + zDataFilt**2)
                        
                        # Calculate jerk
                        xJerkData = scipy.misc.derivative(xDataFilt)
                        yJerkData = scipy.misc.derivative(yDataFilt)
                        zJerkData = scipy.misc.derivative(zDataFilt)

                        # Average value in signal buffer for all three acceleration components (1 each)

                        # Standard deviation

                        # Median absolute deviation

                        # Maximum sample

                        # Minimum sample

                        # Signal magnitude area

                        # Signal magnitude area jerk

                        # Energy measure

                        # Inter-quartile range

                        # Autocorrelation features for all three acceleration components (3 each): 
                        # height of main peak; height and position of second peak

                        # Spectral peak features (12 each): height and position of first 6 peaks

                        # Spectral power features (5 each): total power in 5 adjacent
                        # and pre-defined frequency bands

                        # Assign the corresponding event code to a posture class list
                        posture_class.append(row.Event_Code)

                    self.print_progress_bar(loaded_chunks, max_number_of_chunks, 'Engineering set progress:')
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