from ProsNet.dataset.dataset import Dataset

import pandas as pd
import numpy as np
import math

class EngineeringSet(Dataset):
    def __init__(self):
        super().__init__()

    def create_set(self, epochSize = 15):
        if self.processing_type == 'epoch':
            if self.posture_stack is not None:
                print('WARNING: CODE IS IGNORING SPECIFIED EPOCH SIZE WHEN YOU CREATED POSTURE STAK AND REVERTING TO A WINDOW OF 295 SAMPLES. UPDATE THIS CODE FOR DIFFERENT ENGINEERING SET EPOCH SIZES')
                engineering_set = np.empty((0,295,3), int)
                posture_class = []
                # Load in the accelerometer
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
                            #print('epoch start time is less than the first value in data')
                            continue
                        # If the epoch start time is in the dataset and the end time is in the dataset then extract the data
                        elif row.Start_Time >= chunk.Time.iloc[0] and row.Finish_Time <= chunk.Time.iloc[-1]:
                            current_epoch = chunk[(chunk.Time >= row.Start_Time) & (chunk.Time <= row.Finish_Time)].copy()
                        # If the start time is in the dataset but the end time is not in the dataset then load in the next dataset
                        elif row.Start_Time >= chunk.Time.iloc[0] and row.Finish_Time > chunk.Time.iloc[-1]: # <<< I'M MISSING EVENTS HERE BUT I DONT KNOW HOW TO FIX IT
                            #print('found epoch start time but epoch end time is outside of the dataset')
                            break
                            #is_local_var = "last_epoch" in locals()
                            #if not is_local_var:
                            #    last_epoch = current_epoch
                        # If the start time is greater than the last value in the dataset then load in the next dataset 
                        elif row.Start_Time > chunk.Time.iloc[-1]:
                            #print('epoch start time is greater than the last value in data')
                            break
                        # Assign the accelerometer data to a tensor index (in numpy form, convert to tf later)
                        current_epoch_accel_data = current_epoch[['X','Y','Z']].to_numpy()
                        engineering_set = np.append(engineering_set, [current_epoch_accel_data[:295,:]], axis=0)
                        # Assign the corresponding event code to a posture class list
                        posture_class.append(row.Event_Code)

                    self.print_progress_bar(loaded_chunks, max_number_of_chunks, 'Engineering set progress:')
                    # Contitions for early ending chunking for cropped datasets
                    if loaded_chunks == max_number_of_chunks:
                        break

                posture_class = np.array(posture_class)
                self.dataset = [engineering_set, posture_class]
                self.remove_classes()
            else:
                engineering_set = np.empty((0,295,3), int)
                CHUNKSIZE = 300000
                loaded_chunks = 0
                print(f"Loaded chunk: {loaded_chunks}")
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

                    numOfEpochs = (CHUNKSIZE / epochSize)
                    for i in range(int(numOfEpochs)):
                        current_epoch = chunk.iloc[(15*20)*i:(15*20)*(i+1):,:].copy()
                        current_epoch_accel_data = current_epoch[['X','Y','Z']].to_numpy()
                        # Break if the last epoch is too small
                        if current_epoch_accel_data.shape[0] < 300:
                            break

                        engineering_set = np.append(engineering_set, [current_epoch_accel_data[:295,:]], axis=0)
                    print(f"Loaded chunk: {loaded_chunks}")
                    # For testing this on a subset of data
                    #if loaded_chunks == 2:
                    #    break
                self.dataset = [engineering_set]
        else:
            pass