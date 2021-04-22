from dataset_abc import ABCDataset

import pandas as pd
import numpy as np
import math

class EngineeringSet(ABCDataset):
    def __init__(self):
        self.dataset = None
        self.raw_acceleration_data = None
        self.posture_stack = None
        self.posture_stack_duration = None
        self.posture_stack_epoch_type = None

    def show_set(self):
        print('Engineering Set')
        print('----------')
        #print('Engineering set data')
        #print(self.dataset[0])
        #print('----------')
        #print('Posture Classes')
        #print(self.dataset[1])
        #print('----------')
        if self.posture_stack_epoch_type == 'mixed':
            print('Extracted Set')
            print(f" {len(self.dataset[1])} mixed epochs were extracted from {len(self.posture_stack.index)} total epochs.")
            print('----------')
        elif self.posture_stack_epoch_type == 'pure':
            print('Extracted Set')
            print(f" {len(self.dataset[1])} pure epochs were extracted from {len(self.posture_stack.index)} total epochs.")
            print('----------')

    def get_data(self, activity_monitor):
        self.raw_acceleration_data = activity_monitor.raw_data

    def get_posture_stack(self, posture_stack):
        self.posture_stack = posture_stack.posture_stack
        self.posture_stack_duration = posture_stack.posture_stack_duration
        self.posture_stack_epoch_type = posture_stack.posture_stack_epoch_type

    def create_set(self):
        # Print iterations progress
        def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
            """
            Call in a loop to create terminal progress bar
            @params:
                iteration   - Required  : current iteration (Int)
                total       - Required  : total iterations (Int)
                prefix      - Optional  : prefix string (Str)
                suffix      - Optional  : suffix string (Str)
                decimals    - Optional  : positive number of decimals in percent complete (Int)
                length      - Optional  : character length of bar (Int)
                fill        - Optional  : bar fill character (Str)
                printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
            """
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
            # Print New Line on Complete
            if iteration == total: 
                print()
        
        # Creat empty engineering set
        engineering_set = np.empty((0,295,3), int)
        posture_class = []
        # Load in the accelerometer
        raw_data_file_size = self.posture_stack_duration * 20
        CHUNKSIZE = 100000
        max_number_of_chunks = math.ceil(raw_data_file_size / CHUNKSIZE)
        loaded_chunks = 0
        printProgressBar (loaded_chunks, max_number_of_chunks, 'Engineering set progress:')
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

            printProgressBar (loaded_chunks, max_number_of_chunks, 'Engineering set progress:')
            # Contitions for early ending chunking for cropped datasets
            if loaded_chunks == max_number_of_chunks:
                break

        posture_class = np.array(posture_class)
        self.dataset = [engineering_set, posture_class]
        self.remove_classes()

    def remove_classes(self, classes_to_remove = []): # This needs to be in the posture stack class
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