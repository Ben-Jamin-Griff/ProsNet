from posture_stack_abc import ABCPostureStack

import pandas as pd
import numpy as np
import math
import datetime
#import xlrd

class EpochStack(ABCPostureStack):
    def __init__(self, processing_type='epoch'):
        self.processing_type = processing_type
        self.posture_stack = None

    def get_data(self, activity_monitor):
        self.events_to_process = activity_monitor.event_data

    def show_stack(self):
        print('Posture Stack')
        print('----------')
        print('Posture stack data')
        print(self.posture_stack)
        print('----------')
        print('Unique values')
        print(self.posture_stack.Event_Code.unique())
        print('----------')

    def create_stack(self, stack_type, subset_of_data = None):
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

        if self.processing_type == 'epoch':
            event_data = pd.read_csv(self.events_to_process)
            # subset of data for testing
            if subset_of_data:
                print(f'Using subset of data with {subset_of_data} events')
                event_data = event_data.iloc[:subset_of_data]
            event_data.Time = pd.to_datetime(event_data.Time, unit='d', origin='1899-12-30')
            epochSize = 15
            windowShift = 5
            startTime = event_data.Time.iloc[0]
            endTime = event_data.Time.iloc[-1]
            totalTime = (endTime - startTime).total_seconds()
            numOfEvents = math.ceil(totalTime / windowShift)
            column_names = ['Start_Time', 'Finish_Time', 'Event_Code']
            posture_stack = pd.DataFrame(0, index=np.arange(numOfEvents), columns=column_names)
            for i in range(numOfEvents):
                printProgressBar (i, numOfEvents, 'Creating posture stack progress:')

                posture_stack.iloc[i, 0] = startTime + datetime.timedelta(0,windowShift*i)
                posture_stack.iloc[i, 1] = posture_stack.iloc[i, 0] + datetime.timedelta(0,epochSize)
                current_epoch_startTime = event_data.Time[(event_data.Time <= posture_stack.iloc[i, 0])].tail(1).item()
                current_epoch_endTime = event_data.Time[(event_data.Time <= posture_stack.iloc[i, 1])].tail(1).item()
                current_epoch = event_data[(event_data.Time >= current_epoch_startTime) & (event_data.Time <= current_epoch_endTime)].copy()
                if len(current_epoch.index) == 1:
                    posture_stack.iloc[i, 2] = current_epoch['ActivityCode (0=sedentary 1=standing 2=stepping 2.1=cycling 3.1=primary lying, 3.2=secondary lying 4=non-wear 5=travelling)']
                else:
                    # if mixed events are required
                    if stack_type == 'mixed':
                        # Crop the time of the first and final events
                        first_new_value = current_epoch['Interval (s)'].iloc[0] - ((posture_stack.iloc[i, 0] - current_epoch_startTime).total_seconds())
                        last_new_value = ((posture_stack.iloc[i, 1] - current_epoch_endTime).total_seconds())
                        current_epoch.iloc[0,2]= first_new_value
                        current_epoch.iloc[-1,2] = last_new_value
                        # Work out which is the predominent event
                        activity_codes = current_epoch['ActivityCode (0=sedentary 1=standing 2=stepping 2.1=cycling 3.1=primary lying, 3.2=secondary lying 4=non-wear 5=travelling)'].unique()
                        activity_codes_counter = {}
                        for code in activity_codes:
                            activity_code_dataframe = current_epoch[current_epoch['ActivityCode (0=sedentary 1=standing 2=stepping 2.1=cycling 3.1=primary lying, 3.2=secondary lying 4=non-wear 5=travelling)'] == code]
                            activity_code_counter_value = activity_code_dataframe['Interval (s)'].sum()
                            activity_codes_counter[code] = activity_code_counter_value
                        max_activity_code = max(activity_codes_counter, key=activity_codes_counter.get)
                        # Assign predominent event as the code
                        posture_stack.iloc[i, 2] = max_activity_code
                    # if pure events are required
                    elif stack_type == 'pure':
                        if np.std(current_epoch['ActivityCode (0=sedentary 1=standing 2=stepping 2.1=cycling 3.1=primary lying, 3.2=secondary lying 4=non-wear 5=travelling)'].unique()) == 0:
                            posture_stack.iloc[i, 2] = current_epoch.iloc[0,3]
                        else:
                            posture_stack.iloc[i, 2] = 99
            self.posture_stack = posture_stack
            

