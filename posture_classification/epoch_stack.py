from posture_stack_abc import ABCPostureStack

import pandas as pd
import numpy as np
import math
import datetime
import xlrd

class EpochStack(ABCPostureStack):
    def __init__(self, processing_type='epoch'):
        self.processing_type = processing_type

    def get_data(self, activity_monitor):
        self.events_to_process = activity_monitor.event_data

    def create_stack(self):
        if self.processing_type == 'epoch':
            event_data = pd.read_csv(self.events_to_process)
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
                posture_stack.iloc[i, 0] = startTime + datetime.timedelta(0,windowShift*i)
                posture_stack.iloc[i, 1] = posture_stack.iloc[i, 0] + datetime.timedelta(0,epochSize)
                current_epoch_startTime = event_data.Time[(event_data.Time <= posture_stack.iloc[i, 0])].tail(1).item()
                current_epoch_endTime = event_data.Time[(event_data.Time <= posture_stack.iloc[i, 1])].tail(1).item()
                current_epoch = event_data[(event_data.Time >= current_epoch_startTime) & (event_data.Time <= current_epoch_endTime)]
                if len(current_epoch.index) == 1:
                    posture_stack.iloc[i, 2] = current_epoch['ActivityCode (0=sedentary 1=standing 2=stepping 2.1=cycling 3.1=primary lying, 3.2=secondary lying 4=non-wear 5=travelling)']
                else:
                    breakpoint()
                    # Crop the time of the first and final events

                    # Work out which is the predominent event

                    # Assign predominent event as the code
                    posture_stack.iloc[i, 2] = 69
                    # Question - how can I get pure events out instead?
            print(posture_stack)
            
