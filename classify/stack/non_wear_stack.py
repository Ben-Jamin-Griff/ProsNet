from stack.posture_stack_abc import ABCPostureStack
from helper import Helper

import pandas as pd
import numpy as np
import math
import datetime
from scipy import signal
import resampy

from uos_activpal.io.raw import load_activpal_data
import warnings
warnings.filterwarnings("ignore")

class NonWearStack(ABCPostureStack, Helper):
    def __init__(self, processing_type='epoch'):
        self.processing_type = processing_type
        self.posture_stack = None
        self.posture_stack_duration = None
        self.posture_stack_start_time = None

    def get_data(self, activity_monitor):
        self.raw_acceleration_data = activity_monitor.raw_data

    def show_stack(self):
        ## Not edited
        print('Posture Stack')
        print('----------')
        print('Unique class values')
        print(self.posture_stack.Event_Code.unique())
        print('----------')
        print('Posture stack duration')
        print(f"The posture stacks contains {self.posture_stack_duration} seconds of data.")
        print('----------')

    def create_stack(self, epochSize = 1):
        """
        epochSize = 1 for 1 second epochs
        """
        meta, signals = load_activpal_data(self.raw_acceleration_data)
        total_time = meta.stop_datetime - meta.start_datetime
        total_samples = int(total_time.total_seconds() * 20)
        arr = np.array([meta.start_datetime + datetime.timedelta(seconds=i*0.05) for i in range(total_samples)])
        x = signals[:total_samples,0]
        y = signals[:total_samples,1]
        z = signals[:total_samples,2]
        x_g = ((x/253)-0.5)*4
        y_g = ((y/253)-0.5)*4
        z_g = ((z/253)-0.5)*4

        ##predefined filter coefficients, as found by Jan Brond
        filesf = 20

        A_coeff = np.array(
            [1, -4.1637, 7.5712,-7.9805, 5.385, -2.4636, 0.89238, 0.06361, -1.3481, 2.4734, -2.9257, 2.9298, -2.7816, 2.4777,
            -1.6847, 0.46483, 0.46565, -0.67312, 0.4162, -0.13832, 0.019852])
        B_coeff = np.array(
            [0.049109, -0.12284, 0.14356, -0.11269, 0.053804, -0.02023, 0.0063778, 0.018513, -0.038154, 0.048727, -0.052577,
            0.047847, -0.046015, 0.036283, -0.012977, -0.0046262, 0.012835, -0.0093762, 0.0034485, -0.00080972, -0.00019623])

        def pptrunc(data, max_value):
            '''
            Saturate a vector such that no element's absolute value exceeds max_abs_value.
            Current name: absolute_saturate().
            :param data: a vector of any dimension containing numerical data
            :param max_value: a float value of the absolute value to not exceed
            :return: the saturated vector
            '''
            outd = np.where(data > max_value, max_value, data)
            return np.where(outd < -max_value, -max_value, outd)

        def trunc(data, min_value):
        
            '''
            Truncate a vector such that any value lower than min_value is set to 0.
            Current name zero_truncate().
            :param data: a vector of any dimension containing numerical data
            :param min_value: a float value the elements of data should not fall below
            :return: the truncated vector
            '''

            return np.where(data < min_value, 0, data)

        def runsum(data, length, threshold):
            '''
            Compute the running sum of values in a vector exceeding some threshold within a range of indices.
            Divides the data into len(data)/length chunks and sums the values in excess of the threshold for each chunk.
            Current name run_sum().
            :param data: a 1D numerical vector to calculate the sum of
            :param len: the length of each chunk to compute a sum along, as a positive integer
            :param threshold: a numerical value used to find values exceeding some threshold
            :return: a vector of length len(data)/length containing the excess value sum for each chunk of data
            '''
            
            N = len(data)
            cnt = int(math.ceil(N/length))

            rs = np.zeros(cnt)

            for n in range(cnt):
                for p in range(length*n, length*(n+1)):
                    if p<N and data[p]>=threshold:
                        rs[n] = rs[n] + data[p] - threshold

            return rs

        def counts(data, filesf, B=B_coeff, A=A_coeff):
            '''
            Get activity counts for a set of accelerometer observations.
            First resamples the data frequency to 30Hz, then applies a Butterworth filter to the signal, then filters by the
            coefficient matrices, saturates and truncates the result, and applies a running sum to get the final counts.
            Current name get_actigraph_counts()
            :param data: the vertical axis of accelerometer readings, as a vector
            :param filesf: the number of observations per second in the file
            :param a: coefficient matrix for filtering the signal, as found by Jan Brond
            :param b: coefficient matrix for filtering the signal, as found by Jan Brond
            :return: a vector containing the final counts
            '''
            
            deadband = 0.068
            sf = 30
            peakThreshold = 2.13
            adcResolution = 0.0164
            integN = 10
            gain = 0.965

            #if filesf>sf:
            data = resampy.resample(np.asarray(data), filesf, sf)

            B2, A2 = signal.butter(4, np.array([0.01, 7])/(sf/2), btype='bandpass')
            dataf = signal.filtfilt(B2, A2, data)

            B = B * gain

            #NB: no need for a loop here as we only have one axis in array
            fx8up = signal.lfilter(B, A, dataf)

            fx8 = pptrunc(fx8up[::3], peakThreshold) #downsampling is replaced by slicing with step parameter

            return runsum(np.floor(trunc(np.abs(fx8), deadband)/adcResolution), integN, 0)

        # calculate counts per axis
        c_x = counts(x_g, filesf)
        c_y = counts(y_g, filesf)
        c_z = counts(z_g, filesf)
        c_vm = np.sqrt(c_x**2 + c_y**2 + c_z**2)

        minutes = total_time.total_seconds()/60
        time_arr = np.linspace(0, minutes, num=len(c_vm))

        chunk = pd.DataFrame({'Time':time_arr, 'X':c_x, 'Y':c_y, 'Z':c_z, 'VM':c_vm})
        chunk = chunk.astype(int)

        NonWear = 0 # identifier for logical sorting
        Wear = 1 # identifier for logical sorting

        previousEpochClassification = 0 # The algorithm assumes non-wear prior to the start of recording
        timestep = 60 # Convert seconds (epochs from Actigraph count conversion) to minutes
        amplitudeSensitivity = 15 # original value = 15 % Relates to STEP 3: threshold under which the activity counts will be automatically assumed to be non-wear if detected within a period classified as non-wear
        timeSensitivity = 5 # original value = 5 % Relates to STEP 3: if 2 spikes of activity greater than 'amplitudeSensitivity' occur within 'timeSensitivity' minutes of each other this will be classified as wear
        NonWearData = []

        def IsActivityMaintained(index, chunk, timestep, amplitudeSensitivity, timeSensitivity, Wear, NonWear):
            counter = 1
            checkCountNo = 1
            NonWearDataToAdd = NonWear
            while checkCountNo < 20*timestep: # check the next 20 minutes of counts
                if index + checkCountNo > len(chunk): # if the end of the data has been reached
                    checkCountNo = 20*timestep # escape while loop
                else:
                    if chunk.iloc[index+checkCountNo, 4] <= amplitudeSensitivity: # if the magnitude of the counts for the next epoch in the loop is less than 'amplitudeSensitivity', keep the counter running
                        counter += 1
                        checkCountNo += 1
                    else: # next epoch measures more than 'amplitudeSensitivity' counts, stop the counter and choose whether to continue through the loop or classify as wear
                        if counter <= timeSensitivity*timestep: # it's been less than 'timeSensitivity' minutes since the last spike in activity counts
                            NonWearDataToAdd = Wear # classify as wear
                            checkCountNo = 20*timestep # escape while loop
                        elif counter > timeSensitivity*timestep: # it's been more than 'timeSensitivity' minutes since the last spike in activity counts
                            counter = 1
                            checkCountNo += 1
            return NonWearDataToAdd 

        def check_short_classifications(NonWearData, timestep):
            # check for classification periods lasting less than 10 mins before changing allocation. If the following period is longer then re-allocate as the opposite classification.
            Point1 = 0
            classification = NonWearData[0]
            for c, val in enumerate(NonWearData[1:]):
                if val == classification: # epoch was the same classification as the previous epoch
                    Point2 = c # counter
                else: # the classification has changed
                    if c == 1: # if on 2nd epoch in recording
                        NonWearData[c-1] = math.sqrt((classification-1)**2) # replace previous epoch with the opposite classification
                    elif Point2-Point1 < 10*timestep: # were there less than 10 minutes of the previous classification
                        # If yes...check length of following portion (the new classification)
                        epochCount = 1
                        escape = 0
                        while escape == 0:
                            for check in range(Point2, len(NonWearData), 1):
                                if NonWearData[check] == math.sqrt((classification-1)**2): # if epoch is classified with the new classification
                                    if check == len(NonWearData): # if run out of data to check
                                        escape = 1 # stop checking

                                    epochCount += 1 # keep checking the next epoch
                                else:
                                    escape = 1 # stop checking
                        
                        if epochCount < (Point2-Point1):# check if the next portion is shorter (anomaly)
                            pass
                        else:
                            NonWearData[Point1:Point2] = math.sqrt((classification-1)**2) # change the classification of the previous portion

                    Point1 = c # reset the counters
                    Point2 = c # reset the counters
                    classification = math.sqrt((classification-1)**2); # swap the reference classification
            return NonWearData

        breakpoint()
        for index, row in chunk.iterrows():
            if row['VM'] == 0: # If no counts were recorded for the epoch awaiting classification
                if previousEpochClassification == 0: # Was the previous epoch classified as non wear? If yes...
                    NonWearData.append(NonWear) # If previous epoch was classified as non-wear and no counts were recorded for this epoch - classify this epoch as non-wear
                else: # Was the previous epoch classified as non wear? If no...
                    if len(chunk)-index >= (20*timestep): # Relates to STEP 3 - Check if there are more than 20 mins left in the recording
                        pointsToCheck = index+(20*timestep)-1 # Inspect the next 20 minutes of counts
                    else: # If there are less than 20 minutes left...
                        pointsToCheck = len(chunk)-index # Inspect all the remaining minutes of data
                    
                    if sum(chunk.iloc[index:pointsToCheck, 4]) == 0: # Were any counts detected during the next 20 minutes?
                        NonWearData.append(NonWear) # If no counts were recorded then allocate this epoch as non-wear
                    else: # If there are counts during the next 20 minutes
                        NonWearDataToAdd = IsActivityMaintained(index, chunk, timestep, amplitudeSensitivity, timeSensitivity, Wear, NonWear) # See how long until the next count
                        if NonWearDataToAdd: #check this
                            NonWearData.append(NonWearDataToAdd)
                        if len(NonWearData) == index: # if the epoch hasn't been allocated as wear during the loop, allocate non-wear
                            NonWearData.append(NonWear)

            else: # If counts were recorded for the epoch awaiting classification
                if previousEpochClassification == 1: # was the previous epoch classified as wear? If yes...
                    NonWearData.append(Wear) # if the previous epoch was wear and there is still some activity, classify the current epoch as wear
                elif row['VM'] <= amplitudeSensitivity: # If the previous epoch was classified as non-wear, and the current epoch has less than 'amplitudeSensitivity' counts, it's unlikely that they just put the monitor/prosthesis on
                    NonWearData.append(NonWear)
                else: # if the previous epoch was classified as non-wear but this epoch shows to have more to have more than 'amplitudeSensivity' counts recorded
                    NonWearDataToAdd = IsActivityMaintained(index, chunk, timestep, amplitudeSensitivity, timeSensitivity, Wear, NonWear) # See how long until the next count
                    if NonWearDataToAdd: #check this
                        NonWearData.append(NonWearDataToAdd)
                    if len(NonWearData) == index: # if the epoch hasn't been allocated as wear during the loop, allocate non-wear
                            NonWearData.append(NonWear)
            
            previousEpochClassification = NonWearData[-1] # reset the categorisation of the previous epoch before re-entering the loop for the next epoch
            
            NonWearData = check_short_classifications(NonWearData, timestep)


        old = 0
        if old == 1:
            event_data = pd.read_csv(self.events_to_process)

            event_data.Time = pd.to_datetime(event_data.Time, unit='d', origin='1899-12-30')
            windowShift = epochSize/2
            startTime = event_data.Time.iloc[0]
            self.posture_stack_start_time = startTime
            endTime = event_data.Time.iloc[-1]
            totalTime = ((endTime - startTime).total_seconds()) + event_data['Interval (s)'].iloc[-1]
            self.posture_stack_duration = totalTime
            numOfEvents = math.ceil(totalTime / windowShift)
            column_names = ['Start_Time', 'Finish_Time', 'Event_Code']
            posture_stack = pd.DataFrame(0, index=np.arange(numOfEvents), columns=column_names)
            for i in range(numOfEvents):
                self.print_progress_bar(i+1, numOfEvents, 'Creating posture stack progress:')
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