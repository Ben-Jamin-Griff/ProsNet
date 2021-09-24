from stack.posture_stack_abc import ABCPostureStack
from helper import Helper

import pandas as pd
import numpy as np
import math
import datetime
from scipy import signal
import resampy
import tkinter as tk
from tkinter import filedialog
import random

import matplotlib.pyplot as plt
plt.ion()

from uos_activpal.io.raw import load_activpal_data
import warnings
warnings.filterwarnings("ignore")

class NonWearStack(ABCPostureStack, Helper):
    def __init__(self, processing_type='epoch'):
        self.processing_type = processing_type
        self.posture_stack = None
        self.start_time = None
        self.end_time = None
        self.total_time = None

    def get_data(self, activity_monitor):
        self.raw_acceleration_data = activity_monitor.raw_data

    def show_stack(self):
        ## Not edited
        print('Posture Stack')
        print('----------')
        try:
            self.posture_stack.iloc[:,[0,4,5,6,]].plot(x='Time')
        except:
            self.posture_stack.iloc[:,[0,4,5,]].plot(x='Time')

        plt.savefig('plot' + str(random.randrange(99)) + '.png')
        #plt.close()
        print('----------')

    def create_validation_stack(self, filename = None):
        if filename is not None:
            file_path = filename
        else:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(title = "Load validation data")

        non_wear_data = pd.read_csv(file_path)
        try:
            non_wear_data.start = pd.to_datetime(non_wear_data.start, format="%d/%m/%Y %H:%M")
            non_wear_data.end = pd.to_datetime(non_wear_data.end, format="%d/%m/%Y %H:%M")
        except:
            non_wear_data.start = pd.to_datetime(non_wear_data.start, unit='d', origin='1899-12-30')
            non_wear_data.end = pd.to_datetime(non_wear_data.end, unit='d', origin='1899-12-30')

        datetime_arr = np.array([self.start_time + datetime.timedelta(seconds=i*1) for i in range(len(self.posture_stack))])
        wear = [0]*(len(self.posture_stack))
        chunk = pd.DataFrame({'Time':datetime_arr, 'Wear':wear})
        for index, row in non_wear_data.iterrows():
            # If the epoch
            if row.end < chunk.Time.iloc[0]:
                pass
            # If the epoch
            elif row.start > chunk.Time.iloc[-1]:
                pass
            # If the epoch
            else:
                chunk[(chunk.Time >= row.start) & (chunk.Time <= row.end)] = chunk[(chunk.Time >= row.start) & (chunk.Time <= row.end)].replace([0],1)
        
        chunk['Wear'] = [(element - 1)*-99 for element in chunk['Wear']]
        self.posture_stack['Validation'] = chunk.Wear

    def create_stack(self, subset_of_data = None, min_non_wear = 60):
        """
        subset_of_data = None (assign a percentage value of the raw acc data)
        """
        meta, signals = load_activpal_data(self.raw_acceleration_data)
        total_time = meta.stop_datetime - meta.start_datetime
        total_samples = int(total_time.total_seconds() * 20)
        arr = np.array([meta.start_datetime + datetime.timedelta(seconds=i*0.05) for i in range(total_samples)])
        x = signals[:total_samples,0]
        y = signals[:total_samples,1]
        z = signals[:total_samples,2]

        if subset_of_data:
            print(f'Using subset of {subset_of_data} percent of the data')
            x = x[:math.ceil((subset_of_data/100)*total_samples)]
            y = y[:math.ceil((subset_of_data/100)*total_samples)]
            z = z[:math.ceil((subset_of_data/100)*total_samples)]

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
        datetime_arr = np.array([meta.start_datetime + datetime.timedelta(seconds=i*1) for i in range(len(c_vm))])
        chunk = pd.DataFrame({'Time':datetime_arr, 'X':c_x, 'Y':c_y, 'Z':c_z, 'VM':c_vm})
        #chunk = chunk.astype(int)

        NonWear = 0 # identifier for logical sorting
        Wear = 1 # identifier for logical sorting

        previousEpochClassification = 0 # The algorithm assumes non-wear prior to the start of recording
        timestep = 60 # Convert seconds (epochs from Actigraph count conversion) to minutes
        amplitudeSensitivity = 3 # original value = 15 threshold under which the activity counts will be automatically assumed to be non-wear if detected within a period classified as non-wear
        timeSensitivity = min_non_wear # original value = 5 if 2 spikes of activity greater than 'amplitudeSensitivity' occur within 'timeSensitivity' minutes of each other this will be classified as wear
        NonWearDataToAdd = 99 # assigning this to identify any mistakes
        NonWearData = []

        def IsActivityMaintained(index, chunk, timestep, amplitudeSensitivity, timeSensitivity):
            checkCountNo = 1
            NonWearDataToAdd = 0
            if len(chunk)-index < timeSensitivity*timestep:
                check_duration = len(chunk)-index
            else:
                check_duration = timeSensitivity*timestep

            while checkCountNo < check_duration: # check the next x minutes of counts
                #if index + checkCountNo > len(chunk): # if the end of the data has been reached
                #    checkCountNo = 60*timestep # escape while loop
                #else:
                if chunk.iloc[index+checkCountNo, 4] <= amplitudeSensitivity: # if the magnitude of the counts for the next epoch in the loop is less than 'amplitudeSensitivity', keep the counter running
                    checkCountNo += 1
                else: # next epoch measures more than 'amplitudeSensitivity' counts, stop the counter and choose whether to continue through the loop or classify as wear
                    NonWearDataToAdd = 1 # classify as wear
                    checkCountNo = check_duration # escape while loop
            return NonWearDataToAdd 

        def check_short_classifications(Data, timestep, short_duration):
            # check for classification periods lasting less than x mins before changing allocation. If the following period is longer then re-allocate as the opposite classification.
            NonWearData = Data.copy()
            Point1 = 0
            for c, classification in enumerate(NonWearData):
                self.print_progress_bar(c, len(NonWearData), 'Checking non-wear classifications progress:')
                if c == len(NonWearData)-1:
                    if classification != NonWearData[c-1]: # This should change the last sample if it's wrongly classified. Not sure if this is working?
                        NonWearData[c] = NonWearData[c-1]
                    return NonWearData
                if classification == NonWearData[c+1]: # epoch was the same classification as the previous epoch
                    Point2 = c+1 # counter
                elif NonWearData[c+1] == 0: # the classification has changed and the following classification is non wear (not checking for short wear periods)
                    Point2 = c
                    if c == 0: # if on 2nd epoch in recording
                        NonWearData[c] = math.sqrt((classification-1)**2) # replace previous epoch with the opposite classification
                    elif Point2-Point1 < short_duration*timestep: # were there less than 'short_duration' minutes of the previous classification
                        # If yes...check length of following portion (the new classification)
                        epochCount = 0
                        for check in range(Point2+1, len(NonWearData), 1):
                            if NonWearData[check] == math.sqrt((classification-1)**2): # if epoch is classified with the new classification
                                epochCount += 1 # keep checking the next epoch
                            else:
                                break
                        
                        if epochCount > (Point2-Point1):# check if the next portion is shorter (anomaly)
                            #reakpoint()
                            assign = [math.sqrt((classification-1)**2)]*((Point2-Point1)+1)
                            NonWearData[Point1:Point2+1] = assign # change the classification of the previous portion
                    Point1 = c+1 # reset the first counters
                else:
                    Point1 = c+1 # reset the first counters
                    Point2 = c+1 # counter
    
        for index, row in chunk.iterrows():
            self.print_progress_bar(index+1, len(chunk), 'Creating non-wear stack progress:')
            if row['VM'] == 0: # If no counts were recorded for the epoch awaiting classification
                if previousEpochClassification == 0: # Was the previous epoch classified as non wear? If yes...
                    NonWearData.append(NonWear) # If previous epoch was classified as non-wear and no counts were recorded for this epoch - classify this epoch as non-wear
                else: # Was the previous epoch classified as non wear? If no...

                    if len(chunk)-index >= (timeSensitivity*timestep): # Check if there are more than x mins left in the recording
                        pointsToCheck = index+(timeSensitivity*timestep)#-1 # Inspect the next x minutes of counts
                    else: # If there are less than x minutes left...
                        pointsToCheck = index+(len(chunk)-index) # Inspect all the remaining minutes of data

                    if sum(chunk.iloc[index:pointsToCheck, 4]) == 0: # Were any counts detected during the next x minutes?
                        NonWearData.append(NonWear) # If no counts were recorded then allocate this epoch as non-wear

                    else: # If there are counts during the next x minutes
                        NonWearDataToAdd = IsActivityMaintained(index, chunk, timestep, amplitudeSensitivity, timeSensitivity) # See how long until the next count
                        NonWearData.append(NonWearDataToAdd)

            else: # If counts were recorded for the epoch awaiting classification
                if previousEpochClassification == 1: # was the previous epoch classified as wear? If yes...
                    NonWearData.append(Wear) # if the previous epoch was wear and there is still some activity, classify the current epoch as wear
                elif row['VM'] <= amplitudeSensitivity: # If the previous epoch was classified as non-wear, and the current epoch has less than 'amplitudeSensitivity' counts, it's unlikely that they just put the monitor/prosthesis on
                    NonWearData.append(NonWear)

                else: # if the previous epoch was classified as non-wear but this epoch shows to have more to have more than 'amplitudeSensivity' counts recorded
                    NonWearDataToAdd = IsActivityMaintained(index, chunk, timestep, amplitudeSensitivity, timeSensitivity) # See how long until the next count
                    NonWearData.append(NonWearDataToAdd)
            
            previousEpochClassification = NonWearData[-1] # reset the categorisation of the previous epoch before re-entering the loop for the next epoch
            NonWearDataToAdd = 999

        if len(NonWearData) != len(chunk):
            print('We have a problem! The non wear length does not match the size of the data.')
            breakpoint()

        NonWearDataChecked = check_short_classifications(NonWearData, timestep, int(timeSensitivity/3)) # 10

        #NonWearData = [element * 99 for element in NonWearData]
        NonWearDataChecked = [element * 99 for element in NonWearDataChecked]
        chunk['NonWear'] = NonWearDataChecked
        #chunk['NonWearChecked'] = NonWearDataChecked
        #chunk['Time'] = [element / 60 for element in chunk['Time']]
        self.posture_stack = chunk
        self.start_time = meta.start_datetime
        self.end_time = meta.stop_datetime
        self.total_time = total_time