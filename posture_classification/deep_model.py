from model_abc import ABCModel

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
#import tensorflow as tf

class DeepModel(ABCModel):
    def __init__(self):
        self.model = None
        self.data_to_analyse = None

    def load_model(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        self.model = tf.keras.models.load_model(file_path)
        print(f"Loaded model: {file_path}")
        self.model.summary()
        print('----------')

    def get_data(self, data_holder):
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

        try:
            self.data_to_analyse = data_holder.dataset
        except:
            filename = data_holder.raw_data
            # Creat empty engineering set
            engineering_set = np.empty((0,295,3), int)
            CHUNKSIZE = 300000
            loaded_chunks = 0
            max_number_of_chunks = 100000
            printProgressBar (loaded_chunks, max_number_of_chunks, 'Engineering set progress:')
            for chunk in pd.read_csv(filename, chunksize=CHUNKSIZE):
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

                epochSize = 15
                numOfEpochs = (CHUNKSIZE / epochSize)
                for i in range(int(numOfEpochs)):
                    breakpoint()
                    current_epoch = chunk.iloc[(15*20)*i:(15*20)*(i+1):,:].copy()
                    current_epoch_accel_data = current_epoch[['X','Y','Z']].to_numpy()
                    engineering_set = np.append(engineering_set, [current_epoch_accel_data[:295,:]], axis=0)
                printProgressBar (loaded_chunks, max_number_of_chunks, 'Engineering set progress:')
        self.data_to_analyse = [engineering_set, 0]