from activity_monitor_abc import ABCActivityMonitor
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import time
        
class Activpal(ABCActivityMonitor):
    
    def __init__(self, deviceType='activPAL', raw_data=None):
        self.deviceType = deviceType
        self.raw_data = raw_data

    def load_raw_data(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        self.raw_data = file_path
#        self.raw_data = pd.read_csv(file_path, skiprows=1, delimiter =';' )
        return print(f"Loaded file: {file_path}")

    def load_event_data(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        self.event_data = file_path
        return print(f"Loaded file: {file_path}")
    
    def save(self, output, data):
        return output.write(data)