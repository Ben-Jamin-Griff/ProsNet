from ProsNet.activity_monitor.activity_monitor_abc import ABCActivityMonitor
import tkinter as tk
from tkinter import filedialog
        
class Activpal(ABCActivityMonitor):
    def __init__(self, deviceType='activPAL', raw_data=None, event_data=None):
        self.deviceType = deviceType
        self.raw_data = raw_data
        self.event_data = event_data

    def load_raw_data(self, filename = None):
        if filename is not None:
            file_path = filename
        else:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(title = "Load shank acceleration data")

        self.raw_data = file_path
        print(f"Loaded file: {file_path}")
        print('----------')

    def load_event_data(self, filename = None):
        if filename is not None:
            file_path = filename
        else:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(title = "Load thigh event data")
            
        self.event_data = file_path
        print(f"Loaded file: {file_path}")
        print('----------')