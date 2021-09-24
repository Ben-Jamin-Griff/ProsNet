from activity_monitor.activpal import Activpal
from stack.epoch_stack import EpochStack
from dataset.engineering_set import EngineeringSet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

"""
Running shallow paper plots

This script creates the plots for the shallow paper...
"""

def plot_ensemble(posture_code, posture_label, final_count = 3):
    counter = 0
    data_ensemble = np.empty([295, 3])
    for data, posture in zip(engineering_set.dataset[0], engineering_set.dataset[1]):
        if posture == posture_code:
            counter += 1
            data_ensemble = data_ensemble + data
            if counter == final_count:
                data_ensemble = data_ensemble / final_count
                data_ensemble = ((data_ensemble/253)-0.5)*4
                plt.plot(np.arange(0, 15, 15/295), data_ensemble)
                plt.ylabel('Acceleration gs')
                plt.xlabel('Seconds')
                plt.savefig(posture_label + '_ensemble.png')
                break
    plt.close()

def plot_postures(posture_code, posture_label, final_count = 3):
    counter = 0
    for data, posture in zip(engineering_set.dataset[0], engineering_set.dataset[1]):
        if posture == posture_code:
            counter += 1
            plt.plot(np.arange(0, 15, 15/295), data)
            if counter == final_count:
                plt.ylabel('Acceleration gs')
                plt.xlabel('Seconds')
                plt.savefig(posture_label + '_postures.png')
                break
    plt.close()

'''
epoch_sizes = [15]
raw_data_paths = ["./apc-data/af-data/AF_Shin-AP971770 202a 28May21 3-28pm for 16d 20h 54m-CREA-PA08110254-AccelDataUncompressed.csv"]
event_data_paths = ["./apc-data/af-data/AF_Thigh-AP971728 202a 28May21 3-24pm for 16d 21h-CREA-PA08110254-Events.csv"]
# Load in each participant's data
activPal = Activpal()
activPal.load_raw_data(raw_data_paths[0])
activPal.load_event_data(event_data_paths[0])
posture_stack = EpochStack()
posture_stack.get_data(activPal)
posture_stack.create_stack(stack_type = 'pure', subset_of_data = 100)
engineering_set = EngineeringSet()
engineering_set.get_data(activPal)
engineering_set.get_posture_stack(posture_stack)
engineering_set.create_set()

plot_ensemble(0, 'Sitting')
plot_ensemble(1, 'Standing')
plot_ensemble(2, 'Stepping')
plot_ensemble(3, 'Lying')

plot_postures(0, 'Sitting')
plot_postures(1, 'Standing')
plot_postures(2, 'Stepping')
plot_postures(3, 'Lying')
'''

# Orginal (nice looking plots)
epoch_sizes = [15]
raw_data_paths = ["./apc-data/bg2-data/shank-AP472387 202a 19Sep20 1-00pm for 2d 15m-CREA-PA08110254-AccelDataUncompressed.csv"]
event_data_paths = ["./apc-data/bg2-data/thigh-AP870085 202a 19Sep20 1-00pm for 2d 17m-CREA-PA08110254-Events.csv"]
# Load in each participant's data
activPal = Activpal()
activPal.load_raw_data(raw_data_paths[0])
activPal.load_event_data(event_data_paths[0])
posture_stack = EpochStack()
posture_stack.get_data(activPal)
posture_stack.create_stack(stack_type = 'pure', subset_of_data = 500)
engineering_set = EngineeringSet()
engineering_set.get_data(activPal)
engineering_set.get_posture_stack(posture_stack)
engineering_set.create_set()

plot_ensemble(0, 'Nice_Sitting')
plot_ensemble(1, 'Nice_Standing')
plot_ensemble(2, 'Nice_Stepping')
plot_ensemble(3, 'Nice_Lying')

plot_postures(0, 'Nice_Sitting')
plot_postures(1, 'Nice_Standing')
plot_postures(2, 'Nice_Stepping')
plot_postures(3, 'Nice_Lying')

'''
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id="line-chart"),
])

@app.callback(
    Output("line-chart", "figure"))
def update_line_chart():
    fig = px.line(engineering_set.dataset[0], color=engineering_set.dataset[1])
    return fig

app.run_server(debug=True)
'''