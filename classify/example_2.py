from activity_monitor.activpal import Activpal
from stack.epoch_stack import EpochStack
from dataset.engineering_set import EngineeringSet
from model.deep_model import DeepModel

"""
Make deep predictions

This script provides an example of how to load in data from an activPAL, create an engineering set from the raw acceleration data, make predictions on the set using a pretrained model and saves the predictions to a CSV.
"""

activPal = Activpal()
activPal.load_raw_data()

engineering_set = EngineeringSet()
engineering_set.get_data(activPal)
engineering_set.create_set()
engineering_set.show_set()

model = DeepModel()
model.load_model()
model.get_data(engineering_set)
model.show_set()
model.reshape_set([5,1,59,3])
model.process_epochs()
model.make_prediction()
model.show_predictions()
model.save_predictions('example_2')