from activity_monitor.activpal import Activpal
from stack.epoch_stack import EpochStack
from dataset.engineering_set import EngineeringSet
from model.deep_model import DeepModel

"""
Make shallow predictions

This script provides an example of how to load in data from an activPAL, create a feature set from the raw acceleration data, make predictions on the set using a pretrained model and saves the predictions to a CSV.
"""

activPal = Activpal()
activPal.load_raw_data()

feature_set = FeatureSet()
feature_set.get_data(activPal)
feature_set.create_set()
feature_set.show_set()

model = ShallowModel()
model.load_object()
model.get_data(feature_set)
model.show_set()
model.reassign_classes()
model.remove_classes(4)
model.make_prediction()
model.show_predictions()
model.save_predictions('example_2')
model.plot_postures('predictions')