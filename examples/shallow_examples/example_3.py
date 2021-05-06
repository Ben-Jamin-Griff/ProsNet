from activity_monitor.activpal import Activpal
from stack.epoch_stack import EpochStack
from dataset.feature_set import FeatureSet
from model.shallow_model import ShallowModel

"""
Validate shallow predictions

This script provides an example of how to load in data from an activPAL, create a posture stack using the thigh events, create a feature set from the raw acceleration data and corresponding posture classification codes, make predictions using a pretrained model, compares the predictions to the posture stack and saves the predictions to a CSV.
"""

activPal = Activpal()
activPal.load_raw_data()
activPal.load_event_data()

posture_stack = EpochStack()
posture_stack.get_data(activPal)
posture_stack.create_stack(stack_type = 'mixed')
posture_stack.show_stack()

feature_set = FeatureSet()
feature_set.get_data(activPal)
feature_set.get_posture_stack(posture_stack)
feature_set.create_set()
feature_set.show_set()

model = ShallowModel()
model.load_object()
model.get_data(feature_set)
model.get_postures(feature_set)
model.show_set()
model.reassign_classes()
model.remove_classes(4)
model.make_prediction()
model.show_predictions()
model.show_model_results()
model.save_predictions('example_3')
model.plot_postures('predictions')
model.plot_postures('postures')