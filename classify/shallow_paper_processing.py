from activity_monitor.activpal import Activpal
from stack.epoch_stack import EpochStack
from dataset.feature_set import FeatureSet
from model.shallow_model import ShallowModel

"""
Create a shallow model

This script provides an example of how to load in data from an activPAL, create a posture stack using the thigh events, create a feature set from the raw acceleration data and corresponding posture classification codes, create a model using scikit learn (KNN) and save the model to a pickle object.
"""

# Load in participant 1's data
activPal = Activpal()
activPal.load_raw_data()
activPal.load_event_data()

posture_stack = EpochStack()
posture_stack.get_data(activPal)
posture_stack.create_stack(stack_type = 'pure', subset_of_data = 100, epochSize=15)
posture_stack.show_stack()

feature_set = FeatureSet()
feature_set.get_data(activPal)
feature_set.get_posture_stack(posture_stack)
feature_set.create_set()
feature_set.show_set()

# Load in participant 2's data
activPal_2 = Activpal()
activPal_2.load_raw_data()
activPal_2.load_event_data()

posture_stack_2 = EpochStack()
posture_stack_2.get_data(activPal_2)
posture_stack_2.create_stack(stack_type = 'pure', subset_of_data = 100, epochSize=15)
posture_stack_2.show_stack()

feature_set_2 = FeatureSet()
feature_set_2.get_data(activPal_2)
feature_set_2.get_posture_stack(posture_stack_2)
feature_set_2.create_set()
feature_set_2.show_set()

# Load in participant 3's data
activPal_3 = Activpal()
activPal_3.load_raw_data()
activPal_3.load_event_data()

posture_stack_3 = EpochStack()
posture_stack_3.get_data(activPal_3)
posture_stack_3.create_stack(stack_type = 'pure', subset_of_data = 100, epochSize=15)
posture_stack_3.show_stack()

feature_set_3 = FeatureSet()
feature_set_3.get_data(activPal_3)
feature_set_3.get_posture_stack(posture_stack_3)
feature_set_3.create_set()
feature_set_3.show_set()

# Load in participant 4's data
activPal_4 = Activpal()
activPal_4.load_raw_data()
activPal_4.load_event_data()

posture_stack_4 = EpochStack()
posture_stack_4.get_data(activPal_4)
posture_stack_4.create_stack(stack_type = 'pure', subset_of_data = 100, epochSize=15)
posture_stack_4.show_stack()

feature_set_4 = FeatureSet()
feature_set_4.get_data(activPal_4)
feature_set_4.get_posture_stack(posture_stack_4)
feature_set_4.create_set()
feature_set_4.show_set()

# Combine datasets
feature_set.combine_sets(feature_set_2)
feature_set.combine_sets(feature_set_3)
feature_set.combine_sets(feature_set_4)

# Create a model
model = ShallowModel()
model.get_data(feature_set)
model.get_postures(feature_set)
model.show_set()
model.reassign_classes()
model.remove_classes(4)
model.create_model('knn')
model.save_object('knn')