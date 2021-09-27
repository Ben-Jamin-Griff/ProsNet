from ProsNet import Activpal
from ProsNet import EpochStack
from ProsNet import FeatureSet
from ProsNet import DeepModel

"""
Creating a shallow training set

This script provides an example of how to load in data from an activPAL, create a posture stack using the thigh events and create a feature set from the raw acceleration data and corresponding posture stack codes. This data is saved as a numpy file that can be loaded into one of the notebooks for developing a shallow model. 
"""

activPal = Activpal()
activPal.load_raw_data()
activPal.load_event_data()

posture_stack = EpochStack()
posture_stack.get_data(activPal)
posture_stack.create_stack(stack_type = 'pure')
posture_stack.show_stack()

feature_set = FeatureSet()
feature_set.get_data(activPal)
feature_set.get_posture_stack(posture_stack)
feature_set.create_set()
feature_set.show_set()
feature_set.save_set('example_1', 'feature')