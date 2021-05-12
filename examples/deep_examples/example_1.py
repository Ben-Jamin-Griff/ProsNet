from activity_monitor.activpal import Activpal
from stack.epoch_stack import EpochStack
from dataset.engineering_set import EngineeringSet

"""
Creating a deep training set

This script provides an example of how to load in data from an activPAL, create a posture stack using the thigh events and create an engineering set from the raw acceleration data and corresponding posture stack codes. This data is saved as a numpy file that can be loaded into one of the notebooks for developing a model. 
"""

activPal = Activpal()
activPal.load_raw_data()
activPal.load_event_data()

posture_stack = EpochStack()
posture_stack.get_data(activPal)
posture_stack.create_stack(stack_type = 'pure')
posture_stack.show_stack()

engineering_set = EngineeringSet()
engineering_set.get_data(activPal)
engineering_set.get_posture_stack(posture_stack)
engineering_set.create_set()
engineering_set.show_set()
engineering_set.save_set('example_1', 'engineering')