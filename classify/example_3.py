from activity_monitor.activpal import Activpal
from stack.epoch_stack import EpochStack
from dataset.engineering_set import EngineeringSet
from model.deep_model import DeepModel

"""
Validate deep predictions

This script provides an example of how to load in data from an activPAL, create a posture stack using the thigh events, create an engineering set from the raw acceleration data and corresponding posture classification codes, make predictions using a pretrained model, compares the predictions to the posture stack and saves the predictions to a CSV. 
"""

activPal = Activpal()
activPal.load_raw_data()
activPal.load_event_data()

posture_stack = EpochStack()
posture_stack.get_data(activPal)
posture_stack.create_stack(stack_type = 'mixed')
posture_stack.show_stack()

engineering_set = EngineeringSet()
engineering_set.get_data(activPal)
engineering_set.get_posture_stack(posture_stack)
engineering_set.create_set()
engineering_set.show_set()
engineering_set.save_set('example_3')
#engineering_set.load_set('example_3')

model = DeepModel()
model.load_model()
model.get_data(engineering_set)
model.get_postures(engineering_set)
model.show_set()
model.reshape_set([5,1,59,3])
model.process_epochs()
model.make_prediction()
model.show_predictions()
model.show_model_results()