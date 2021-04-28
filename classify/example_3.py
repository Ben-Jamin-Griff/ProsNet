from activity_monitor.activpal import Activpal
from stack.epoch_stack import EpochStack
from dataset.engineering_set import EngineeringSet
from model.deep_model import DeepModel

"""
This script provides an example of how to load in data from an activPAL, create a posture stack using the thigh events, create an engineering set from the raw acceleration data and corresponding posture classification codes, make predictions using a pretrained model, compares the predictions to the posture stack and saves the predictions to a CSV. 
"""

activPal = Activpal()
activPal.load_raw_data()

engineering_set = EngineeringSet()
engineering_set.get_data(activPal)
engineering_set.create_set()
engineering_set.show_set()

ml_model = DeepModel()
ml_model.load_model()
ml_model.get_data(engineering_set)
ml_model.show_set()
ml_model.reshape_set([5,1,59,3])
ml_model.process_epochs()
ml_model.make_prediction()
ml_model.show_predictions()