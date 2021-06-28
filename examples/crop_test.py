from activity_monitor.activpal import Activpal
from stack.epoch_stack import EpochStack
from dataset.feature_set import FeatureSet
from model.deep_model import DeepModel

"""
Testing the new cropping non-wear function

"""

activPal = Activpal()
activPal.load_raw_data('C:/Users/ANS292/OneDrive - University of Salford/Code Projects/apc/data/dz-data/DZ_Shin-AP971719 202a 28May21 3-37pm for 10d 21h 3m.datx')
activPal.load_event_data('C:/Users/ANS292/OneDrive - University of Salford/Code Projects/apc/data/dz-data/DZ_Thigh-AP971731 202a 28May21 3-34pm for 10d 21h 2m-CREA-PA08110254-Events.csv')

posture_stack = EpochStack()
posture_stack.get_data(activPal)
posture_stack.create_stack(stack_type = 'pure', subset_of_data = 5000)
posture_stack.show_stack()
posture_stack.remove_epochs(filename = 'C:/Users/ANS292/OneDrive - University of Salford/Code Projects/apc/data/dz-data/DZ_non-wear_test.csv')
posture_stack.show_stack()