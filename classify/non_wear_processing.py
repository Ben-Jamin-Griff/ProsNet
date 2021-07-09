from activity_monitor.activpal import Activpal
from stack.epoch_stack import EpochStack
from stack.non_wear_stack import NonWearStack

"""
Analyse Non-wear Data

This script provides..
"""

activPal = Activpal()
activPal.load_raw_data()
#activPal.load_event_data()

non_wear_stack = NonWearStack()
non_wear_stack.get_data(activPal)
non_wear_stack.create_stack()
non_wear_stack.show_stack()

posture_stack = EpochStack()
posture_stack.get_data(activPal)
posture_stack.create_stack(stack_type = 'mixed', epochSize = 1) # creating a 1 second epoch stack to compare with non-wear
posture_stack.show_stack()

breakpoint