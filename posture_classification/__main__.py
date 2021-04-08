from activpal import Activpal
from epoch_stack import EpochStack
from engineering_set import EngineeringSet

if __name__ == '__main__':
    activPal = Activpal()
    activPal.load_raw_data()
    activPal.load_event_data()

    posture_stack = EpochStack()
    posture_stack.get_data(activPal)
    posture_stack.create_stack(stack_type = 'pure', subset_of_data = 500) # stack_type =  'mixed' & 'pure', subset_of_data = int of event dataset length or None
    posture_stack.show_stack()

    engineering_set = EngineeringSet()
    engineering_set.get_data(activPal)
    engineering_set.get_posture_stack(posture_stack)
    engineering_set.create_set()