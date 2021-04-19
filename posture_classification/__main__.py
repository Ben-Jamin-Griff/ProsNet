from activpal import Activpal
from epoch_stack import EpochStack
from engineering_set import EngineeringSet

if __name__ == '__main__':
    #
    activPal = Activpal()
    activPal.load_raw_data()
    activPal.load_event_data()

    posture_stack = EpochStack()
    posture_stack.get_data(activPal)
    posture_stack.create_stack(stack_type = 'mixed', subset_of_data = None) # stack_type =  'mixed' & 'pure', subset_of_data = int of event dataset length or None
    posture_stack.show_stack()

    engineering_set = EngineeringSet()
    engineering_set.get_data(activPal)
    engineering_set.get_posture_stack(posture_stack)
    engineering_set.create_set()
    engineering_set.show_set()
    engineering_set.save_set('3_mixed')

    posture_stack2 = EpochStack()
    posture_stack2.get_data(activPal)
    posture_stack2.create_stack(stack_type = 'pure', subset_of_data = None) # stack_type =  'mixed' & 'pure', subset_of_data = int of event dataset length or None
    posture_stack2.show_stack()

    engineering_set2 = EngineeringSet()
    engineering_set2.get_data(activPal)
    engineering_set2.get_posture_stack(posture_stack2)
    engineering_set2.create_set()
    engineering_set2.show_set()
    engineering_set2.save_set('3_pure')