from activpal import Activpal
from epoch_stack import EpochStack

if __name__ == '__main__':
    activPal = Activpal()
    activPal.load_raw_data()
    activPal.load_event_data()

    posture_stack = EpochStack()
    posture_stack.get_data(activPal)
    posture_stack.create_stack()