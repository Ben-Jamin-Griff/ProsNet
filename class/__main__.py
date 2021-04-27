from activpal import Activpal
from epoch_stack import EpochStack
from engineering_set import EngineeringSet
from deep_model import DeepModel

if __name__ == '__main__':

    activPal = Activpal()
    activPal.load_raw_data()
    #activPal.load_event_data()

    # Testing ML model
    ml_model = DeepModel()
    ml_model.load_model()
    ml_model.get_data(activPal)
    ml_model.show_set()
    ml_model.reshape_set([5,1,59,3])
    ml_model.process_epochs()
    ml_model.make_prediction()
    ml_model.show_predictions()
    breakpoint()

    # ---

    posture_stack = EpochStack()
    posture_stack.get_data(activPal)
    posture_stack.create_stack(stack_type = 'mixed', subset_of_data = None) # stack_type =  'mixed' & 'pure', subset_of_data = int of event dataset length or None
    posture_stack.show_stack()

    engineering_set = EngineeringSet()
    engineering_set.get_data(activPal)
    engineering_set.get_posture_stack(posture_stack)
    engineering_set.create_set()
    engineering_set.show_set()
    engineering_set.save_set('test')
    #engineering_set.load_set('test')

    ml_model = DeepModel()
    ml_model.load_model('my_model.ht5')
    ml_model.get_data()
    ml_model.predict_postures()
    ml_model.save_results
    

