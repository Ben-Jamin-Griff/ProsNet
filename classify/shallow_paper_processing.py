from activity_monitor.activpal import Activpal
from stack.epoch_stack import EpochStack
from dataset.feature_set import FeatureSet
from model.shallow_model import ShallowModel

"""
Create a shallow model

This script provides an example of how to load in data from an activPAL, create a posture stack using the thigh events, create a feature set from the raw acceleration data and corresponding posture classification codes, create a model using scikit learn (KNN) and save the model to a pickle object.
"""

epoch_sizes = [5, 15, 30, 60, 120, 180]

raw_data_paths = [
    "C:/Users/ANS292/OneDrive - University of Salford/Code Projects/apc/data/home-data-collection-5/shank-AP472387 202a 19Sep20 1-00pm for 2d 15m.datx",
    "C:/Users/ANS292/OneDrive - University of Salford/Code Projects/apc/data/home-data-collection-6/shank-AP472387 202a 29Jan21 3-15pm for 7d.datx",
    "C:/Users/ANS292/OneDrive - University of Salford/Code Projects/apc/data/home-data-collection-7/shank-AP472387 202a 5Feb21 9-50pm for 3d 10h 8m.datx",
    "C:/Users/ANS292/OneDrive - University of Salford/Code Projects/apc/data/icl-data-1/DH_shank-AP872481 202a 7Dec20 10-45am for 4d 7h 5m.datx"
]

event_data_paths = [
    'C:/Users\ANS292/OneDrive - University of Salford/Code Projects/apc/data/home-data-collection-5/thigh-AP870085 202a 19Sep20 1-00pm for 2d 17m-CREA-PA08110254-Events.csv',
    'C:/Users\ANS292/OneDrive - University of Salford/Code Projects/apc/data/home-data-collection-6/thigh-AP872479 202a 29Jan21 3-15pm for 7d-CREA-PA08110254-Events.csv',
    'C:/Users\ANS292/OneDrive - University of Salford/Code Projects/apc/data/home-data-collection-7/thigh-AP872479 202a 5Feb21 9-50pm for 3d 22h 10m-CREA-PA08110254-Events.csv',
    'C:/Users\ANS292/OneDrive - University of Salford/Code Projects/apc/data/icl-data-1/thigh-AP870085 202a 7Dec20 10-47am for 4d 7h 7m-CREA-PA08110254-Events.csv'
]

for i in range(len(epoch_sizes)):

    print(f'Creating a model with {epoch_sizes[i]} second epochs')
    print('---------------------')

    feature_set = FeatureSet()

    for k in range(len(raw_data_paths)):
        # Load in each participant's data
        loop_activPal = Activpal()
        loop_activPal.load_raw_data(raw_data_paths[k])
        loop_activPal.load_event_data(event_data_paths[k])

        loop_posture_stack = EpochStack()
        loop_posture_stack.get_data(loop_activPal)
        loop_posture_stack.create_stack(stack_type = 'pure', subset_of_data = None, epochSize=epoch_sizes[i])
        ###posture_stack.remove_epochs(filename = 'C:/Users/ANS292/OneDrive - University of Salford/Code Projects/apc/data/dz-data/DZ_non-wear_test.csv')
        loop_posture_stack.show_stack()

        loop_feature_set = FeatureSet()
        loop_feature_set.get_data(loop_activPal)
        loop_feature_set.get_posture_stack(loop_posture_stack)
        loop_feature_set.create_set()
        loop_feature_set.show_set()

        # Combine datasets
        feature_set.combine_sets(loop_feature_set)

    # Create a model
    model = ShallowModel()
    model.get_data(feature_set)
    model.get_postures(feature_set)
    model.show_set()
    model.reassign_classes()
    model.remove_classes(4)
    model.remove_classes(5)
    object_name = 'knn_epoch_window_' + str(epoch_sizes[i])
    model.create_model('knn', save_model_results = object_name)

    #https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html