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
    "./data/af-data/AF_Shin-AP971770 202a 28May21 3-28pm for 16d 20h 54m.datx",
    "./data/aw-data/AW_Shin-AP971754 202a 28May21 3-57pm for 10d 46m.datx",
    "./data/bg1-data/shank-AP472387 202a 5Feb21 9-50pm for 3d 10h 8m.datx",
    "./data/bg2-data/shank-AP472387 202a 19Sep20 1-00pm for 2d 15m.datx",
    "./data/dh-data/DH_shank-AP872481 202a 7Dec20 10-45am for 4d 7h 5m.datx",
    "./data/dz-data/DZ_Shin-AP971719 202a 28May21 3-37pm for 10d 21h 3m.datx",
    "./data/jd-data/JD_Shin-AP971756 202a 22Jun21 3-25pm for 20d.datx",
    "./data/jm-data/JM_Shank-AP472387 202a 28May21 10-50am for 10d.datx",
    "./data/js-data/shank-AP472387 202a 29Jan21 3-15pm for 7d.datx",
    "./data/ls-data/LS_Shin-AP971750 202a 28May21 3-46pm for 16d 20h 41m.datx",
    "./data/mw-data/MW_Shin-AP971765 202a 29Jun21 11-34am for 20d.datx",
    "./data/nb-data/NB_Shin-AP971733 202a 29Jun21 11-07am for 15d 27m.datx",
    "./data/pk-data/PK_Shin-AP971757 202a 22Jun21 3-35pm for 13d 16h 52m.datx",
    "./data/sg-data/SG_Shin-AP971752 202a 2Jun21 12-12pm for 12d 7m.datx",
    "./data/sr-data/SR_Shin-AP971736 202a 25Jun21 11-43am for 18d 23h 57m.datx",
]

event_data_paths = [
    "./data/af-data/AF_Thigh-AP971728 202a 28May21 3-24pm for 16d 21h-CREA-PA08110254-Events.csv",
    "./data/aw-data/AW_Thigh-AP971753 202a 28May21 3-53pm for 10d 52m-CREA-PA08110254-Events.csv",
    "./data/bg1-data/thigh-AP872479 202a 5Feb21 9-50pm for 3d 22h 10m-CREA-PA08110254-Events.csv",
    "./data/bg2-data/thigh-AP870085 202a 19Sep20 1-00pm for 2d 17m-CREA-PA08110254-Events.csv",
    "./data/dh-data/thigh-AP870085 202a 7Dec20 10-47am for 4d 7h 7m-CREA-PA08110254-Events.csv",
    "./data/dz-data/DZ_Thigh-AP971731 202a 28May21 3-34pm for 10d 21h 2m-CREA-PA08110254-Events.csv",
    "./data/jd-data/JD_Thigh-AP971771 202a 22Jun21 3-22pm for 20d-CREA-PA08110254-Events.csv",
    "./data/jm-data/JM_Thigh-AP872479 202a 28May21 10-47am for 10d-CREA-PA08110254-Events.csv",
    "./data/js-data/thigh-AP872479 202a 29Jan21 3-15pm for 7d-CREA-PA08110254-Events.csv",
    "./data/ls-data/LS_Thigh-AP971772 202a 28May21 3-42pm for 16d 20h 49m-CREA-PA08110254-Events.csv",
    "./data/mw-data/MW_Thigh-AP971720 202a 29Jun21 11-31am for 20d-CREA-PA08110254-Events.csv",
    "./data/nb-data/NB_Thigh-AP971748 202a 29Jun21 11-05am for 15d 31m-CREA-PA08110254-Events.csv",
    "./data/pk-data/PK_Thigh-AP971722 202a 22Jun21 3-32pm for 5d 14h 35m-CREA-PA08110254-Events.csv",
    "./data/sg-data/SG_Thigh-AP971764 202a 2Jun21 12-09pm for 8d 6h 47m-CREA-PA08110254-Events.csv",
    "./data/sr-data/SR_Thigh-AP971766 202a 25Jun21 11-40am for 18d 23h 58m-CREA-PA08110254-Events.csv",
]

non_wear_data_paths = [
    "./data/af-data/AF_non-wear.csv",
    "./data/aw-data/AW_non-wear.csv",
    None,
    None,
    None,
    "./data/dz-data/DZ_non-wear.csv",
    "./data/jd-data/JD_non-wear.csv",
    "./data/jm-data/JM_non-wear.csv",
    None,
    "./data/ls-data/LS_non-wear.csv",
    "./data/mw-data/MW_non-wear.csv",
    "./data/nb-data/NB_non-wear.csv",
    "./data/pk-data/PK_non-wear.csv",
    "./data/sg-data/SG_non-wear.csv",
    "./data/sr-data/SR_non-wear.csv",
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
        loop_posture_stack.remove_epochs(filename = non_wear_data_paths[k])
        loop_posture_stack.show_stack()

        loop_feature_set = FeatureSet()
        loop_feature_set.get_data(loop_activPal)
        loop_feature_set.get_posture_stack(loop_posture_stack)
        loop_feature_set.create_set()
        loop_feature_set.show_set()

        # Combine datasets
        feature_set.combine_sets(loop_feature_set)

    # Saving feature set
    feature_set.save_set('set_size_' + str(epoch_sizes[i]), 'feature')
    
    # Old code for creating a model
    #model = ShallowModel()
    #model.get_data(feature_set)
    #model.get_postures(feature_set)
    #model.show_set()
    #model.reassign_classes()
    #model.remove_classes(4)
    #model.remove_classes(5)
    #object_name = 'knn_epoch_window_' + str(epoch_sizes[i])
    #model.create_model('knn', save_model_results = object_name)

    #https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html