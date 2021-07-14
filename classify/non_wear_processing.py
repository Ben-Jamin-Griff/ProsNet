from activity_monitor.activpal import Activpal
from stack.epoch_stack import EpochStack
from stack.non_wear_stack import NonWearStack

"""
Analyse Non-wear Data

This script provides..
"""

raw_data_paths = [
    "./fab-data/pilot-data-14-02-2021/pal/FAB Pilot 2-FABPilot-AP473889 202a 18Aug20 6-00pm for 6d 5h 1m.datx",
    "./fab-data/pilot-data-14-02-2021/pal/FABPilotSB-FABPilo3-AP473889 202a 6Sep20 6-00pm for 1d 23h 38m.datx",
    "./fab-data/pilot-data-14-02-2021/pal/Pilot4HAT-FABPilo2-AP476687 202a 6Sep20 6-00pm for 8d.datx",
]

validation_data_paths = [
    "./fab-data/pilot-data-14-02-2021/converted-diaries/FAB Pilot 2-FABPilot-AP473889 202a_validation.csv",
    "./fab-data/pilot-data-14-02-2021/converted-diaries/FABPilotSB_validation.csv",
    "./fab-data/pilot-data-14-02-2021/converted-diaries/Pilot4HAT_validation.csv",

]

abs_error_mins = []
per_error = []
per_agreement = []

for i in range(len(raw_data_paths)):

    activPal = Activpal()
    activPal.load_raw_data(raw_data_paths[i])

    non_wear_stack = NonWearStack()
    non_wear_stack.get_data(activPal)
    non_wear_stack.create_stack(subset_of_data = None)
    non_wear_stack.create_validation_stack(validation_data_paths[i])
    non_wear_stack.show_stack()

    # Comparing validation with detection
    validation_non_wear = non_wear_stack.posture_stack.loc[non_wear_stack.posture_stack.Validation == 0]
    detection_prediction_breakdown = validation_non_wear.NonWear.value_counts()

    abs_error_mins.append(detection_prediction_breakdown[99]/60)
    per_error.append((detection_prediction_breakdown[99]/len(validation_non_wear))*100)
    per_agreement.append((detection_prediction_breakdown[0]/len(validation_non_wear))*100)

breakpoint()
print('Done!')

"""
This is for posture classification work
We currently have an issue with the below code because there is a 50% overlap of each window so it create double the number of windows needed (for a 1 second window I could just remove every other value...)
posture_stack = EpochStack()
posture_stack.get_data(activPal)
posture_stack.create_stack(stack_type = 'mixed', epochSize = 1) # creating a 1 second epoch stack to compare with non-wear
posture_stack.show_stack()
"""