from activity_monitor.activpal import Activpal
from stack.epoch_stack import EpochStack
from stack.non_wear_stack import NonWearStack

"""
Analyse Non-wear Data

This script provides..
"""

raw_data_paths = [
    "./fab-data/pilot-data-14-02-2021/pal/Pilot4HAT-FABPilo2-AP476687 202a 6Sep20 6-00pm for 8d.datx",
    "./fab-data/pilot-data-14-02-2021/pal/FAB Pilot 2-FABPilot-AP473889 202a 18Aug20 6-00pm for 6d 5h 1m.datx",
    "./fab-data/pilot-data-14-02-2021/pal/FABPilotSB-FABPilo3-AP473889 202a 6Sep20 6-00pm for 1d 23h 38m.datx",
    "./fab-data/pilot-data-14-02-2021/pal/Pilot BTS FAB-FAB rod-AP476666 202a 28Mar20 10-40pm for 14h 3m.datx",
]

validation_data_paths = [
    "./fab-data/pilot-data-14-02-2021/converted-diaries/Pilot4HAT_validation.csv",
    "./fab-data/pilot-data-14-02-2021/converted-diaries/FAB Pilot 2-FABPilot-AP473889 202a_validation.csv",
    "./fab-data/pilot-data-14-02-2021/converted-diaries/FABPilotSB_validation.csv",
    "./fab-data/pilot-data-14-02-2021/converted-diaries/Pilot Notes.csv",
]

abs_error_mins = []
per_error = []
per_agreement = []

"""
Things that will be useful in the outcome report.
---
How many non wear events were detected / recorded
A plot to show the non-wear periods detected and reported

Resources
---
https://stackoverflow.com/questions/51864730/python-what-is-the-process-to-create-pdf-reports-with-charts-from-a-db

"""

for i in range(len(raw_data_paths)):

    activPal = Activpal()
    activPal.load_raw_data(raw_data_paths[i])

    non_wear_stack = NonWearStack()
    non_wear_stack.get_data(activPal)
    non_wear_stack.create_stack(subset_of_data = None) # percentage of data
    non_wear_stack.create_validation_stack(validation_data_paths[i])
    non_wear_stack.show_stack()

    # Comparing validation with detection
    total_samples = len(non_wear_stack.posture_stack)

    agree_non_wear = non_wear_stack.posture_stack.loc[(non_wear_stack.posture_stack.Validation == 0) & (non_wear_stack.posture_stack.NonWear == 0)]
    agree_wear = non_wear_stack.posture_stack.loc[(non_wear_stack.posture_stack.Validation == 99) & (non_wear_stack.posture_stack.NonWear == 99)]
    abs_agree = len(agree_non_wear) + len(agree_wear)
    abs_error = total_samples - abs_agree

    abs_error_mins.append(abs_error/60)
    per_error.append((abs_error/total_samples)*100)
    per_agreement.append((abs_agree/total_samples)*100)
    
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