from activity_monitor.activpal import Activpal
from stack.epoch_stack import EpochStack
from stack.non_wear_stack import NonWearStack

"""
Analyse Non-wear Data

This script provides...
"""

#raw_data_paths = [
#    "./fab-data/pilot-data-14-02-2021/pal/Pilot4HAT-FABPilo2-AP476687 202a 6Sep20 6-00pm for 8d.datx",
#    "./fab-data/pilot-data-14-02-2021/pal/FAB Pilot 2-FABPilot-AP473889 202a 18Aug20 6-00pm for 6d 5h 1m.datx",
#    "./fab-data/pilot-data-14-02-2021/pal/FABPilotSB-FABPilo3-AP473889 202a 6Sep20 6-00pm for 1d 23h 38m.datx",
#    "./fab-data/pilot-data-14-02-2021/pal/Pilot BTS FAB-FAB rod-AP476666 202a 28Mar20 10-40pm for 14h 3m.datx",
#]

#validation_data_paths = [
#    "./fab-data/pilot-data-14-02-2021/converted-diaries/Pilot4HAT_validation.csv",
#    "./fab-data/pilot-data-14-02-2021/converted-diaries/FAB Pilot 2-FABPilot-AP473889 202a_validation.csv",
#    "./fab-data/pilot-data-14-02-2021/converted-diaries/FABPilotSB_validation.csv",
#    "./fab-data/pilot-data-14-02-2021/converted-diaries/Pilot Notes.csv",
#]

raw_data_paths = [
    #"./fab-data/project-data-03-06-2021/pal/B AP971623 202a 9May21 5-00pm for 5d.datx",
    "./fab-data/project-data-03-06-2021/pal/BY 6-12-2020-6-12-202-AP473889 202a 6Dec20 9-54am for 5d 22h 59m.datx",
    "./fab-data/project-data-03-06-2021/pal/BY AP473889 202a 13Mar21 7-00pm for 1d 5h 27m.datx",
    #"./fab-data/project-data-03-06-2021/pal/F AP971623 202a 6Jun21 5-00pm for 5d.datx",
    "./fab-data/project-data-03-06-2021/pal/H AP473889 202a 21Feb21 6-00pm for 5d 16h.datx",
    #"./fab-data/project-data-03-06-2021/pal/K A-Baby 1-AP476687 202a 15Mar20 9-40am for 7d 46m.datx",
    #"./fab-data/project-data-03-06-2021/pal/K AP476666 202a 16Feb21 12-00am for 3d 12h.datx",
    #"./fab-data/project-data-03-06-2021/pal/K B-Baby 2-AP476666 202a 15Mar20 9-49am for 5d 22h 26m.datx",
    "./fab-data/project-data-03-06-2021/pal/KT 22-11-2020-AP476666 202a 22Nov20 6-00pm for 8d.datx",
    "./fab-data/project-data-03-06-2021/pal/SDK 3rd-AP473889 202a 15Feb21 12-00am for 6d 11h 55m.datx",
    #"./fab-data/project-data-03-06-2021/pal/SJ 29-11-FABpilo7-AP476687 202a 29Nov20 6-00pm for 8d.datx",
    "./fab-data/project-data-03-06-2021/pal/Tz AP476687 202a 14Feb21 12-00am for 5d 12h.datx",
]

validation_data_paths = [
    #"./fab-data/project-data-03-06-2021/converted-diaries/Be_validation.csv",
    "./fab-data/project-data-03-06-2021/converted-diaries/BY 6-12-20_validation.csv",
    "./fab-data/project-data-03-06-2021/converted-diaries/BY_validation.csv",
    "./fab-data/project-data-03-06-2021/converted-diaries/H_validation.csv",
    #"./fab-data/project-data-03-06-2021/converted-diaries/K_validation.csv",
    "./fab-data/project-data-03-06-2021/converted-diaries/KT_validation.csv",
    "./fab-data/project-data-03-06-2021/converted-diaries/SDK_validation.csv",
    #"./fab-data/project-data-03-06-2021/converted-diaries/TU_validation.csv",
    "./fab-data/project-data-03-06-2021/converted-diaries/Tz_validation.csv",
]

abs_error_mins = []
per_error = []
per_agreement = []

"""
Things that will be useful in the outcome report.
---
How many non wear events were detected / recorded
A plot to show the non-wear periods detected and reported

*** Am I getting rid of all the non-wear data in the actual dataset as well as in the validation set?
*** Could I be comparing pre-experiment data that should not count?
- Put in a non-wear validation event which starts a year ago and ends at the start of the session
- Remove any acceleration data before and after the data collection has started and ended

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

g = open("non_wear_testy.txt", "a")
print('Absolute error in minutes', file=g)
print(abs_error_mins, file=g)
print('-----------', file=g)
print('Percentage error', file=g)
print(per_error, file=g)
print('-----------', file=g)
print('Percentage agreement', file=g)
print(per_agreement, file=g)
print('-----------', file=g)
g.close() 

print('Done!')