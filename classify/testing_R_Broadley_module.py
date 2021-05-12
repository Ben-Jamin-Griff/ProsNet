from uos_activpal.io.raw import load_activpal_data
from activity_monitor.activpal import Activpal
from dataset.engineering_set import EngineeringSet

import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd

meta, signals = load_activpal_data('C:/Users/ANS292/OneDrive - University of Salford/Code Projects/apc/data/icl-data-2/shank-shank-AP472387 202a 7Dec20 1-19pm for 2h 57m.datx')
total_time = meta.stop_datetime - meta.start_datetime
total_samples = total_time.seconds * 20

arr = np.array([meta.start_datetime + datetime.timedelta(seconds=i*0.05) for i in range(total_samples)])
x = signals[:total_samples,0]
y = signals[:total_samples,1]
z = signals[:total_samples,2]
df = pd.DataFrame({'Time':arr, 'X':x, 'Y':y, 'Z':z})

#activPal = Activpal()
#activPal.load_raw_data('C:/Users/ANS292/OneDrive - University of Salford/Code Projects/apc/data/icl-data-2/shank-shank-AP472387 202a 7Dec20 1-19pm for 2h 57m-CREA-PA08110254-AccelDataUncompressed.csv')
#engineering_set = EngineeringSet()
#engineering_set.get_data(activPal)
#engineering_set.create_set()
#new_data = np.reshape(engineering_set.dataset[0], (209155,3))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(new_data)
ax2.plot(new_signals)
plt.show()