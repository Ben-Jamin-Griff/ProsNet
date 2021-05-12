from uos_activpal.io.raw import load_activpal_data
from activity_monitor.activpal import Activpal
from dataset.engineering_set import EngineeringSet

def convert_signals(signals):
    x = np.trim_zeros(signals[:,0], 'b')
    y = np.trim_zeros(signals[:,1], 'b')
    z = np.trim_zeros(signals[:,2], 'b')
    new_signals = np.concatenate([[x], [y], [z]])
    new_signals = np.transpose(new_signals)
    return new_signals

meta, signals = load_activpal_data('C:/Users/ANS292/OneDrive - University of Salford/Code Projects/apc/data/icl-data-2/shank-shank-AP472387 202a 7Dec20 1-19pm for 2h 57m.datx')

print(signals.shape)
print(meta)

activPal = Activpal()
activPal.load_raw_data('C:/Users/ANS292/OneDrive - University of Salford/Code Projects/apc/data/icl-data-2/shank-shank-AP472387 202a 7Dec20 1-19pm for 2h 57m-CREA-PA08110254-AccelDataUncompressed.csv')
engineering_set = EngineeringSet()
engineering_set.get_data(activPal)
engineering_set.create_set()

import numpy as np
import matplotlib.pyplot as plt

new_data = np.reshape(engineering_set.dataset[0], (209155,3))
new_signals = convert_signals(signals)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(new_data)
ax2.plot(new_signals)
plt.show()