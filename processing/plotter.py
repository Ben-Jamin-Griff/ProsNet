import numpy as np
import matplotlib.pyplot as plt

class Plotter():
    def __init__(self):
        pass

    def plot_signal(self, array, title):
        plt.plot(array, c = 'hotpink')
        plt.title(title)
        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")

    def plot_postures(self, type):
        if type == 'postures':
            my_data = self.postures
            title = 'Postures'
        elif type == 'predictions':
            my_data = self.predictions
            title = 'Predictions'

        x = self.posture_stack_start_time
        seconds = ((x.hour * 60 + x.minute) * 60) + x.second

        START_TIME = seconds/(60*60*24)
        
        days = (len(my_data)*5)/60/60/24 # 5 seconds because the moving window slides over the data... Converting it from epochs to seconds
        tod_offset = (2*np.pi) * START_TIME
        # creating an array containing the radian values
        rads = np.arange(0 + tod_offset, ((2*days) * np.pi) + tod_offset, ((2*days) * np.pi)/len(my_data))
        MAKERSIZE = 10
        fig, ax = plt.subplots(figsize=(8,8))
        ax = plt.subplot(111, projection='polar')
        for count, value in enumerate(rads):
            try:
                r = value + 20
                clr = my_data[count]
                if clr == 0:
                    plt.polar(value, r, 'y.', ms=MAKERSIZE)
                elif clr == 1:
                    plt.polar(value, r, 'g.', ms=MAKERSIZE)
                elif clr == 2:
                    plt.polar(value, r, 'r.', ms=MAKERSIZE)
                elif clr == 3:
                    plt.polar(value, r, 'b.', ms=MAKERSIZE)
            except:
                pass
        ax.set_xticklabels(['Midnight', '3am', '6am', '9am', 'Midday', '3pm', '6pm', '9pm'])
        ax.set_theta_zero_location("N")  # theta=0 at the top
        ax.set_theta_direction(-1)  # theta increasing clockwise
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        ax.set_title(title, va='bottom')
        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")
