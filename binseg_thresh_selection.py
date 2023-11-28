import numpy as np
import basefuncs_multCPP as bfmcpp
import matplotlib.pyplot as plt

num_curves = 400
change_locs = [100, 200, 300]
time_points = np.linspace(0, 2*np.pi, 100) 
run = 1
delta_vals = [22]
alpha = 0.1
bootrepeats = 100
block_length_l = 9

#---------------------------------------------------------------------------------------------------#
#-             Parameter selection for binary segmentation threshold                               -#
#---------------------------------------------------------------------------------------------------#

generate_data = bfmcpp.gen_data(num_curves, change_locs, time_points)
median_l2_norm_squared = bfmcpp.calculate_median_l2_norm_squared(generate_data)
thresh_Xi_n = median_l2_norm_squared*(3*np.log(num_curves))**(1/2)
print(thresh_Xi_n)


## visualise the data # uncomment following lines
#plt.plot(generate_data.reshape(-1))
#for cps in change_locs:
#    plt.axvline(x=cps*100, color='red', linestyle='--', linewidth= 2)
#plt.show()
