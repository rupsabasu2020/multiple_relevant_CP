import numpy as np
import fMA_1_base as fMA1
import basefuncs_multCPP as bfmcpp
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import decimal
plt.rcParams["axes.grid"] = True
np.set_printoptions(threshold=np.inf)  # <-- displays all elements of the array
decimal.getcontext().prec = 10

# ISSUES:::
# 
# (1) Fix method of choosing binseg_thresh
# (2) Fix code for running simulations
# (3) update for generated data from bfmcpp... !! needs changing of the Change_point_test for no values of block-length and binseg_thresh




#---------------------------------------------------------------------------------------------------#
#-                        Easy example for single run of simulation                                -#
#---------------------------------------------------------------------------------------------------#


num_curves = 300
time_points = np.linspace(0, 2*np.pi, num=100)  # Create 100 time points for each curve
true_cp = [100, 200]
binseg_thresh = 4
rel_size = [20]
alpha = 0.1    # parameter, level of the test
bootrepeats = 100

def change_point_Test(run, num_curves, time_points, change_locs,  delta_size, alpha, boot_repeats, block_length= None, bin_seg_thresh= None, c_value= 0.001):
    """
    function to run simulations for the methodology 
    --
    --
    --
    --

    """
    errors_eta = fMA1.fMA1(num_curves)
    #----------------generate data----------------------------------#
    sine_curves = np.sin(time_points) + np.cos(time_points) 
    noisy_curves = 9.126*sine_curves  + errors_eta #+ noise
    #---------------add change point to data -----------------------#
    for idx in change_locs:

        noisy_curves[idx:, 0:16] +=   [2, 5, 9, 10,  12, 15, 25, 25, 25, 22, 15, 12, 10, 9, 5, 2] #continuous constant to simulate change 

    #------------ binary segmentation ------------------------------#
    
    if bin_seg_thresh is None:
        bin_seg_thresh = bfmcpp.calculate_median_l2_norm_squared(noisy_curves)
    if block_length is None:
        block_length = bfmcpp.w_funcs_q(h_val = 9, timeseries=noisy_curves, weight_function='qs')

    binseg_cp  = bfmcpp.binary_seg_init(noisy_curves, bin_seg_thresh)  # not using at all
#    print(binseg_cp)
    #------------- change point test -------------------------------#
    global_decisions, local_decisions = bfmcpp.reject(noisy_curves, binseg_cp, Delta= delta_size, level_alpha=alpha, repeats=boot_repeats, const_c = c_value , binseg_thresh=bin_seg_thresh, l= block_length)
    return  binseg_cp, local_decisions# , global_decisions # uncomment to get global decisions



binsegChanges, _ = change_point_Test(1, num_curves, time_points, true_cp, rel_size, alpha, boot_repeats=100, block_length=9)



#---------------------------------------------------------------------------------------------------#
#-                       Run above multiple simulations of above example                           -#
#-------------------------------------- using multiple processing ----------------------------------#
#---------------------------------------------------------------------------------------------------#

num_repeats = 1  # adjust accordingly for  larger number of simulations
if __name__ == '__main__':
    num_cores =  mp.cpu_count()
    with mp.Pool(num_cores) as p:
       arguments = [(i, num_curves, time_points, true_cp,  binseg_thresh, rel_size, alpha, bootrepeats, 9) for i in range(num_repeats)]
       results = p.starmap(change_point_Test, arguments)

    relevant_cp = [c for cl,tl in results for c,t in zip (cl,tl) if t == 1]   # results --> tuple .. each tuple--> a tuple of changes and decisions --> each change is coupled with corresponding decision.. 
    relevantChanges = [element/num_curves for element in relevant_cp]
    print(len(relevantChanges))
    
    plt.hist(relevantChanges, bins=np.linspace(0, 1, 60), alpha=0.95, color = 'green', weights= [1/len(relevantChanges) for _ in relevantChanges])
    plt.vlines(x=[loc/num_curves for loc in true_cp], ymin=0, ymax=1, linestyles='dashed', colors='r', label="True change locations")
    plt.xlim([0, 1])
    plt.ylabel('Relative frequency')
    plt.show()