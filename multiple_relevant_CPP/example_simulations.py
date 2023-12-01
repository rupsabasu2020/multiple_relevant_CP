import numpy as np
import basefuncs_multCPP as bfmcpp
import multiprocessing as mp
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


num_curves = 600
time_points = np.linspace(0, 2*np.pi, num=100)  # Create 100 time points for each curve
true_cp = [150, 400]
#binseg_thresh = 4
rel_size = [15]
alpha = 0.1    # parameter, level of the test
bootrepeats = 100

def change_point_Test(run, num_curves, time_points, change_locs,  delta_size, alpha, boot_repeats, block_length= None, bin_seg_thresh= None, c_value= 0.1):
    """
    function to run simulations for the methodology 
    --
    --
    --
    --

    """

    #----------------generate data----------------------------------#
    noisy_curves = bfmcpp.gen_data(num_curves, change_locs, time_points)

    #------------ binary segmentation ------------------------------#
    
    if bin_seg_thresh is None:
        bin_seg_thresh = bfmcpp.calculate_median_l2_norm_squared(noisy_curves)

    if block_length is None:
        block_length = bfmcpp.w_funcs_q(h_val = 9, timeseries=noisy_curves, weight_function='qs')

    binseg_cp  = bfmcpp.binary_seg_init(noisy_curves, factor=2, min_distance=20)  # not using at all

    local_decisions = []
    if len(binseg_cp) <= 5:
        #------------- change point test -------------------------------#
        global_decisions, local_decisions = bfmcpp.reject(noisy_curves, binseg_cp, Delta= delta_size, level_alpha=alpha, repeats=boot_repeats, const_c = c_value, block_length_boot= block_length)
        local_decisions = local_decisions.tolist()
    return  binseg_cp, local_decisions#, global_decisions # uncomment to get global decisions



#binsegChanges, decisions, global_decision = change_point_Test(1, num_curves, time_points, true_cp, rel_size, alpha, boot_repeats=100)
#print(decisions, global_decision)


#---------------------------------------------------------------------------------------------------#
#-                       Run above multiple simulations of above example                           -#
#-------------------------------------- using multiple processing ----------------------------------#
#---------------------------------------------------------------------------------------------------#

num_repeats = 1  # adjust accordingly for  larger number of simulations
if __name__ == '__main__':
    num_cores =  mp.cpu_count()
    with mp.Pool(num_cores) as p:
       arguments = [(i, num_curves, time_points, true_cp, rel_size, alpha, bootrepeats) for i in range(num_repeats)]
       results = p.starmap(change_point_Test, arguments)
    print(results)
    relevant_cp = [c for cl,tl in results for c,t in zip (cl,tl) if t == 1]   # results --> tuple .. each tuple--> a tuple of changes and decisions --> each change is coupled with corresponding decision.. 
    relevantChanges = [element/num_curves for element in relevant_cp]
    #print(len(relevantChanges))
    
    #plt.hist(relevantChanges, bins=np.linspace(0, 1, 60), alpha=0.95, color = 'green', weights= [1/len(relevantChanges) for _ in relevantChanges])
    #plt.vlines(x=[loc/num_curves for loc in true_cp], ymin=0, ymax=1, linestyles='dashed', colors='r', label="True change locations")
    #plt.xlim([0, 1])
    #plt.ylabel('Relative frequency')
    #plt.show()
