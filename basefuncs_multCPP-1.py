import numpy as np
import random
from scipy.linalg import eigh


#---------------------------------------------------------------------------------------------------#
#-                           Initial binary segmentation                                           -#
#---------------------------------------------------------------------------------------------------#

def U_hat(s, fdata, l, r, N):
    """ from eq. 4.6 of the paper, generate sequential statistic 
    # ----- for multiple CP test, ensure that end of data is at next CP  ----- #
    """
    res = (1/(r-l)) * (np.sum(fdata[ l :(int(s * N)), :], axis=0) - ((int(s * N) - l) / (r - l)) * np.sum(fdata[l:r-1, :], axis=0))   # configure start to be prev. cp
    return res



def cusum(data_mat, begin, end, N, thresh_val):  #, thresh_val, cp_list,
    """ 
    #----generate statistic d_hat as in eq. 4.12 and estimate change-point----#
    #----not being used in CP testing  -----#
     """
    
    if begin >= end:
        return -1
    
    dataPoints = end - begin
    sValues = np.linspace(float(begin)/float(N), float(end - 1)/float(N), num=dataPoints)
    M_hat = []
    for i in range(dataPoints):  # scanning over possible values of s
        U = U_hat(sValues[i], data_mat, begin, end, N)
        #t_argmax = np.argmax(np.abs(U))  # sup over t     
        hat_M =  np.max(np.abs(U))  #np.abs(U[t_argmax])  
        M_hat.append(hat_M)
    #plt.plot(M_hat)
    #plt.vlines(x = np.argmax(M_hat), ymin=0, ymax=20)
    #plt.show()
    pos = -1
    if np.max(M_hat) > thresh_val:   # binseg thresh for detecting
        #print(np.max(M_hat), thresh_val)
        pos = np.argmax(M_hat)
    if pos > 0: 
        return pos + begin 
    else: 
        return pos #sorted(cp_list), M_hat[pos]






def binary_seg_init(data, thresh):
    """Binary segmentation for change-point detection."""
    N = len(data[:, 0])
    cp = [0, N]  # Initial interval
#    cp2 = [0, N]
    return binary_seg_rec(data, thresh, cp, 0, N, N)


def binary_seg_rec(data, thresh, cp, begin, end, N):
    cp_location = cusum(data, begin, end, N, thresh)
    if cp_location > 0:
        cp.append(cp_location)
        cp.sort()
        cp = binary_seg_rec(data, thresh, cp, begin, cp_location - 1, N)
        cp = binary_seg_rec(data, thresh, cp, cp_location + 1, end, N)
    return cp

#---------------------------------------------------------------------------------------------------#
#-                           Step 2: Relevant change points                                        -#
#---------------------------------------------------------------------------------------------------#

def mult_test_stat(list_of_cp, fdata, binseg_thresh):
    """
    #--- 
    """
    test_stat_i = {}
    cp_i = list_of_cp[1:-1]
    for each_cp in range(len(cp_i)):
        prev_cp= list_of_cp[list_of_cp.index(cp_i[each_cp])-1]
        next_cp = list_of_cp[list_of_cp.index(cp_i[each_cp])+1]
        datamatrix = fdata[ prev_cp: next_cp]    #takes care of calculating the statistic between prev. and next. change points
        _, test_stat_i[str(cp_i[each_cp])] = cusum(datamatrix, binseg_thresh, [])
    return test_stat_i

def mean_funcs(func_data, est_changepoint):
    # useful for computing extremal sets
    mu_diff = []
    cp_consideration = est_changepoint[1:-1]
    for cp in cp_consideration:
        cp_m1 = est_changepoint[est_changepoint.index(cp)-1]
        cp_p1 = est_changepoint[est_changepoint.index(cp)+1]
        fdata1 = func_data[cp_m1: cp-cp_m1, :]
        fdata2 = func_data[cp-cp_m1:cp_p1, :]          # select data before/after change-point
        mu1_hat = np.mean(fdata1, axis = 0)
        mu2_hat = np.mean(fdata2, axis = 0)
        deviation_mu = mu2_hat- mu1_hat
        mu_diff.append(deviation_mu)
    return mu_diff

def extremal_points(mu_diff, c, n, M_hat, est_cp):
    """ within curves """
    c_thresh = c*np.log(n)
    E_upper = []
    E_lower = []
    d_hat = []   # store d_hat corresponding to each CP.
    sValues = np.linspace(0, n, num = n)
    sValues = sValues/n             # s_hat has to be in (0,1)
    for cp in range(1,len(est_cp)-1):
        statistic = M_hat[str(est_cp[cp])]   # M_(n,i)
        d_hat_i = statistic
        upper_thresh = d_hat_i - c_thresh/np.sqrt(n)
        lower_thresh = - d_hat_i + c_thresh/np.sqrt(n)
        E_plus = np.argwhere(mu_diff[cp-1] >= upper_thresh)      # arguments which are extremal points
        E_minus = np.argwhere(mu_diff[cp-1] <= lower_thresh)    # arguments where the difference is  significant
        E_upper.append(E_plus), E_lower.append(E_minus)
        d_hat.append(d_hat_i)
    return E_upper, E_lower, d_hat #, E_p_shade, E_m_shade


def bootstrap_func(func_data, est_cp, mu_diff, E_upper, E_lower, l, M=2):
    """
    #----function for estimate quantiles  of eq. 4.14 by generating T_n of eq. 4.20
    #----note that in this function the data matrix is transformed after the est. change-point
    # -- n1: estimated change-point.

    #--- some previous code might not run because of not being provided with the arg l 
    """
    N = len(func_data[:, 0])
    #l = int(N**(1/4))    #remove this from the arguments of functions...
    T_max = []
    ni = []
    s_tild_hat = []
    sValues = np.linspace(0, N, num = N)/N
    for i in range(1, len(est_cp)-1): 
        cp = est_cp[i]
        cp_m1= est_cp[i-1]
        cp_p1= est_cp[i+1]
        E_plus, E_minus = E_upper[i-1], E_lower[i-1]    
        s_hat = sValues[int(cp)]                      #  Estimated change-point between 0, 1
        s_tilde =  (s_hat- sValues[int(cp_m1)])/(sValues[int(cp_p1)-1]- sValues[int(cp_m1)]) 
        fdata = func_data[cp_m1: cp_p1, :]  # slicing the data pre and post cp
        s_tild_hat.append(s_tilde)
        n_i = est_cp[i+1]- est_cp[i-1]
        ni.append(n_i)
        fdata[cp- cp_m1:, :] = fdata[cp-cp_m1:, :]- mu_diff[i-1]        # Y_j centering data after est. change-point with mu_diff,  check immediately after eq. 4.15
        rv = np.random.normal(0, 1, N*M)
        epsilon_hat = fdata - np.mean(fdata, axis = 0)   # first term from paper
        epsilon_star = []
        """ l is bandwidth parameter, rv is multiplier random variable ! """
        for k in range(1, n_i-l+1):
            quant  = (1/np.sqrt(l))*np.sum(epsilon_hat[k: (k+l), :], axis = 0)   # second factor need not be included... 
            quant = quant*rv[k]
            epsilon_star.append(quant)      # sequentially appending from for loop
        epsilon_star = np.asarray(epsilon_star)

        B_1 = np.multiply((1/np.sqrt(n_i)), np.sum(epsilon_star, axis = 0))
        B_s = np.multiply((1/np.sqrt(n_i)), np.sum(epsilon_star[int(cp_m1):int(cp)], axis = 0))
        W_hat = B_s - np.multiply((s_tilde), B_1)    #(1/n_i)*        # quantity for bootstrap repetitions ! # stilde here
        
        if len(E_plus) > 0:    
            T_plus =  np.max(W_hat[E_plus])  # sup over each extremal set
        else:
            T_plus = float("-inf")
        
        if len(E_minus) > 0:
            T_minus = np.max(-W_hat[E_minus])
        else:
            T_minus = float("-inf")
        
        W_factor_test = max(T_plus, T_minus)            # max over the two different extremals...!
        T = W_factor_test#*(1/s_hat*(1-s_hat))    # rescaling wrong and from previous paper.... 
        T_max.append(T)  # over which max_i will be taken   
    
    if len(T_max) > 0:
        max_value = max(T_max)
        # Use the maximum value as needed
    else:
    # Handle the case when the sequence is empty
        max_value = 1000000
    
    return max_value, ni, s_tild_hat








def reject(data_mat, change_points,  Delta, level_alpha, repeats, const_c, binseg_thresh, l,  M = 2):

    N = len(data_mat[:,0])     # number of functional data 
    T_star = []
    mu_diff = mean_funcs(data_mat, change_points) # list of mu_diffs
    test_stat =  mult_test_stat(change_points, data_mat, binseg_thresh)
    E_upper, E_lower, _ = extremal_points(mu_diff, const_c, N, test_stat, change_points)
    for _ in range(repeats): 
        data_mat_copy = np.copy(data_mat)
        T_st, n_i, s_Tilde= bootstrap_func(data_mat_copy, change_points, mu_diff, E_upper, E_lower, l, M)
        T_star.append(T_st)   #value of bootstrap samples

    quantile = np.quantile(np.asarray(T_star), 1-level_alpha)
    s_tilde_fact = np.multiply(np.array(s_Tilde), (1- np.array(s_Tilde)))
    teststat_list = list(test_stat.values())
    test_dec = {}
    for delta in Delta:
        rhs_1 = np.array(teststat_list)- np.multiply(s_tilde_fact, delta)     # coord-wise multiply
        rhs = np.multiply(np.sqrt(n_i), rhs_1)
        #print(max(rhs), quantile)
        global_rej = ((max(rhs))>= quantile)*1
        rej = (rhs>= quantile)
        rej = rej*1
        test_dec[str(delta)] = global_rej
    return rej  #test_dec #rej, global_rej# , vars_boots   





def change_point_Test(run, num_curves, time_points, change_locs, bin_seg_thresh, delta_size, alpha, boot_repeats, block_length, c_value= 0.001):
    """
    #------(1) generates synthetic data and detects change points by binary segmentation followed by testing for the detected change points
    #----- (2) runs only for multiprocessing stuff
    """
    #---- add after seed---#
    # for multiple runs add # np.random.seed(seed_val) and set seed_val as the index of run
    """
    #### uncomment for independent errors
    std_noise = 2
    start_value, end_value = 0, 0
    n_steps = 99
    volatility = std_noise
    # Initialize a matrix to store the Brownian bridges
    bridge_matrix = np.zeros((num_curves, n_steps + 1))
    for i_row in range(num_curves):
        bridge_matrix[i_row, :] = brownian_bridge(start_value, end_value, n_steps, volatility)
    """
    
    #seed_val = np.random.seed(run)
    errors_eta = fMA1(num_curves)
    #----------------generate data----------------------------------#
    sine_curves = np.sin(time_points) + np.cos(time_points) 
    noisy_curves = 9.126*sine_curves  + errors_eta #+ noise
    #mean_curve = np.mean(noisy_curves, axis= 0)
    #---------------add change point to data -----------------------#
    for idx in change_locs:

        noisy_curves[idx:, 0:16] +=   [2, 5, 9, 10,  12, 15, 25, 25, 25, 22, 15, 12, 10, 9, 5, 2] #continuous constant to simulate change 
    #------------ binary segmentation ------------------------------#
    binseg_cp, _ = binary_seg(noisy_curves, bin_seg_thresh)  # not using at all

    #------------- change point test -------------------------------#
    test_decisions = reject(noisy_curves, binseg_cp, Delta= delta_size, level_alpha=alpha, repeats=boot_repeats, const_c = c_value , binseg_thresh=bin_seg_thresh, l= block_length)
    
    
    return  binseg_cp, test_decisions #relevant_cp, global_rej


