import numpy as np
import random

# ---- FINAL VERSION OF REQUIRED BASE FUNCTIONS ----------------#


#---------------------------------------------------------------------------------------------------#
#--                           Initial binary segmentation                                         --#
#---------------------------------------------------------------------------------------------------#

def U_hat(s, fdata, l, r, N):
    """  
    statistic to initially segment data
    Parameters:
        --
        --
        --
        --

    """
    res = (1/(r-l)) * (np.sum(fdata[ l :(int(s * N)), :], axis=0) - ((int(s * N) - l) / (r - l)) * np.sum(fdata[l:r-1, :], axis=0))   # configure start to be prev. cp
    return res



def cusum(data_mat, begin, end, N, thresh_val):  #, thresh_val, cp_list,
    """ 
    function to find multiple change locations using U_hat
    Parameters : 
        --
        --
        --
        --  
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
    pos = -1
    if np.max(M_hat) > thresh_val:   # binseg thresh for detecting

        pos = np.argmax(M_hat)
    
    if pos > 0: 
        return pos + begin, M_hat[pos] 
    else: 
        return pos, M_hat[pos] #sorted(cp_list)






def binary_seg_init(data, thresh):
    """Binary segmentation for change-point detection."""
    N = len(data[:, 0])
    cp = [0, N]  # Initial interval
    return binary_seg_rec(data, thresh, cp, 0, N, N)


def binary_seg_rec(data, thresh, cp, begin, end, N):
    cp_location, _ = cusum(data, begin, end, N, thresh)
    if cp_location > 0:
        cp.append(cp_location)
        print(cp_location)
        cp.sort()
        cp = binary_seg_rec(data, thresh, cp, begin, cp_location - 1, N)
        cp = binary_seg_rec(data, thresh, cp, cp_location + 1, end, N)
    return cp

#---------------------------------------------------------------------------------------------------#
#-                           Step 2: Relevant change points                                        -#
#---------------------------------------------------------------------------------------------------#

def mult_test_stat(list_of_cp, fdata, binseg_thresh):
    """
    compute test statistic for each change point:
        --
        --
        --
        --
    """
    test_stat_i = {}
    cp_i = list_of_cp[1:-1]
    for each_cp in range(len(cp_i)):
        prev_cp= list_of_cp[list_of_cp.index(cp_i[each_cp])-1]
        next_cp = list_of_cp[list_of_cp.index(cp_i[each_cp])+1]
        datamatrix = fdata[ prev_cp: next_cp]    #takes care of calculating the statistic between prev. and next. change points
        N = len(datamatrix[:, 0])
        _, test_stat_i[str(cp_i[each_cp])] = cusum(datamatrix, begin=0, end=N, N=N, thresh_val=binseg_thresh)   # value of the test statistic at the supremum
    return test_stat_i

def mean_funcs(func_data, est_changepoint):
    """
    compute differences in mean before and after change
        --
        --
        --
        --
    """
    mu_diff = []
    cp_consideration = est_changepoint[1:-1]   # removing end points i.e., 0 and n
    for cp in cp_consideration:
        cp_m1 = est_changepoint[est_changepoint.index(cp)-1]  #prev cp
        cp_p1 = est_changepoint[est_changepoint.index(cp)+1]  #next cp
        fdata1 = func_data[cp_m1: cp-cp_m1, :]                #data from previous until cp
        fdata2 = func_data[cp-cp_m1:cp_p1, :]          # data from cp until next cp
        mu1_hat = np.mean(fdata1, axis = 0)            # mean curve prior segment
        mu2_hat = np.mean(fdata2, axis = 0)            # mean curve next segment             
        deviation_mu = mu2_hat- mu1_hat                # difference in mean
        mu_diff.append(deviation_mu)                   # appending to list for multiple CP
    return mu_diff

def extremal_points(mu_diff, c, n, M_hat, est_cp):
    """
    Extremal points within the curves
        --
        --
        --
        --
    """
    c_thresh = c*np.log(n)          # within RHS of extremal points definition
    E_upper = []
    E_lower = []
    d_hat = []   # store d_hat corresponding to each CP.

    for cp in range(1,len(est_cp)-1):
        d_hat_i = M_hat[str(est_cp[cp])]   # Test statistic corresponding to CP
        upper_thresh = d_hat_i - c_thresh/np.sqrt(n)  # RHS E_plus
        lower_thresh = - d_hat_i + c_thresh/np.sqrt(n) # RHS E_minus
        E_plus = np.argwhere(mu_diff[cp-1] >= upper_thresh)     # arguments which are extremal points
        E_minus = np.argwhere(mu_diff[cp-1] <= lower_thresh)    # arguments where the difference is  significant
        E_upper.append(E_plus), E_lower.append(E_minus)
        d_hat.append(d_hat_i)
    return E_upper, E_lower, d_hat #, E_p_shade, E_m_shade


def bootstrap_func(func_data, est_cp, mu_diff, E_upper, E_lower, l, M=2):
    """
    Bootstrapping the threshold quantiles
        --
        --
        --
        --
    """
    N = len(func_data[:, 0])
    T_max = []
    ni = []
    s_tild_hat = []
    sValues = np.linspace(0, 1, num = N)
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



#---------------------------------------------------------------------------------------------------#
#-                           Step 2 (continued): Relevant change points, decision function         -#
#---------------------------------------------------------------------------------------------------#




def reject(data_mat, change_points,  Delta, level_alpha, repeats, const_c, binseg_thresh, l,  M = 2):
    """
    Test procedure for multiple changes

        --
        --
        --
        --
    """
    
    N = len(data_mat[:,0])     # number of samples of functional data 
    T_star = []
    mu_diff = mean_funcs(data_mat, change_points) #  mean difference before and after change locations
    test_stat =  mult_test_stat(change_points, data_mat, binseg_thresh)   # Test statistic at each change location
    E_upper, E_lower, _ = extremal_points(mu_diff, const_c, N, test_stat, change_points)  # E_plus and E_minus (the extremal points)
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
    return test_dec, rej




#---------------------------------------------------------------------------------------------------#
#-               Ancillary functions I: choice of block-length for the bootstrap                   -#
#---------------------------------------------------------------------------------------------------#
# ----------- Not required if block-length l is known ----------------------------------------------#








#---------------------------------------------------------------------------------------------------#
#-               Ancillary functions II: choice of threshold for binary segmentation               -#
#---------------------------------------------------------------------------------------------------#
# ----------- Not required if binary segmentation threshold is known via other methods -------------#





