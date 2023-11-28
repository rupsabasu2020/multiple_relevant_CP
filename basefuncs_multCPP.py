import numpy as np
import random
import fMA_1_base as fMA1

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




def reject(data_mat, change_points,  Delta, level_alpha, repeats, const_c, binseg_thresh= None, block_length_boot= None,  M = 2):
    """
    Test procedure for multiple changes

        --
        --
        --
        --
    """
    # Calculate or use user-defined binseg_thresh
    if binseg_thresh is None:
        binseg_thresh = calculate_median_l2_norm_squared(data_mat)
    if block_length_boot is None:
        block_length_boot = w_funcs_q(h_val = 9, timeseries=data_mat, weight_function='qs')

    N = len(data_mat[:,0])     # number of samples of functional data 
    T_star = []
    mu_diff = mean_funcs(data_mat, change_points) #  mean difference before and after change locations
    test_stat =  mult_test_stat(change_points, data_mat, binseg_thresh)   # Test statistic at each change location
    E_upper, E_lower, _ = extremal_points(mu_diff, const_c, N, test_stat, change_points)  # E_plus and E_minus (the extremal points)
    for _ in range(repeats): 
        data_mat_copy = np.copy(data_mat)
        T_st, n_i, s_Tilde= bootstrap_func(data_mat_copy, change_points, mu_diff, E_upper, E_lower, block_length_boot, M)
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


def bartlett_window(window_length= 21):
    if window_length % 2 == 0:
        raise ValueError("Window length should be odd for Bartlett window")
    return np.bartlett(window_length)

def parzen_window(window_length= 21):
    if window_length % 2 == 0:
        raise ValueError("Window length should be odd for Parzen window")
    parzen_window = np.zeros(window_length)
    half_width = window_length // 2
    for n in range(half_width):
        parzen_window[half_width - n] = 1 - 6 * (n / half_width) ** 2 + 6 * (n / half_width) ** 3
        parzen_window[half_width + n] = 2 * (1 - n / half_width) ** 3
    return parzen_window

def tukey_hanning_window(window_length= 21):
    tukey_hanning = np.zeros(window_length)
    center = (window_length - 1) / 2
    x = np.linspace(-center, center, window_length)

    # Define the Tukey-Hanning window as (1 + cos(πx)) / 2 for |x| ≤ 1 and 0 otherwise
    tukey_hanning[np.abs(x) <= 1] = 0.5 * (1 + np.cos(np.pi * x[np.abs(x) <= 1]))

    return tukey_hanning

def quadratic_spectral_weight(window_length =21):
    if window_length < 3:
        raise ValueError("Window length should be at least 3 for the quadratic spectral weight function")
    quadratic_weight = np.zeros(window_length)
    center = (window_length - 1) / 2
    x = np.linspace(-center, center, window_length)
    for i in range(len(x)):
        if x[i] == 0:
            quadratic_weight[i] = 0
        else:
            quadratic_weight[i] = (25 / (12 * np.pi**2 * x[i]**2)) * (np.sin(6 * np.pi * x[i] / 5) / (6 * np.pi * x[i] / 5) - np.cos(6 * np.pi * x[i] / 5))

    return quadratic_weight






# mean squared error... for selection of bandwidth of the bootstrap

def weighted_corr(h_val, data, weights):
    """gives estimate of covariance, denoted by C_hat"""
    weighted_corrs = []
    data_centered = data-np.mean(data, axis = 0)
    for l in range(1, h_val):
        gamma_l = np.matmul(data_centered[:, :-l-1], data_centered[:, l+1:].T)
        K = weights[l] * gamma_l
        weighted_corrs.append(K)
    w_corrs = np.asarray(weighted_corrs)
    summing_l = np.sum(w_corrs, axis = 0) # summing over l
    return summing_l


def compute_c_0(w_corrs, weight_func, q, N):
    norm_Chat_1st = (q*(np.linalg.norm(w_corrs))**2)**(1/(1+2*q))
    norm_Chat_2te =  ((np.linalg.norm(w_corrs)**2 + np.trace(w_corrs)**2) * np.linalg.norm(weight_func)**2)**(-1/(1+2*q))
    c_0 = norm_Chat_1st*norm_Chat_2te
    opt_h = c_0*(N**(1/(1+2*q)))
    return opt_h




def w_funcs_q(h_val, timeseries, weight_function='qs'):
    """
    Optimal bandwidth with respect to different weight functions

    Parameters:
    - h_val: Bandwidth parameter
    - timeseries: Time series data
    - runs: Number of runs
    - weight_function: Choice of weight function ('bartlett', 'parzon', 'tukey', 'qs')

    Returns:
    Optimal bandwidth for the specified weight function
    """
    if weight_function == 'bartlett':
        bartlett_weights = bartlett_window(h_val)
        bartlett_estC = weighted_corr(h_val, timeseries.T, bartlett_weights)
        return compute_c_0(bartlett_estC, bartlett_weights, 1, len(timeseries[:, 0]))
    elif weight_function == 'parzon':
        parzon_weights = parzen_window(h_val)
        parzon_estC = weighted_corr(h_val, timeseries.T, parzon_weights)
        return compute_c_0(parzon_estC, parzon_weights, 2, len(timeseries[:, 0]))
    elif weight_function == 'tukey':
        tukey_weights = tukey_hanning_window(h_val)
        tukey_estC = weighted_corr(h_val, timeseries.T, tukey_weights)
        return compute_c_0(tukey_estC, tukey_weights, 2, len(timeseries[:, 0]))
    elif weight_function == 'qs':
        qs_weights = quadratic_spectral_weight(h_val)
        qs_estC = weighted_corr(h_val, timeseries.T, qs_weights)
        return compute_c_0(qs_estC, qs_weights, 2, len(timeseries[:, 0]))
    else:
        raise ValueError("Invalid weight function. Choose from 'bartlett', 'parzon', 'tukey', 'qs'.")





#---------------------------------------------------------------------------------------------------#
#-               Ancillary functions II: choice of threshold for binary segmentation               -#
#---------------------------------------------------------------------------------------------------#
# ----------- Not required if binary segmentation threshold is known via other methods -------------#





def calculate_squared_l2_norm(data):
    differences = np.diff(data, axis=0)
    squared_l2_norms = np.sum(differences**2, axis=0)#/2
    return squared_l2_norms

def calculate_median_l2_norm_squared(data):
    squared_l2_norms = calculate_squared_l2_norm(data)
    sigma_n_2 = np.median(squared_l2_norms)
    sigma_n = sigma_n_2#**(1/2)
    return sigma_n

#---------------------------------------------------------------------------------------------------#
#-               Ancillary functions III: generate data for testing the method                     -#
#---------------------------------------------------------------------------------------------------#
# ----------- Not required if other choice of data is to be generated ------------------------------#
# we generate fMA1 data here as this fulfills assumptions in our paper 



def gen_data(num_curves, change_locs, time_points):
    #np.random.seed(seed_val)
    errors_eta = fMA1.fMA1(num_curves)
    #----------------generate data----------------------------------#
    sine_curves = np.sin(time_points) + np.cos(time_points)
    noisy_curves = 20*sine_curves + errors_eta   #+ noise

    #mean_curve = np.mean(noisy_curves, axis= 0)
    #---------------add change point to data -----------------------#
    for idx in change_locs:
        if len(change_locs) <= 2:
            noisy_curves[idx:, 0:16] +=   [2, 5, 9, 10,  12, 15, 22, 25, 25, 22, 15, 12, 10, 9, 5, 2] #continuous constant to simulate change 
        elif len(change_locs) > 2 and idx >= change_locs[2]:
            noisy_curves[idx:, 0:16] -= [2, 5, 9, 10, 12, 15, 22, 25, 25, 22, 15, 12, 10, 9, 5, 2]
    return noisy_curves