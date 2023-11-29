import numpy as np
import random
import fMA_1_base as fMA1
import matplotlib.pyplot as plt

# ---- FINAL VERSION OF REQUIRED BASE FUNCTIONS ----------------#


#---------------------------------------------------------------------------------------------------#
#--                           Initial binary segmentation                                         --#
#---------------------------------------------------------------------------------------------------#


def U_hat(s, fdata, l, r, N):
    """ CUSUM statistic to recover change points
    
    Parameters: 
        -- s: s in (l, r) and the time point under consideration
        --fdata: functional data in matrix form with dimensions (N, 100)   # 100 can be any other value
        -- l: start of the sequence in (1, N)
        -- r: end of the sequence in (1, N)
        -- N: sample size of the full sequence
    """
    res = (1/(r-l)) * (np.sum(fdata[ l :(int(s * N)), :], axis=0) - ((int(s * N) - l) / (r - l)) * np.sum(fdata[l:r-1, :], axis=0))   # configure start to be prev. cp
    return res

def cusum(fdata, begin, end, N, thresh_val):  #, thresh_val, cp_list,
    """ iteratively using the CUSUM statistic above to recover locations of changes 

    Parameters: 
        --fdata:
        --begin:
        --end:
        --N: 
        --thresh_val:

    """
    
    if begin >= end:    # defining new l and r of the sequence from recursive binseg
        return -1
    
    dataPoints = end - begin
    sValues = np.linspace(float(begin)/float(N), float(end - 1)/float(N), num=dataPoints)
    M_hat = []
    for i in range(dataPoints):  # scanning over possible values of s
        U = U_hat(sValues[i], fdata, begin, end, N)
        l2_normU = np.linalg.norm(U, ord=2)   # l2 norm
        M_hat.append(l2_normU)
    pos= -1
    if np.max(M_hat)> thresh_val:
        pos= np.argmax(M_hat)
    if pos >0:
        return pos + begin
    else:
        return pos
    
def binary_seg_init(data, thresh):
    """Binary segmentation for change-point detection."""
    N = len(data[:, 0])
    cp = [0, N]  # Initial interval
    #cp2 = [0, N]
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





def mult_test_stat(data_mat, N, cp_i):  #, thresh_val, cp_list,
    """ Gets the test statistic of each change point. As this is in step 2, hat_M is the sup-norm.
    Parameters:
        -- : data_mat : functional data in dimension (N* 100)
        -- begin : previous change location in (1, N)
        -- end : next change location in (1, N)
        -- N : total length of the sequence
        -- cp_i : all change locations detected by Binary segmentation
    """
    test_stat_i = {}
    for cp_index in range(1, len(cp_i)-1):
        begin = cp_i[cp_index-1]
        end = cp_i[cp_index +1]
        dataPoints = end - begin
        sValues = np.linspace(float(begin)/float(N), float(end - 1)/float(N), num=dataPoints)
        M_hat = []
        for spoints_i in range(dataPoints):  # scanning over possible values of s
            U = U_hat(sValues[spoints_i], data_mat, begin, end, N)
            hat_M =  np.max(np.abs(U))  #sup over t for fixed s
            M_hat.append(hat_M) # store this for all s

        #plt.plot(M_hat)
        #plt.vlines(x = np.argmax(M_hat), ymin=0, ymax=20)
        #plt.show()
        # print(M_hat[cp_i[cp_index]-begin])
        test_stat_i[str(cp_i[cp_index])] = M_hat[cp_i[cp_index]-begin]#*dataPoints
    return  test_stat_i





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
    h_hat = []
    sValues = np.linspace(0, 1, num = N)
    fdata = np.copy(func_data)  
    
    for i in range(1, len(est_cp)-1): 
        fdata = np.copy(func_data)
        #----------locations of current, previous and next cp -------#
        cp = est_cp[i]   # current cp
        cp_m1= est_cp[i-1] # previous cp
        cp_p1= est_cp[i+1] # next cp
        #-----------extremal sets corresponding to the cp -----------#
        E_plus, E_minus = E_upper[i-1], E_lower[i-1]    
        s_hat = sValues[int(cp)]                      #  Estimated change-point between 0, 1
        h_hat_s =  (s_hat- sValues[int(cp_m1)])/(sValues[int(cp_p1)-1]- sValues[int(cp_m1)])    # rescaling for the correct interval
        n_i = cp_p1- cp_m1
        ni.append(n_i), h_hat.append(h_hat_s)
        #----------adjusting the data for respective cp -------------#
        fdata = fdata[cp_m1: cp_p1, :]
        mu_diff[i-1] = mu_diff[i-1].reshape((1, 100))
        fdata[cp- cp_m1:, :] -= mu_diff[i-1]
        rv = np.random.normal(0, 1, N*M)
        epsilon_hat = fdata - l*np.mean(fdata, axis = 0)   # first term from paper from the bootstrap process
        epsilon_star = []
        """ l is bandwidth parameter, rv is multiplier random variable ! """
        for k in range(1, n_i-l+1):
            quant  = (1/np.sqrt(l))*np.sum(epsilon_hat[k: (k+l), :], axis = 0)   # second factor need not be included... 
            quant = quant*rv[k]    
            epsilon_star.append(quant)      # sequentially appending from for loop

        epsilon_star = np.asarray(epsilon_star)
        B_1 = np.multiply((1/np.sqrt(n_i)), np.sum(epsilon_star, axis = 0))   # full segment under considerations
        B_s = np.multiply((1/np.sqrt(n_i)), np.sum(epsilon_star[:int(cp)], axis = 0))

        W_hat = B_s - np.multiply((h_hat_s), B_1)#*(1/n_i)        # quantity for bootstrap repetitions ! # stilde here
        
        """
        #print(int(h_hat_s*N))
        #W_hat_rescaled = np.multiply((1/np.sqrt(n_i)), np.sum(epsilon_star[:int(h_hat_s*N)], axis = 0))-np.multiply((h_hat_s), B_1)

        if len(E_plus) > 0:    
            T_plus =  np.max(W_hat_rescaled[E_plus])  # sup over each extremal set
        else:
            T_plus = float("-inf")
        
        if len(E_minus) > 0:
            T_minus = np.max(-W_hat_rescaled[E_minus])
        else:
            T_minus = float("-inf")
        
        W_factor_test = max(T_plus, T_minus)            # max over the two different extremals...!
        T = W_factor_test#*(1/s_hat*(1-s_hat))    # rescaling wrong and from previous paper.... 
        T_max.append(T)  # over which max_i will be taken    

        """
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
    
    return max_value, ni, h_hat



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
    test_stat =  mult_test_stat(data_mat, N, change_points)   # Test statistic at each change location
    E_upper, E_lower, _ = extremal_points(mu_diff, const_c, N, test_stat, change_points)  # E_plus and E_minus (the extremal points)

    for _ in range(repeats): 
        data_mat_copy = np.copy(data_mat)
        T_st, n_i, h_hat= bootstrap_func(data_mat_copy, change_points, mu_diff, E_upper, E_lower, block_length_boot, M)
        T_star.append(T_st)   #value of bootstrap samples

    quantile = np.quantile(T_star, 1-level_alpha)
    h_hat_all = np.multiply(np.array(h_hat), (1- np.array(h_hat)))
    teststat_list = list(test_stat.values())
    test_dec = {}
    for delta in Delta:
        rhs_1 = np.array(teststat_list)- np.multiply(h_hat_all, delta)     # coord-wise multiply
        rhs = np.multiply(np.sqrt(n_i), rhs_1)
        print(max(rhs), quantile)
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
    return int(opt_h)




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
    squared_l2_norms = np.sum(differences**2, axis=0)/2
    return squared_l2_norms

def calculate_median_l2_norm_squared(data):
    n = len(data[:, 0])
    squared_l2_norms = calculate_squared_l2_norm(data)
    sigma_n_2 = np.median(squared_l2_norms)
    sigma_n = sigma_n_2**(1/2)
    thresh = sigma_n*np.sqrt(3*np.log(n))
    return thresh

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




