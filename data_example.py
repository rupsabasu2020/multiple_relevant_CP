import numpy as np
import matplotlib.pyplot as plt
import basefuncs_multCPP as bfmcpp
import pandas as pd



#### parameter values for testing
delta_size = 1
alpha = 0.1
boot_repeats = 100
bin_seg_thresh =1
#import datamat directly
datamat = pd.read_csv('/Users/rupsabasu/Documents/DATASETS/dataMat/indoor_fatigue/RightKnee/3.csv') #3 #2
datavals = datamat.values

segs, M_hat, cusumstat = bfmcpp.binary_seg_init(datavals, bin_seg_thresh)
relevant_cp, global_rej = bfmcpp.reject(datavals, segs, Delta= [delta_size], level_alpha=alpha, repeats=boot_repeats, const_c= 0.01)
detected_cp = segs[1:]
