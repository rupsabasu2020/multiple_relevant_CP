import numpy as np
import matplotlib.pyplot as plt
import basefuncs_multCPP as bfmcpp
import pandas as pd



#### parameter values for testing
delta_size = 1
alpha = 0.1
boot_repeats = 100
bin_seg_thresh =7
#import datamat directly
datamat = pd.read_csv('/Users/rupsabasu/Documents/DATASETS/dataMat/indoor_fatigue/RightKnee/3.csv') #3 #2
datavals = datamat.values

segs = bfmcpp.binary_seg_init(datavals, bin_seg_thresh)
test_reject = bfmcpp.reject(datavals, segs, [delta_size], alpha, boot_repeats, 0.01 )
print(test_reject)
