import numpy as np
import basefuncs_multCPP as bfmcpp
import matplotlib.pyplot as plt
import multiprocessing as mp
import matplotlib.gridspec as gridspec

#---------------------------------------------------------------------------------------------------#
#-                        parameter selection for bootstrap block-length                           -#
#---------------------------------------------------------------------------------------------------#

def w_funcs_q(h_val, timeseries, runs):
    """Optimal bandwidth with respect to different weight functions"""
    #timeseries = gen_data(num_curves, cp_locs, time_points)
    #--- bartlett ---#
    bartlett_weights = bfmcpp.bartlett_window(h_val)
    bartlett_estC = bfmcpp.weighted_corr(h_val, timeseries.T, bartlett_weights)
    bartlett_optH = bfmcpp.compute_c_0(bartlett_estC, bartlett_weights, 1, len(timeseries[:, 0]))
    #---- parzon----#
    parzon_weights = bfmcpp.parzen_window(h_val)
    parzon_estC = bfmcpp.weighted_corr(h_val, timeseries.T, parzon_weights)
    parzon_optH= bfmcpp.compute_c_0(parzon_estC, parzon_weights, 2, len(timeseries[:, 0]))
    #----tukey-hanning ---#
    tukey_weights = bfmcpp.tukey_hanning_window(h_val)
    tukey_estC = bfmcpp.weighted_corr(h_val, timeseries.T, tukey_weights)
    tukey_optH = bfmcpp.compute_c_0(tukey_estC, tukey_weights, 2, len(timeseries[:, 0]))
    #----quadratic-spectral ----#
    qs_weights = bfmcpp.quadratic_spectral_weight(h_val)
    qs_estC = bfmcpp.weighted_corr(h_val, timeseries.T, qs_weights)
    qs_optH = bfmcpp.compute_c_0(qs_estC, qs_weights, 2, len(timeseries[:, 0]))
    return bartlett_optH, parzon_optH, tukey_optH, qs_optH



num_curves = 1000
cp_locs = [0, 500, num_curves]
time_points = np.linspace(0, 2*np.pi, 100)  
ts_data = bfmcpp.gen_data(num_curves, cp_locs, time_points)
h_0 = 11   #(initial bandwidth)
order_window = 2




if __name__ == '__main__':

    h_vals = [3, 63, 101, 209]    # multiple h_0 (for trying out single ones, just comment this out and tweak h_0)
    num_runs = 20
    num_cores = 6  # mp.cpu_count()
    res_all = {}
    res_parzen = []
    for h_values in h_vals:
        with mp.Pool(num_cores) as p:
            results = p.starmap(w_funcs_q, [(h_values, ts_data, runs) for runs in range(num_runs)] )
        res_all[str(h_values)] = results


    # Create a figure with 3 rows and 2 columns of subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 6))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1])
    # Create the overarching plot in the first row, spanning both columns
    ax_overarching = plt.subplot(gs[2, :])
    # You can use ax_overarching to create the plot that spans both columns.
    ax_overarching.plot(ts_data.reshape(-1))

    for i, (key, values) in enumerate(res_all.items()):
        row = i // 2
        col = i % 2
        ax = axs[row, col]
        ax.set_title('Initial bandwidth ' + str(key))
        
        # Transpose the data for plotting
        transposed_values = list(map(list, zip(*values)))
        # Create box plots for the values within parentheses
        ax.boxplot(transposed_values, labels=['Barlett', 'Parzen', 'Tukey-Hanning', 'Quadratic Spectral'])

    # Adjust the layout of subplots
    plt.tight_layout()
    plt.show()