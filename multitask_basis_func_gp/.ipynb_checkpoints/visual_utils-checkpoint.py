import numpy as np
import math
from matplotlib import pyplot as plt
from .gen_funcs import *
from .basis_funcs import *
from .utils import *

# Shows coeff vs time graph for given coeff_idx
def show_coeff_time_graph(cid, times, coeff_arr_1, coeff_arr_2, coeff_idx, label_1 = "Prediction", label_2 = "Ground Truth", use_relu=True):
    fig, ax = plt.subplots()
    
    # Check shapes
    assert(times.shape[-1] == coeff_arr_1.shape[-2], \
           "Sizes of coeff_arr_1 {} and times {} are not compatible".format(coeff_arr_1.shape[-2], times.shape[-1]))
    assert(times.shape[-1] == coeff_arr_2.shape[-2], \
           "Sizes of coeff_arr_2 {} and times {} are not compatible".format(coeff_arr_2.shape[-2], times.shape[-1]))
    
    # Plotting data
    coeff_to_plot_1 = transform_coeffs(make_nonzeroD_tensor(coeff_arr_1[:, coeff_idx]), use_relu)
    coeff_to_plot_2 = transform_coeffs(make_nonzeroD_tensor(coeff_arr_2[:, coeff_idx]), use_relu)
    ax.plot(times, coeff_to_plot_1, color="blue") # rescale duals while plotting
    ax.plot(times, coeff_to_plot_2, color="orange")

    # Label plots
    plt.legend([label_1, label_2])
    plt.xlabel("Time")
    plt.ylabel("Coefficient at Index " + str(coeff_idx))
    plt.title("Campaign Id " + str(cid) + ", Index " + str(coeff_idx))

# Shows the coeff vs time graph for single coeff_array and multiple indices n coeff_idx_list
def show_coeff_graphs(cid, times, coeff_array, coeff_idx_list, use_relu = True):
    fig, ax = plt.subplots()

    # Check shape
    assert(times.shape[-1] == coeff_array.shape[-2], \
           "Sizes of coeff_array {} and times {} are not compatible".format(coeff_array.shape[-2], times.shape[-1]))
    
    # Plotting data
    for coeff_idx in coeff_idx_list:
        coeff_to_plot = transform_coeffs(coeff_array[:, coeff_idx], use_relu)
        ax.plot(times, coeff_to_plot) # rescale duals while plotting\

    # Label plot
    plt.xlabel("Time")
    plt.ylabel("Coefficient")
    plt.legend(coeff_idx_list)
    plt.title("Campaign Id " + str(cid) + ", Index {}".format(coeff_idx_list))

# Show dual-kpi curves from Exalibur data given as a dataframe
def show_train_dual_kpi_curves(df, cid, time_list, dual_scale_factor, grid_size, kpi="Spend"):
    fig, ax = plt.subplots()

    # Get kpi values and plot dual-kpi curves
    for time in time_list:
        scaled_dual_x, kpi_y, scale = gen_unif_grid_dual_kpi(df, cid, time, dual_scale_factor, grid_size, kpi=kpi)
        ax.plot(scaled_dual_x*scale, kpi_y)

    # Label plot
    plt.xlabel("Dual")
    plt.ylabel(kpi)
    ax.legend(time_list)
    plt.title("Campaign Id " + str(cid) + ", Times " + str(time_list))

# Shows prior and posterior dual kpi curves and update points
def show_prior_post_dual_kpi_curves(model, likelihood, scale_df, cid, time_list, dual_scale_factor, grid_size, basis_funcs, kpi="Spend", use_relu=True):
    fig, ax = plt.subplots()

    colors = ["red", "blue", "green", "black", "brown", "pink"]
    scale = get_scale(scale_df, cid, dual_scale_factor)
    legend = []
    for idx, time in enumerate(time_list):

        scaled_dual_x = np.linspace(0, 1, grid_size)

        prior_y = gen_kpi_vals_single_time(model, likelihood, basis_funcs,\
                                           scaled_dual_x, time, kpi=kpi, mode = "prior", use_relu=use_relu).detach()
        post_y = gen_kpi_vals_single_time(model, likelihood, basis_funcs,\
                                          scaled_dual_x, time, kpi=kpi, mode = "posterior", use_relu=use_relu).detach()
        
        ax.plot(scaled_dual_x*scale, prior_y, color = colors[idx])
        ax.plot(scaled_dual_x*scale, post_y, color = colors[idx], linestyle = "dashed")
        legend += ["Prior Hr " + str(time)]
        legend += ["Posterior Hr " + str(time)]

    for idx in range(model.update_basis_inputs.shape[-1]):
        ax.scatter(model.update_basis_inputs[idx]*scale, model.update_labels[idx], color=colors[idx])
        legend += ["New point Hr " + str(int(model.update_inputs[idx]))]

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Dual")
    plt.ylabel(kpi)
    
    # Plotting new data
    plt.title("Campaign Id " + str(cid) + ", Time " + str(time_list))