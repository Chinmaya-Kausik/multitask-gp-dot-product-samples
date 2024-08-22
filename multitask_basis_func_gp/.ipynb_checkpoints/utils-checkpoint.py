import numpy as np
import math
from matplotlib import pyplot as plt
import torch
import gpytorch
from tqdm import tqdm
from sklearn.linear_model import Ridge
from scipy.interpolate import griddata
from .gen_funcs import *
from .basis_funcs import *
import gc

relu = torch.nn.ReLU()

# Generate the true coefficients from curve data using a uniform grid ot the original grid
def gen_true_coeffs(df, scale_df, cid, time_list, dual_scale_factor, grid_size, basis_funcs, kpi="Spend", grid = "uniform"):
    if torch.is_tensor(time_list):
        time_array = time_list.detach().numpy()
    else:
        time_array = np.array(time_list)
    true_coeffs = np.zeros([time_array.shape[0], len(basis_funcs.func_list)])
    
    for idx, time in enumerate(time_array):

        # Get interpolated kpi values for a uniform grid
        if (grid == "uniform"):
            scaled_dual_x, kpi_y, scale = gen_unif_grid_dual_kpi(df, scale_df, cid, time, dual_scale_factor, grid_size, kpi=kpi)
        elif (grid == "original"):
            scaled_dual_x, kpi_y, scale = get_orig_grid_dual_kpi(df, scale_df, cid, time, dual_scale_factor, kpi="Spend")
        else:
            raise NotImplementedError("Not implemented any other grid yet")
        
        # Fitting with data
        reg = Ridge(positive=True, alpha=0.1)
        basis_func_outputs = basis_funcs(scaled_dual_x)
        reg.fit(basis_func_outputs, kpi_y)
        true_coeffs[idx, :] = reg.coef_
        
    return torch.tensor(true_coeffs, dtype=torch.float)

# Generate kpi values using a model for given input duals
def gen_kpi_vals_single_time(model, likelihood, basis_funcs, scaled_dual_x, time, kpi="Spend", mode = "prior", use_relu = True): 

    time = int(time) # Sometimes it comes as a tensor

    # Prior vs posterior mode
    if (mode == "prior"):
        coeff_model_at_time = model([time])
    else:
        coeff_model_at_time = model.dot_prod_posterior([time])

    # Predict using mean coefficients
    mean_coeffs = transform_coeffs(coeff_model_at_time.mean.squeeze(), use_relu)
    basis_func_outputs = make_nonzeroD_tensor(basis_funcs(scaled_dual_x))
    pred_y = basis_func_outputs @ mean_coeffs
        
    return pred_y

# Get dual-KPI grid with uniform grid of dual inputs
def gen_unif_x_y_grid2D(x,y,N2):
    grid_x =  np.linspace(x.min(),x.max(),N2)
    grid_y= griddata(x, y, grid_x, method='linear')
    
    return grid_x, grid_y #train_x_reduced, train_y_reduced

# Get dual-kpi grid with custom dual inputs
def gen_custom_x_y_grid2D(x, y, custom_x):
    return custom_x, griddata(x, y, custom_x, method = "linear")

# Get scaling factor for Campaign Id cid using dataframe of scaling factors
def get_scale(scale_df, cid, dual_scale_factor):
    scale_df_cid = scale_df[scale_df["CampaignId"] == cid]
    scale = scale_df_cid["BaseDual"].values[0]*dual_scale_factor

    return scale

# Get the original dual-KPI grid from dataframe
def get_orig_grid_dual_kpi(df, scale_df, cid, time, dual_scale_factor, kpi="Spend"):
    time = int(time) # Sometimes it comes as a tensor
    
    # Slice the dataframe
    df_cid = df[df["CampaignId"] == cid]
    df_cid_time = df_cid[df_cid["Time"]==time]
    scale = get_scale(scale_df, cid, dual_scale_factor)

    # Get raw dual grid and kpis
    scaled_dual_x = df_cid_time["Dual"].to_numpy()/scale
    kpi_y = df_cid_time[kpi].to_numpy()

    return scaled_dual_x, kpi_y, scale

# Get a uniform dual-kpi grid from dataframe
def gen_unif_grid_dual_kpi(df, scale_df, cid, time, dual_scale_factor, grid_size, kpi="Spend"):
    
    scaled_dual_x, kpi_y, scale = get_orig_grid_dual_kpi(df, scale_df, cid, time, dual_scale_factor, kpi="Spend")

    # Get new grid and interpolate
    new_scaled_dual_x = np.linspace(scaled_dual_x.min(),scaled_dual_x.max(), grid_size)
    new_scaled_dual_x, kpi_y = gen_custom_x_y_grid2D(scaled_dual_x,kpi_y,new_scaled_dual_x)
    return new_scaled_dual_x, kpi_y, scale

# Accessory utility for debugging linear operator operations
def diagnose(linop):
    print("Matrix = {}\n\n Eig vals = {}".format(linop.to_dense().shape, torch.linalg.eigvalsh(linop.to_dense())))

# Accessory utility for debugging linear operator operations
def lprint(linop):
    print("Matrix = {}".format(linop.to_dense().shape))

# Processes ints and lists and converts them in to a tensor with ndimension >= 1
def make_nonzeroD_tensor(stuff):
    if stuff is None:
        return torch.tensor([], dtype = torch.float)
    elif (not torch.is_tensor(stuff)):
        tensor_stuff = torch.tensor(stuff, dtype=torch.float)
    else:
        tensor_stuff = stuff.float()
        
    if (tensor_stuff.ndimension() == 0):
        return tensor_stuff.unsqueeze(-1)
    else:
        return tensor_stuff

# Applies relu to coefficient vectors element-wise
def transform_coeffs(coeffs, use_relu=True):
    if use_relu:
        return relu(coeffs)
    else:
        return coeffs

# Obtain observed times/duals/kpis from dataframe df_real and campaign cid
def get_real_vals_hourly(df_real, scale_df, cid, dual_scale_factor, kpi="Spend"):
    
    df_real_cid = df_real[df_real["CampaignId"] == cid]
    scale = get_scale(scale_df, cid, dual_scale_factor)
    
    update_times = make_nonzeroD_tensor(df_real_cid["Time"].values)
    update_duals_scaled = make_nonzeroD_tensor(df_real_cid["BaseDual"].values)/scale
    update_kpis = make_nonzeroD_tensor(np.float64(df_real_cid["Spend"].values))

    return update_times, update_duals_scaled, update_kpis

# For a given update_sigma
# Compute error in posterior predictions made using data from previous times [0, ... t-1] , aggregated across all times t
def compute_post_pred_error(df_real, scale_df, cid, model, likelihood, update_sigma, dual_scale_factor,\
                            basis_funcs, max_time = 24, kpi="Spend", use_relu = True, verbose = False):
    
    update_inputs, update_basis_inputs, update_labels = \
        get_real_vals_hourly(df_real, scale_df, cid, dual_scale_factor, kpi="Spend")
    model.update_inputs = torch.cat((model.update_inputs, update_inputs))
    model.update_basis_inputs = torch.cat((model.update_basis_inputs, update_basis_inputs))
    model.update_labels = torch.cat((model.update_labels, update_labels))
    
    model.update_sigma = torch.nn.Parameter(update_sigma)
    
    preds = model.large_input_dot_prod_posterior_means(max_time, verbose = verbose)
    num_datapoints = update_labels.squeeze().shape[0]
    preds = preds[-num_datapoints:]
    pred_error = torch.mean(torch.square(preds.squeeze() - update_labels.squeeze()))

    return pred_error

# For model's current update_sigma value
# Compute error in posterior predictions made using data from previous times [0, ... t-1] , aggregated across all times t
def compute_post_pred_error_implicit_sigma(df_real, scale_df, cid, model, likelihood, dual_scale_factor, 
                                           basis_funcs, max_time = 24, kpi="Spend", use_relu = True, verbose = False):
    
    update_inputs, update_basis_inputs, update_labels = \
        get_real_vals_hourly(df_real, scale_df, cid, dual_scale_factor, kpi="Spend")
    model.update_inputs = torch.cat((model.update_inputs, update_inputs))
    model.update_basis_inputs = torch.cat((model.update_basis_inputs, update_basis_inputs))
    model.update_labels = torch.cat((model.update_labels, update_labels))
    
    preds = model.large_input_dot_prod_posterior_means(max_time, verbose = verbose)
    num_datapoints = update_labels.squeeze().shape[0]
    preds = preds[-num_datapoints:]
    pred_error = torch.mean(torch.square(preds.squeeze() - update_labels.squeeze()))

    return pred_error

# Generate array of errors for given grid of update_sigma values
def gen_error_array(df_real, scale_df, cid, model, likelihood,\
                sigma_grid, dual_scale_factor, basis_funcs, max_time = 24, \
                    kpi="Spend", use_relu = True, verbose = False, per_compute_verbose = False):
    with torch.no_grad():
        error_array = torch.zeros(sigma_grid.shape[0])
        for idx, update_sigma in tqdm(enumerate(sigma_grid), disable = not verbose):
            error_array[idx] = compute_post_pred_error(df_real, scale_df, cid, model, likelihood,\
                                              update_sigma, dual_scale_factor, basis_funcs, max_time = max_time, kpi="Spend",\
                                                use_relu = True, verbose = per_compute_verbose)
    return error_array

# Perform grid search using a given grid of update_sigma values
def gridsearch(df_real, scale_df, cid, model, likelihood,\
                sigma_grid, dual_scale_factor, basis_funcs, max_time = 24,\
               kpi="Spend", use_relu = True, verbose = False, per_compute_verbose = False):
    error_array = gen_error_array(df_real, scale_df, cid, model, likelihood,\
                sigma_grid, dual_scale_factor, basis_funcs, max_time = max_time, kpi=kpi, use_relu = use_relu,\
                                  verbose = verbose, per_compute_verbose = per_compute_verbose)

    best_sigma_idx = torch.argmin(error_array)
    return sigma_grid[best_sigma_idx]

# Find optimal model hyperparameters using training data
def hyperparam_optim(model, likelihood, train_t, train_coeffs, num_train_iter, lr = 0.1, scheduler_gamma = 0.999, verbose = False):
    model.train()
    likelihood.train()
    
    # Use the adam optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = scheduler_gamma)
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in tqdm(range(num_train_iter), disable = not verbose):
        optimizer.zero_grad()
        output = model(train_t)
        loss = -mll(output, train_coeffs)
        loss.backward()
        optimizer.step()
        scheduler.step()

# Use gradients to optimize update_sigma
def update_sigma_grad_opt(df_real, scale_df, cid, model, likelihood, dual_scale_factor, basis_funcs, num_train_iter,\
                          max_time = 24, lr = 0.1, scheduler_gamma = 0.999, init_update_sigma = 1.0, verbose = False, per_compute_verbose = False):
    
    lr = lr
    scheduler_gamma = scheduler_gamma
    model.update_sigma = torch.nn.Parameter(torch.tensor(init_update_sigma))
    
    for name, param in model.named_parameters():
        param.requires_grad = False
    model.update_sigma.requires_grad = True
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # Includes GaussianLikelihood parameters
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = scheduler_gamma)

    for i in tqdm(range(num_train_iter), disable = not verbose):
        optimizer.zero_grad()
        loss = compute_post_pred_error_implicit_sigma(df_real, scale_df, cid, model, likelihood,\
                                                  dual_scale_factor, basis_funcs, max_time = max_time,\
                                                      kpi="Spend", use_relu = True, verbose = per_compute_verbose)
        loss.backward()
        optimizer.step()
        scheduler.step()
