import numpy as np
import math
import torch
import gpytorch
from tqdm import tqdm
from sklearn.linear_model import Ridge
from scipy.interpolate import griddata
from .gen_funcs import *
from .basis_funcs import *
import gc

relu = torch.nn.ReLU()

# Function to log CPU memory usage
def log_cpu_memory():
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"RSS: {memory_info.rss / (1024 * 1024):.2f} MB, VMS: {memory_info.vms / (1024 * 1024):.2f} MB")


# Function to log GPU memory usage
def log_gpu_memory():
    import psutil
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    print(f"Allocated Memory: {allocated:.2f} MB, Reserved Memory: {reserved:.2f} MB")

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

# Generate the true coefficients from curve data using a uniform grid ot the original grid
def gen_true_coeffs(df, scale_df, inst_id, time_tensor, x_scale_factor, grid_size, basis_funcs, target_name="Spend", grid = "uniform"):
    if not torch.is_tensor(time_tensor):
        time_tensor = torch.tensor(time_tensor, dtype=torch.float)
    true_coeffs = np.zeros([time_tensor.shape[0], len(basis_funcs.func_list)])
    
    bad_idx = []
    for idx, time in enumerate(time_tensor):

        # Get interpolated y values for a uniform grid
        if (grid == "uniform"):
            scaled_x, y_init, scale = gen_unif_grid_x_y(df, scale_df, inst_id, time, x_scale_factor, grid_size, target_name=target_name)
        elif (grid == "original"):
            scaled_x, y_init, scale = get_orig_grid_x_y(df, scale_df, inst_id, time, x_scale_factor, target_name="Spend")
        else:
            raise NotImplementedError("Not implemented any other grid yet")
        
        if scaled_x.size == 0:
            bad_idx.append(idx)
        else:
            # Fitting with data
            reg = Ridge(positive=True, alpha=0.1)
            basis_func_outputs = basis_funcs(scaled_x)
            reg.fit(basis_func_outputs, y_init)
            true_coeffs[idx, :] = reg.coef_

    # Remove bad indices
    if not bad_idx:
        true_coeffs = np.delete(true_coeffs, bad_idx, axis=0)
        time_tensor = torch.tensor([time for idx, time in enumerate(time_tensor) if idx not in bad_idx], dtype=torch.float)
    
    return time_tensor, torch.tensor(true_coeffs, dtype=torch.float)

# Get x-y grid with uniform grid of x inputs
def gen_unif_x_y_grid2D(x,y,N2):
    grid_x =  np.linspace(x.min(),x.max(),N2)
    grid_y= griddata(x, y, grid_x, method='linear')
    
    return grid_x, grid_y #train_x_reduced, train_y_reduced

# Get x-y grid with custom x inputs
def gen_custom_x_y_grid2D(x, y, custom_x):
    return custom_x, griddata(x, y, custom_x, method = "linear")

# Get scaling factor for Campaign Id inst_id using dataframe of scaling factors
def get_scale(scale_df, inst_id, x_scale_factor):
    scale_df_inst_id = scale_df[scale_df["CampaignId"] == inst_id]
    scale = scale_df_inst_id["Basex"].values[0]*x_scale_factor

    return scale

# Get the original x-y grid from dataframe
def get_orig_grid_x_y(df, scale_df, inst_id, time, x_scale_factor, target_name="Spend"):
    time = int(time) # Sometimes it comes as a tensor
    
    # Slice the dataframe
    df_inst_id = df[df["CampaignId"] == inst_id]
    df_inst_id_time = df_inst_id[df_inst_id["Time"]==time]
    scale = get_scale(scale_df, inst_id, x_scale_factor)

    # Get raw x grid and ys
    scaled_x = df_inst_id_time["x"].to_numpy()/scale
    y_init = df_inst_id_time[target_name].to_numpy()

    return scaled_x, y_init, scale

# Get a uniform x-y grid from dataframe
def gen_unif_grid_x_y(df, scale_df, inst_id, time, x_scale_factor, grid_size, target_name="Spend"):
    
    scaled_x, y_init, scale = get_orig_grid_x_y(df, scale_df, inst_id, time, x_scale_factor, target_name="Spend")

    if scaled_x.size == 0:
        return scaled_x, y_init, scale
    else:
        # Get new grid and interpolate
        new_scaled_x = np.linspace(scaled_x.min(),scaled_x.max(), grid_size)
        new_scaled_x, y_init = gen_custom_x_y_grid2D(scaled_x,y_init,new_scaled_x)
        return new_scaled_x, y_init, scale

# Obtain observed times/xs/ys from dataframe df_real and campaign inst_id
def get_real_vals(df_real, scale_df, inst_id, x_scale_factor, target_name="Spend"):
    
    df_real_inst_id = df_real[df_real["CampaignId"] == inst_id]
    scale = get_scale(scale_df, inst_id, x_scale_factor)
    
    update_times = make_nonzeroD_tensor(df_real_inst_id["Time"].values)
    update_xs_scaled = make_nonzeroD_tensor(df_real_inst_id["Basex"].values)/scale
    update_ys = make_nonzeroD_tensor(np.float64(df_real_inst_id["Spend"].values))

    return update_times, update_xs_scaled, update_ys

# Generate y values using a model for given input xs
def gen_y_vals_single_time(model, likelihood, basis_funcs, scaled_x, time, target_name="Spend", mode = "prior", use_relu = True, verbose = False): 

    time = int(time) # Sometimes it comes as a tensor

    # Prior vs posterior vs iterative posterior mode 
    # Iterative posterior has dynamic programming and memoization for speed
    if (mode == "prior"):
        mean_coeffs_at_time = model([time]).mean.squeeze()
    elif (mode == "posterior"):
        mean_coeffs_at_time = model.dot_prod_posterior([time]).mean.squeeze()
    else:
        current_dist = model.__call__(torch.arange(model.max_time).float())
        update_inputs = model.update_inputs.squeeze()
        update_dot_prod_vecs = make_nonzeroD_tensor(model.basis_funcs(model.update_basis_inputs)).squeeze()
        update_labels = model.update_labels.squeeze()
        for idx, current_input in tqdm(enumerate(update_inputs), disable = not verbose):
            current_dot_prod_vec = update_dot_prod_vecs[idx, :]
            current_label = update_labels[idx]
            current_dist = model.large_input_dot_prod_posterior_helper_bf(current_dist, current_input, current_dot_prod_vec, current_label)

        mean_coeffs_at_time = current_dist.mean.squeeze()[time]

    # Predict using mean coefficients
    mean_coeffs_transformed = transform_coeffs(mean_coeffs_at_time, use_relu)
    basis_func_outputs = make_nonzeroD_tensor(basis_funcs(scaled_x))
    pred_y = basis_func_outputs @ mean_coeffs_transformed
        
    return pred_y

# Function to make posterior predictions on update_inputs while sequentially updating the posterior 
# Posterior is updated with the update_labels and update_basis_inputs seen so far before making the next prediction
# Model is initialized with its existing update data, and the inital update data is restored at the end so that the model is unchanged
def make_post_preds(update_inputs, update_basis_inputs, update_labels, model, likelihood, x_scale_factor,\
                            basis_funcs, target_name="Spend", use_relu = True, verbose = False):
    
    init_update_inputs, init_update_basis_inputs, init_update_labels = model.update_inputs, model.update_basis_inputs, model.update_labels
    model.update_inputs = torch.cat((model.update_inputs, update_inputs))
    model.update_basis_inputs = torch.cat((model.update_basis_inputs, update_basis_inputs))
    model.update_labels = torch.cat((model.update_labels, update_labels))

    preds = model.large_input_dot_prod_posterior_means_bf(use_relu = use_relu, verbose = verbose)
    num_datapoints = update_labels.numel()
    preds = preds[-num_datapoints:]

    model.update_inputs, model.update_basis_inputs, model.update_labels = init_update_inputs, init_update_basis_inputs, init_update_labels

    return preds

# Function to make predictions using only the model's prior
def make_prior_preds(update_inputs, update_basis_inputs, model, likelihood,\
                            basis_funcs, target_name="Spend", use_relu = True, verbose = False):
    
    y_preds = []
    for idx, input in enumerate(update_inputs):
        y_preds.append(gen_y_vals_single_time(model, likelihood, basis_funcs, update_basis_inputs[idx:idx+1], input,\
                                                target_name = target_name, mode = "prior", use_relu = use_relu, verbose = verbose))
        
    
    return torch.tensor(y_preds, dtype = torch.float)

# Function to make predictions at the daily level using prevalibur data
def make_prev_preds(df_prev, df_real, inst_id, target_name="Spend"):
    df_prev_inst_id = df_prev[df_prev["CampaignId"] == inst_id]
    df_real_inst_id = df_real[df_real["CampaignId"] == inst_id]

    xs_chosen = df_real_inst_id["Basex"].to_numpy()

    prev_inst_id_xs = df_prev_inst_id["x"].to_numpy()
    prev_inst_id_ys = df_prev_inst_id[target_name].to_numpy()

    y_prev_preds = griddata(prev_inst_id_xs, prev_inst_id_ys, xs_chosen, method = "linear")
    
    return y_prev_preds

# Function to make predictions at the hourly level using prevalibur data
def make_prev_preds_hourly(df_prev, df_real, inst_id, target_name="Spend"):
    df_prev_inst_id = df_prev[df_prev["CampaignId"] == inst_id]
    df_real_inst_id = df_real[df_real["CampaignId"] == inst_id]

    time_list = df_real_inst_id["Time"].unique()

    prev_preds = np.zeros(0)
    for time in time_list:
        df_real_inst_id_time = df_real_inst_id[df_real_inst_id["Time"] == time]
        xs_chosen = df_real_inst_id_time["Basex"].to_numpy()

        df_prev_inst_id_time = df_prev_inst_id[df_prev_inst_id["Time"] == time]
        prev_inst_id_time_xs = df_prev_inst_id_time["x"].to_numpy()
        prev_inst_id_time_ys = df_prev_inst_id_time[target_name].to_numpy()

        y_prev_preds = griddata(prev_inst_id_time_xs, prev_inst_id_time_ys, xs_chosen, method = "linear")
        prev_preds = np.append(prev_preds, y_prev_preds)
    
    return prev_preds

# Function to compute the error of making predictions using the model's prior
def compute_prior_pred_error(df_real_check, scale_df, inst_id, model, likelihood, x_scale_factor, basis_funcs, target_name="Spend", use_relu = True, verbose = False):

    update_inputs, update_basis_inputs, update_labels = get_real_vals(df_real_check, scale_df, inst_id, x_scale_factor, target_name = target_name)
    preds = make_prior_preds(update_inputs, update_basis_inputs, model, likelihood,\
                            basis_funcs, target_name = target_name, use_relu = use_relu, verbose = verbose)
    pred_error = torch.mean(torch.square(preds.squeeze() - update_labels.squeeze()))

    return pred_error

# Function to compute the error of making predictions at hourly granularity using prevalibur data
def compute_prevalibur_error_hourly(df_prev, df_real, inst_id, target_name="Spend"):    
    y_prev_preds = make_prev_preds_hourly(df_prev, df_real, inst_id, target_name = target_name)
    true_y_vals = df_real[target_name].to_numpy()
    true_y_vals = true_y_vals.astype(float)
    prev_error_array = np.square(y_prev_preds - true_y_vals)

    prev_error = np.mean(prev_error_array)
    
    return prev_error

# Function to compute the error of making predictions at daily granularity using prevalibur data
def compute_prevalibur_error(df_prev, df_real, inst_id, target_name="Spend"):
    y_prev_preds = make_prev_preds(df_prev, df_real, inst_id, target_name = target_name)
    true_y_vals = df_real[target_name].to_numpy()
    true_y_vals = true_y_vals.astype(float)
    prev_error = np.mean(np.square(y_prev_preds - true_y_vals))

    return prev_error


# For a given update_sigma
# Compute error in posterior predictions made using data from previous times [0, ... t-1] , aggregated across all times t
def compute_post_pred_error(df_real_update, df_real_check, scale_df, inst_id, model, likelihood, update_sigma, x_scale_factor,\
                            basis_funcs, target_name="Spend", use_relu = True, verbose = False):
    
    model.log_update_sigma = torch.nn.Parameter(torch.tensor(math.log(update_sigma)))

    pred_error = compute_post_pred_error_implicit_sigma(df_real_update, df_real_check, scale_df, inst_id, model, likelihood, x_scale_factor, 
                                           basis_funcs, target_name=target_name, use_relu = use_relu, verbose = verbose)

    return pred_error

# For model's current update_sigma value
# Compute error in posterior predictions made using data from previous times [0, ... t-1] , aggregated across all times t
def compute_post_pred_error_implicit_sigma(df_real_update, df_real_check, scale_df, inst_id, model, likelihood, x_scale_factor, 
                                           basis_funcs, target_name="Spend", use_relu = True, verbose = False):
    
    if df_real_update is None:
        empty_tensor = torch.tensor([], dtype = torch.float)
        model.update_inputs, model.update_basis_inputs, model.update_labels = empty_tensor, empty_tensor, empty_tensor
    else:
        model.update_inputs, model.update_basis_inputs, model.update_labels = get_real_vals(df_real_update, scale_df, inst_id, x_scale_factor, target_name = target_name)
    
    update_inputs, update_basis_inputs, update_labels = \
        get_real_vals(df_real_check, scale_df, inst_id, x_scale_factor, target_name = target_name)
    preds = make_post_preds(update_inputs, update_basis_inputs, update_labels, model, likelihood, x_scale_factor,\
                        basis_funcs, target_name=target_name, use_relu = use_relu, verbose = verbose)
    pred_error = torch.mean(torch.square(preds.squeeze() - update_labels.squeeze()))

    return pred_error


# Generate array of errors for given grid of update_sigma values
def gen_error_array(df_real_update, df_real_check, scale_df, inst_id, model, likelihood,\
                sigma_grid, x_scale_factor, basis_funcs, \
                    target_name="Spend", use_relu = True, verbose = False, per_compute_verbose = False):
    
    # Prevent computation graph from accumulating
    # for name, param in model.named_parameters():
    #     param.requires_grad = False

    if df_real_update is None:
        empty_tensor = torch.tensor([], dtype = torch.float)
        model.update_inputs, model.update_basis_inputs, model.update_labels = empty_tensor, empty_tensor, empty_tensor
    else:
        model.update_inputs, model.update_basis_inputs, model.update_labels = get_real_vals(df_real_update, scale_df, inst_id, x_scale_factor, target_name = target_name)
    
    update_inputs, update_basis_inputs, update_labels = \
        get_real_vals(df_real_check, scale_df, inst_id, x_scale_factor, target_name = target_name)
    

    with torch.no_grad():
        error_array = torch.zeros(sigma_grid.shape[0])
        for idx, update_sigma in tqdm(enumerate(sigma_grid), disable = not verbose):
            model.log_update_sigma = torch.nn.Parameter(torch.tensor(math.log(update_sigma)))
            current_num_posterior_crashes = model.num_posterior_crashes
            preds = make_post_preds(update_inputs, update_basis_inputs, update_labels, model, likelihood, x_scale_factor,\
                        basis_funcs, target_name=target_name, use_relu = use_relu, verbose = per_compute_verbose)

            error_array[idx] = torch.mean(torch.square(preds.squeeze() - update_labels.squeeze()))
            if model.num_posterior_crashes > current_num_posterior_crashes:
                error_array[idx] = torch.inf # If posterior crashes, set error to infinity
            
            # log_cpu_memory()
            gc.collect()
            torch.cuda.empty_cache()
            
    return error_array

# Get the right scaling for the y to initialize a sigma_grid for gridsearch or to set an initial update_sigma for gradient-based optimization
def get_y_scaling(df_real, scale_df, inst_id, y_scale, model, likelihood, x_scale_factor, basis_funcs, target_name="Spend", use_relu = True,\
                    min_power = -2, y_scalings = torch.logspace(-3, 0, 15), verbose = True):

    update_inputs, update_basis_inputs, update_labels = get_real_vals(df_real, scale_df, inst_id, x_scale_factor)

    # Prevent computation graph from accumulating
    for name, param in model.named_parameters():
        param.requires_grad = False

    check_x = update_basis_inputs[0]
    for idx, y_scaling in tqdm(enumerate(y_scalings), disable = not verbose):

        lowest_update_sigma = target_name_scaling*y_scale*10**min_power
        model.log_update_sigma = torch.nn.Parameter(torch.tensor(math.log(lowest_update_sigma)))

        try:
            with torch.no_grad():
                dist = model.dot_prod_posterior(torch.arange(model.max_time).float())
                full_mean, full_covar = dist.loc, torch.flatten(dist.lazy_covariance_matrix.to_dense())
                if torch.isnan(full_mean).any() or torch.isnan(full_covar).any():
                    continue
                compute_post_pred_error_implicit_sigma(None, df_real, scale_df, inst_id, model, likelihood, x_scale_factor, basis_funcs,\
                                                    target_name=target_name, use_relu = use_relu, verbose = False)
                assert model.num_posterior_crashes == 0
                
            # Memory management
            gc.collect()
            torch.cuda.empty_cache()
            break
        except:
            model.num_posterior_crashes = 0
            model.posterior_crash_sigmas = []
            continue
    if idx >= target_name_scalings.numel() - 2:
        y_scaling = target_name_scalings[-1]
    else:
        y_scaling = target_name_scalings[idx+2]

    return y_scaling

# Perform grid search using a given grid of update_sigma values
def best_update_sigma_grid_search(df_real_update, df_real_check, scale_df, inst_id, model, likelihood,\
                sigma_grid, x_scale_factor, basis_funcs,\
               target_name="Spend", use_relu = True, verbose = False, per_compute_verbose = False):
    
    for name, param in model.named_parameters():
        param.requires_grad = False
    with torch.no_grad():
        error_array = gen_error_array(df_real_update, df_real_check, scale_df, inst_id, model, likelihood,\
                    sigma_grid, x_scale_factor, basis_funcs, target_name=target_name, use_relu = use_relu,\
                                    verbose = verbose, per_compute_verbose = per_compute_verbose)
        # Memory management
        gc.collect()
        torch.cuda.empty_cache()

    best_sigma_idx = torch.argmin(error_array)
    return sigma_grid[best_sigma_idx]

# Find optimal model hyperparameters using training data 
def prior_hyperparam_optim(model, likelihood, train_t, train_coeffs, num_train_iter, lr = 0.1, scheduler_gamma = 0.999, verbose = False):
    if (train_t is None) or (train_t.numel() == 0):
        pass
    else:
        for name, param in model.named_parameters():
            param.requires_grad = True
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
def best_update_sigma_grad_opt(df_real_update, df_real_check, scale_df, inst_id, model, likelihood, x_scale_factor, basis_funcs, y, num_train_iter,\
                          lr = 0.1, scheduler_gamma = 0.999, init_update_sigma = 1.0, use_relu = True, 
                          verbose = False, per_compute_verbose = False):
    
    lr = lr
    scheduler_gamma = scheduler_gamma
    model.log_update_sigma = torch.nn.Parameter(torch.tensor(math.log(init_update_sigma)))
    
    for name, param in model.named_parameters():
        param.requires_grad = False
    model.log_update_sigma.requires_grad = True
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # Includes GaussianLikelihood parameters
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = scheduler_gamma)

    loss_array = torch.zeros(num_train_iter)

    for i in tqdm(range(num_train_iter), disable = not verbose):
        current_update_sigma = model.update_sigma.detach().clone()
        optimizer.zero_grad()

        loss = compute_post_pred_error_implicit_sigma(df_real_update, df_real_check, scale_df, inst_id, model, likelihood,\
                                                  x_scale_factor, basis_funcs, \
                                                      target_name="Spend", use_relu = use_relu, verbose = per_compute_verbose)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_array[i] = float(loss.detach().clone())
        try:
            dist = model.dot_prod_posterior(torch.arange(model.max_time).float())
            full_mean, full_covar = dist.loc, torch.flatten(dist.lazy_covariance_matrix.to_dense())
            if torch.isnan(full_mean).any() or torch.isnan(full_covar).any():
                raise exception("Nans in posterior")
        except:
            model.log_update_sigma = torch.nn.Parameter(torch.tensor(math.log(current_update_sigma)))
            break
        gc.collect()
        torch.cuda.empty_cache()

    return loss_array
