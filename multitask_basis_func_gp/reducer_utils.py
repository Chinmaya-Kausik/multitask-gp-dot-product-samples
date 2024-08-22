import os
import numpy as np
import math
import random
import torch
import gpytorch
import pandas as pd
from .gen_funcs import discrete_sigmoid_gap_based
from .basis_funcs import BasisFuncs
from .utils import *
from .visual_utils import *
from .multitask_basis_func_gp import MultitaskBasisFuncGPModel
from .dot_product_prediction_strategy import *

from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Various lists for model prep modes, used in prep_model_likelihood
custom_mean_prep_modes = ["exc_mean_post_only", "exc_mean_post_only_hyp_opt"]
hyp_opt_prep_modes = ["exc_post", "post_only_hyp_opt", "exc_mean_post_only_hyp_opt", "no_eval"]
set_train_to_none_later_prep_modes = ["post_only_hyp_opt", "exc_mean_post_only_hyp_opt"]
no_eval_prep_modes = ["no_eval"]

"""
Note on model_prep_mode and inference_mode:
The model_prep_mode values only affect how the model is prepped. 
The inference_mode parameter controls whether we use the prior, posterior or just the original excalibur data to make predictions.

We list the various model_prep_mode values here:
1. exc_post: Main mode. Prior from excalibur data, posterior using real data.
2. post_only: Zero prior and standard covariance, posterior using real data.
3. post_only_hyp_opt: Zero prior and standard covariance followed by hyperparameter optimization using excalibur data, posterior using real data.\
    No exact inference using excalibur data.
4. exc_mean_post_only: Custom mean from excalibur data but standard covariance, posterior using real data.
5. exc_mean_post_only_hyp_opt: Custom mean from excalibur data but standard covariance followed by hyperparameter optimization using excalibur data,\
    posterior using real data. No exact inference using excalibur data.
6. no_eval: Similar to but subtly different from post_only_hyp_opt. \
    Hyperparameter optimization is done with prior data but then train mode is used instead of eval mode, so excalibur data is not used for exact inference.\
    The difference is this - in post_only_hyp_opt, we set both train_t, train_coeffs to None, while in exc_post they would come from the excalibur curves. In no_eval,\
    we don't call the function to perform inference using train_t, train_coeffs at all.

We list the various inference_mode values here:
1. exc: The model is completely ignored, and the excalibur data is used to make predictions
2. prior: The model's prior is used to make predictions
3. post: The model's posterior is used to make predictions.
"""

# Class to output a custom mean that evaluates a function f(x) to produce the mean
class CustomMean(gpytorch.means.Mean):
    def __init__(self, f):
        super().__init__()
        self.custom_function = f
    def forward(self, x):
        return self.custom_function(x)

# Outputs the custom mean class given by the mapping train_t to train_coeffs
# Needs all training inputs to be present
def exc_mean_coeff(train_t, train_coeffs, max_time):
    if not train_t.numel() == max_time:
        raise ValueError("train_t must be a tensor of integers from 0 to max_time")
    def f(time):
        time = time.int()
        if time.numel() == 1:
            return torch.tensor(train_coeffs[time], dtype=torch.float)
        else:
            return torch.tensor(train_coeffs[time].squeeze(), dtype=torch.float)
    
    return CustomMean(f)

# Prepares the model and likelihood with a prior initialized by df_prev, according to the chosen model_prep_mode
# df_real is used to determine the scaling of duals
def prep_model_likelihood(df_prev, df_real, scale_df, inst_id, basis_funcs, rank, max_time, kpi, train_t, use_relu,\
                prior_lr, prior_sched_gamma, num_prior_train_iter, \
                    dual_scale_factor, grid_size, model_prep_mode = "exc_post", has_safe_posterior = False, verbose = False):

    num_basis = len(basis_funcs.func_list)
    y_scale = float(df_real[df_real["InstanceId"] == inst_id][kpi].to_numpy().mean())
    y_scale = y_scale if (y_scale > 0) else float(df_prev[df_prev["InstanceId"] == inst_id][kpi].to_numpy().mean())
    y_scale = y_scale if (y_scale > 0) else 10**(-4)
    init_update_sigma = y_scale

    train_t, train_coeffs = gen_true_coeffs(df_prev, scale_df, inst_id, train_t, dual_scale_factor, grid_size, basis_funcs = basis_funcs,\
                                            kpi=kpi, grid = "uniform")
    # Choose mean and covariance module for the vector-valued coeff GP
    if model_prep_mode in custom_mean_prep_modes:
        mean_module = exc_mean_coeff(train_t, train_coeffs, max_time)
    else:
        mean_module = gpytorch.means.MultitaskMean(ConstantMean(), num_tasks=num_basis)
    covar_module = gpytorch.kernels.MultitaskKernel(MaternKernel(), num_tasks=num_basis, rank=rank)

    if model_prep_mode not in hyp_opt_prep_modes:
        train_t, train_coeffs = None, None
    # Set up "likelihood" (essentially takes GP output distribution f(x) and gives an output distribution by adding appropriate noise)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_basis)

    # Initialize model
    model = MultitaskBasisFuncGPModel(train_t, train_coeffs, num_basis, likelihood, init_update_sigma,\
                                      basis_funcs, max_time, mean_module, covar_module, has_safe_posterior = has_safe_posterior)

    model.train()
    likelihood.train() 

    # Optimize hyperparameters for the prior
    prior_hyperparam_optim(model, likelihood, train_t, train_coeffs, num_prior_train_iter, lr = prior_lr, scheduler_gamma = prior_sched_gamma,
                           verbose=verbose)

    if model_prep_mode in set_train_to_none_later_prep_modes:
        model.train_inputs, model.train_targets = None, None

    if model_prep_mode not in no_eval_prep_modes:
        model.eval()
        likelihood.eval()
    
    return model, likelihood

# Get the optimal update_sigma using a gridsearch
def get_optimal_sigma_grid_search(df_prev, df_real, scale_df, inst_id, basis_funcs, rank, max_time, kpi, train_t, use_relu,\
                prior_lr, prior_sched_gamma, num_prior_train_iter, \
                    dual_scale_factor, grid_size, y_scalings, sigma_grid_size, min_y_scaling_power, max_y_scaling_power,\
                        model_prep_mode = "exc_post", has_safe_posterior = False, verbose = False):
    
    y_scale = float(df_real[df_real["InstanceId"] == inst_id][kpi].to_numpy().mean())
    y_scale = y_scale if (y_scale > 0) else float(df_prev[df_prev["InstanceId"] == inst_id][kpi].to_numpy().mean())
    y_scale = y_scale if (y_scale > 0) else 10**(-4)


    model, likelihood = prep_model_likelihood(df_prev, df_real, scale_df, inst_id, basis_funcs, rank, max_time, kpi, train_t, use_relu,\
                prior_lr, prior_sched_gamma, num_prior_train_iter, dual_scale_factor, grid_size,\
                    model_prep_mode = model_prep_mode, has_safe_posterior = has_safe_posterior, verbose = verbose)
    
    # Set up empty update data
    empty_tensor = torch.tensor([], dtype=torch.float)
    model.update_inputs, model.update_basis_inputs, model.update_labels = empty_tensor, empty_tensor, empty_tensor

    with torch.no_grad():
        # Find bets kpi scaling
        y_scaling = get_y_scaling(df_real, scale_df, inst_id, y_scale, model, likelihood, dual_scale_factor, basis_funcs, kpi=kpi, use_relu = use_relu,\
                        min_power = min_y_scaling_power, y_scalings = y_scalings, verbose = verbose)

        # Set up empty update data
        empty_tensor = torch.tensor([], dtype=torch.float)
        model.update_inputs, model.update_basis_inputs, model.update_labels = empty_tensor, empty_tensor, empty_tensor
        
        sigma_grid = torch.logspace(min_y_scaling_power, max_y_scaling_power, sigma_grid_size)*y_scale*y_scaling

        # Directly get update_sigma with min error on real data
        best_sigma = best_update_sigma_grid_search(None, df_real, scale_df, inst_id, model, likelihood,\
                        sigma_grid, dual_scale_factor, basis_funcs,\
                    kpi=kpi, use_relu = use_relu, verbose = verbose, per_compute_verbose = False)
    
    return best_sigma, sigma_grid, model.num_posterior_crashes, model.posterior_crash_sigmas

# Primarily for debugging, get the error arrays for training and testing data over the sigma_grid generated during gridsearch
def get_train_test_error_array(df_prev, df_real, df_real_test, scale_df, inst_id, basis_funcs, rank, max_time, kpi, train_t, use_relu,\
                prior_lr, prior_sched_gamma, num_prior_train_iter, \
                    dual_scale_factor, grid_size, y_scalings, sigma_grid_size, min_y_scaling_power, max_y_scaling_power,\
                        model_prep_mode = "exc_post", has_safe_posterior = False, verbose = False):
    
    y_scale = float(df_real[df_real["InstanceId"] == inst_id][kpi].to_numpy().mean())
    y_scale = y_scale if (y_scale > 0) else float(df_prev[df_prev["InstanceId"] == inst_id][kpi].to_numpy().mean())
    y_scale = y_scale if (y_scale > 0) else 10**(-4)


    model, likelihood = prep_model_likelihood(df_prev, df_real, scale_df, inst_id, basis_funcs, rank, max_time, kpi, train_t, use_relu,\
                prior_lr, prior_sched_gamma, num_prior_train_iter, dual_scale_factor, grid_size,\
                    model_prep_mode = model_prep_mode, has_safe_posterior = has_safe_posterior, verbose = verbose)
    
    # Set up empty update data
    empty_tensor = torch.tensor([], dtype=torch.float)
    model.update_inputs, model.update_basis_inputs, model.update_labels = empty_tensor, empty_tensor, empty_tensor

    with torch.no_grad():
        # Find best kpi scaling
        y_scaling = get_y_scaling(df_real, scale_df, inst_id, y_scale, model, likelihood, dual_scale_factor, basis_funcs, kpi=kpi, use_relu = use_relu,\
                        min_power = min_y_scaling_power, y_scalings = y_scalings, verbose = verbose)

        # Set up empty update data
        empty_tensor = torch.tensor([], dtype=torch.float)
        model.update_inputs, model.update_basis_inputs, model.update_labels = empty_tensor, empty_tensor, empty_tensor
        
        sigma_grid = torch.logspace(min_y_scaling_power, max_y_scaling_power, sigma_grid_size)*y_scale*y_scaling

        # Directly get update_sigma with min error on real data
        error_array = gen_error_array(None, df_real, scale_df, inst_id, model, likelihood,\
                                    sigma_grid, dual_scale_factor, basis_funcs,\
                                        kpi=kpi, use_relu = use_relu, verbose = verbose, per_compute_verbose = False)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        error_array_test = gen_error_array(df_real, df_real_test, scale_df, inst_id, model, likelihood,\
                                    sigma_grid, dual_scale_factor, basis_funcs,\
                                        kpi=kpi, use_relu = use_relu, verbose = verbose, per_compute_verbose = False)
    
    return sigma_grid, error_array, error_array_test, model.num_posterior_crashes, model.posterior_crash_sigmas

# Get the optimal update_sigma using a gridsearch
def get_optimal_sigma_grad_opt(df_prev, df_real, scale_df, inst_id, basis_funcs, rank, max_time, kpi, train_t, use_relu,\
                prior_lr, prior_sched_gamma, num_prior_train_iter, dual_scale_factor, grid_size, y_scalings,\
                    min_y_scaling_power, post_lr, post_sched_gamma, num_post_train_iter,\
                    model_prep_mode = "exc_post", has_safe_posterior = False, verbose = False):

    y_scale = float(df_real[df_real["InstanceId"] == inst_id][kpi].to_numpy().mean())
    y_scale = y_scale if (y_scale > 0) else float(df_prev[df_prev["InstanceId"] == inst_id][kpi].to_numpy().mean())
    y_scale = y_scale if (y_scale > 0) else 10**(-4)


    model, likelihood = prep_model_likelihood(df_prev, df_real, scale_df, inst_id, basis_funcs, rank, max_time, kpi, train_t, use_relu,\
                prior_lr, prior_sched_gamma, num_prior_train_iter, dual_scale_factor, grid_size,\
                    model_prep_mode = model_prep_mode, has_safe_posterior = has_safe_posterior, verbose = verbose)
    
    # Set up empty update data
    empty_tensor = torch.tensor([], dtype=torch.float)
    model.update_inputs, model.update_basis_inputs, model.update_labels = empty_tensor, empty_tensor, empty_tensor

    # Find bets kpi scaling
    y_scaling = get_y_scaling(df_real, scale_df, inst_id, y_scale, model, likelihood, dual_scale_factor, basis_funcs, kpi=kpi, use_relu = use_relu,\
                    min_power = min_y_scaling_power, y_scalings = y_scalings, verbose = verbose)
    init_update_sigma = y_scale*y_scaling*10**min_y_scaling_power

    # Set up empty update data
    empty_tensor = torch.tensor([], dtype=torch.float)
    model.update_inputs, model.update_basis_inputs, model.update_labels = empty_tensor, empty_tensor, empty_tensor

    best_update_sigma_grad_opt(None, df_real, scale_df, inst_id, model, likelihood,\
                    dual_scale_factor, basis_funcs, kpi, num_post_train_iter, post_lr, post_sched_gamma,\
                    init_update_sigma=init_update_sigma, use_relu=use_relu, verbose = verbose, per_compute_verbose = False)

    return model.update_sigma.detach(), model.num_posterior_crashes, model.posterior_crash_sigmas

# Make a grid of kpi predictions at time "time" over a scaled dual grid given by "scaled_dual_x"
# Probably should make this an unscaled grid?
def make_dual_grid_single_time(df_prev, df_real, scale_df, scaled_dual_x, inst_id, time, basis_funcs, rank, max_time, kpi, train_t, use_relu,\
                prior_lr, prior_sched_gamma, num_prior_train_iter, dual_scale_factor, grid_size, best_sigma,\
                    model_prep_mode = "exc_post", inference_mode = "post", has_safe_posterior = False, verbose = False):
    
    latest_date = df_real["Date"].max()

    if inference_mode == "exc":
        if (max_time == 7):
            scaled_dual_x, kpi_exc_preds, _ = get_orig_grid_dual_kpi(df_prev, scale_df, inst_id, time, dual_scale_factor, kpi=kpi)
            df_latest_date = df_real[df_real["Date"] == latest_date]
            kpi_exc_error = compute_excalibur_error(df_prev, df_latest_date, inst_id, kpi)
            return kpi_exc_preds, kpi_exc_error, 0, []
        else:
            scaled_dual_x, kpi_exc_preds, _ = get_orig_grid_dual_kpi(df_prev, scale_df, inst_id, time, dual_scale_factor, kpi=kpi)
            df_latest_date = df_real[df_real["Date"] == latest_date]
            kpi_exc_error = compute_excalibur_error_hourly(df_prev, df_latest_date, inst_id, kpi)
            return kpi_exc_preds, kpi_exc_error, 0, []
            

    model, likelihood = prep_model_likelihood(df_prev, df_real, scale_df, inst_id, basis_funcs, rank, max_time, kpi, train_t, use_relu,\
                prior_lr, prior_sched_gamma, num_prior_train_iter, dual_scale_factor, grid_size,\
                    model_prep_mode = model_prep_mode, has_safe_posterior = has_safe_posterior, verbose = verbose)
    
    with torch.no_grad():
        if inference_mode == "prior":
            kpi_preds = gen_kpi_vals_single_time(model, likelihood, basis_funcs, scaled_dual_x, time, kpi, mode = "prior",\
                                                use_relu = use_relu, verbose = verbose)

            df_latest_date = df_real[df_real["Date"] == latest_date]
            kpi_prior_pred_error = compute_prior_pred_error(df_latest_date, scale_df, inst_id, model, likelihood, dual_scale_factor,\
                                                            basis_funcs, kpi, use_relu = use_relu, verbose = verbose)
            
            return kpi_preds, kpi_prior_pred_error, model.num_posterior_crashes, model.posterior_crash_sigmas


        
        # Set update_sigma to best_sigma
        model.log_update_sigma = torch.nn.Parameter(torch.log(best_sigma))
        
        # Set up update data
        model.update_inputs, model.update_basis_inputs, model.update_labels = get_real_vals(df_real, scale_df, inst_id, dual_scale_factor, kpi)

        # Get kpi preds from posterior grid
        kpi_preds = gen_kpi_vals_single_time(model, likelihood, basis_funcs, scaled_dual_x, time, kpi, mode = "iter", use_relu = use_relu, verbose = verbose) 

        # Reset update data for error computation, just to be safe
        model.update_inputs, model.update_basis_inputs, model.update_labels = get_real_vals(df_real, scale_df, inst_id, dual_scale_factor, kpi)

        df_latest_date = df_real[df_real["Date"] == latest_date]
        df_before_latest_date = df_real[df_real["Date"] != latest_date]

        # Get error for latest date
        kpi_post_latest_error = compute_post_pred_error_implicit_sigma(df_before_latest_date, df_latest_date, scale_df, inst_id,\
                                                        model, likelihood, dual_scale_factor, basis_funcs, kpi, use_relu = use_relu, verbose = verbose)

    return kpi_preds, kpi_post_latest_error, model.num_posterior_crashes, model.posterior_crash_sigmas

# The wrapped up version of make_post_preds from utils
# Function to make posterior predictions on update_inputs from df_real_check while sequentially updating the posterior 
# Posterior is updated with the update_labels and update_basis_inputs seen so far from df_real_update and df_real_check before making the next prediction
# Model is initialized with training data from df_prev and update data from df_real_update
def make_test_preds(df_prev, df_real, df_real_test, scale_df, inst_id, basis_funcs, rank, max_time, kpi, train_t, use_relu,\
                prior_lr, prior_sched_gamma, num_prior_train_iter, dual_scale_factor, grid_size, best_sigma,\
                    model_prep_mode = "exc_post", inference_mode = "post", has_safe_posterior = False, verbose = False):
    
    latest_date = df_real["Date"].max()

    if inference_mode == "exc":
        if (max_time == 7):
            kpi_exc_preds = make_exc_preds(df_prev, df_real_test, inst_id, kpi)
            return kpi_exc_preds, 0, []
        else:
            kpi_exc_preds = make_exc_preds_hourly(df_prev, df_real_test, inst_id, kpi)
            return kpi_exc_preds, 0, []
            

    model, likelihood = prep_model_likelihood(df_prev, df_real, scale_df, inst_id, basis_funcs, rank, max_time, kpi, train_t, use_relu,\
                prior_lr, prior_sched_gamma, num_prior_train_iter, dual_scale_factor, grid_size,\
                    model_prep_mode = model_prep_mode, has_safe_posterior = has_safe_posterior, verbose = verbose)
    
    with torch.no_grad():
        if inference_mode == "prior":
            update_inputs, update_basis_inputs, update_labels = get_real_vals(df_real_test, scale_df, inst_id, dual_scale_factor, kpi = kpi)
            kpi_prior_preds = make_prior_preds(update_inputs, update_basis_inputs, model, likelihood,\
                                basis_funcs, kpi, use_relu, verbose = verbose)
            
            return kpi_prior_preds, model.num_posterior_crashes, model.posterior_crash_sigmas


        
        # Set update_sigma to best_sigma
        model.log_update_sigma = torch.nn.Parameter(torch.log(best_sigma))
        
        # Set up update data
        model.update_inputs, model.update_basis_inputs, model.update_labels = get_real_vals(df_real, scale_df, inst_id, dual_scale_factor, kpi)

        # Get kpi preds from posterior grid
        update_inputs, update_basis_inputs, update_labels = get_real_vals(df_real_test, scale_df, inst_id, dual_scale_factor, kpi = kpi)
        kpi_post_preds = make_post_preds(update_inputs, update_basis_inputs, update_labels, model, likelihood, dual_scale_factor,\
                            basis_funcs, kpi=kpi, use_relu = use_relu, verbose = verbose)

    return kpi_post_preds, model.num_posterior_crashes, model.posterior_crash_sigmas
    
# Get the mean squared error over df_real_test if the model is prepped using excalibur data from df_prev and real observations from df_real
def get_errors(df_prev, df_real, df_real_test, scale_df, inst_id, basis_funcs, rank, max_time, kpi, train_t, use_relu,\
                prior_lr, prior_sched_gamma, num_prior_train_iter, dual_scale_factor, grid_size, best_sigma,\
                    model_prep_mode = "exc_post", inference_mode = "post", has_safe_posterior = False, verbose = False):
    
    if inference_mode == "exc":
        if (max_time == 7):
            kpi_exc_error = compute_excalibur_error(df_prev, df_real_test, inst_id, kpi)
            return kpi_exc_error, 0, []
        else:
            kpi_exc_error = compute_excalibur_error_hourly(df_prev, df_real_test, inst_id, kpi)
            return kpi_exc_error, 0, []


    model, likelihood = prep_model_likelihood(df_prev, df_real, scale_df, inst_id, basis_funcs, rank, max_time, kpi, train_t, use_relu,\
                prior_lr, prior_sched_gamma, num_prior_train_iter, dual_scale_factor, grid_size,\
                    model_prep_mode = model_prep_mode, has_safe_posterior = has_safe_posterior, verbose = verbose)

    with torch.no_grad():
        if inference_mode == "prior":
            kpi_prior_pred_error = compute_prior_pred_error(df_real_test, scale_df, inst_id, model, likelihood, dual_scale_factor,\
                                                            basis_funcs, kpi, use_relu = use_relu, verbose = verbose)
            
            return kpi_prior_pred_error, model.num_posterior_crashes, model.posterior_crash_sigmas
        
        # Set update_sigma to best_sigma
        model.log_update_sigma = torch.nn.Parameter(torch.log(best_sigma))
        
        # Set up update data
        model.update_inputs, model.update_basis_inputs, model.update_labels = get_real_vals(df_real, scale_df, inst_id, dual_scale_factor, kpi)

        # Get error for df_real_test
        kpi_post_error = compute_post_pred_error_implicit_sigma(df_real, df_real_test, scale_df, inst_id,\
                            model, likelihood, dual_scale_factor, basis_funcs, kpi, use_relu = use_relu, verbose = verbose)

    return kpi_post_error, model.num_posterior_crashes, model.posterior_crash_sigmas