import numpy as np
import math
import torch
import gpytorch
import pandas as pd
from .gen_funcs import *
from .utils import *

from gpytorch import settings
import warnings
from gpytorch.utils.warnings import GPInputWarning
from gpytorch.utils.generic import length_safe_zip
from gpytorch.models import ExactGP
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal

# Returns an instance of DotProductPredictionStrategy
def dot_product_prediction_strategy(update_dot_prod_vecs, update_labels, update_sigma):
    return DotProductPredictionStrategy(update_dot_prod_vecs, update_labels, update_sigma)

class DotProductPredictionStrategy(object):
    """
    A class to implement the update rule for vector valued GPs with dot product samples.
    This is a part of the modeling of observations y_j(t_j) = sum_{i=1}^F coeff_i(t)V_{j,i} + epsilon,
    given vectors V_j, where epsilon is iid Gaussian scalar noise, and the vector-valued map 
    t -> (coeff_i(t))_i represents the vector-valued GP

    Attributes
    ----------
    update_dot_prod_vecs : tensor
        Tensor of vectors with which dot products were taken, with vectors in the last dimension - V in our notation
    update_labels : tensor
        A one dimensional tensor of observed outputs y
    update_sigma : float
        The std dev of the signal used for the update rule, balances between faith in prior and faith in signal
    vec_dim: int
        Stores the number of basis functions
    num_update: int
        Stores the number of update values

    Methods
    -------
    phi_update_test(test_test_covar, test_update_covar, update_update_covar, f_d)
        Computes a technical qty phi used by the update rule

    exact_prediction(joint_mean, joint_covar)
        Returns the posterior mean and covar given the joint prior mean and covar for the test and update values

    exact_predictive_mean(test_mean, update_mean, test_test_covar, test_update_covar, update_update_covar, f_d)
        Helper function for exact_prediction for computing the posterior mean

    exact_predictive_covar(test_test_covar, test_update_covar, update_update_covar, f_d)
        Helper function for exact_prediction for computing the posterior covar
    """
    
    def __init__(self, update_dot_prod_vecs, update_labels, update_sigma):
        """
        Parameters
        ----------
        update_dot_prod_vecs : tensor
            Tensor of vectors with which dot products were taken, with vectors in the last dimension - V in our notation
        update_labels : tensor
            A one dimensional tensor of observed outputs y
        update_sigma : float
            The std dev of the signal used for the update rule, balances between faith in prior and faith in signal
        """
        
        self.update_dot_prod_vecs = update_dot_prod_vecs
        self.update_labels = update_labels
        self.update_sigma = update_sigma
        assert(self.update_dot_prod_vecs.shape[:-1] == update_labels.shape, "Shape of update_dot_prod_vecs {} not \
            the compatible with the shape of update labels {}".format(self.update_dot_prod_vecs.shape, update_labels.shape))
        
        self.vec_dim = self.update_dot_prod_vecs.shape[-1]
        self.num_update = self.update_labels.numel()
        
    def get_fantasy_strategy(self, inputs, targets, full_inputs, full_targets, full_output, **kwargs):
        raise NotImplementedError("Fantasy observation updates not yet supported for updates using dot products")

    # Computes phi, a technical (num_update, num_update) "scaling matrix" used in the update rule
    def phi_update_test(self, test_test_covar, test_update_covar, update_update_covar, f_d, test_test_covar_root):
        """
        Helper for exact_prediction
        """
        
        Idm = torch.ones(self.num_update)

        # We compute (A B_inv A) using (R^T R) for R = L.solve(A) and B = LL^T, where R = L_tt_inv_tu here
        # This maintains psd nature despite numerical instability
        L_tt_inv_tu = torch.linalg.solve_triangular(test_test_covar_root,\
                                                    test_update_covar, upper=False)
        
        update_given_test_covar = update_update_covar + (-1)*L_tt_inv_tu.transpose(-1,-2) @ L_tt_inv_tu

        # Same cholesky -> L then R = L.solve(A) then R^T R as above
        inter = torch.linalg.cholesky((self.update_sigma**2)*Idm + f_d.transpose(-1,-2) @ update_given_test_covar @ f_d)
        L_inter_inv_fd_ugt = torch.linalg.solve_triangular(inter,\
                                                           (f_d.transpose(-1,-2) @ update_given_test_covar), upper=False)
        
        update_sigma_1 = update_given_test_covar + (-1)*L_inter_inv_fd_ugt.transpose(-1,-2) @ L_inter_inv_fd_ugt

        return Idm + (-1/self.update_sigma**2)* f_d.transpose(-1,-2) @ update_sigma_1 @ f_d

    # Returns the posterior mean and covariance given a joint mean and covariance for test inputs and posterior update inputs
    def exact_prediction(self, joint_mean, joint_covar):
        """
        Returns the posterior mean and covariance given a joint mean and covariance for test inputs and posterior update inputs
        
        Parameters
        ----------
        joint_mean : tensor
            The joint mean for the test updates and update inputs
        joint_covar : tensor
            The joint covariance for the test updates and update inputs
        """
        
        # Split the joint mean and joint covariance into test and update based terms
        slice_update_test = self.num_update*self.vec_dim
        test_mean = (joint_mean[..., slice_update_test :].unsqueeze(-1))
        update_mean = (joint_mean[..., : slice_update_test].unsqueeze(-1))
        test_test_covar = (joint_covar[..., slice_update_test :, slice_update_test :]).to_dense()
        update_update_covar = (joint_covar[..., : slice_update_test, : slice_update_test]).to_dense()
        test_update_covar = (joint_covar[..., slice_update_test :, : slice_update_test]).to_dense()

        # Set up and arrange the vectors given for the dot product based posterior update
        f_d = torch.zeros([self.num_update, self.num_update*self.vec_dim])
        for i in range(self.num_update):
            f_d[i, i*self.vec_dim: (i+1)*self.vec_dim] = self.update_dot_prod_vecs[i, :]       
        f_d = f_d.transpose(-1,-2)

        test_test_covar_root = torch.linalg.cholesky(test_test_covar)
        return (
            self.exact_predictive_mean(test_mean, update_mean, test_test_covar, test_update_covar, 
                                       update_update_covar, f_d, test_test_covar_root).squeeze(),
            self.exact_predictive_covar(test_test_covar, test_update_covar, update_update_covar, f_d, test_test_covar_root)
        )

    # Returns posterior mean
    def exact_predictive_mean(self, test_mean, update_mean, test_test_covar, test_update_covar, update_update_covar, f_d, test_test_covar_root):
        """
        Helper for exact_prediction
        """
        
        phi = self.phi_update_test(test_test_covar, test_update_covar, update_update_covar, f_d, test_test_covar_root)

        # Difference between actual output and mean output
        delta_y = self.update_labels.reshape(self.update_labels.shape[-1], 1) - f_d.transpose(-1,-2) @ update_mean

        posterior_covar = self.exact_predictive_covar(test_test_covar, test_update_covar, update_update_covar, f_d)
        post_mean_intermediate_1 = torch.linalg.solve_triangular(test_test_covar,\
                            test_update_covar @ f_d @ phi @ delta_y, upper = False)
        post_mean_intermediate_2 = torch.linalg.solve_triangular(test_test_covar.transpose(-1,-2),\
                            post_mean_intermediate_1, upper = False)
        posterior_mean = test_mean + (1/self.update_sigma**2)*posterior_covar @ post_mean_intermediate_2

        return posterior_mean

    # Returns posterior covariance
    def exact_predictive_covar(self, test_test_covar, test_update_covar, update_update_covar, f_d, test_test_covar_root):
        """
        Helper for exact_prediction
        """
        
        Idm = torch.ones(self.num_update)

        # Compute phi and an intermediate qty designed to keep computed matrices psd despite numerical instability
        phi = self.phi_update_test(test_test_covar, test_update_covar, update_update_covar, f_d, test_test_covar_root)
        L_tt_inv_tu_fd = torch.linalg.solve_triangular(test_test_covar_root, (test_update_covar @ f_d), upper=False)

        # Phi may not be easily invertible, although in practice it typically seems to be
        try:
            phi_inv = torch.linalg.inv(phi)
        except:
            print("Unable to invert phi with eigvals {}".format(torch.linalg.eigvals(phi)))

        # Compute another intermediate qty designed to keep computed matrices psd despite numerical instability
        inter_2 = self.update_sigma**2*phi_inv +  L_tt_inv_tu_fd.transpose(-1, -2) @L_tt_inv_tu_fd
        inter_2_root = torch.linalg.cholesky(inter_2)
        L_inter_2_inv_fd_tu = torch.linalg.solve_triangular(inter_2_root,\
                                                            (f_d.transpose(-1, -2) @ test_update_covar.transpose(-1, -2)), upper = False)
        
        posterior_covar = test_test_covar + (-1)*L_inter_2_inv_fd_tu.transpose(-1,-2) @ L_inter_2_inv_fd_tu
        
        return posterior_covar