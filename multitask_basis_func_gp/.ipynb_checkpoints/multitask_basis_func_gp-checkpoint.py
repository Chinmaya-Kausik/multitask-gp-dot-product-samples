import numpy as np
import math
import torch
import gpytorch
from .gen_funcs import *
from .basis_funcs import *
from .utils import *
from .dot_product_prediction_strategy import *

from gpytorch import settings
import warnings
from gpytorch.utils.warnings import GPInputWarning
from gpytorch.utils.generic import length_safe_zip
from gpytorch.models import ExactGP
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel
from linear_operator import to_dense, to_linear_operator

# We will use the simplest form of GP model, exact inference
class MultitaskDotProdUpdateGPModel(ExactGP):
    """
    A class that implements a custom GP model for vector valued GPs alpha with dot product samples.
    This is a part of the modeling of observations y(t_j) = sum_{i=1}^F alpha_i(t)V_{j,i} + epsilon,
    given vectors V_j, where epsilon is iid Gaussian scalar noise, and the vector-valued map 
    t -> (coeff_i(t))_i represents the vector-valued GP.
    
    Accepts two kinds of data: 
        1. Direct training data with input values t_j and output vectors alpha_i(t_j)
        2. Dot product update data with input values t_j, vectors V_j for taking dot products with and outputs y(t_j)
    We feed the first kind at initialization, while the second kind is initialized as None and updated manually later.

    Attributes
    ----------
    direct_train_inputs: Tensor
        Inputs to the vector valued GP of coefficients for direct training of the GP
    direct_train_vecs: Tensor
        Coefficient vectors corresponding to direct_train_inputs, used for direct training of the GP
    update_inputs: tensor
        Inputs to the vector valued GP for posterior update, where the first dimension is always 1, t in our notation
    update_dot_prod_vecs : Tensor
        Tensor of vectors with which dot products were taken, with vectors in the last dimension - V in our notation
    update_labels : Tensor
        Observed outputs y for posterior update
    update_sigma : float
        Std dev of the signal used for the update rule, balances between faith in prior and faith in signal
    mean_module: gpytorch.means.MultitaskMean
        Multitask mean instance for the mean of the vector-valued GP of coefficients
    covar_module: gpytorch.means.MultitaskKernel
        Multitask kernel instance for the covar kernel of the vector-valued GP of coefficients

    Methods
    -------
    forward(t)
        Returns the prior distribution before using direct training data
    dot_prod_posterior(*args, **kwargs)
        Returns the posterior distribution for vectors after using both training and posterior update data
    """
    
    def __init__(self, direct_train_inputs, direct_train_vecs, vec_dim, likelihood, update_sigma, mean_module, covar_module):
        """
        Parameters
        --------
        direct_train_inputs: Tensor
            Inputs to the vector valued GP of coefficients for direct training of the GP
        direct_train_vecs: Tensor
            Coefficient vectors corresponding to direct_train_inputs, used for direct training of the GP
        vec_dim: int
            Dimension of the vectors
        update_inputs: tensor
            Inputs to the vector valued GP for posterior update, where the first dimension is always 1, t in our notation
        update_dot_prod_vecs : Tensor
            Tensor of vectors with which dot products were taken, with vectors in the last dimension - V in our notation
        update_labels : Tensor
            Observed outputs y for posterior update
        update_sigma : float
            Std dev of the signal used for the update rule, balances between faith in prior and faith in signal
        mean_module: gpytorch.means.MultitaskMean
            Multitask mean instance for the mean of the vector-valued GP of coefficients
        covar_module: gpytorch.means.MultitaskKernel
            Multitask kernel instance for the covar kernel of the vector-valued GP of coefficients
        """
        direct_train_inputs_tensor = make_nonzeroD_tensor(direct_train_inputs)
        direct_train_vecs_tensor = make_nonzeroD_tensor(direct_train_vecs)
        super(MultitaskDotProdUpdateGPModel, self).__init__(direct_train_inputs, direct_train_vecs, likelihood)
        
        self.update_inputs = torch.tensor([], dtype = torch.float)
        self.update_dot_prod_vecs = torch.tensor([], dtype = torch.float)
        self.update_labels = torch.tensor([], dtype = torch.float)
        self.update_sigma = torch.nn.Parameter(torch.tensor(update_sigma, dtype = torch.float))
        
        if (direct_train_vecs_tensor.numel() > 0):
            assert(vec_dim == direct_train_vecs.shape[-1], "Vector dimension vec_dim = {} does not match\
            dimension of training vectors = {}".format(vec_dim, direct_train_vecs.shape[-1]))
            
        self.vec_dim = vec_dim
        self.mean_module = mean_module
        self.covar_module = covar_module

    # Returns the prior distribution before using direct training data
    def forward(self, t):
        """
        Parameters
        ----------
        t: float or Tensor
            The input to the vector-valued GP

        Returns
        ---------
        The prior distribution of vectors before using direct training data
        """
        
        mean_t = self.mean_module(t)
        covar_t = self.covar_module(t)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_t, covar_t)

    def __call__(self, *args, **kwargs):
        args = [make_nonzeroD_tensor(i) for i in args]
        return super(MultitaskDotProdUpdateGPModel, self).__call__(*args, **kwargs)

    # Returns the posterior distribution after using both training and posterior update data
    def dot_prod_posterior(self, *args, **kwargs):
        """
        Parameters
        ----------
        args, kwargs
            Typically just the input at which one wants to evaluate the vector valued GP

        Returns
        ---------
        The posterior distribution of vectors after using both training and posterior update data
        """
        # Make everything a 1D or higher order tensor
        self.update_inputs = make_nonzeroD_tensor(self.update_inputs)
        self.update_dot_prod_vecs = make_nonzeroD_tensor(self.update_dot_prod_vecs)
        self.update_labels = make_nonzeroD_tensor(self.update_labels)
        
        # If no update information, just use prior
        if ((self.update_dot_prod_vecs.numel()==0) or (self.update_labels.numel()==0) or (self.update_inputs.numel()==0)):
            return self.__call__(*args, **kwargs)
        # Code for posterior
        else:
            
            # Processing update_inputs in the gpytorch style
            if (self.update_inputs.ndimension() == 1):
                update_inputs = [self.update_inputs.unsqueeze(-1)]
            else:
                update_inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in self.update_inputs]\
                    if self.update_inputs.numel()>0 else []
            inputs = [make_nonzeroD_tensor(i) for i in args]
            inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in inputs]
            
            # Checking if using eval mode on training data by accident
            if settings.debug.on():
                if all(torch.equal(update_input, input) for update_input, input in length_safe_zip(update_inputs, inputs)):
                    warnings.warn(
                        "The input matches the stored training data. Did you forget to call model.train()?",
                        GPInputWarning,
                    )
            
            # Create the update strategy for getting posteriors
            self.update_strategy = dot_product_prediction_strategy(
                update_dot_prod_vecs = self.update_dot_prod_vecs,
                update_labels=self.update_labels,
                update_sigma=self.update_sigma
            )
    
            # Concatenate the input to the training input
            full_inputs = []
            batch_shape = update_inputs[0].shape[:-2]
            for update_input, input in length_safe_zip(update_inputs, inputs):
                if batch_shape != update_input.shape[:-2]:
                    batch_shape = torch.broadcast_shapes(batch_shape, update_input.shape[:-2])
                    update_input = update_input.expand(*batch_shape, *update_input.shape[-2:])
                if batch_shape != input.shape[:-2]:
                    batch_shape = torch.broadcast_shapes(batch_shape, input.shape[:-2])
                    update_input = update_input.expand(*batch_shape, *update_input.shape[-2:])
                    input = input.expand(*batch_shape, *input.shape[-2:])
                full_inputs.append(torch.cat([update_input, input], dim=-2))
    
            # Get the joint distribution for training/test data
            full_output = self.__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix
    
            # Determine the shape of the joint distribution
            batch_shape = full_output.batch_shape
            joint_shape = full_output.event_shape
            tasks_shape = joint_shape[1:]  # For multitask learning
            test_shape = torch.Size([joint_shape[0] - self.update_labels.shape[0], *tasks_shape])
    
            # Make the prediction
            with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
                (
                    predictive_mean,
                    predictive_covar,
                ) = self.update_strategy.exact_prediction(full_mean, full_covar)
    
            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
            return full_output.__class__(predictive_mean, predictive_covar)

    def large_input_dot_prod_posterior_means(self, max_time, verbose = False):

        # Make everything a 1D or higher order tensor
        self.update_inputs = make_nonzeroD_tensor(self.update_inputs)
        self.update_dot_prod_vecs = make_nonzeroD_tensor(self.update_dot_prod_vecs)
        self.update_labels = make_nonzeroD_tensor(self.update_labels)
        
        mean_list = torch.zeros(self.update_inputs.squeeze().shape)
        current_dist = self.__call__(torch.arange(max_time).float())
        for idx, current_input in tqdm(enumerate(self.update_inputs.squeeze()), disable = not verbose):
            current_time = int(current_input)
            current_mean = current_dist.mean[current_time]
            current_dot_prod_vec = self.update_dot_prod_vecs[idx]
            current_label = self.update_labels.squeeze()[idx]
            mean_list[idx] = self.update_dot_prod_vecs[idx] @ current_mean
            current_dist = self.large_input_dot_prod_posterior_helper(current_dist, current_input, current_dot_prod_vec, current_label)
        return mean_list

    def large_input_dot_prod_posterior_helper(self, previous_dist, new_input, new_dot_prod_vec, new_label):
        update_inputs = make_nonzeroD_tensor(new_input)
        update_dot_prod_vecs = make_nonzeroD_tensor(new_dot_prod_vec).unsqueeze(-2)
        update_labels = make_nonzeroD_tensor(new_label)
            
        # Create the update strategy for getting posteriors
        update_strategy = dot_product_prediction_strategy(
            update_dot_prod_vecs = update_dot_prod_vecs,
            update_labels= update_labels,
            update_sigma= self.update_sigma
        )

        mean_24, covar_24 = previous_dist.loc, previous_dist.lazy_covariance_matrix.to_dense()
        update_time = int(update_inputs[0])
        full_mean = torch.cat((mean_24[update_time*45: (update_time+1)*45], mean_24), 0)
        addl_covar = torch.cat((covar_24[update_time*45: (update_time+1)*45, update_time*45: (update_time+1)*45],\
                               covar_24[update_time*45: (update_time+1)*45, :]), 1)
        covar_24_modified = torch.cat((covar_24[:, update_time*45: (update_time+1)*45], covar_24), 1)
        full_covar = torch.cat((addl_covar, covar_24_modified), 0)

        # Determine the shape of the joint distribution
        test_shape = previous_dist.event_shape
        
        # Make the prediction
        with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
            (
                predictive_mean,
                predictive_covar,
            ) = update_strategy.exact_prediction(full_mean, to_linear_operator(full_covar))

        # Reshape predictive mean to match the appropriate event shape
        predictive_mean = predictive_mean.view(*test_shape).contiguous()
        return previous_dist.__class__(predictive_mean, predictive_covar)


    # Helper function to return the joint mean and covar of test and update data
    def full_mean_covar(self, *args, **kwargs):
        # Processing update_inputs in the gpytorch style
        if (self.update_inputs.ndimension() == 1):
            update_inputs = [self.update_inputs.unsqueeze(-1)]
        else:
            update_inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in self.update_inputs]\
                if self.update_inputs.numel()>0 else []
        inputs = [make_nonzeroD_tensor(i) for i in args]
        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in inputs]
        
        # Checking if using eval mode on training data by accident
        if settings.debug.on():
            if all(torch.equal(update_input, input) for update_input, input in length_safe_zip(update_inputs, inputs)):
                warnings.warn(
                    "The input matches the stored training data. Did you forget to call model.train()?",
                    GPInputWarning,
                )
        
        # Create the update strategy for getting posteriors
        self.update_strategy = dot_product_prediction_strategy(
            update_dot_prod_vecs = self.update_dot_prod_vecs,
            update_labels=self.update_labels,
            update_sigma=self.update_sigma
        )

        # Concatenate the input to the training input
        full_inputs = []
        batch_shape = update_inputs[0].shape[:-2]
        for update_input, input in length_safe_zip(update_inputs, inputs):
            if batch_shape != update_input.shape[:-2]:
                batch_shape = torch.broadcast_shapes(batch_shape, update_input.shape[:-2])
                update_input = update_input.expand(*batch_shape, *update_input.shape[-2:])
            if batch_shape != input.shape[:-2]:
                batch_shape = torch.broadcast_shapes(batch_shape, input.shape[:-2])
                update_input = update_input.expand(*batch_shape, *update_input.shape[-2:])
                input = input.expand(*batch_shape, *input.shape[-2:])
            full_inputs.append(torch.cat([update_input, input], dim=-2))

        # Get the joint distribution for training/test data
        full_output = self.__call__(*full_inputs, **kwargs)
        if settings.debug().on():
            if not isinstance(full_output, MultivariateNormal):
                raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
        full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

# Instantiate the DotProductUpdate model to incorporate basis functions
class MultitaskBasisFuncGPModel(MultitaskDotProdUpdateGPModel):
    """
     A class that implements a custom GP model for updating y(t,d) curves with constraints on behavior wrt d.
     We model y(t,d) = sum_{i=1}^F coeff_i(t)f_i(d) + epsilon, where epsilon is iid Gaussian scalar noise, 
     f_i are deterministic functions with desirable properties compatible with our constraints
     (typically monotonicity), and the vector-valued map t -> (coeff_i(t))_i represents the vector-valued GP.
     
     Builds off of MultitaskDotProdUpdateGPModel, needs chosen basis functions to be fed at initialization.
     Accepts two kinds of data: 
         1. Direct training data with input values t_j and corresponding coefficients coeff_i(t_j)
         2. Dot product update data with input values t_j, basis inputs d_j and outputs y(t_j)
     We feed the first kind at initialization, while the second kind is initialized as None and updated manually later.


    Attributes
    ----------
    direct_train_inputs: Tensor
        Inputs to the vector valued GP of coefficients for direct training of the GP
    direct_train_coeffs: Tensor
        Coefficient vectors corresponding to direct_train_inputs, used for direct training of the GP
    update_inputs: tensor
        Inputs to the vector valued GP for posterior update, where the first dimension is always 1, t in our notation
    update_basis_inputs : Tensor
        Tensor of vectors with which dot products were taken, with vectors in the last dimension - V in our notation
    update_labels : Tensor
        Observed outputs y for posterior update
    update_sigma : float
        Std dev of the signal used for the update rule, balances between faith in prior and faith in signal
    basis_funcs : BasisFuncs
        BasisFuncs instance representing the basis functions used
    mean_module: gpytorch.means.MultitaskMean
        Multitask mean instance for the mean of the vector-valued GP of coefficients
    covar_module: gpytorch.means.MultitaskKernel
        Multitask kernel instance for the covar kernel of the vector-valued GP of coefficients

    Methods
    -------
    dot_prod_posterior(*args, **kwargs)
        Returns the posterior mean of kpis after using both training and posterior update data, overwrites the version for the
        superclass to incorporate basis function inputs and take means
    """

    # Initialize as per MultitaskDotProdUpdateGPModel and store basis_funcs
    def __init__(self, direct_train_inputs, direct_train_coeffs, vec_dim, likelihood, update_sigma, basis_funcs, mean_module, covar_module):
        """
        Parameters
        ----------
        direct_train_inputs: Tensor
            Inputs to the vector valued GP of coefficients for direct training of the GP
        direct_train_coeffs: Tensor
            Coefficient vectors corresponding to direct_train_inputs, used for direct training of the GP
        update_sigma : float
            Std dev of the signal used for the update rule, balances between faith in prior and faith in signal
        basis_funcs : BasisFuncs
            BasisFuncs instance representing the basis functions used
        mean_module: gpytorch.means.MultitaskMean
            Multitask mean instance for the mean of the vector-valued GP of coefficients
        covar_module: gpytorch.means.MultitaskKernel
            Multitask kernel instance for the covar kernel of the vector-valued GP of coefficients
        """
        
        super(MultitaskBasisFuncGPModel, self).__init__(direct_train_inputs, direct_train_coeffs, vec_dim, likelihood, update_sigma, 
                                                       mean_module, covar_module)
        
        self.update_basis_inputs = torch.tensor([], dtype = torch.float)
        self.basis_funcs = basis_funcs

    # Modify dot_prod_posterior from MultitaskDotProdUpdateGPModel to use update_basis_inputs and basis_funcs
    # instead of update_dot_prod_vecs
    def dot_prod_posterior(self, *args, **kwargs):
        """
        Parameters
        ----------
        args, kwargs
            Typically just the input at which one wants to evaluate the vector valued GP

        Returns
        ---------
        The posterior distribution of the GP of coeffs after using both training and posterior update data
        """
        
        self.update_dot_prod_vecs = make_nonzeroD_tensor(self.basis_funcs(self.update_basis_inputs))
        
        # Calls the dot prod posterior of the vector valued GP super class above
        return super(MultitaskBasisFuncGPModel, self).dot_prod_posterior(*args, **kwargs)

    def dot_prod_posterior_mean_vals(self, inputs, basis_inputs):
        """
        Parameters
        --------
        inputs: Tensor
            inputs to the GP
        basis_inputs: Tensor
            Inputs to the basis functions for taking dot products with the GP

        Returns
        ----------
            The posterior means of GP(input) dot basis_funcs(basis_input) as we range over zip(inputs, basis_inputs)
            Here, the GP is our GP of coeffs represented by this model.
        """
        
        inputs = make_nonzeroD_tensor(inputs)
        basis_inputs = make_nonzeroD_tensor(basis_inputs)

        coeff_model_at_inputs = self.dot_prod_posterior(inputs)
        coeff_likelihood_at_inputs = self.likelihood(coeff_model_at_inputs)

        basis_func_outputs = self.basis_funcs(basis_inputs)
        mean_post_coeffs = coeff_model_at_inputs.mean.squeeze().detach().numpy()
        mean_pred_y = basis_func_outputs @ mean_post_coeffs

        # Set up and arrange the vectors given for the dot product based posterior update
        num_basis_inputs = basis_inputs.shape[-1]
        f_d = torch.zeros([self.num_update, self.num_update*self.vec_dim])
        for i in range(self.num_update):
            f_d[i, i*self.vec_dim: (i+1)*self.vec_dim] = self.update_dot_prod_vecs[i, :]       
        f_d = torch.tensor(f_d, dtype = torch.float).transpose(-1,-2)
        return pred_y

    def large_input_dot_prod_posterior_means(self, max_time, verbose = False):
        self.update_dot_prod_vecs = make_nonzeroD_tensor(self.basis_funcs(self.update_basis_inputs))

        return super(MultitaskBasisFuncGPModel, self).large_input_dot_prod_posterior_means(max_time, verbose = verbose)
        
        