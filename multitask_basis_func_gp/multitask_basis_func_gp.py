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
from gpytorch.models.exact_prediction_strategies import prediction_strategy


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
    train_inputs: Tensor
            Inputs to the vector valued GP of coefficients for direct training of the GP
    train_targets: Tensor
        Coefficient vectors corresponding to train_inputs, used for direct training of the GP
    vec_dim: int
        Dimension of the vectors
    likelihood: gpytorch.likelihoods.Likelihood
        Likelihood instance to be used with the model (see GPyTorch documentation for more details)
    update_inputs: tensor
        Inputs to the vector valued GP for posterior update, where the first dimension is always 1, t in our notation
    update_dot_prod_vecs : Tensor
        Tensor of vectors with which dot products were taken, with vectors in the last dimension - V in our notation
    update_labels : Tensor
        Observed outputs y for posterior update
    log_update_sigma: torch.nn.Parameter
        Used to store the log of the update_sigma for gradient reasons
    update_sigma : float
        Std dev of the signal used for the update rule, balances between faith in prior and faith in signal
    mean_module: gpytorch.means.Mean
        Multitask mean instance for the mean of the vector-valued GP of coefficients
    covar_module: gpytorch.means.Kernel
        Multitask kernel instance for the covar kernel of the vector-valued GP of coefficients
    has_safe_posterior: bool
        Whether or not to compute the posterior in safe mode (fall back to prior if things break)
    num_posterior_crashes: int
        Number of times the posterior has crashed in safe mode
    posterior_crash_sigmas: list[float]
        List of update_sigma values where the posterior has crashed

    Methods
    -------
    forward(t)
        Returns the prior distribution before using direct training data
    dot_prod_posterior(*args, **kwargs)
        Returns the posterior distribution for vectors after using both training and posterior update data
    large_input_dot_prod_posterior_means(input_tensor, use_relu = True, verbose = False)
        Returns a tensor mean_list with values\
            mean_list_i = posterior_mean(update_label_i | update_data_1... update_data_{i-1}, update_input_i, update_basis_input_i)\
            Dynamically computes the posterior distribution at inputs in input tensor to sequentially add the update points in the model\
            and make predictions for the next update point as one runs through update_inputs.\
                The assumption is that all update_input values lie in input_tensor.
    large_input_dot_prod_posterior(query_inputs, input_tensor, verbose = False)
        Returns the posterior distribution over query_inputs, but is computed much faster than dot_prod_posterior when update_inputs is large
    large_input_dot_prod_posterior_helper(previous_dist, input_tensor, new_input, new_dot_prod_vec, new_label)
        Helper function to create a new distribution over the universal tensor of inputs (input_tensor) given an initial such distribution (previous_dist)\
            and a single new datapoint (new_input, new_dot_prod_vec, new_label)
    """
    
    def __init__(self, direct_train_inputs, direct_train_vecs, vec_dim, likelihood, update_sigma, mean_module, covar_module, has_safe_posterior = False):
        """
        Parameters
        --------
        direct_train_inputs: Tensor
            Inputs to the vector valued GP of coefficients for direct training of the GP
        direct_train_vecs: Tensor
            Coefficient vectors corresponding to direct_train_inputs, used for direct training of the GP
        vec_dim: int
            Dimension of the vectors
        likelihood: gpytorch.likelihoods.Likelihood
            Likelihood instance to be used with the model (see GPyTorch documentation for more details)
        update_inputs: tensor
            Inputs to the vector valued GP for posterior update, where the first dimension is always 1, t in our notation
        update_dot_prod_vecs : Tensor
            Tensor of vectors with which dot products were taken, with vectors in the last dimension - V in our notation
        update_labels : Tensor
            Observed outputs y for posterior update
        log_update_sigma: torch.nn.Parameter
            Used to store the log of the update_sigma for gradient reasons
        update_sigma : float
            Std dev of the signal used for the update rule, balances between faith in prior and faith in signal
        mean_module: gpytorch.means.Mean
            Multitask mean instance for the mean of the vector-valued GP of coefficients
        covar_module: gpytorch.means.Kernel
            Multitask kernel instance for the covar kernel of the vector-valued GP of coefficients
        has_safe_posterior: bool
            Whether or not to compute the posterior in safe mode (fall back to prior if things break)
        """
        direct_train_inputs_tensor = make_nonzeroD_tensor(direct_train_inputs)
        direct_train_vecs_tensor = make_nonzeroD_tensor(direct_train_vecs)
        super(MultitaskDotProdUpdateGPModel, self).__init__(direct_train_inputs_tensor, direct_train_vecs_tensor, likelihood)
        
        self.update_inputs = torch.tensor([], dtype = torch.float)
        self.update_dot_prod_vecs = torch.tensor([], dtype = torch.float)
        self.update_labels = torch.tensor([], dtype = torch.float)
        self.log_update_sigma = torch.nn.Parameter(torch.tensor(math.log(update_sigma), dtype = torch.float)) if (update_sigma > 0) \
            else torch.nn.Parameter(torch.tensor(math.log(10**(-3)), dtype = torch.float))
        
        if (direct_train_vecs_tensor.numel() > 0):
            assert(vec_dim == direct_train_vecs.shape[-1], "Vector dimension vec_dim = {} does not match\
            dimension of training vectors = {}".format(vec_dim, direct_train_vecs.shape[-1]))
            
        self.vec_dim = vec_dim
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.has_safe_posterior = has_safe_posterior
        self.num_posterior_crashes = 0
        self.posterior_crash_sigmas = []
        
    @property
    def update_sigma(self):
        return torch.exp(self.log_update_sigma)

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
                # In production, we want to use the safe posterior if the posterior is crashing
                if self.has_safe_posterior:
                    try:
                        (
                            predictive_mean,
                            predictive_covar,
                        ) = self.update_strategy.exact_prediction(full_mean, full_covar)
                        assert not (torch.isnan(full_mean).any() or torch.isnan(full_covar).any())
                    # If the posterior crashes, we use the prior instead
                    except:
                        self.num_posterior_crashes += 1
                        self.posterior_crash_sigmas.append(float(self.update_sigma))
                        return self.__call__(*args, **kwargs)
                # Otherwise, we just use the regular posterior
                else:
                    (
                        predictive_mean,
                        predictive_covar,
                    ) = self.update_strategy.exact_prediction(full_mean, full_covar)

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
            return full_output.__class__(predictive_mean, predictive_covar)

    # Uses dynamic programming to quickly compute the tensor mean_list with values\
    # mean_list_i = posterior_mean(update_label_i | update_data_1... update_data_{i-1}, update_input_i, update_basis_input_i)
    # even if update_inputs is a large tensor
    def large_input_dot_prod_posterior_means(self, input_tensor, use_relu = True, verbose = False):
        """
        Parameters
        ----------
        input_tensor: Tensor
            Exhaustive tensor of possible inputs to the model
        use_relu: bool
            Whether or not to use a relu mapping to make predictions
        verbose: bool
            Whether or not to be verbose (set to True for debugging)

        Returns
        ---------
        A tensor mean_list with values\
        mean_list_i = posterior_mean(update_label_i | update_data_1... update_data_{i-1}, update_input_i, update_basis_input_i)
        """

        # Make everything a 1D or higher order tensor
        self.update_inputs = make_nonzeroD_tensor(self.update_inputs)
        self.update_dot_prod_vecs = make_nonzeroD_tensor(self.update_dot_prod_vecs)
        self.update_labels = make_nonzeroD_tensor(self.update_labels)
        
        # Initialize list of predicted means as zeros
        mean_list = torch.zeros(self.update_inputs.squeeze().shape)

        # Initialize prior distribution over the universal tensor of possible inputs, input_tensor
        current_dist = self.__call__(input_tensor)
        # Iteratively update the distribution using each new point
        for idx, current_input in tqdm(enumerate(self.update_inputs.squeeze()), disable = not verbose):
            current_input_idx = input_tensor.squeeze().eq(current_input).nonzero().item()
            current_mean_coeffs = current_dist.mean[current_input_idx]
            current_dot_prod_vec = self.update_dot_prod_vecs[idx]
            current_label = self.update_labels.squeeze()[idx]

            current_mean_coeffs = transform_coeffs(current_mean_coeffs, use_relu = use_relu)
            mean_list[idx] = self.update_dot_prod_vecs[idx] @ current_mean_coeffs
            current_dist = self.large_input_dot_prod_posterior_helper(current_dist, input_tensor, current_input, current_dot_prod_vec, current_label)

        return mean_list
    
    # Uses dynamic programming to compute the posterior at query inputs quickly, even if update_inputs is a large tensor
    def large_input_dot_prod_posterior(self, query_inputs, input_tensor, verbose = False):
        """
        Parameters
        ----------
        query_inputs: Tensor
            Tensor of inputs to query the posterior distribution at
        input_tensor: Tensor
            Exhaustive tensor of possible inputs to the model
        use_relu: bool
            Whether or not to use a relu mapping to make predictions
        verbose: bool
            Whether or not to be verbose (set to True for debugging)

        Returns
        ---------
        The posterior distribution at query_inputs, computed much faster than dot_prod_posterior
        """

        # Make everything a 1D or higher order tensor
        self.update_inputs = make_nonzeroD_tensor(self.update_inputs)
        self.update_dot_prod_vecs = make_nonzeroD_tensor(self.update_dot_prod_vecs)
        self.update_labels = make_nonzeroD_tensor(self.update_labels)

        # Initialize prior distribution over the universal tensor of possible inputs, input_tensor
        current_dist = self.__call__(input_tensor)
        # Iteratively update the distribution using each new point
        for idx, current_input in tqdm(enumerate(self.update_inputs.squeeze()), disable = not verbose):
            current_dot_prod_vec = self.update_dot_prod_vecs[idx]
            current_label = self.update_labels.squeeze()[idx]
            current_dist = self.large_input_dot_prod_posterior_helper(current_dist, input_tensor, current_input, current_dot_prod_vec, current_label)

        # Find list of (possibly repeating) indices of query_inputs in input_tensor
        query_indices = [input_tensor.squeeze().eq(q).nonzero().item() for q in query_inputs]

        # Get distribution for specific tensor of query_inputs
        current_mean, current_covar = current_dist.loc, current_dist.lazy_covariance_matrix.to_dense()
        dim = self.vec_dim
        full_mean = torch.zeros(0)
        full_covar = torch.zeros([0, len(query_indices)*dim])

        for new_idx in query_indices:
            full_mean = torch.cat((full_mean, current_mean[new_idx*dim: (new_idx+1)*dim]), 0)
            addl_covar = torch.zeros([dim, 0])
            for q in query_indices:
                addl_covar = torch.cat((addl_covar, current_covar[new_idx*dim: (new_idx+1)*dim, q*dim: (q+1)*dim]), 1)
            full_covar = torch.cat((full_covar, addl_covar), 0)

        full_mean = full_mean.view([len(query_indices), dim]).contiguous()
        return current_dist.__class__(full_mean, full_covar)

    # Helper function that computes the next distribution over a universal input_tensor of all possible inputs\
    # given a single new datapoint new_input, new_dot_prod_vec, new_label
    def large_input_dot_prod_posterior_helper(self, previous_dist, input_tensor, new_input, new_dot_prod_vec, new_label):
        """
        Parameters
        ----------
        previous_dist: gpytorch.distributions.MultitaskMultivariateNormal
            Distribution to treat as prior
        input_tensor: Tensor
            Exhaustive tensor of possible inputs to the model
        new_input: [float, Tensor]
            Input for the single new data point
        new_dot_prod_vec: [float, Tensor]
            Basis input for the single new data point
        new_label: [float, Tensor]
            Label, i.e. output, for the single new data point

        Returns
        ---------
        The posterior distribution over input_tensor, updating previous_dist with the new data point
        """

        update_inputs = make_nonzeroD_tensor(new_input)
        update_dot_prod_vecs = make_nonzeroD_tensor(new_dot_prod_vec).unsqueeze(-2)
        update_labels = make_nonzeroD_tensor(new_label)
            
        # Create the update strategy for getting posteriors
        update_strategy = dot_product_prediction_strategy(
            update_dot_prod_vecs = update_dot_prod_vecs,
            update_labels= update_labels,
            update_sigma= self.update_sigma
        )

        current_mean, current_covar = previous_dist.loc, previous_dist.lazy_covariance_matrix.to_dense()
        new_idx = input_tensor.squeeze().eq(new_input).nonzero().item()
        dim = self.vec_dim
        full_mean = torch.cat((current_mean[new_idx*dim: (new_idx+1)*dim], current_mean), 0)
        addl_covar = torch.cat((current_covar[new_idx*dim: (new_idx+1)*dim, new_idx*dim: (new_idx+1)*dim],\
                               current_covar[new_idx*dim: (new_idx+1)*dim, :]), 1)
        current_covar_modified = torch.cat((current_covar[:, new_idx*dim: (new_idx+1)*dim], current_covar), 1)
        full_covar = torch.cat((addl_covar, current_covar_modified), 0)

        # Determine the shape of the joint distribution
        test_shape = previous_dist.event_shape
        
        # Make the prediction
        with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
            # In production, we want to use the safe posterior if the posterior is crashing
            if self.has_safe_posterior:
                try:
                    (
                        predictive_mean,
                        predictive_covar,
                    ) = update_strategy.exact_prediction(full_mean, to_linear_operator(full_covar))
                    assert not (torch.isnan(full_mean).any() or torch.isnan(full_covar).any())
                # If the posterior crashes, we use the prior instead
                except:
                    self.num_posterior_crashes += 1
                    self.posterior_crash_sigmas.append(float(self.update_sigma))
                    train_output = super(ExactGP, self).__call__(input_tensor)

                    (
                        predictive_mean,
                        predictive_covar,
                    ) = (train_output.loc, train_output.lazy_covariance_matrix)
            # Otherwise, we just use the regular posterior
            else:
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
    direct_train_vecs: Tensor
        Coefficient vectors corresponding to direct_train_inputs, used for direct training of the GP
    vec_dim: int
        Dimension of the vectors
    update_inputs: tensor
        Inputs to the vector valued GP for posterior update, where the first dimension is always 1, t in our notation
    update_basis_inputs : Tensor
        Tensor of vectors with which dot products were taken, with vectors in the last dimension - V in our notation
    update_labels : Tensor
        Observed outputs y for posterior update
    log_update_sigma: torch.nn.Parameter
        Used to store the log of the update_sigma for gradient reasons
    update_sigma : float
        Std dev of the signal used for the update rule, balances between faith in prior and faith in signal
    basis_funcs : BasisFuncs
        BasisFuncs instance representing the basis functions used
    mean_module: gpytorch.means.Mean
        Multitask mean instance for the mean of the vector-valued GP of coefficients
    covar_module: gpytorch.means.Kernel
        Multitask kernel instance for the covar kernel of the vector-valued GP of coefficients
    has_safe_posterior: bool
        Whether or not to compute the posterior in safe mode (fall back to prior if things break)
    num_posterior_crashes: int
        Number of times the posterior has crashed in safe mode
    posterior_crash_sigmas: list[float]
        List of update_sigma values where the posterior has crashed

    Methods
    -------
    forward(t)
        Returns the prior distribution before using direct training data
    dot_prod_posterior(*args, **kwargs)
        Returns the posterior distribution for vectors after using both training and posterior update data
    large_input_dot_prod_posterior_means_bf(use_relu = True, verbose = False)
        Returns a tensor mean_list with values\
            mean_list_i = posterior_mean(update_label_i | update_data_1... update_data_{i-1}, update_input_i, update_basis_input_i)\
            Dynamically computes the posterior distribution at inputs in range(max_time) to sequentially add the update points in the model\
            and make predictions for the next update point as one runs through update_inputs.\
                The assumption is that all update_input values lie in range(max_time).
    large_input_dot_prod_posterior(query_inputs, verbose = False)
        Returns the posterior distribution over query_inputs, but is computed much faster than dot_prod_posterior when update_inputs is large
    large_input_dot_prod_posterior_helper(previous_dist, new_input, new_dot_prod_vec, new_label)
        Helper function to create a new distribution over the universal tensor of inputs (torch.arange(max_time))\
            given an initial such distribution (previous_dist)\
            and a single new datapoint (new_input, new_dot_prod_vec, new_label)
    """

    # Initialize as per MultitaskDotProdUpdateGPModel and store basis_funcs
    def __init__(self, direct_train_inputs, direct_train_coeffs, vec_dim, likelihood, update_sigma, basis_funcs, max_time,\
                mean_module, covar_module, has_safe_posterior = False):
        """
        Parameters
        ----------
        direct_train_inputs: Tensor
            Inputs to the vector valued GP of coefficients for direct training of the GP
        direct_train_vecs: Tensor
            Coefficient vectors corresponding to direct_train_inputs, used for direct training of the GP
        vec_dim: int
            Dimension of the vectors
        update_inputs: tensor
            Inputs to the vector valued GP for posterior update, where the first dimension is always 1, t in our notation
        update_basis_inputs : Tensor
            Tensor of vectors with which dot products were taken, with vectors in the last dimension - V in our notation
        update_labels : Tensor
            Observed outputs y for posterior update
        log_update_sigma: torch.nn.Parameter
            Used to store the log of the update_sigma for gradient reasons
        update_sigma : float
            Std dev of the signal used for the update rule, balances between faith in prior and faith in signal
        basis_funcs : BasisFuncs
            BasisFuncs instance representing the basis functions used
        mean_module: gpytorch.means.Mean
            Multitask mean instance for the mean of the vector-valued GP of coefficients
        covar_module: gpytorch.means.Kernel
            Multitask kernel instance for the covar kernel of the vector-valued GP of coefficients
        has_safe_posterior: bool
            Whether or not to compute the posterior in safe mode (fall back to prior if things break)
        """
        
        super(MultitaskBasisFuncGPModel, self).__init__(direct_train_inputs, direct_train_coeffs, vec_dim, likelihood, update_sigma, 
                                                       mean_module, covar_module, has_safe_posterior = has_safe_posterior)
        
        self.update_basis_inputs = torch.tensor([], dtype = torch.float)
        self.basis_funcs = basis_funcs
        self.max_time = max_time

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

    # Modify large_input_dot_prod_posterior_means from MultitaskDotProdUpdateGPModel to use use update_basis_inputs and basis_funcs\
    # instead of update_dot_prod_vecs
    # Description: Uses dynamic programming to quickly compute the tensir mean_list with values\
    # mean_list_i = posterior_mean(update_label_i | update_data_1... update_data_{i-1}, update_input_i, update_basis_input_i)
    # even if update_inputs is a large tensor
    def large_input_dot_prod_posterior_means_bf(self, use_relu = True, verbose = False):
        """
        Parameters
        ----------
        use_relu: bool
            Whether or not to use a relu mapping to make predictions
        verbose: bool
            Whether or not to be verbose (set to True for debugging)

        Returns
        ---------
        A tensor mean_list with values\
        mean_list_i = posterior_mean(update_label_i | update_data_1... update_data_{i-1}, update_input_i, update_basis_input_i)
        """
        self.update_dot_prod_vecs = make_nonzeroD_tensor(self.basis_funcs(self.update_basis_inputs))

        time_tensor = torch.arange(self.max_time).float()

        return super(MultitaskBasisFuncGPModel, self).large_input_dot_prod_posterior_means(time_tensor, use_relu = use_relu, verbose = verbose)

    # Modify large_input_dot_prod_posterior from MultitaskDotProdUpdateGPModel to use use update_basis_inputs and basis_funcs\
    # instead of update_dot_prod_vecs
    # Description: Returns the posterior distribution over query_inputs, but is computed much faster than dot_prod_posterior when update_inputs is large
    def large_input_dot_prod_posterior_bf(self, query_inputs, verbose=False):
        """
        Parameters
        ----------
        query_inputs: Tensor
            Tensor of inputs to query the posterior distribution at
        use_relu: bool
            Whether or not to use a relu mapping to make predictions
        verbose: bool
            Whether or not to be verbose (set to True for debugging)

        Returns
        ---------
        The posterior distribution at query_inputs, computed much faster than dot_prod_posterior
        """
        
        self.update_dot_prod_vecs = make_nonzeroD_tensor(self.basis_funcs(self.update_basis_inputs))
        input_tensor = torch.arange(self.max_time).float()
        return super(MultitaskBasisFuncGPModel, self).large_input_dot_prod_posterior(query_inputs, input_tensor, verbose)
    
    # Modify large_input_dot_prod_posterior_helper from MultitaskDotProdUpdateGPModel to use use update_basis_inputs and basis_funcs\
    # instead of update_dot_prod_vecs
    # Description: Helper function to create a new distribution over the universal tensor of inputs (torch.arange(max_time))\
    # given an initial such distribution (previous_dist)\
    # and a single new datapoint (new_input, new_dot_prod_vec, new_label)
    def large_input_dot_prod_posterior_helper_bf(self, previous_dist, new_input, new_basis_input, new_label):
        """
        Parameters
        ----------
        previous_dist: gpytorch.distributions.MultitaskMultivariateNormal
            Distribution to treat as prior
        new_input: [float, Tensor]
            Input for the single new data point
        new_dot_prod_vec: [float, Tensor]
            Basis input for the single new data point
        new_label: [float, Tensor]
            Label, i.e. output, for the single new data point

        Returns
        ---------
        The posterior distribution over input_tensor, updating previous_dist with the new data point
        """

        new_dot_prod_vec = make_nonzeroD_tensor(self.basis_funcs(new_basis_input))
        input_tensor = torch.arange(self.max_time).float()
        return super(MultitaskBasisFuncGPModel, self).large_input_dot_prod_posterior_helper(previous_dist, input_tensor,\
                                                                                            new_input, new_dot_prod_vec, new_label)