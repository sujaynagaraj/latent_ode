"""Transformer for generating multivariate missingness in complete datasets"""
import numpy as np
from sklearn.base import TransformerMixin
from scipy import stats
import torch

import lib.utils as utils

from data_ampute_utils import MAR_mask, MNAR_mask_logistic, MNAR_mask_quantiles, MNAR_self_mask_logistic
def data_ampute_batch_collate(batch, time_steps, args, device, data_type = "train"):
   
    batch = torch.stack(batch)
    data_dict = {
        "data": batch, 
        "time_steps": time_steps}


    data_dict = data_ampute_split_and_subsample_batch_updated(data_dict, args, data_type = data_type)

    return data_dict

 
def data_ampute_split_and_subsample_batch_updated(data_dict, args, data_type = "train"):
    if data_type == "train":
        # Training set
        if args.extrap:
            processed_dict = utils.split_data_extrap(data_dict, dataset = args.dataset)
        else:
            processed_dict = utils.split_data_interp(data_dict)

    else:
        # Test set
        if args.extrap:
            processed_dict = utils.split_data_extrap(data_dict, dataset = args.dataset)
        else:
            processed_dict = utils.split_data_interp(data_dict)

    # add mask
    processed_dict = utils.add_mask(processed_dict)


    if args.extrap:
        raise Exception("Have not implemented extrapolation mode for data ampute collate!")

    observed_data = processed_dict["observed_data"]
    n_traj, n_timepoints, n_dims = observed_data.shape

    flattened_data = observed_data.reshape(n_traj, -1)

    flattened_data = flattened_data.cpu().numpy()

    if args.mcar:
        missing_mask = 1 - produce_NA(flattened_data, args.p_miss, mecha="MCAR")["mask"]
    elif args.mnar:
        missing_mask = 1 - produce_NA(flattened_data, p_miss=args.p_miss, mecha="MNAR", opt="logistic", p_obs=args.p_obs)["mask"]

    missing_mask = missing_mask.reshape(n_traj, n_timepoints, n_dims)
    
    device = processed_dict["observed_data"].device

    processed_dict["observed_mask"] = torch.tensor(missing_mask).float().to(device)
    # apply missing mask
    processed_dict["observed_data"] *= processed_dict["observed_mask"]
    processed_dict["observed_data"] = processed_dict["observed_data"].float()

    processed_dict["mask_predicted_data"] = None

    return processed_dict
 
def produce_NA(X, p_miss, mecha="MCAR", opt=None, p_obs=None, q=None):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values. 
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    mecha : str, 
            Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str, 
         For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
    p_obs : float
            If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
    q : float
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.
    
    Returns
    ----------
    A dictionnary containing:
    'X_init': the initial data matrix.
    'X_incomp': the data with the generated missing values.
    'mask': a matrix indexing the generated missing values.s
    """
    
    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
    
    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X, p_miss, q, 1-p_obs).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X, p_miss).double()
    else:
        mask = (torch.rand(X.shape) < p_miss).double()
    
    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan
    
    return {'X_init': X.double(), 'X_incomp': X_nas.double(), 'mask': mask}