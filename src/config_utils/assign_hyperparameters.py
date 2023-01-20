from typing import Any
import itertools

import numpy as np
import pandas as pd
from scipy.stats import qmc
from omegaconf.errors import ConfigAttributeError

from permutation.models.hyperparameters import HParams


def generate_sobol(dimensions: int, power: int) -> np.ndarray:
    """
    Wrapper for sobol number generator from scipy

    Arguments
    --------
    dimensions : Number of variables (e.g. features) to generate numbers for
    power : The number of samples to generate as a logarithm in base 2 (i.e. n=2^power)

    Returns
    --------
    sample: numpy array of sobol indeces
    """
    generator = qmc.Sobol(d=dimensions, scramble=False)
    sample = generator.random_base2(m=power)
    return sample


def generate_sobol_hparams_df(
    lower_bounds: list[int | float],
    upper_bounds: list[int | float],
    parameter_names: list[str],
    power: int = 4,
) -> pd.DataFrame:
    """
    Generates a dataframe of hyperparameters values of size len(lower_bound), 2^power

    Arguments
    --------
    lower_bounds : List of numbers that coorespond to the lower bounds of each parameter
    upper_bounds : List of numbers that coorespond to the upperer bounds of each parameter
    parameter_names : Names of the parameters (e.g. features) that are being sampled
    power :  The number of samples to generate as a logarithm in base 2 (i.e. n=2^power)

    Returns
    --------
    : dataframe of sampled parameter values
    """
    check_list_lengths(lower_bounds, upper_bounds)
    sample = generate_sobol(len(lower_bounds), power)
    parameter_array = qmc.scale(sample, lower_bounds, upper_bounds)
    return pd.DataFrame(parameter_array, columns=parameter_names)


def fix_types(df: pd.DataFrame, types: dict[str, str]) -> pd.DataFrame:
    """
    Cast objects to ints (rounded rather than truncated) and floats based on types dict

    Arguments
    --------
    df : DataFrame to cast
    types : Dictionary of "feature": "type" strings

    Returns
    --------
    rounded: Dataframe with fixed types
    """
    round_dict = {key: 0 for key, value in types.items() if value == "int"}
    rounded = df.round(round_dict)
    return rounded.astype(types)


def include_permutations(
    list_of_lists: list[list[str]], param_names: list[str], df: pd.DataFrame
) -> pd.DataFrame:
    """
    Permutes discrete hyperparameter values before adding them to hyperparameter samples dataframe

    Arguments
    --------
    list_of_lists : List of discrete parameters and their possible values
    param_names : Names of the parameters (e.g. features) that are in the sampled df
    df : Dataframe of hyperparameters to append to

    Returns
    --------
    : Hyperparmeter dataframe with new permuted discrete values
    """
    permutations = list(itertools.product(*list_of_lists))
    df["temp"] = pd.Series([permutations] * len(df))
    temp_df = df.explode("temp")
    temp_df.reset_index(inplace=True, drop=True)
    df_with_permutations = temp_df.join(pd.DataFrame([*temp_df.temp], temp_df.index, param_names))
    df_with_permutations.drop(columns=["temp"], inplace=True)
    return df_with_permutations


def check_list_lengths(l1: list[Any], l2: list[Any]) -> None:
    """
    Raise a value error if l1 and l2 are not the same length
    """
    try:
        assert len(l1) == len(l2)
    except AssertionError:
        raise ValueError("Lists must be the same length.")


def add_constant_params(names: list[str], values: list[Any], df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds any constant parameter values to the hyperparameter dataframe
    """
    for name, value in zip(names, values):
        df[name] = value
    return df


def build_hparams_df(hparam_cfg) -> pd.DataFrame:
    """ 
    Generates a dataframe of permuted hyperparameter sample values based on values in the config files
    """
    temp_df = _handle_continuous_config(hparam_cfg)
    temp_df_discrete = _handle_discrete_config(hparam_cfg, temp_df)
    hparam_df = _handle_static_config(hparam_cfg, temp_df_discrete)
    return hparam_df


def _handle_continuous_config(param_cfg):
    """
    Reads in continuous parameters from config file and generates a dataframe of randomly generated samples from those parameters

    Returns
    --------
    : Dataframe with sampled values for each continuous parameter
    """
    try:
        cont_params = param_cfg.continuous
    except ConfigAttributeError:
        return pd.DataFrame()

    log_names = []
    lower_bounds = []
    upper_bounds = []
    names = []
    for param, param_dict in cont_params.items():
        names.append(param)
        search = param_dict["search"]
        if search == "linear":
            lower, upper = param_dict["range"]
        elif search == "log":
            log_names.append(param)
            lower, upper = np.log(param_dict["range"])
        else:
            raise ValueError(f"Search: {search} is not implemented")
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    temp_df = generate_sobol_hparams_df(lower_bounds, upper_bounds, names)

    temp_df[log_names] = temp_df[log_names].apply(lambda x: np.exp(x))
    type_dict = {param: param_dict["type"] for param, param_dict in cont_params.items()}
    fixed_df = fix_types(temp_df, type_dict)
    return fixed_df


def _handle_discrete_config(param_cfg, hparam_df) -> pd.DataFrame:
    """
    Reads in discrete parameters from config file and appends permutations of them to the hyperparameter dataframe

    Returns
    --------
    : Dataframe with sampled values for each continuous and discrete parameter
    """
    try:
        discrete_params = param_cfg.discrete
    except ConfigAttributeError:
        return hparam_df

    param_names, val_lists = zip(*discrete_params.items())
    with_perm_df = include_permutations(val_lists, param_names, hparam_df)
    return with_perm_df


def _handle_static_config(param_cfg, hparam_df) -> pd.DataFrame:
    """
    Reads in static parameters from config file and appends them to the hyperparameter dataframe

    Returns
    --------
    : Dataframe with sampled values for each continuous and discrete parameter along with static parameter values
    """
    try:
        static_params = param_cfg.static
    except ConfigAttributeError:
        return hparam_df

    names, values = zip(*static_params.items())
    withstatic_hparam_df = add_constant_params(names, values, hparam_df)
    return withstatic_hparam_df


def assign_hyperparameters(hparam_cfg) -> list[HParams]:
    """
    Creates a list of sampled hyperparmeters for a models from a set of config files
    """
    hparam_df = build_hparams_df(hparam_cfg)
    params = [*hparam_df.to_dict(orient="index").values()]
    return [HParams(hparam_dict) for hparam_dict in params]
