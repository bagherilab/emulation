from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc


def generate_sobol(dimensions: int, power: int) -> np.ndarray:
    """wrapper for sobol number generator from scipy

    Arguments
    --------
    dimensions : Number of variables (e.g. features) to generate numbers for
    y : Pandas series containing <response> variable

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
    todo
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
    roudned: Dataframe with fixed types
    """
    round_dict = {key: 0 for key, value in types.items() if value == "int"}
    rounded = df.round(round_dict)
    return rounded.astype(types)


def include_permutations(
    list_of_lists: list[list[str]], df: pd.DataFrame, param_names: list[str]
) -> pd.DataFrame:
    permutations = list(itertools.product(*list_of_lists))
    df["temp"] = pd.Series([permutations] * len(df))
    temp_df = df.explode("temp")
    temp_df.reset_index(inplace=True)
    temp_df.join(pd.DataFrame([*temp_df.temp], temp_df.index, param_names))
    df_with_permutations = temp_df.drop(columns=["temp"])
    return df_with_permutations


def check_list_lengths(l1: list[Any], l2: list[Any]) -> None:
    """
    raise a value error if l1 and l2 are not the same length
    """
    try:
        assert len(l1) == len(l2)
    except AssertionError:
        raise ValueError("Lists must be the same length.")


def build_hparams_df():
    pass


if __name__ == "__main__":
    pass
