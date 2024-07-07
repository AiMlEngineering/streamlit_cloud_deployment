# import library and module
import copy
import os
import pickle
import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import seaborn as sns
import statsmodels.stats.api as sms
from feature_engine.discretisation import (DecisionTreeDiscretiser,
                                           EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from feature_engine.encoding import OrdinalEncoder
from feature_engine.imputation import RandomSampleImputer
from scipy.stats import (anderson, ks_2samp, kstest, mannwhitneyu, normaltest,
                         shapiro)
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import (Binarizer, KBinsDiscretizer,
                                   QuantileTransformer)
from sklearn.tree import DecisionTreeRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

# for reproducibility, split size
seed = 0
split_size = 0.3


# function for loading data from directory path
def loading_data(*, file_path: str, file_name: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the directory containing the file.
    - file_name (str): The name of the CSV file to load.

    Returns:
    - pd.DataFrame: The loaded DataFrame.
    """
    # Check the type of variables passed in the function
    if not isinstance(file_path, str) or not isinstance(file_name, str):
        raise ValueError("Both file_path and file_name should be strings.")

    # Load and copy the dataset
    file_full_path = os.path.join(file_path, file_name)
    dataframe = pd.read_csv(Path(file_full_path)).copy()

    # Return a copy of dataframe
    return dataframe


# function for loading data from directory path
def loading_data_method2(
    *, file_path: str, file_name: str, dtype_dict: dict
) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the directory containing the file.
    - file_name (str): The name of the CSV file to load.
    - dtype_dict (dict): The dictionary of data type for each columns.

    Returns:
    - pd.DataFrame: The loaded DataFrame.
    """
    # Check the type of variables passed in the function
    if not isinstance(file_path, str) or not isinstance(file_name, str):
        raise ValueError("Both file_path and file_name should be strings.")
    if not isinstance(dtype_dict, dict):
        raise ValueError("dtype_dict should be dictionary.")

    # Load and copy the dataset
    file_full_path = os.path.join(file_path, file_name)
    dataframe = pd.read_csv(Path(file_full_path), dtype=dtype_dict).copy()

    # Return a copy of dataframe
    return dataframe


# function to loading file from directory path
def get_variable_list_by_loading_file(*, file_path: str, file_name: str) -> list:
    """
    Load a list from a file using pickle.

    Parameters:
    - file_path (str): The path to the directory containing the file.
    - file_name (str): The name of the file to load.

    Returns:
    - list: The loaded list from the file.
    """
    # Check the type of variables passed in the function
    if not isinstance(file_path, str) or not isinstance(file_name, str):
        raise ValueError("Both file_path and file_name should be strings.")

    # Create a pathlib.Path object for better path manipulation
    file_path_obj = Path(file_path)

    # Join path components using os.path.join
    file_full_path = os.path.join(file_path_obj, file_name)

    try:
        # Load the list from the file using pickle
        with open(file_full_path, "rb") as file:
            miss_var = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_full_path}")
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

    # Return list containing string values
    return miss_var


# function to get the probability of some observation can be miss
def missing_information_probability(*, input_data: pd.DataFrame, mvar: str) -> dict:
    """
    Calculate the probability of missing values in 'mvar' based on other variables.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - mvar (str): The target variable for which missing values are considered.

    Returns:
    - dict: A dictionary of probabilities for missing values in 'mvar' based on other variables.
    """
    # Copy input data
    data = input_data.copy()
    data_imput_nan = input_data.copy()
    nan_imputer = 20000000000

    # Imput missing value in data set with 20000000000
    data_imput_nan = data_imput_nan.replace(np.nan, nan_imputer)

    # get dataframe with only NaN value under a single 'mvar' column
    # and replace NaN value by 20000000000
    data_nan = data[data[mvar].isnull()]
    data_nan = data_nan.replace(np.nan, nan_imputer)

    # Initialize dictionnary
    frame_dict = {}

    for var in data.columns.to_list():
        if var != mvar:
            try:
                # Calculate probability
                tmp = (
                    data_nan.groupby(var)[mvar].count()
                    / data_imput_nan.groupby(var)[mvar].count()
                )

                # When a category in 'var' doesn't have any missing value in 'mvar',
                # replace the numerator of tmp by 0 ==> tmp = 0 for this category
                frame_dict[var] = tmp.to_frame().dropna()
            except ZeroDivisionError:
                frame_dict[var] = pd.DataFrame(data={mvar: [0]}, index=[np.nan])

    # Return dictionary containing probability values
    return frame_dict


# function to get missing variable witch have missing value at random
def missing_at_random(*, input_data: pd.DataFrame, miss_var_list: list) -> tuple:
    """
    Identify variables with missing values at random (MAR) and not MAR.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_var_list (list): List of variables with missing values.

    Returns:
    - tuple: A tuple containing two lists - one for MAR variables and one for not MAR variables.
    """
    # Copy input data
    data = input_data.copy()

    # Initialize lists of variables with missing values at random (MAR) and not MAR
    MAR_list = []
    not_MAR_list = []

    # Get the probability of an observation being missing for each variable
    miss_info_prob_dict = {
        mvar: missing_information_probability(input_data=data, mvar=mvar)
        for mvar in miss_var_list
    }

    # Check if probability frame of mvar contains unique element or not
    for mvar in miss_var_list:
        same_value_column = []
        for var in data.columns.to_list():
            if var != mvar:
                miss_info_prob_frame = miss_info_prob_dict[mvar][var]
                if (
                    miss_info_prob_frame[mvar]
                    .eq(miss_info_prob_frame[mvar].iloc[0])
                    .all()
                    .item()
                    is True
                ):
                    same_value_column.append(miss_info_prob_frame[mvar].iloc[0])

        # Check if mvar is MAR variable
        if len(same_value_column) == len(data.columns.to_list()) - 1:
            not_MAR_list.append(mvar)
        else:
            MAR_list.append(mvar)

    # Return tuple of lists
    return (MAR_list, not_MAR_list)


# function to get missing missing information percentage
def missing_information_percentage(*, input_data: pd.DataFrame, mvar: str) -> dict:
    """
    Calculate the percentage of missing values in 'mvar' based on other variables in the DataFrame.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - mvar (str): The target variable for which missing values are considered.

    Returns:
    - dict: A dictionary containing percentages for missing values in 'mvar'.
    """
    # Copy input data
    data = input_data.copy()
    nan_imputer = 20000000000

    # Get DataFrame with only NaN values under a single 'mvar' column
    # and replace NaN values with 20000000000
    data_nan = data[data[mvar].isnull()]
    data_nan = data_nan.replace(np.nan, nan_imputer)

    # Initialize dictionary
    frame_dict = {}

    # Calculate the percentage of missing values for each variable
    for var in data.columns.to_list():
        if var != mvar:
            if not data_nan.empty:
                tmp = data_nan.groupby(var)[mvar].count() / float(data_nan.shape[0])
            else:
                tmp = pd.Series(index=[np.nan], data=[0.0])
            frame_dict[var] = tmp.to_frame().dropna()

    # Return dictionary containing percentage values
    return frame_dict


# function to get missing variable witch have missing value at completely random
def missing_completely_at_random(
    *, input_data: pd.DataFrame, miss_var_list: list
) -> tuple:
    """
    Identify variables with missing values completely at random (MCAR) and not MCAR.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_var_list (list): List of variables with missing values.

    Returns:
    - tuple: A tuple containing two lists - one for MCAR variables and one for not MCAR variables.
    """
    # Copy input data
    data = input_data.copy()

    # Initialize lists of variables for missing values completely at random (MCAR) and not MCAR
    MCAR_list = []
    not_MCAR_list = []

    # Get the percentage of an observation being missing
    miss_info_perc_dict = {}
    for mvar in miss_var_list:
        miss_info_perc_dict[mvar] = missing_information_percentage(
            input_data=data, mvar=mvar
        )

    # Check if percentage frame of mvar contains unique element or not
    for mvar in miss_var_list:
        same_value_column = []
        for var in data.columns.to_list():
            if var != mvar:
                miss_info_perc_frame = miss_info_perc_dict[mvar][var]
                if (
                    miss_info_perc_frame[mvar]
                    .eq(miss_info_perc_frame[mvar].iloc[0])
                    .all()
                    .item()
                    is True
                ):
                    same_value_column.append(miss_info_perc_frame[mvar].iloc[0])

        # Check if mvar is MCAR variable
        if len(same_value_column) == len(data.columns.to_list()) - 1:
            if same_value_column.count(same_value_column[0]) == len(same_value_column):
                MCAR_list.append(mvar)
            else:
                not_MCAR_list.append(mvar)
        else:
            not_MCAR_list.append(mvar)

    # Return tuple of lists
    return (MCAR_list, not_MCAR_list)


# function to perform Kolmogorov Smirnov Test
def kolmogorov_smirnov_test_for_same_distribution(
    *, imput_numerical_dataframe: pd.DataFrame, same_ditribution_test_threshold: float
) -> tuple:
    """
    Perform the Kolmogorov Smirnov Test, to check if 2 variables have or not the same distributions.
    Args:
       imput_numerical_dataframe: pd.DataFrame.
       same_ditribution_test_threshold: float
    Outputs:
        u_statistic: Float. U statisitic for the test.
        p_value: Float.

    Parameters:
    - imput_numerical_dataframe (pd.DataFrame): The input numerical DataFrame with imput value.
    - same_ditribution_test_threshold (float): A threshold for nomality test.

    Returns:
    - u_statistic (float): A float of u_statistic.
    - p_value (float): A float of p_value.
    - tuple: A tuple of lists.
    """
    # Check the type of variables passed in the function
    if not isinstance(imput_numerical_dataframe, pd.DataFrame):
        raise ValueError(
            "imput_numerical_dataframe should be pandas dataframe contains only"
            "numerical features with imput values."
        )
    if not isinstance(same_ditribution_test_threshold, float):
        raise ValueError("same_ditribution_test_threshold should be a float")

    # List of normal same and different distribution variable
    same_dist_var = []
    diff_dist_var = []
    same_dist_pval = []
    diff_dist_pval = []
    same_dist_stat = []
    diff_dist_stat = []
    same_dist_var_list = []
    diff_dist_var_list = []
    all_dist_var = []
    all_dist_pval = []
    all_dist_stat = []
    all_dist_var_list = []

    # Copy input data
    data_num = imput_numerical_dataframe.copy()

    # Perform Kolmogorov Smirnov Test for each column
    for var1 in data_num.columns.values.tolist():
        for var2 in data_num.columns.values.tolist():
            if var1 != var2 and var1 in var2:
                # drop nan values in column
                if data_num[var1].isnull().mean() != 0:
                    ref_frame = data_num[[var1]].copy().dropna()
                else:
                    ref_frame = data_num[[var1]].copy()

                if data_num[var2].isnull().mean() != 0:
                    new_frame = data_num[[var2]].copy().dropna()
                else:
                    new_frame = data_num[[var2]].copy()

                # perform test
                stat, p_value = ks_2samp(ref_frame[var1], new_frame[var2])

                # distinct distribution with same_ditribution_test_threshold=0.05
                all_dist_var.append([var1, var2])
                all_dist_pval.append(p_value)
                all_dist_stat.append(stat)
                all_dist_var_list.append(var1)

                if p_value < same_ditribution_test_threshold:
                    diff_dist_var.append([var1, var2])
                    diff_dist_pval.append(p_value)
                    diff_dist_stat.append(stat)
                    diff_dist_var_list.append(var1)
                else:
                    same_dist_var.append([var1, var2])
                    same_dist_pval.append(p_value)
                    same_dist_stat.append(stat)
                    same_dist_var_list.append(var1)

    # return results as tuple of list
    return (
        same_dist_var,
        same_dist_pval,
        same_dist_stat,
        same_dist_var_list,
        diff_dist_var,
        diff_dist_pval,
        diff_dist_stat,
        diff_dist_var_list,
        all_dist_var,
        all_dist_pval,
        all_dist_stat,
        all_dist_var_list,
    )


# function to perform Mann Whitney U Test
def mann_whitney_u_test_for_same_distribution(
    *, imput_numerical_dataframe: pd.DataFrame, same_ditribution_test_threshold: float
) -> tuple:
    """
    Perform the Mann Whitney U Test, to check if 2 variables have or not the same distributions.
    Args:
       imput_numerical_dataframe: pd.DataFrame.
       same_ditribution_test_threshold: float
    Outputs:
        u_statistic: Float. U statisitic for the test.
        p_value: Float.

    Parameters:
    - imput_numerical_dataframe (pd.DataFrame): The input numerical DataFrame with imput value.
    - same_ditribution_test_threshold (float): A threshold for nomality test.

    Returns:
    - u_statistic (float): A float of u_statistic.
    - p_value (float): A float of p_value.
    - tuple: A tuple of lists.
    """
    # Check the type of variables passed in the function
    if not isinstance(imput_numerical_dataframe, pd.DataFrame):
        raise ValueError(
            "imput_numerical_dataframe should be pandas dataframe contains only"
            "numerical features with imput values."
        )
    if not isinstance(same_ditribution_test_threshold, float):
        raise ValueError("same_ditribution_test_threshold should be a float")

    # List of normal same and different distribution variable
    same_dist_var = []
    diff_dist_var = []
    same_dist_pval = []
    diff_dist_pval = []
    same_dist_stat = []
    diff_dist_stat = []
    same_dist_var_list = []
    diff_dist_var_list = []
    all_dist_var = []
    all_dist_pval = []
    all_dist_stat = []
    all_dist_var_list = []

    # Copy input data
    data_num = imput_numerical_dataframe.copy()

    # Perform Agostino-Pearson normality test for each column
    for var1 in data_num.columns.values.tolist():
        for var2 in data_num.columns.values.tolist():
            if var1 != var2 and var1 in var2:
                # drop nan values in column
                if data_num[var1].isnull().mean() != 0:
                    ref_frame = data_num[[var1]].copy().dropna()
                else:
                    ref_frame = data_num[[var1]].copy()

                if data_num[var2].isnull().mean() != 0:
                    new_frame = data_num[[var2]].copy().dropna()
                else:
                    new_frame = data_num[[var2]].copy()

                # perform test
                stat, p_value = mannwhitneyu(ref_frame[var1], new_frame[var2])

                # distinct distribution with normality_test_threshold=0.05
                all_dist_var.append([var1, var2])
                all_dist_pval.append(p_value)
                all_dist_stat.append(stat)
                all_dist_var_list.append(var1)

                if p_value < same_ditribution_test_threshold:
                    diff_dist_var.append([var1, var2])
                    diff_dist_pval.append(p_value)
                    diff_dist_stat.append(stat)
                    diff_dist_var_list.append(var1)
                else:
                    same_dist_var.append([var1, var2])
                    same_dist_pval.append(p_value)
                    same_dist_stat.append(stat)
                    same_dist_var_list.append(var1)

    # return results as tuple of list
    return (
        same_dist_var,
        same_dist_pval,
        same_dist_stat,
        same_dist_var_list,
        diff_dist_var,
        diff_dist_pval,
        diff_dist_stat,
        diff_dist_var_list,
        all_dist_var,
        all_dist_pval,
        all_dist_stat,
        all_dist_var_list,
    )


# function to sort the output of same distribution test
def sort_output_of_test_for_same_distribution(
    *,
    Same_Dist_Test_Res: tuple,
    imputer_values_list: list,
    same_ditribution_test_threshold: float,
) -> tuple:
    """
    Sort and retain only necessary variable from result of the same distributions test.
    Args:
       Same_Dist_Test_Res: tuple.
       imputer_values_list: list
       same_ditribution_test_threshold: float
    Outputs:
        tuple: tuple

    Parameters:
    - Same_Dist_Test_Res (tuple): The input tuple.
    - imputer_values_list : list
    - same_ditribution_test_threshold (float): A threshold for nomality test.

    Returns:
    - tuple (tuple): A tuple of variable, p_value, statistic and variable.
    """
    # Check the type of variables passed in the function
    if not isinstance(Same_Dist_Test_Res, tuple):
        raise ValueError("Same_Dist_Test_Res should be tuple.")
    if not isinstance(imputer_values_list, list):
        raise ValueError("imputer_values_list should be list.")
    if not isinstance(same_ditribution_test_threshold, float):
        raise ValueError("same_ditribution_test_threshold should be a float")

    # List of normal same and different distribution variable
    same_dist_var = []
    diff_dist_var = []
    same_dist_pval = []
    diff_dist_pval = []
    same_dist_stat = []
    diff_dist_stat = []
    same_dist_var_list = []
    diff_dist_var_list = []
    all_dist_var = []
    all_dist_pval = []
    all_dist_stat = []
    all_dist_var_list = []

    # Copy input tuple
    Same_Dist_Test_Res_copy = copy.copy(Same_Dist_Test_Res)

    # Create list of name, pvalue and statistical for variable witch have same distribution
    duplicate_list = []
    for count, elem in enumerate(Same_Dist_Test_Res_copy[0]):
        for imp_val in imputer_values_list:
            if elem not in duplicate_list and (
                elem[1] == f"{elem[0]}_{imp_val}" or elem[0] == f"{elem[1]}_{imp_val}"
            ):
                same_dist_var.append(elem)
                same_dist_pval.append(Same_Dist_Test_Res_copy[1][count])
                same_dist_stat.append(Same_Dist_Test_Res_copy[2][count])
                same_dist_var_list.append(Same_Dist_Test_Res_copy[3][count])

                # lits to avoid redundant somme like var1*var2 and var2*var1
                x1, x2 = elem[0], elem[1]
                duplicate_list.append([x1, x2])
                duplicate_list.append([x2, x1])

    # Create list of name, pvalue and statistical for variable witch have different distribution
    duplicate_list = []
    for count, elem in enumerate(Same_Dist_Test_Res_copy[4]):
        for imp_val in imputer_values_list:
            if elem not in duplicate_list and (
                elem[1] == f"{elem[0]}_{imp_val}" or elem[0] == f"{elem[1]}_{imp_val}"
            ):
                diff_dist_var.append(elem)
                diff_dist_pval.append(Same_Dist_Test_Res_copy[5][count])
                diff_dist_stat.append(Same_Dist_Test_Res_copy[6][count])
                diff_dist_var_list.append(Same_Dist_Test_Res_copy[7][count])

                # lits to avoid redundant somme like var1*var2 and var2*var1
                x1, x2 = elem[0], elem[1]
                duplicate_list.append([x1, x2])
                duplicate_list.append([x2, x1])

    # Create list of name, pvalue and statistical for all variable distribution in data set
    duplicate_list = []
    for count, elem in enumerate(Same_Dist_Test_Res_copy[8]):
        for imp_val in imputer_values_list:
            if elem not in duplicate_list and (
                elem[1] == f"{elem[0]}_{imp_val}" or elem[0] == f"{elem[1]}_{imp_val}"
            ):
                all_dist_var.append(elem)
                all_dist_pval.append(Same_Dist_Test_Res_copy[9][count])
                all_dist_stat.append(Same_Dist_Test_Res_copy[10][count])
                all_dist_var_list.append(Same_Dist_Test_Res_copy[11][count])

                # lits to avoid redundant somme like var1*var2 and var2*var1
                x1, x2 = elem[0], elem[1]
                duplicate_list.append([x1, x2])
                duplicate_list.append([x2, x1])

    # return results as tuple of list
    return (
        same_dist_var,
        same_dist_pval,
        same_dist_stat,
        same_dist_var_list,
        diff_dist_var,
        diff_dist_pval,
        diff_dist_stat,
        diff_dist_var_list,
        all_dist_var,
        all_dist_pval,
        all_dist_stat,
        all_dist_var_list,
    )


# function to sort the output of same distribution test
def same_distribution_pvalue_frame(*, Same_Dist_Test_Res_Sorted: tuple) -> tuple:
    """
    Sort and retain only necessary variable from result of the same distributions test.
    Args:
       Same_Dist_Test_Res_Sorted: tuple.
    Outputs:
        tuple: tuple

    Parameters:
    - Same_Dist_Test_Res_Sorted (tuple): The input tuple.

    Returns:
    - tuple (tuple): A tuple of frame list.
    """
    # Check the type of variables passed in the function
    if not isinstance(Same_Dist_Test_Res_Sorted, tuple):
        raise ValueError("Same_Dist_Test_Res_Sorted should be tuple.")

    # List of normal same and different distribution variable
    same_dist_list_frame = []
    diff_dist_list_frame = []
    all_dist_list_frame = []

    # Copy input tuple
    Same_Dist_Test_Res_Sorted_copy = copy.copy(Same_Dist_Test_Res_Sorted)

    # Create frame tuple for variable witch have same distribution
    for count, elem in enumerate(Same_Dist_Test_Res_Sorted_copy[0]):
        frame = pd.DataFrame(
            Same_Dist_Test_Res_Sorted_copy[1][count], columns=[elem[0]], index=[elem[1]]
        )
        same_dist_list_frame.append(frame)

    # Create frame tuple for variable witch have different distribution
    for count, elem in enumerate(Same_Dist_Test_Res_Sorted_copy[4]):
        frame = pd.DataFrame(
            Same_Dist_Test_Res_Sorted_copy[5][count], columns=[elem[0]], index=[elem[1]]
        )
        diff_dist_list_frame.append(frame)

    # Create frame tuple for all variable distribution
    for count, elem in enumerate(Same_Dist_Test_Res_Sorted_copy[8]):
        frame = pd.DataFrame(
            Same_Dist_Test_Res_Sorted_copy[9][count], columns=[elem[0]], index=[elem[1]]
        )
        all_dist_list_frame.append(frame)

    # return results as tuple of frame list
    return (same_dist_list_frame, diff_dist_list_frame, all_dist_list_frame)


# function to perform Agostino Pearson test
def agostino_pearson_normality_test(
    *,
    imput_numerical_dataframe: pd.DataFrame,
    num_var_list: list,
    normality_test_threshold: float,
) -> tuple:
    """
    Perform the Agostino-Pearson normality Test, to check if variable follows normal distributions.
    Args:
       imput_numerical_dataframe: pd.DataFrame.
       num_var_list: list
       normality_test_threshold: float
    Outputs:
        u_statistic: Float. U statisitic for the test.
        p_value: Float.

    Parameters:
    - imput_numerical_dataframe (pd.DataFrame): The input numerical DataFrame with imput value.
    - num_var_list (list): A list of numerical variables in dataframe.
    - normality_test_threshold (float): A threshold for nomality test.

    Returns:
    - u_statistic (float): A float of u_statistic.
    - p_value (float): A float of p_value.
    - tuple (tuple): A tuple of list
    """
    # Check the type of variables passed in the function
    if not isinstance(imput_numerical_dataframe, pd.DataFrame):
        raise ValueError(
            "imput_numerical_dataframe should be pandas dataframe contains only"
            "numerical features with imput values."
        )
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be list of strings.")
    if not isinstance(normality_test_threshold, float):
        raise ValueError("normality_test_threshold should be a float")

    # List of normal distribution and skewed distribution variable
    norm_dist_var = []
    skewed_dist_var = []
    norm_dist_pval = []
    skewed_dist_pval = []
    norm_dist_stat = []
    skewed_dist_stat = []

    # Copy input data
    data_num = imput_numerical_dataframe[num_var_list].copy()

    # Perform Agostino-Pearson normality test for each column
    for var in num_var_list:
        # drop nan values in column
        if data_num[var].isnull().mean() != 0:
            test_frame = data_num[[var]].copy().dropna()
        else:
            test_frame = data_num[[var]].copy()

        # perform test
        stat, p_value = normaltest(test_frame[var])

        # distinct distribution with normality_test_threshold=0.05
        if p_value < normality_test_threshold:
            skewed_dist_var.append(var)
            skewed_dist_pval.append(p_value)
            skewed_dist_stat.append(stat)
        else:
            norm_dist_var.append(var)
            norm_dist_pval.append(p_value)
            norm_dist_stat.append(stat)

    # return results as tuple of list
    return (
        norm_dist_var,
        norm_dist_pval,
        norm_dist_stat,
        skewed_dist_var,
        skewed_dist_pval,
        skewed_dist_stat,
    )


# function to perform Anderson Darling test
def anderson_darling_normality_test(
    *, imput_numerical_dataframe: pd.DataFrame, num_var_list: list
) -> tuple:
    """
    Perform the Anderson-Darling normality Test, to check if variable follows normal distributions.
    Args:
       imput_numerical_dataframe: pd.DataFrame.
       num_var_list: list
    Outputs:
        u_statistic: Float. U statisitic for the test.
        p_value: Float.

    Parameters:
    - imput_numerical_dataframe (pd.DataFrame): The input numerical DataFrame with imput value.
    - num_var_list (list): A list of numerical variables in dataframe.

    Returns:
    - u_statistic (float): A float of u_statistic.
    - p_value (float): A float of p_value.
    - tuple (tuple): A tuple of list
    """
    # Check the type of variables passed in the function
    if not isinstance(imput_numerical_dataframe, pd.DataFrame):
        raise ValueError(
            "imput_numerical_dataframe should be pandas dataframe contains only"
            "numerical features with imput values."
        )
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be list of strings.")

    # List of normal distribution and skewed distribution variable
    norm_dist_var = []
    skewed_dist_var = []
    norm_dist_pval = []
    skewed_dist_pval = []
    norm_dist_stat = []
    skewed_dist_stat = []

    # Copy input data
    data_num = imput_numerical_dataframe[num_var_list].copy()

    # Perform Agostino-Pearson normality test for each column
    for var in num_var_list:
        # drop nan values in column
        if data_num[var].isnull().mean() != 0:
            test_frame = data_num[[var]].copy().dropna()
        else:
            test_frame = data_num[[var]].copy()

        # perform test
        result = anderson(test_frame[var])

        # distinct distribution with result.significance_level[2]=0.05
        if result.statistic < result.significance_level[2]:
            norm_dist_var.append(var)
            norm_dist_pval.append(result.significance_level[2])
            norm_dist_stat.append(result.statistic)
        else:
            skewed_dist_var.append(var)
            skewed_dist_pval.append(result.significance_level[2])
            skewed_dist_stat.append(result.statistic)

    # return results as tuple of list
    return (
        norm_dist_var,
        norm_dist_pval,
        norm_dist_stat,
        skewed_dist_var,
        skewed_dist_pval,
        skewed_dist_stat,
    )


# function to perform Shapiro-Wilk test
def shapiro_wilk_normality_test(
    *,
    imput_numerical_dataframe: pd.DataFrame,
    num_var_list: list,
    normality_test_threshold: float,
) -> tuple:
    """
    Perform the Shapiro-Wilk normality Test, to check if variable follows normal distributions.
    Args:
       imput_numerical_dataframe: pd.DataFrame.
       num_var_list: list
       normality_test_threshold: float
    Outputs:
        u_statistic: Float. U statisitic for the test.
        p_value: Float.

    Parameters:
    - imput_numerical_dataframe (pd.DataFrame): The input numerical DataFrame with imput value.
    - num_var_list (list): A list of numerical variables in dataframe.
    - normality_test_threshold (float): A threshold for nomality test.

    Returns:
    - u_statistic (float): A float of u_statistic.
    - p_value (float): A float of p_value.
    - tuple (tuple): A tuple of list
    """
    # Check the type of variables passed in the function
    if not isinstance(imput_numerical_dataframe, pd.DataFrame):
        raise ValueError(
            "imput_numerical_dataframe should be pandas dataframe contains only"
            "numerical features with imput values."
        )
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be list of strings.")
    if not isinstance(normality_test_threshold, float):
        raise ValueError("normality_test_threshold should be a float")

    # List of normal distribution and skewed distribution variable
    norm_dist_var = []
    skewed_dist_var = []
    norm_dist_pval = []
    skewed_dist_pval = []
    norm_dist_stat = []
    skewed_dist_stat = []

    # Copy input data
    data_num = imput_numerical_dataframe[num_var_list].copy()

    # Perform Agostino-Pearson normality test for each column
    for var in num_var_list:
        # drop nan values in column
        if data_num[var].isnull().mean() != 0:
            test_frame = data_num[[var]].copy().dropna()
        else:
            test_frame = data_num[[var]].copy()

        # perform test
        stat, p_value = shapiro(test_frame[var])

        # distinct distribution with normality_test_threshold=0.05
        if p_value < normality_test_threshold:
            skewed_dist_var.append(var)
            skewed_dist_pval.append(p_value)
            skewed_dist_stat.append(stat)
        else:
            norm_dist_var.append(var)
            norm_dist_pval.append(p_value)
            norm_dist_stat.append(stat)

    # return results as tuple of list
    return (
        norm_dist_var,
        norm_dist_pval,
        norm_dist_stat,
        skewed_dist_var,
        skewed_dist_pval,
        skewed_dist_stat,
    )


# function for Normality test
def residuals_normality_test(
    *, reference_data: pd.DataFrame, normality_test_threshold: float
) -> tuple:
    """
    Perform 4 normality Test, to check if variable follows normal distributions.

    Parameters:
    - reference_data (pd.DataFrame): The input DataFrame.
    - normality_test_threshold (float): The variance test threshold.

    Returns:
    - tuple: A tuple of list.
    """
    # Check the type of variables passed in the function
    if not isinstance(reference_data, pd.DataFrame):
        raise ValueError("reference_data should be pandas dataframe.")
    if not isinstance(normality_test_threshold, float):
        raise ValueError("normality_test_threshold should be float.")

    # Make a copy of input data
    ref_data = reference_data.copy()

    # Initialize list
    # List of normal distribution and skewed distribution variable
    agpr_norm_dist_var = []
    agpr_skewed_dist_var = []
    agpr_norm_dist_pval = []
    agpr_skewed_dist_pval = []
    agpr_norm_dist_stat = []
    agpr_skewed_dist_stat = []

    and_norm_dist_var = []
    and_skewed_dist_var = []
    and_norm_dist_pval = []
    and_skewed_dist_pval = []
    and_norm_dist_stat = []
    and_skewed_dist_stat = []

    shap_norm_dist_var = []
    shap_skewed_dist_var = []
    shap_norm_dist_pval = []
    shap_skewed_dist_pval = []
    shap_norm_dist_stat = []
    shap_skewed_dist_stat = []

    ks_norm_dist_var = []
    ks_skewed_dist_var = []
    ks_norm_dist_pval = []
    ks_skewed_dist_pval = []
    ks_norm_dist_stat = []
    ks_skewed_dist_stat = []

    for var in ref_data.columns.values.tolist():
        # perform Agostino Pearson test
        agpr_stat, agpr_p_value = normaltest(ref_data[var])
        # perform Anderson-Darling test
        result = anderson(ref_data[var])
        # perform Shapiro Wilk test
        shap_stat, shap_p_value = shapiro(ref_data[var])
        # perform Kolmogorov-Smirnov test
        ks_stat, ks_p_value = kstest(ref_data[var], "norm")

        # distinct distribution with normality_test_threshold=0.05
        if agpr_p_value < normality_test_threshold:
            agpr_skewed_dist_var.append(var)
            agpr_skewed_dist_pval.append(agpr_p_value)
            agpr_skewed_dist_stat.append(agpr_stat)
        else:
            agpr_norm_dist_var.append(var)
            agpr_norm_dist_pval.append(agpr_p_value)
            agpr_norm_dist_stat.append(agpr_stat)

        # distinct distribution with result.significance_level[2]=0.05
        if result.statistic < result.significance_level[2]:
            and_norm_dist_var.append(var)
            and_norm_dist_pval.append(result.significance_level[2])
            and_norm_dist_stat.append(result.statistic)
        else:
            and_skewed_dist_var.append(var)
            and_skewed_dist_pval.append(result.significance_level[2])
            and_skewed_dist_stat.append(result.statistic)

        # distinct distribution with normality_test_threshold=0.05
        if shap_p_value < normality_test_threshold:
            shap_skewed_dist_var.append(var)
            shap_skewed_dist_pval.append(shap_p_value)
            shap_skewed_dist_stat.append(shap_stat)
        else:
            shap_norm_dist_var.append(var)
            shap_norm_dist_pval.append(shap_p_value)
            shap_norm_dist_stat.append(shap_stat)

        # distinct distribution with normality_test_threshold=0.05
        if ks_p_value < normality_test_threshold:
            ks_skewed_dist_var.append(var)
            ks_skewed_dist_pval.append(ks_p_value)
            ks_skewed_dist_stat.append(ks_stat)
        else:
            ks_norm_dist_var.append(var)
            ks_norm_dist_pval.append(ks_p_value)
            ks_norm_dist_stat.append(ks_stat)

    # Return result as tuple
    return (
        agpr_norm_dist_var,
        agpr_norm_dist_pval,
        agpr_norm_dist_stat,
        agpr_skewed_dist_var,
        agpr_skewed_dist_pval,
        agpr_skewed_dist_stat,
        and_norm_dist_var,
        and_norm_dist_pval,
        and_norm_dist_stat,
        and_skewed_dist_var,
        and_skewed_dist_pval,
        and_skewed_dist_stat,
        shap_norm_dist_var,
        shap_norm_dist_pval,
        shap_norm_dist_stat,
        shap_skewed_dist_var,
        shap_skewed_dist_pval,
        shap_skewed_dist_stat,
        ks_norm_dist_var,
        ks_norm_dist_pval,
        ks_norm_dist_stat,
        ks_skewed_dist_var,
        ks_skewed_dist_pval,
        ks_skewed_dist_stat,
    )


# function for Homoscedasticity test
def homoscedasticity_test(
    *,
    reference_data: pd.DataFrame,
    new_data: pd.DataFrame,
    same_variance_test_threshold: float,
) -> tuple:
    """
    Test Homoscedasticity in the data set.

    Parameters:
    - reference_data (pd.DataFrame): The input DataFrame.
    - new_data (pd.DataFrame): The input DataFrame.
    - same_variance_test_threshold (float): The variance test threshold.

    Returns:
    - tuple: A tuple of list.
    """
    # Check the type of variables passed in the function
    if not isinstance(reference_data, pd.DataFrame) or not isinstance(
        new_data, pd.DataFrame
    ):
        raise ValueError("reference_data and new_data should be pandas dataframe.")
    if not isinstance(same_variance_test_threshold, float):
        raise ValueError("same_variance_test_threshold should be float.")

    # Make a copy of input data
    ref_data = reference_data.copy()
    new0_data = new_data.copy()

    # Initialize list
    bar_same_variance_var_list = []
    bar_same_variance_pvalue_list = []
    bar_diff_variance_var_list = []
    bar_diff_variance_pvalue_list = []
    lev_same_variance_var_list = []
    lev_same_variance_pvalue_list = []
    lev_diff_variance_var_list = []
    lev_diff_variance_pvalue_list = []
    gfq_same_variance_var_list = []
    gfq_same_variance_pvalue_list = []
    gfq_diff_variance_var_list = []
    gfq_diff_variance_pvalue_list = []

    for var1 in ref_data.columns.values.tolist():
        for var2 in new0_data.columns.values.tolist():
            # variables do not follow normal law or we do not know the law
            # Variance equal test
            # Barlett test for variable we don't know probability law.
            # you can make this test for any number of variable
            bar_result = scipy.stats.bartlett(ref_data[var1], new0_data[var2])
            # Levene test for variable we don't know probability law.
            # you can make this test for any number of variable
            lev_result = scipy.stats.levene(
                ref_data[var1], new0_data[var2], center="median", proportiontocut=0.05
            )
            # # het_goldfeldquandt test for variable
            # gfq_result = sm.stats.diagnostic.het_goldfeldquandt(
            #     new0_data[[var2]],
            #     ref_data[[var1]],
            #     alternative='increasing'
            # )
            # het_goldfeldquandt test for variable
            gfq_result = sms.het_goldfeldquandt(
                y=new0_data[[var2]], x=ref_data[[var1]], alternative="increasing"
            )
            if bar_result.pvalue >= same_variance_test_threshold:
                bar_same_variance_var_list.append([var1, var2])
                bar_same_variance_pvalue_list.append(bar_result.pvalue)
            else:
                bar_diff_variance_var_list.append([var1, var2])
                bar_diff_variance_pvalue_list.append(bar_result.pvalue)

            if lev_result.pvalue >= same_variance_test_threshold:
                lev_same_variance_var_list.append([var1, var2])
                lev_same_variance_pvalue_list.append(lev_result.pvalue)
            else:
                lev_diff_variance_var_list.append([var1, var2])
                lev_diff_variance_pvalue_list.append(lev_result.pvalue)

            if gfq_result[1] >= same_variance_test_threshold:
                gfq_same_variance_var_list.append([var1, var2])
                gfq_same_variance_pvalue_list.append(gfq_result[1])
            else:
                gfq_diff_variance_var_list.append([var1, var2])
                gfq_diff_variance_pvalue_list.append(gfq_result[1])

    # Return result as tuple
    return (
        bar_same_variance_var_list,
        bar_same_variance_pvalue_list,
        bar_diff_variance_var_list,
        bar_diff_variance_pvalue_list,
        lev_same_variance_var_list,
        lev_same_variance_pvalue_list,
        lev_diff_variance_var_list,
        lev_diff_variance_pvalue_list,
        gfq_same_variance_var_list,
        gfq_same_variance_pvalue_list,
        gfq_diff_variance_var_list,
        gfq_diff_variance_pvalue_list,
    )


# # function to impute NaN value with arbitrary value imputation using pandas
# def arbitrary_value_imputation_with_pandas(
#     *, input_data: pd.DataFrame,
#     imputer_values_list: list,
#     miss_num_var_list: list
# ) -> pd.DataFrame:
#     """
#     Imput missing value with arbitrary value imputation methods.

#     Parameters:
#     - input_data (pd.DataFrame): The input DataFrame.
#     - imputer_values_list (list): The list of imputer values.
#     - miss_num_var_list (list): The list of numerical variable with missing values.

#     Returns:
#     - pd.DataFrame: A pandas frame contains initial feature and new imputing feature .
#     """
#     # Check the type of variables passed in the function
#     if not isinstance(input_data, pd.DataFrame):
#         raise ValueError("input_data should be pandas dataframe.")
#     if not isinstance(imputer_values_list, list) or not isinstance(miss_num_var_list, list):
#         raise ValueError("Both imputer_values and miss_num_var should be a list of strings.")

#     # Make a copy of a part of the input data wich contains
#     # only numerical variable with missing values
#     imputed_data_num = input_data[miss_num_var_list].copy()

#     # Impute missing value using pandas
#     for var in miss_num_var_list:
#         for imp_val in imputer_values_list:
#             new_var_name = f"{var}_{imp_val}"
#             imputed_data_num[new_var_name] = imputed_data_num[var].fillna(imp_val)

#     # Return result as tuple
#     return imputed_data_num


# function to impute NaN value with arbitrary value imputation using pandas
def arbitrary_value_imputation_with_pandas(
    *, input_data: pd.DataFrame, imputer_values_list: list, missing_var_list: list
) -> pd.DataFrame:
    """
    Imput missing value with arbitrary value imputation methods.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - imputer_values_list (list): The list of imputer values.
    - missing_var_list (list): The list of variable with missing values.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature .
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(imputer_values_list, list) or not isinstance(
        missing_var_list, list
    ):
        raise ValueError(
            "Both imputer_values_list and missing_var_list should be a list of strings."
        )

    # Make a copy of a part of the input data wich contains
    # only numerical variable with missing values
    imputed_data = input_data[missing_var_list].copy()
    imp_i = 0

    # Impute missing value using pandas
    for var in missing_var_list:
        imp_val = imputer_values_list[imp_i]
        new_var_name = f"{var}_{str(imp_val)}"
        imputed_data[new_var_name] = imputed_data[var].fillna(imp_val)
        imp_i = imp_i + 1

    # Return result as dataframe
    return imputed_data


# function to generate imput value as combination of 9
def replace_with_nines(value: float) -> float:
    """
    Replace a numerical value with the smallest power of 10
    that is greater than the original value.

    Parameters:
    - value: numerical value to be replaced

    Returns:
    - replaced_value: value replaced with the smallest power of 10
    """
    # Check the type of variables passed in the function
    if not isinstance(value, float):
        raise ValueError("value should be a float.")

    # copy of first value
    copy_value = value

    # replace operation
    if value < 9:
        res_value = 9
    else:
        res_value = 10 ** int(np.log10(value) + 1) - 1

    # checking step
    if res_value == copy_value:
        value = value + 1
        # replace operation
        if value < 9:
            res_value = 9
        else:
            res_value = 10 ** int(np.log10(value) + 1) - 1

    # return the result as float
    return res_value


# function to plot distribution of initial and imput variable for numerical variable
# witch contains missing values in dataset
def initial_and_imput_variable_distribution_plots_for_numerical_variable_old(
    *,
    imput_numerical_dataframe: pd.DataFrame,
    miss_num_var_list: list,
    imputer_values_list: list,
    size_of_figure: tuple,
) -> None:
    """
    Plot distribution of initial and imput variables distribution for numerical features.

    Parameters:
    - imput_numerical_dataframe (pd.DataFrame): The input numerical DataFrame with imput value.
    - miss_num_var_list (list): A list of numerical variables contains missing values in dataframe.
    - imputer_values_list (list): A list of values using to imput missing values in dataframe.
    - size_of_figure (tuple): A tuple of values using to sizing the figure.

    Returns:
    - None
    """
    # Check the type of variables passed in the function
    if not isinstance(imput_numerical_dataframe, pd.DataFrame):
        raise ValueError(
            "imput_numerical_dataframe should be pandas dataframe contains only"
            "numerical features with imput values."
        )
    if not isinstance(miss_num_var_list, list) or not isinstance(
        imputer_values_list, list
    ):
        raise ValueError(
            "Both miss_num_var_list and imputer_values_list should be list of strings."
        )

    # Copy input data
    data = imput_numerical_dataframe.copy()

    # rows and lines of plot figure
    nrow = len(miss_num_var_list)
    ncol = len(imputer_values_list) * 2

    # original and imput variable distribution plot
    fig = plt.figure(figsize=size_of_figure)

    for var in miss_num_var_list:
        # list of var+imputer_values_list
        var_plus_imputervalueslist = []

        # place of plot in figure
        nplace = 1
        ax = fig.add_subplot(nrow, ncol, nplace)

        # original variable distribution
        data[var].plot(kind="kde", ax=ax)
        # add legends
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc="best")
        # add title
        plt.title("Distribution of " + var)

        # list of var+imputer_values_list
        var_plus_imputervalueslist.append(var)

        for imp_val in imputer_values_list:
            nplace = nplace + 1
            new_var_name = f"{var}_{imp_val}"
            ax = fig.add_subplot(nrow, ncol, nplace)

            # imput variable distribution
            data[new_var_name].plot(kind="kde", ax=ax)
            # add legends
            lines, labels = ax.get_legend_handles_labels()
            ax.legend(lines, labels, loc="best")
            # add title
            plt.title("Distribution of " + new_var_name)
            # list of var+imputer_values_list
            var_plus_imputervalueslist.append(new_var_name)

        nplace = nplace + 1
        ax = fig.add_subplot(nrow, ncol, nplace)

        for all_var in var_plus_imputervalueslist:
            # original and imput variable distribution
            data[all_var].plot(kind="kde", ax=ax)

            # add legends
            lines, labels = ax.get_legend_handles_labels()
            ax.legend(lines, labels, loc="best")
            # add title
            plt.title("Distribution of " + var + " and " + new_var_name)

    # add space between subplots using padding
    fig.tight_layout(pad=5.0)

    plt.show()

    # Return result as None
    return None


# function to plot distribution of initial and imput variable for numerical variable
# witch contains missing values in dataset
def initial_and_imput_variable_distribution_plots_for_numerical_variable(
    *, imput_numerical_dataframe: pd.DataFrame,
    miss_num_var_list: list,
    imputer_values_list: list,
    size_of_figure: tuple
) -> None:
    """
    Plot distribution of initial and imput variables distribution for numerical features.

    Parameters:
    - imput_numerical_dataframe (pd.DataFrame): The input numerical DataFrame with imput value.
    - miss_num_var_list (list): A list of numerical variables contains missing values in dataframe.
    - imputer_values_list (list): A list of values using to imput missing values in dataframe.
    - size_of_figure (tuple): A tuple of values using to sizing the figure.

    Returns:
    - None
    """
    # Check the type of variables passed in the function
    if not isinstance(imput_numerical_dataframe, pd.DataFrame):
        raise ValueError(
            "imput_numerical_dataframe should be pandas dataframe contains only"
            "numerical features with imput values."
        )
    if not isinstance(miss_num_var_list, list) or not isinstance(imputer_values_list, list):
        raise ValueError(
            "Both miss_num_var_list and imputer_values_list should be list of strings."
        )

    # Copy input data
    data = imput_numerical_dataframe.copy()

    # rows and lines of plot figure
    nrow = len(miss_num_var_list)
    ncol = len(imputer_values_list)*3

    # original and imput variable distribution plot
    fig = plt.figure(figsize=size_of_figure)

    for var in miss_num_var_list:
        # list of var+imputer_values_list
        var_plus_imputervalueslist = []

        # place of plot in figure
        nplace = 1
        ax = fig.add_subplot(nrow, ncol, nplace)

        # original variable distribution
        data[var].plot(kind="kde", ax=ax)
        # add legends
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc="best")
        # add title
        plt.title('Distribution of '+var)

        # list of var+imputer_values_list
        var_plus_imputervalueslist.append(var)

        for imp_val in imputer_values_list:
            nplace = nplace+1
            new_var_name = f"{var}_{str(imp_val)}"
            ax = fig.add_subplot(nrow, ncol, nplace)

            # imput variable distribution
            data[new_var_name].plot(kind="kde", ax=ax)
            # add legends
            lines, labels = ax.get_legend_handles_labels()
            ax.legend(lines, labels, loc="best")
            # add title
            plt.title('Distribution of '+new_var_name)
            # list of var+imputer_values_list
            var_plus_imputervalueslist.append(new_var_name)

        nplace = nplace+1
        ax = fig.add_subplot(nrow, ncol, nplace)

        for all_var in var_plus_imputervalueslist:
            # original and imput variable distribution
            data[all_var].plot(kind="kde", ax=ax)

            # add legends
            lines, labels = ax.get_legend_handles_labels()
            ax.legend(lines, labels, loc="best")
            # add title
            plt.title('Distribution of '+var+' and '+new_var_name)

    # add space between subplots using padding
    fig.tight_layout(pad=5.0)

    plt.show()

    # Return result as tuple
    return None


# function to plot distribution of initial and imput variable for numerical variable
# witch contains missing values in dataset
def distribution_plots_for_numerical_variable_with_and_without_nan_value(
    *,
    imput_numerical_dataframe_with_nan: pd.DataFrame,
    imput_numerical_dataframe_without_nan: pd.DataFrame,
    num_var_list: list,
    size_of_figure: tuple,
) -> None:
    """
    Plot distribution of initial and imput variables distribution for numerical features.

    Parameters:
    - imput_numerical_dataframe (pd.DataFrame): The input DataFrame with nan value.
    - imput_numerical_dataframe_without_nan (pd.DataFrame): The input DataFrame without nan value.
    - num_var_list (list): A list of numerical variables in dataframe.
    - size_of_figure (tuple): A tuple of values using to sizing the figure.

    Returns:
    - None
    """
    # Check the type of variables passed in the function
    if not isinstance(
        imput_numerical_dataframe_with_nan, pd.DataFrame
    ) or not isinstance(imput_numerical_dataframe_without_nan, pd.DataFrame):
        raise ValueError(
            "imput_numerical_dataframe_with_nan and imput_numerical_dataframe_without_nan"
            "should be pandas dataframe contains onlynumerical features with imput values."
        )
    if not isinstance(num_var_list, list):
        raise ValueError("Both num_var_list should be list of strings.")

    # Copy input data
    data_with_nan = imput_numerical_dataframe_with_nan.copy()
    data_without_nan = imput_numerical_dataframe_without_nan.copy()

    # rows and lines of plot figure
    nrow = len(num_var_list)
    ncol = 2

    # original and imput variable distribution plot
    fig = plt.figure(figsize=size_of_figure)

    for var in num_var_list:
        # place of plot in figure
        nplace = 1
        ax = fig.add_subplot(nrow, ncol, nplace)

        # plot distribution of data without nan value
        data_without_nan[var].plot(kind="kde", ax=ax)
        # plot distribution of data contains nan value
        data_with_nan[var].plot(kind="kde", ax=ax)
        # add legends
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc="best")
        # add title
        plt.title(
            "Distribution of " + f"{var}_without_nan" + " and " + f"{var}_with_nan"
        )

        # # place of plot in figure
        # nplace = 1
        # ax = fig.add_subplot(nrow, ncol, nplace)

        # # plot distribution of data without nan value
        # data_without_nan[var].plot(kind="kde", ax=ax)
        # # add legends
        # lines, labels = ax.get_legend_handles_labels()
        # ax.legend(lines, labels, loc="best")
        # # add title
        # plt.title('Distribution of '+f"{var}_without_nan")

        # nplace = nplace+1
        # ax = fig.add_subplot(nrow, ncol, nplace)

        # # plot distribution of data contains nan value
        # data_with_nan[var].plot(kind="kde", ax=ax)

        # # add legends
        # lines, labels = ax.get_legend_handles_labels()
        # ax.legend(lines, labels, loc="best")
        # # add title
        # plt.title('Distribution of '+f"{var}_with_nan")

    # add space between subplots using padding
    fig.tight_layout(pad=5.0)

    plt.show()

    # Return result as None
    return None


# function save variable as pickle file into dedicated folder
def save_variable_into_directory_old(
    *, input_var: t.Union[float, int, str, list], file_path: str, file_name: str
) -> None:
    """
    Save a list to a file using pickle.

    Parameters:
    - input_var (list): The list to be saved.
    - file_path (str): The path to the directory where the file will be saved.
    - file_name (str): The name of the file.

    Returns:
    - None
    """
    # Check the type of variables passed in the function
    if (
        not isinstance(input_var, float)
        and not isinstance(input_var, int)
        and not isinstance(input_var, str)
        and not isinstance(input_var, list)
    ):
        raise ValueError("input_var should be float, integer, strings or list.")
    if not isinstance(file_path, str) or not isinstance(file_name, str):
        raise ValueError("Both file_path and file_name should be strings.")

    # Make a copy of the  input list variable
    var = input_var

    # Save the list to a file using pickle
    file_full_path = Path(file_path) / file_name
    try:
        with open(file_full_path, "wb") as file:
            pickle.dump(var, file)
    except (FileNotFoundError, PermissionError) as e:
        raise IOError(f"Error saving the file {file_full_path}: {e}")

    # Return results as None
    return None


# function save variable as pickle file into dedicated folder
def save_variable_into_directory(
    *, input_var: t.Union[float, int, str, list],
    file_path: str,
    file_name: str
) -> None:
    """
    Save a list to a file using pickle.

    Parameters:
    - input_var (list): The list to be saved.
    - file_path (str): The path to the directory where the file will be saved.
    - file_name (str): The name of the file.

    Returns:
    - None
    """
    # Check the type of variables passed in the function
    if (
        not isinstance(input_var, float)
        and not isinstance(input_var, int)
        and not isinstance(input_var, str)
        and not isinstance(input_var, list)
    ):
        raise ValueError("input_var should be float, integer, strings or list.")
    if not isinstance(file_path, str) or not isinstance(file_name, str):
        raise ValueError("Both file_path and file_name should be strings.")

    # Make a copy of the  input list variable
    var = input_var

    # Save the list to a file using pickle
    file_full_path = Path(file_path) / file_name
    try:
        with open(file_full_path, 'wb') as file:
            pickle.dump(var, file)
    except (FileNotFoundError, PermissionError) as e:
        raise IOError(f"Error saving the file {file_full_path}: {e}")

    # Return results as None
    return None


# function to impute NaN value with end tail value imputation using pandas
def end_tail_imputation_using_IQR_with_pandas(
    *, input_data: pd.DataFrame, miss_num_var_list: list
) -> pd.DataFrame:
    """
    Imput missing value with end tail imputation methods.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_num_var_list (list): The list of numerical variable with missing values.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_num_var_list, list):
        raise ValueError("miss_num_var should be a list of strings.")

    # Make a copy of a part of the input data wich contains only
    # numerical variable with missing values
    imputed_data_num = input_data[miss_num_var_list].copy()

    # Impute missing value using pandas
    for var in miss_num_var_list:
        IQR = imputed_data_num[var].quantile(0.75) - imputed_data_num[var].quantile(
            0.25
        )
        new_var_name = f"{var}_{str(IQR)}"
        imputed_data_num[new_var_name] = imputed_data_num[var].fillna(IQR)

    # Return result as dataframe
    return imputed_data_num


# function to impute NaN value with arbitrary value imputation using pandas
def arbitrary_value_imputation_with_pandas_method2(
    *, input_data: pd.DataFrame,
    imputer_values_list: list,
    missing_var_list: list
) -> pd.DataFrame:
    """
    Imput missing value with arbitrary value imputation methods.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - imputer_values_list (list): The list of imputer values.
    - missing_var_list (list): The list of variable with missing values.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature .
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(imputer_values_list, list) or not isinstance(missing_var_list, list):
        raise ValueError(
            "Both imputer_values_list and missing_var_list should be a list of strings."
        )

    # Make a copy of a part of the input data wich contains only
    # numerical variable with missing values
    imputed_data = input_data[missing_var_list].copy()
    original_imputed_data = input_data[missing_var_list].copy()

    # Impute missing value using pandas
    for var in missing_var_list:
        for imp_val in imputer_values_list:
            new_var_name = f"{var}_{str(imp_val)}"
            imputed_data[new_var_name] = imputed_data[var].fillna(imp_val)
            original_imputed_data[var] = original_imputed_data[var].fillna(imp_val)

    # Return result as tuple
    return imputed_data, original_imputed_data


# function to plot distribution of initial and imput variable for categorical variable
# witch contains missing values in dataset
def initial_and_imput_variable_distribution_plots_for_categorical_variable(
    *,
    imput_categorical_dataframe: pd.DataFrame,
    miss_cat_var_list: list,
    imputer_values_list: list,
    size_of_figure: tuple,
) -> None:
    """
    Plot distribution of initial and imput variables distribution for categorical features.

    Parameters:
    - imput_categorical_dataframe (pd.DataFrame): The input categorical DataFrame with imput value.
    - miss_cat_var_list (list): A list of categorical variables contains missing values.
    - imputer_values_list (list): A list of values using to imput missing values.
    - size_of_figure (tuple): A tuple of values using to sizing the figure.

    Returns:
    - None
    """
    # Check the type of variables passed in the function
    if not isinstance(imput_categorical_dataframe, pd.DataFrame):
        raise ValueError(
            "imput_categorical_dataframe should be pandas dataframe contains only"
            "categorical features with imput values."
        )
    if not isinstance(miss_cat_var_list, list) or not isinstance(
        imputer_values_list, list
    ):
        raise ValueError(
            "Both miss_cat_var_list and imputer_values_list should be list of strings."
        )

    # Copy input data
    data = imput_categorical_dataframe.copy()

    # rows and lines of plot figure
    nrow = len(miss_cat_var_list)
    ncol = len(imputer_values_list) * 3

    # original and imput variable distribution plot
    fig = plt.figure(figsize=size_of_figure)

    for var in miss_cat_var_list:
        # list of var+imputer_values_list
        imputervalueslist = []

        nplace = 1
        ax = fig.add_subplot(nrow, ncol, nplace)

        # data[var].value_counts().plot.bar()
        sns.countplot(x=var, data=data[var].to_frame(), dodge=True)
        plt.title("Distribution of " + var)

        # list of var+imputer_values_list
        # var_plus_imputervalueslist.append(var)

        for imp_val in imputer_values_list:
            nplace = nplace + 1
            new_var_name = f"{var}_{imp_val}"
            ax = fig.add_subplot(nrow, ncol, nplace)

            # data[new_var_name].value_counts().plot.bar()
            sns.countplot(
                x=new_var_name, data=data[new_var_name].to_frame(), dodge=True
            )
            plt.title("Distribution of " + new_var_name)
            # list of var+imputer_values_list
            imputervalueslist.append(new_var_name)

        # Plot bars for each category with different colors
        nplace = nplace + 1
        ax = fig.add_subplot(nrow, ncol, nplace)

        # original and imput variable distribution
        data_for_last_plot = data.apply(pd.Series.value_counts)
        data_for_last_plot.plot(kind="bar", ax=ax)
        # add title
        plt.title("Distribution of " + var + " and " + new_var_name)
        plt.ylabel("count")

    # add space between subplots using padding
    fig.tight_layout(pad=5.0)

    plt.show()

    # Return result as None
    return None


# function to impute NaN value with median value imputation using pandas
def median_imputation_with_pandas(
    *, input_data: pd.DataFrame, miss_num_var_list: list
) -> pd.DataFrame:
    """
    Imput missing value with median imputation methods.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_num_var_list (list): The list of numerical variable with missing values.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature .
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_num_var_list, list):
        raise ValueError("miss_num_var_list should be a list of strings.")

    # Make a copy of a part of the input data wich contains only
    # numerical variable with missing values
    imputed_data_num = input_data[miss_num_var_list].copy()

    # Impute missing value using pandas
    for var in miss_num_var_list:
        med = imputed_data_num[var].median()
        new_var_name = f"{var}_{str(med)}"
        imputed_data_num[new_var_name] = imputed_data_num[var].fillna(med)

    # Return result as dataframe
    return imputed_data_num


# function to impute NaN value with mean value imputation using pandas
def mean_imputation_with_pandas(
    *, input_data: pd.DataFrame, miss_num_var_list: list
) -> pd.DataFrame:
    """
    Imput missing value with mean imputation methods.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_num_var_list (list): The list of numerical variable with missing values.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_num_var_list, list):
        raise ValueError("miss_num_var_list should be a list of strings.")

    # Make a copy of a part of the input data wich contains only
    # numerical variable with missing values
    imputed_data_num = input_data[miss_num_var_list].copy()

    # Impute missing value using pandas
    for var in miss_num_var_list:
        mea = imputed_data_num[var].mean()
        new_var_name = f"{var}_{str(mea)}"
        imputed_data_num[new_var_name] = imputed_data_num[var].fillna(mea)

    # Return result as dataframe
    return imputed_data_num


# function to add missing indicator to data set using pandas
def add_missing_indicator_with_pandas(
    *, input_data: pd.DataFrame, miss_var_list: list
) -> pd.DataFrame:
    """
    Add missing indicator to dataframe.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_var_list (list): The list of variable with missing values.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and missing indicator variables.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_var_list, list):
        raise ValueError("miss_var_list should be a list of strings.")

    # Make a copy of dataset
    data_with_missing_indicator = input_data[miss_var_list].copy()

    # create indicator names
    indicators = [f"{var}_nan" for var in miss_var_list]

    # Add missing indicators to data set
    data_with_missing_indicator[indicators] = (
        data_with_missing_indicator[miss_var_list].isna().astype(int)
    )

    # Return result as dataframe
    return data_with_missing_indicator


# function to impute NaN value with random sample value imputation using pandas
def random_sample_imputation_for_numerical_variable_with_pandas(
    *, input_data: pd.DataFrame, miss_num_var_list: list, imputer_var_name_suffixe: str
) -> pd.DataFrame:
    """
    Imput missing value with random sample imputation methods.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_num_var_list (list): The list of numerical variable with missing values.
    - imputer_var_name_suffixe (str): The suffixe name of variable imputed.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature .
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_num_var_list, list):
        raise ValueError("miss_num_var_list should be a list of strings.")
    if not isinstance(imputer_var_name_suffixe, str):
        raise ValueError("imputer_var_name_suffixe should be strings.")

    # Make a copy of a part of the input data wich contains
    # only numerical variable with missing values
    imputed_data_num = input_data[miss_num_var_list].copy()

    for var in miss_num_var_list:
        # extract the random sample to fill the na:
        random_sample_train = (
            imputed_data_num[var]
            .dropna()
            .sample(imputed_data_num[var].isnull().sum(), random_state=seed)
        )
        # create new variable
        new_var_name = f"{var}_{imputer_var_name_suffixe}"
        imputed_data_num[new_var_name] = imputed_data_num[var]
        # pandas needs to have the same index in order to merge datasets
        random_sample_train.index = imputed_data_num[
            imputed_data_num[var].isnull()
        ].index
        # replace the NA in the newly created variable
        imputed_data_num.loc[imputed_data_num[new_var_name].isnull(), new_var_name] = (
            random_sample_train
        )

    # Return result pandas dataframe
    return imputed_data_num


# function to impute NaN value with random sample value imputation using feature engine
def random_sample_imputation_for_numerical_variable_with_feature_engine_old(
    *, input_data: pd.DataFrame, miss_num_var_list: list, imputer_var_name_suffixe: str
) -> pd.DataFrame:
    """
    Imput missing value with random sample imputation methods.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_num_var_list (list): The list of numerical variable with missing values.
    - imputer_var_name_suffixe (str): The suffixe name of variable imputed.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature .
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_num_var_list, list):
        raise ValueError("miss_num_var_list should be a list of strings.")
    if not isinstance(imputer_var_name_suffixe, str):
        raise ValueError("imputer_var_name_suffixe should be strings.")

    # Make a copy of a part of the input data wich contains
    # only numerical variable with missing values
    imputed_data_num = input_data[miss_num_var_list].copy()

    # we call the imputer from feature-engine
    imputer = RandomSampleImputer(variables=miss_num_var_list, random_state=29)

    # we fit the imputer
    imputer.fit(imputed_data_num)
    imputed_data_num_tr = imputer.transform(imputed_data_num)

    for var in miss_num_var_list:
        # create new variable
        new_var_name = f"{var}_{imputer_var_name_suffixe}"
        imputed_data_num[new_var_name] = imputed_data_num_tr[var]

    # Return result pandas dataframe
    return imputed_data_num


# function to impute NaN value with random sample value imputation using feature engine
def random_sample_imputation_for_numerical_variable_with_feature_engine(
    *, input_data: pd.DataFrame,
    miss_num_var_list: list,
    imputer_var_name_suffixe: str
) -> pd.DataFrame:
    """
    Imput missing value with random sample value imputation methods.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_num_var_list (list): The list of numerical variable with missing values.
    - imputer_var_name_suffixe (str): The suffixe name of variable imputed.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature .
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_num_var_list, list):
        raise ValueError("miss_num_var_list should be a list of strings.")
    if not isinstance(imputer_var_name_suffixe, str):
        raise ValueError("imputer_var_name_suffixe should be strings.")

    # from feature-engine
    from feature_engine.imputation import RandomSampleImputer

    # Make a copy of a part of the input data wich contains only
    # numerical variable with missing values
    imputed_data_num = input_data[miss_num_var_list].copy()

    # we call the imputer from feature-engine
    imputer = RandomSampleImputer(variables=miss_num_var_list, random_state=29)

    # we fit the imputer
    imputer.fit(imputed_data_num)
    imputed_data_num_tr = imputer.transform(imputed_data_num)

    for var in miss_num_var_list:
        # create new variable
        new_var_name = f"{var}_{imputer_var_name_suffixe}"
        imputed_data_num[new_var_name] = imputed_data_num_tr[var]

    # Return result pandas dataframe
    return imputed_data_num, imputed_data_num_tr


# function to impute NaN value with random sample value imputation using pandas
def random_sample_imputation_for_numerical_feature_with_pandas(
    *, input_data: pd.DataFrame, miss_num_var_list: list, imputer_var_name_suffixe: str
) -> pd.DataFrame:
    """
    Imput missing value with random sample imputation methods.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_num_var_list (list): The list of numerical variable with missing values.
    - imputer_var_name_suffixe (str): The suffixe name of variable imputed.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature .
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_num_var_list, list):
        raise ValueError("miss_num_var_list should be a list of strings.")
    if not isinstance(imputer_var_name_suffixe, str):
        raise ValueError("imputer_var_name_suffixe should be strings.")

    # Make a copy of a part of the input data wich contains
    # only numerical variable with missing values
    imputed_data_num = input_data[miss_num_var_list].copy()

    for var in miss_num_var_list:
        # extract the random sample to fill the na:
        random_sample_train = (
            imputed_data_num[var]
            .dropna()
            .sample(imputed_data_num[var].isnull().sum(), random_state=seed)
        )
        # create new variable
        new_var_name = f"{var}_{imputer_var_name_suffixe}"
        imputed_data_num[new_var_name] = imputed_data_num[var]
        # pandas needs to have the same index in order to merge datasets
        random_sample_train.index = imputed_data_num[
            imputed_data_num[var].isnull()
        ].index
        # replace the NA in the newly created variable
        imputed_data_num.loc[imputed_data_num[new_var_name].isnull(), new_var_name] = (
            random_sample_train
        )

    # Return result pandas dataframe
    return imputed_data_num


# function to impute NaN value with mean imputation per group using pandas
def mean_imputation_per_group_with_pandas(
    *, input_data: pd.DataFrame, miss_num_var_list: list, cat_var_for_grouping: str
) -> pd.DataFrame:
    """
    Imput missing value with mean imputation per group methods.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_num_var_list (list): The list of numerical variable with missing values.
    - cat_var_for_grouping (str): The categorical variable to use for grouping.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature .
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_num_var_list, list):
        raise ValueError("miss_num_var_list should be a list of strings.")
    if not isinstance(cat_var_for_grouping, str):
        raise ValueError("cat_var_for_grouping should be strings.")

    # Make a copy of a part of the input data wich contains only
    # numerical variable with missing values
    data = input_data.copy()
    new_var_name_list = []

    for var in miss_num_var_list:
        # create new variable and append it into new list
        new_var_name = f"{var}_meanipg"
        new_var_name_list.append(new_var_name)
        data[new_var_name] = data[var]

    # Create imputation dictionary
    imputation_dict = {}
    for i in data[cat_var_for_grouping].unique():
        imputation_dict[i] = (
            data[data[cat_var_for_grouping] == i][new_var_name_list].mean().to_dict()
        )

    # Replace missing data
    for i in imputation_dict.keys():
        data[data[cat_var_for_grouping] == i] = data[
            data[cat_var_for_grouping] == i
        ].fillna(imputation_dict[i])

    # Make a copy of a part of the input data wich contains only
    # numerical variable with missing values
    imputed_data_num = data[miss_num_var_list + new_var_name_list].copy()

    # Return result pandas dataframe
    return imputed_data_num


# function to impute NaN value with median imputation per group using pandas
def median_imputation_per_group_with_pandas(
    *, input_data: pd.DataFrame, miss_num_var_list: list, cat_var_for_grouping: str
) -> pd.DataFrame:
    """
    Imput missing value with median imputation per group methods.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_num_var_list (list): The list of numerical variable with missing values.
    - cat_var_for_grouping (str): The categorical variable to use for grouping.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_num_var_list, list):
        raise ValueError("miss_num_var_list should be a list of strings.")
    if not isinstance(cat_var_for_grouping, str):
        raise ValueError("cat_var_for_grouping should be strings.")

    # Make a copy of a part of the input data wich contains only
    # numerical variable with missing values
    data = input_data.copy()
    new_var_name_list = []

    for var in miss_num_var_list:
        # create new variable and append it into new list
        new_var_name = f"{var}_medianipg"
        new_var_name_list.append(new_var_name)
        data[new_var_name] = data[var]

    # Create imputation dictionary
    imputation_dict = {}
    for i in data[cat_var_for_grouping].unique():
        imputation_dict[i] = (
            data[data[cat_var_for_grouping] == i][new_var_name_list].median().to_dict()
        )

    # Replace missing data
    for i in imputation_dict.keys():
        data[data[cat_var_for_grouping] == i] = data[
            data[cat_var_for_grouping] == i
        ].fillna(imputation_dict[i])

    # Make a copy of a part of the input data wich contains only
    # numerical variable with missing values
    imputed_data_num = data[miss_num_var_list + new_var_name_list].copy()

    # Return result pandas dataframe
    return imputed_data_num


# function to group variable for mean/median imputation using pandas
def grouping_variable_for_mean_median_imputation_per_group_with_pandas(
    *, input_data: pd.DataFrame, miss_num_var_list: list, cat_var_for_grouping: str
) -> pd.DataFrame:
    """
    Grouping variable for mean/median imputation per group.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_num_var_list (list): The list of numerical variable with missing values.
    - cat_var_for_grouping (str): The categorical variable to use for grouping.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature .
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_num_var_list, list):
        raise ValueError("miss_num_var_list should be a list of strings.")
    if not isinstance(cat_var_for_grouping, str):
        raise ValueError("cat_var_for_grouping should be strings.")

    # Make a copy of a part of the input data wich contains only
    # numerical variable with missing values
    data = input_data.copy()
    new_var_name_list = []

    for var in miss_num_var_list:
        # create new variable and append it into new list
        new_var_name = f"{var}_groupvar"
        new_var_name_list.append(new_var_name)
        data[new_var_name] = data[var]

        # Create a grouping variable
        for i, lab in enumerate(data[cat_var_for_grouping].unique()):
            # re-group variable to imput based on grouping variable
            data[new_var_name] = np.where(
                data[cat_var_for_grouping].isin([lab]), i, data[new_var_name]
            )

    # Make a copy of a part of the input data wich contains only
    # numerical variable with missing values
    grouping_data_num = data[miss_num_var_list + new_var_name_list].copy()

    # Return result pandas dataframe
    return grouping_data_num


# function to impute NaN value with KNN imputation using pandas
def KNN_imputation_with_pandas(
    *, input_data: pd.DataFrame, miss_num_var_list: list
) -> pd.DataFrame:
    """
    Imput missing value with KNN imputation methods.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_num_var_list (list): The list of numerical variable with missing values.

    Returns:
    - tuple: A tuple contains pandas frame with initial feature and new imputing feature .
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_num_var_list, list):
        raise ValueError("miss_num_var_list should be a list of strings.")

    # Make a copy of a part of the input data wich contains
    # only numerical variable with missing values
    data = input_data[miss_num_var_list].copy()
    new_var_name_list = []

    # Set imputer associated to each method
    imputer_KNN = KNNImputer(
        n_neighbors=5,  # the number of neighbours K
        weights="distance",  # the weighting factor
        metric="nan_euclidean",  # the metric to find the neighbours
        add_indicator=False,  # whether to add a missing indicator
    )

    # fit all imputer
    imputer_KNN.fit(data)

    # imput NaN values
    data_knn = imputer_KNN.transform(data)

    for var in miss_num_var_list:
        # create new variable and append it into new list
        new_var_name = f"{var}_KNN"
        new_var_name_list.append(new_var_name)

    # creating the dataframe
    df_data_knn = pd.DataFrame(data=data_knn, columns=new_var_name_list)

    # concatenate dataframe horizontaly
    imput_data_knn = pd.concat([data, df_data_knn], axis=1)

    # Return result as pandas dataframe
    return imput_data_knn


# function to impute NaN value with median imputation per group using pandas
def MICE_and_missForest_imputation_with_pandas_old(
    *, input_data: pd.DataFrame, miss_num_var_list: list
) -> tuple:
    """
    Imput missing value with MICE and missForest imputation methods.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_num_var_list (list): The list of numerical variable with missing values.

    Returns:
    - tuple: A tuple contains pandas frame with initial feature and new imputing feature.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_num_var_list, list):
        raise ValueError("miss_num_var_list should be a list of strings.")

    # Make a copy of a part of the input data wich contains only
    # numerical variable with missing values
    data = input_data[miss_num_var_list].copy()
    new_var_name_list = []

    # Set imputer associated to each method
    imputer_bayes = IterativeImputer(
        estimator=BayesianRidge(), max_iter=10, random_state=seed
    )

    imputer_knn = IterativeImputer(
        estimator=KNeighborsRegressor(n_neighbors=5), max_iter=10, random_state=seed
    )

    imputer_nonLin = IterativeImputer(
        estimator=DecisionTreeRegressor(max_features="sqrt", random_state=0),
        max_iter=500,
        random_state=seed,
    )

    imputer_missForest = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=10, random_state=0),
        max_iter=100,
        random_state=seed,
    )

    # fit all imputer
    imputer_bayes.fit(data)
    imputer_knn.fit(data)
    imputer_nonLin.fit(data)
    imputer_missForest.fit(data)

    # imput NaN values
    data_bayes = imputer_bayes.transform(data)
    data_knn = imputer_knn.transform(data)
    data_nonLin = imputer_nonLin.transform(data)
    data_missForest = imputer_missForest.transform(data)

    for var in miss_num_var_list:
        # create new variable and append it into new list
        new_var_name = f"{var}_MICEmissForest"
        new_var_name_list.append(new_var_name)

    # creating the dataframe
    df_data_bayes = pd.DataFrame(data=data_bayes, columns=new_var_name_list)
    df_data_knn = pd.DataFrame(data=data_knn, columns=new_var_name_list)
    df_data_nonLin = pd.DataFrame(data=data_nonLin, columns=new_var_name_list)
    df_data_missForest = pd.DataFrame(data=data_missForest, columns=new_var_name_list)

    # concatenate dataframe horizontaly
    imput_data_bayes = pd.concat([data, df_data_bayes], axis=1)
    imput_data_knn = pd.concat([data, df_data_knn], axis=1)
    imput_data_nonLin = pd.concat([data, df_data_nonLin], axis=1)
    imput_data_missForest = pd.concat([data, df_data_missForest], axis=1)

    # Return result as a tuple of pandas dataframe
    return (imput_data_bayes, imput_data_knn, imput_data_nonLin, imput_data_missForest)


# function to impute NaN value with MICE and MissForest using pandas
def MICE_and_missForest_imputation_with_pandas(
    *, input_data: pd.DataFrame,
    miss_num_var_list: list
) -> tuple:
    """
    Imput missing value with random sample value imputation methods.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_num_var_list (list): The list of numerical variable with missing values.

    Returns:
    - tuple: A tuple contains pandas frame with initial feature and new imputing feature .
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_num_var_list, list):
        raise ValueError("miss_num_var_list should be a list of strings.")

    # Make a copy of a part of the input data wich contains only
    # numerical variable with missing values
    data = input_data[miss_num_var_list].copy()
    new_var_name_list_bayes = []
    new_var_name_list_knn = []
    new_var_name_list_nonLin = []
    new_var_name_list_missForest = []

    # Set imputer associated to each method
    imputer_bayes = IterativeImputer(
        estimator=BayesianRidge(),  # the estimator to predict the NA
        initial_strategy='mean',  # how will NA be imputed in step 1
        max_iter=100,  # number of cycles
        imputation_order='ascending',  # the order in which to impute the variables
        n_nearest_features=None,  # whether to limit the number of predictors
        skip_complete=True,  # whether to ignore variables without NA
        random_state=seed
    )

    imputer_knn = IterativeImputer(
        estimator=KNeighborsRegressor(n_neighbors=5),
        max_iter=100,
        random_state=seed)

    imputer_nonLin = IterativeImputer(
        estimator=DecisionTreeRegressor(max_features='sqrt', random_state=0),
        max_iter=100,
        random_state=seed)

    imputer_missForest = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=10, random_state=0),
        max_iter=100,
        random_state=seed)

    # fit all imputer
    imputer_bayes.fit(data)
    imputer_knn.fit(data)
    imputer_nonLin.fit(data)
    imputer_missForest.fit(data)

    # imput NaN values
    data_bayes = imputer_bayes.transform(data)
    data_knn = imputer_knn.transform(data)
    data_nonLin = imputer_nonLin.transform(data)
    data_missForest = imputer_missForest.transform(data)

    for var in miss_num_var_list:
        # create new variable and append it into new list
        new_var_name_bayes = f"{var}_bayes"
        new_var_name_knn = f"{var}_knn"
        new_var_name_nonLin = f"{var}_nonLin"
        new_var_name_missForest = f"{var}_missForest"
        new_var_name_list_bayes.append(new_var_name_bayes)
        new_var_name_list_knn.append(new_var_name_knn)
        new_var_name_list_nonLin.append(new_var_name_nonLin)
        new_var_name_list_missForest.append(new_var_name_missForest)

    # creating the dataframe
    df_data_bayes = pd.DataFrame(data=data_bayes, columns=new_var_name_list_bayes)
    df_data_knn = pd.DataFrame(data=data_knn, columns=new_var_name_list_knn)
    df_data_nonLin = pd.DataFrame(data=data_nonLin, columns=new_var_name_list_nonLin)
    df_data_missForest = pd.DataFrame(data=data_missForest, columns=new_var_name_list_missForest)

    # concatenate dataframe horizontaly
    imput_data_bayes = pd.concat([data, df_data_bayes], axis=1)
    imput_data_knn = pd.concat([data, df_data_knn], axis=1)
    imput_data_nonLin = pd.concat([data, df_data_nonLin], axis=1)
    imput_data_missForest = pd.concat([data, df_data_missForest], axis=1)

    # Return result as a tuple of pandas dataframe
    return imput_data_bayes, imput_data_knn, imput_data_nonLin, imput_data_missForest


# function to impute NaN value with frequent category imputation using pandas
def frequent_category_imputation_with_pandas(
    *, input_data: pd.DataFrame, miss_cat_var_list: list
) -> pd.DataFrame:
    """
    Imput missing value with frequent category imputation methods.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_cat_var_list (list): The list of categorical variable with missing values.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_cat_var_list, list):
        raise ValueError("miss_cat_var_list should be a list of strings.")

    # Make a copy of a part of the input data wich contains
    # only categorical variable with missing values
    imputed_data_cat = input_data[miss_cat_var_list].copy()

    # Capture the mode of the variables in
    # a dictionary
    imputation_dict = (
        imputed_data_cat[miss_cat_var_list].copy().mode().iloc[0].to_dict()
    )

    # Impute missing value using pandas
    imputed_data_cat.fillna(imputation_dict, inplace=True)

    # Initialize imputed variable list
    imputed_list = []

    # Create a new column in data set
    for var in miss_cat_var_list:
        new_var_name = f"{var}_mod"
        imputed_list.append(new_var_name)

    # Rename imputed_data_cat columns
    imputed_data_cat.columns = imputed_list

    # data imputed
    dataframe_imputed = pd.concat(
        [input_data[miss_cat_var_list].copy(), imputed_data_cat], axis=1
    )

    # Return result as dataframe
    return dataframe_imputed


# function to impute NaN value with random sample imputation using pandas
def random_sample_imputation_for_categorical_feature_with_pandas(
    *, input_data: pd.DataFrame, miss_cat_var_list: list
) -> pd.DataFrame:
    """
    Imput missing value with random sample imputation methods.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_cat_var_list (list): The list of categorical variable with missing values.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature .
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_cat_var_list, list):
        raise ValueError("miss_cat_var_list should be a list of strings.")

    # Make a copy of a part of the input data wich contains only
    # categorical variable with missing values
    imputed_data_cat = input_data[miss_cat_var_list].copy()

    # Impute missing value using pandas
    for var in miss_cat_var_list:
        # extract the random sample to fill the na:
        random_sample_data = (
            imputed_data_cat[var]
            .dropna()
            .sample(imputed_data_cat[var].isnull().sum(), random_state=seed)
        )
        new_var_name = f"{var}_rsi"
        imputed_data_cat[new_var_name] = imputed_data_cat[var]
        # pandas needs to have the same index in order to merge datasets
        random_sample_data.index = imputed_data_cat[
            imputed_data_cat[var].isnull()
        ].index
        # replace the NA in the newly created variable
        imputed_data_cat.loc[imputed_data_cat[new_var_name].isnull(), new_var_name] = (
            random_sample_data
        )

    # Return result as dataframe
    return imputed_data_cat


# Let's find out the proportion of missing observations per variable.
def missing_values_proportion(
    *, input_data: pd.DataFrame, miss_var_list: list
) -> pd.DataFrame:
    """
    Get missing values proportion in the data set.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_var_list (list): The numerical variable.

    Returns:
    - pd.DataFrame: A pandas frame contains columns with missing values and their proportion.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_var_list, list):
        raise ValueError("miss_var_list should be a list.")

    # Make a copy of a part of the input data wich contains
    # only numerical variable with missing values
    data = input_data.copy()

    # Calculate the proportion of missing (as we did in section 3)
    # using the isnull() and mean() methods from pandas.
    data_na = data[miss_var_list].isnull().mean()

    # Transform the array into a dataframe.
    data_na = pd.DataFrame(data_na.reset_index())

    # Add column names to the dataframe.
    data_na.columns = ["nan_variable", "nan_fraction"]

    # Order the dataframe according to proportion of na per variable.
    data_na.sort_values(by="nan_fraction", ascending=False, inplace=True)

    # Return result pandas dataframe
    return data_na


# function to impute NaN value with random sample imputation using pandas
def random_sample_imputation_for_categorical_variable_with_pandas(
    *, input_data: pd.DataFrame,
    miss_cat_var_list: list
) -> pd.DataFrame:
    """
    Imput missing value with random sample imputation methods.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_cat_var_list (list): The list of categorical variable with missing values.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature .
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_cat_var_list, list):
        raise ValueError("miss_cat_var_list should be a list of strings.")

    # Make a copy of a part of the input data wich contains only
    # categorical variable with missing values
    imputed_data_cat = input_data[miss_cat_var_list].copy()

    # Impute missing value using pandas
    for var in miss_cat_var_list:
        # extract the random sample to fill the na:
        random_sample_data = (
            imputed_data_cat[var].dropna().sample(
                imputed_data_cat[var].isnull().sum(), random_state=seed
            )
        )
        new_var_name = f"{var}_rsi"
        imputed_data_cat[new_var_name] = imputed_data_cat[var]
        # pandas needs to have the same index in order to merge datasets
        random_sample_data.index = imputed_data_cat[imputed_data_cat[var].isnull()].index
        # replace the NA in the newly created variable
        imputed_data_cat.loc[
            imputed_data_cat[new_var_name].isnull(), new_var_name
        ] = random_sample_data

    # Return result as tuple
    return imputed_data_cat


# Complete Case Analysis (CCA) Imputation
def complete_case_analysis_imputation_with_pandas(
    *,
    input_data: pd.DataFrame,
    miss_var_list: list,
    missing_proportion_threshold: float,
) -> pd.DataFrame:
    """
    Imput missing value with complete case analysis imputation methods.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - miss_var_list (list): The list of numerical variables.
    - missing_proportion_threshold (float): The threshold for linearity of variables.

    Returns:
    - pd.DataFrame: A pandas frame without all nan rows.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(miss_var_list, list):
        raise ValueError("num_var_list should be a list of strings.")
    if not isinstance(missing_proportion_threshold, float):
        raise ValueError("missing_proportion_threshold should be a float.")

    # Make a copy of a part of the input data wich contains
    # only categorical variable with missing values
    data = input_data.copy()

    # Capture variables with less than 5% NA
    # in a list.
    vars_cca = [
        var
        for var in miss_var_list
        if data[var].isnull().mean() < missing_proportion_threshold
    ]

    # Create the complete case dataset,
    # in other words, remove observations with na in any variable.
    data_cca = data.dropna(subset=vars_cca)

    # Return result as dataframe
    return data_cca


# function to plot distribution and Q-Q plot of data
def distribution_diagnostic_plots(
    *, dataframe: pd.DataFrame, target: str, bins: int, figsize: tuple
) -> None:
    """
    Create distribution diagnostic plots for a variable.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame.
    - target (str): The target variable.
    - bins (int, optional): Number of bins for the histogram. Default is 20.
    - figsize (tuple, optional): Figure size for the plots. Default is (8, 4).

    Returns:
    - None
    """
    # Check the type of variables passed in the function
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe should be pandas dataframe.")
    if not isinstance(target, str):
        raise ValueError("target should be a string.")
    if not isinstance(bins, int):
        raise ValueError("bins should be an integer.")
    if not isinstance(figsize, tuple):
        raise ValueError("figsize should be tuple.")

    # Copy input data
    data = dataframe.copy()

    # Get numerical variables
    num_var_list = [
        var for var in data.columns if data[var].dtype != "O" and var != target
    ]

    # Plot the distribution of numerical variables
    for var in num_var_list:
        # Plot histogram and Q-Q plot
        plt.figure(figsize=figsize)

        # Histogram
        plt.subplot(1, 3, 1)
        plt.title("Histogram")
        fig = data[var].dropna().hist(bins=bins)
        fig.set_ylabel("Number of " + var)
        fig.set_xlabel(var)

        # Q-Q plot
        plt.subplot(1, 3, 2)
        plt.title("Q-Q plot")
        fig = stats.probplot(data[var].dropna(), dist="norm", plot=plt)

        # boxplot
        plt.subplot(1, 3, 3)
        sns.boxplot(y=data[var].dropna())
        plt.title('Boxplot')

        plt.show()


# function to apply QuantileTransformer() for numerical variable transformation
# this transformation + RobustScaler() tends to be suitable for continuous numerical variable
# and for reduce the impact of (marginal) outliers
# In order to improve the model performance, this transformer can also applied to target
# and fit model witch is suitable for target transformed. Don't forget in production
# to apply inverse of the transformer to the prediction of the model to obtain the real target
def quantile_transformer_method_for_variable_transformation(
    *, input_data: pd.DataFrame, num_var_list: list
) -> pd.DataFrame:
    """
    Apply quantile transformation to numerical skewed variable.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - num_var_list (list): The list of numerical variable.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and transformed feature.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be a list of strings.")

    # Define transformer
    scaler = QuantileTransformer(output_distribution="normal", random_state=seed)

    # Make a copy of a part of the input data wich contains
    # only numerical variable with missing values
    data = input_data[num_var_list].copy()
    new_var_name_list = []

    for var in num_var_list:
        # create new variable and append it into new list
        new_var_name = f"{var}_quant"
        new_var_name_list.append(new_var_name)

    # Apply transformation to data
    scaler.fit(data)
    data_ndarray = scaler.transform(data)

    # creating the dataframe
    data_frame = pd.DataFrame(data=data_ndarray, columns=new_var_name_list)

    # concatenate dataframe horizontaly
    data_transformed = pd.concat([data, data_frame], axis=1)

    # Return result pandas dataframe
    return data_transformed


# function to collect positive numerical variables in dataset
def get_positive_numerical_variable(*, input_data: pd.DataFrame, target: str) -> list:
    """
    Identify positive numerical variables in a DataFrame.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - target (str): The target variable.

    Returns:
    - list: A list of positive numerical variable names.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(target, str):
        raise ValueError("target should be strings.")

    # Make a copy of the input data
    data = input_data.copy()

    # Get numerical variables
    numerical_variables = [
        var for var in data.columns.tolist() if data[var].dtype != "O" and var != target
    ]
    positive_numerical_variables = [
        var for var in numerical_variables if (data[var] > 0).any()
    ]

    # Return results as list
    return positive_numerical_variables


# function to apply logarithm for numerical variable transformation
# this transformation tends to deal with positive variable with a right-skewed distribution
def log_transformer_method_for_variable_transformation(
    *, input_data: pd.DataFrame, num_var_list: list
) -> pd.DataFrame:
    """
    Apply log transformation for skewed variable.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - num_var_list (list): The list of numerical variable.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be a list of strings.")

    # Make a copy of a part of the input data
    data = input_data[num_var_list].dropna().copy()
    new_var_name_list = []

    # Initialize dictionary
    new_data_dict = {}

    for var in num_var_list:
        # create new variable and append it into new list
        new_var_name = f"{var}_log"
        new_var_name_list.append(new_var_name)

        if (data[var] == 0).any():
            # Apply transformation to data contains 0
            data[new_var_name] = np.log(data[var] + 1)
        else:
            # Apply transformation to data don't contains 0
            new_data_dict[new_var_name] = np.log(data[var])

    # concatenate dataframe horizontaly
    data_transformed = pd.concat(
        [data.copy(), pd.DataFrame.from_dict(new_data_dict)], axis=1
    )

    # Return result pandas dataframe
    return data_transformed


# function to apply reciprocal for numerical variable transformation
# this transformation tends to useful when we have ratio
def reciprocal_transformer_method_for_variable_transformation(
    *, input_data: pd.DataFrame, num_var_list: list
) -> pd.DataFrame:
    """
    Apply reciprocal transformer for skewed variable.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - num_var_list (list): The list of numerical variable.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be a list of strings.")

    # Make a copy of a part of the input data
    data = input_data[num_var_list].dropna().copy()
    new_var_name_list = []

    # Initialize dictionary
    new_data_dict = {}

    for var in num_var_list:
        # create new variable and append it into new list
        new_var_name = f"{var}_rec"
        new_var_name_list.append(new_var_name)

        if (data[var] == 0).any():
            # don't Apply reciprocal transformation to data contains 0
            pass
        else:
            # Apply transformation to data don't contains 0
            new_data_dict[new_var_name] = np.reciprocal(data[var])

    # concatenate dataframe horizontaly
    data_transformed = pd.concat(
        [data.copy(), pd.DataFrame.from_dict(new_data_dict)], axis=1
    )

    # Return result pandas dataframe
    return data_transformed


# function to apply square root for numerical variable transformation
# this transformation tends to suitable for variable with count
def square_root_transformer_method_for_variable_transformation(
    *, input_data: pd.DataFrame, num_var_list: list
) -> pd.DataFrame:
    """
    Apply square root transformers for skewed variable.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - num_var_list (list): The list of numerical variable.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be a list of strings.")

    # Make a copy of a part of the input data
    data = input_data[num_var_list].dropna().copy()
    new_var_name_list = []

    # Initialize dictionary
    new_data_dict = {}

    for var in num_var_list:
        # create new variable and append it into new list
        new_var_name = f"{var}_sqrt"
        new_var_name_list.append(new_var_name)

        if (data[var] < 0).any():
            # don't Apply reciprocal transformation to negative data
            pass
        else:
            # Apply transformation to data don't contains 0
            new_data_dict[new_var_name] = np.round(np.sqrt(data[var]), 2)

    # concatenate dataframe horizontaly
    data_transformed = pd.concat(
        [data.copy(), pd.DataFrame.from_dict(new_data_dict)], axis=1
    )

    # Return result pandas dataframe
    return data_transformed


# function to apply arcsin for numerical variable transformation
# this transformation help in dealing with probabilities, percentages, and proportions
def arcsin_transformer_method_for_variable_transformation(
    *, input_data: pd.DataFrame, num_var_list: list
) -> pd.DataFrame:
    """
    Applt arcsin transformer for skewed variable.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - num_var_list (list): The list of numerical variable.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be a list of strings.")

    # Make a copy of a part of the input data
    data = input_data[num_var_list].dropna().copy()
    new_var_name_list = []

    # Initialize dictionary
    new_data_dict = {}

    for var in num_var_list:
        # create new variable and append it into new list
        new_var_name = f"{var}_arcsin"
        new_var_name_list.append(new_var_name)

        if (data[var] < 0).any():
            # don't Apply reciprocal transformation to negative data
            pass
        else:
            # Apply transformation to data don't contains 0
            new_data_dict[new_var_name] = np.arcsin(np.sqrt(data[var]))

    # concatenate dataframe horizontaly
    data_transformed = pd.concat(
        [data.copy(), pd.DataFrame.from_dict(new_data_dict)], axis=1
    )

    # Return result pandas dataframe
    return data_transformed


# function to apply Box-cox for numerical variable transformation
# this transformation help to find the best transformation between
# logarithm, reciprocal and square root
def boxcox_transformer_method_for_variable_transformation(
    *, input_data: pd.DataFrame, num_var_list: list
) -> pd.DataFrame:
    """
    Apply boxcox function to transform numerical skewed variable.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - num_var_list (list): The list of numerical variable.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be a list of strings.")

    # Make a copy of a part of the input data
    data = input_data[num_var_list].dropna().copy()
    new_var_name_list = []

    # Initialize dictionary
    new_data_dict = {}
    param_dict = {}

    for var in num_var_list:
        # create new variable and append it into new list
        new_var_name = f"{var}_boxcox"
        new_var_name_list.append(new_var_name)

        if (data[var] <= 0).any():
            # don't Apply reciprocal transformation to negative data or data contains 0
            pass
        else:
            # Apply transformation to strictly positive data
            new_data_dict[new_var_name] = stats.boxcox(data[var])[0]
            param_dict[new_var_name] = stats.boxcox(data[var])[1]

    # concatenate dataframe horizontaly
    data_transformed = pd.concat(
        [data.copy(), pd.DataFrame.from_dict(new_data_dict)], axis=1
    )

    # Return result pandas dataframe
    return data_transformed


# function to apply Yeo-Johnson for numerical variable transformation
# this transformation extend Box-cox transformation to zero and negative variable
# In order to improve the model performance, this transformer can also applied to target
# and fit model witch is suitable for target transformed. Don't forget in production to apply
# inverse of the transformer to the prediction of the model to obtain the real target
def yeojohnson_transformer_method_for_variable_transformation(
    *, input_data: pd.DataFrame, num_var_list: list
) -> pd.DataFrame:
    """
    Apply yeojohnson function to transform numerical skewed variable.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - num_var_list (list): The list of numerical variable.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be a list of strings.")

    # Make a copy of a part of the input data
    data = input_data[num_var_list].copy()
    new_var_name_list = []

    # Initialize dictionary
    new_data_dict = {}
    param_dict = {}

    for var in num_var_list:
        # create new variable and append it into new list
        new_var_name = f"{var}_yeojoh"
        new_var_name_list.append(new_var_name)
        # Apply transformation to strictly positive data
        new_data_dict[new_var_name] = stats.yeojohnson(data[var])[0]
        param_dict[new_var_name] = stats.yeojohnson(data[var])[1]

    # concatenate dataframe horizontaly
    data_transformed = pd.concat(
        [data.copy(), pd.DataFrame.from_dict(new_data_dict)], axis=1
    )

    # Return result pandas dataframe
    return data_transformed


# Unsupervised discretization methods
# Equal width discretisation
# Equal frequency discretization
# kbinsdiscretizer discretization


# function to apply Equal width discretisation for numerical variable transformation
# (discretization + encoding + reorder)
# this transformation tends to be suitable for continuous numerical variable
def equal_width_discretisation_for_variable_transformation(
    *, input_data: pd.DataFrame, num_var_list: list, number_of_bins: int
) -> pd.DataFrame:
    """
    Apply discretization method to handle skewed variable.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - num_var_list (list): The list of numerical variable.
    - number_of_bins (int): The number of bins or interval for discretization.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be a list of strings.")
    if not isinstance(number_of_bins, int):
        raise ValueError("number_of_bins should be an integer.")

    # Define transformer
    disc = EqualWidthDiscretiser(bins=number_of_bins, variables=num_var_list)

    # Make a copy of a part of the input data wich contains
    # only numerical variable with missing values
    data = input_data[num_var_list].copy()
    new_var_name_list = []

    for var in num_var_list:
        # create new variable and append it into new list
        new_var_name = f"{var}_eqwid"
        new_var_name_list.append(new_var_name)

    # Apply transformation to data
    disc.fit(data)
    data_frame = disc.transform(data)

    # rename the column of dataframe
    data_frame.columns = new_var_name_list

    # concatenate dataframe horizontaly
    data_transformed = pd.concat([data, data_frame], axis=1)

    # Return result pandas dataframe
    return data_transformed


# function to apply Equal frequency discretisation for numerical variable transformation
# (discretization + encoding + reorder)
# this transformation tends to be suitable for continuous numerical variable
def equal_frequency_discretisation_for_variable_transformation(
    *, input_data: pd.DataFrame, num_var_list: list, number_of_frequence: int
) -> pd.DataFrame:
    """
    Apply discretization method to handle skewed variable.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - num_var_list (list): The list of numerical variable.
    - number_of_frequence (int): The number of frequence for discretization.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be a list of strings.")
    if not isinstance(number_of_frequence, int):
        raise ValueError("number_of_frequence should be an integer.")

    # Define transformer
    disc = EqualFrequencyDiscretiser(
        q=number_of_frequence, variables=num_var_list, return_boundaries=False
    )

    # Make a copy of a part of the input data wich contains
    # only numerical variable with missing values
    data = input_data[num_var_list].copy()
    new_var_name_list = []

    for var in num_var_list:
        # create new variable and append it into new list
        new_var_name = f"{var}_eqfreq"
        new_var_name_list.append(new_var_name)

    # Apply transformation to data
    disc.fit(data)
    data_frame = disc.transform(data)

    # rename the column of dataframe
    data_frame.columns = new_var_name_list

    # concatenate dataframe horizontaly
    data_transformed = pd.concat([data, data_frame], axis=1)

    # Return result pandas dataframe
    return data_transformed


# function to apply KBinsDiscretizer() for numerical variable transformation
# (discretization + encoding + reorder)
# this transformation tends to be suitable for continuous numerical variable
def kbinsdiscretizer_for_variable_transformation(
    *, input_data: pd.DataFrame, num_var_list: list, number_of_bins: int
) -> pd.DataFrame:
    """
    Apply discretization method to handle skewed variable.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - num_var_list (list): The list of numerical variable.
    - number_of_bins (int): The number of bins or interval for discretization.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be a list of strings.")
    if not isinstance(number_of_bins, int):
        raise ValueError("number_of_bins should be an integer.")

    # Define transformer
    disc = KBinsDiscretizer(
        n_bins=number_of_bins, encode="ordinal", strategy="kmeans", subsample=None
    )

    # Make a copy of a part of the input data wich contains
    # only numerical variable with missing values
    data = input_data[num_var_list].copy()
    new_var_name_list = []

    for var in num_var_list:
        # create new variable and append it into new list
        new_var_name = f"{var}_kbin"
        new_var_name_list.append(new_var_name)

    # Apply transformation to data
    disc.fit(data)
    data_ndarray = disc.transform(data)

    # creating the dataframe
    data_frame = pd.DataFrame(data=data_ndarray, columns=new_var_name_list)

    # concatenate dataframe horizontaly
    data_transformed = pd.concat([data, data_frame], axis=1)

    # Return result pandas dataframe
    return data_transformed


# Supervised discretization methods
# Discretization using decision trees
# Binarization


# function to apply decision trees discretisation for numerical variable transformation
# (discretization + encoding + reorder)
# this transformation tends to be suitable for continuous numerical variable
def decision_trees_discretisation_for_variable_transformation(
    *, input_data: pd.DataFrame, num_var_list: list, target_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply discretization method to handle skewed variable.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - num_var_list (list): The list of numerical variable.
    - target_data (pd.DataFrame): The target DataFrame.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame) or not isinstance(
        target_data, pd.DataFrame
    ):
        raise ValueError("Both input_data and target_data should be pandas dataframe.")
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be a list of strings.")

    # Define transformer
    disc = DecisionTreeDiscretiser(
        cv=5,
        scoring="accuracy",
        variables=num_var_list,
        regression=False,
        param_grid={"max_depth": [1, 2, 3], "min_samples_leaf": [10, 4]},
    )

    # Make a copy of a part of the input data wich contains
    # only numerical variable with missing values
    data = input_data[num_var_list].copy()
    target = target_data  # .values.flatten().copy()
    new_var_name_list = []

    for var in num_var_list:
        # create new variable and append it into new list
        new_var_name = f"{var}_tree"
        new_var_name_list.append(new_var_name)

    # Apply transformation to data
    disc.fit(data, target)
    data_frame = disc.transform(data)

    # rename the column of dataframe
    data_frame.columns = new_var_name_list

    # concatenate dataframe horizontaly
    data_transformed = pd.concat([data, data_frame], axis=1)

    # Return result pandas dataframe
    return data_transformed


# function to apply Binarization for numerical variable transformation
# this transformation tends to be suitable for continuous numerical variable
def binarization_for_variable_transformation(
    *, input_data: pd.DataFrame, num_var_list: list
) -> pd.DataFrame:
    """
    Apply binarization method to handle skewed variable.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - num_var_list (list): The list of numerical variable.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new imputing feature.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be a list of strings.")

    # Define transformer
    disc = Binarizer(threshold=0).set_output(transform="pandas")

    # Make a copy of a part of the input data wich contains
    # only numerical variable with missing values
    data = input_data[num_var_list].copy()
    new_var_name_list = []

    for var in num_var_list:
        # create new variable and append it into new list
        new_var_name = f"{var}_binariz"
        new_var_name_list.append(new_var_name)

    # Apply transformation to data
    disc.fit(data)
    data_frame = disc.transform(data)

    # rename the column of dataframe
    data_frame.columns = new_var_name_list

    # concatenate dataframe horizontaly
    data_transformed = pd.concat([data, data_frame], axis=1)

    # Return result pandas dataframe
    return data_transformed


# function to reoder variable transformed (discretization + encoding + reorder)
# this transformation tends to be suitable for continuous numerical variable
def reorder_variable_by_target(
    *, input_data: pd.DataFrame, target_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Reordering of the variable.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - target_data (pd.DataFrame): The target DataFrame.

    Returns:
    - pd.DataFrame: A pandas frame contains reorder data.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame) or not isinstance(
        target_data, pd.DataFrame
    ):
        raise ValueError("Both input_data and target_data should be pandas dataframe.")

    # Define the reorder tools
    reorder = OrdinalEncoder(encoding_method="ordered", ignore_format=True)

    # Make a copy of a part of the input data wich contains
    # only numerical variable with missing values
    data = input_data.copy()
    target = target_data.values.flatten().copy()

    # Let's try to improve it with ordered ordinal encoding by target
    data_reorder = reorder.fit_transform(data, target)

    # Return result pandas dataframe
    return data_reorder


# function to find limits of outliers with IQR method
# suitable for skewed variable
# normaly fold = 1.5, but for extrem skewed variable fold = 3.0
def find_limits_with_IQR(
    *, input_data: pd.DataFrame, variable: str, fold: float
) -> tuple:
    """
    Find the limits of outliers with IQR method.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - variable (str): The numerical variable for study.
    - fold (float): The fold for limit.

    Returns:
    - tuple: A tuple contains lower_limit and upper_limit.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(variable, str):
        raise ValueError("variable should be a strings.")
    if not isinstance(fold, float):
        raise ValueError("fold should be float.")

    # To calcul the IQR
    IQR = input_data[variable].quantile(0.75) - input_data[variable].quantile(0.25)

    # Calcul lower_limit and upper_limit
    lower_limit = input_data[variable].quantile(0.25) - (IQR * fold)
    upper_limit = input_data[variable].quantile(0.75) + (IQR * fold)

    # Return result as tuple
    return (lower_limit, upper_limit)


# suitable for variable with normaly distribution
# normaly fold = 3
# function to find limits of outliers with Normal distribution method
def find_limits_with_normal_distribution(
    *, input_data: pd.DataFrame, variable: str, fold: float
) -> tuple:
    """
    Find the limits of outliers with normaly distribution method.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - variable (str): The numerical variable for study.
    - fold (float): The fold for limit.

    Returns:
    - tuple: A tuple contains lower_limit and upper_limit.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(variable, str):
        raise ValueError("variable should be a strings.")
    if not isinstance(fold, float):
        raise ValueError("fold should be float.")

    # Calcul lower_limit and upper_limit
    lower_limit = input_data[variable].mean() - (input_data[variable].std() * fold)
    upper_limit = input_data[variable].mean() + (input_data[variable].std() * fold)

    # Return result as tuple
    return (lower_limit, upper_limit)


# function to find limits of outliers with quantile method
def find_limits_with_quantile(*, input_data: pd.DataFrame, variable: str) -> tuple:
    """
    Find the limits of outliers with quantile method.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - variable (str): The numerical variable for study.

    Returns:
    - tuple: A tuple contains lower_limit and upper_limit.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(variable, str):
        raise ValueError("variable should be a strings.")

    # Calcul lower_limit and upper_limit
    lower_limit = input_data[variable].quantile(0.05)
    upper_limit = input_data[variable].quantile(0.95)

    # Return result as tuple
    return (lower_limit, upper_limit)


# function to remove or censor outliers with pandas
def remove_outliers(
    *, input_data: pd.DataFrame, variable: str, lower_limit: float, upper_limit: float
) -> pd.DataFrame:
    """
    Remove outliers in the data set.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - variable (str): The numerical variable.
    - lower_limit (float): The lower limit.
    - upper_limit (float): The upper limit.

    Returns:
    - pd.DataFrame: A pandas frame contains data without outliers.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(variable, str):
        raise ValueError("variable should be a strings.")
    if not isinstance(lower_limit, float) or not isinstance(upper_limit, float):
        raise ValueError("Both lower_limit and upper_limit should be float.")

    # Make a copy of a part of the input data wich contains
    # only numerical variable with missing values
    data = input_data.copy()

    # Remove outliers on the left
    # ge: greater or equal than ==> True/False vector
    inliers = data[variable].ge(lower_limit)
    data = data.loc[inliers]

    # Remove outliers on the right
    # le: lower or equal than ==> True/False vector
    inliers = data[variable].le(upper_limit)
    data = data.loc[inliers]

    # data without outliers for variable
    data_without_outliers = data

    # Return result pandas dataframe
    return data_without_outliers


# function to capping outliers with pandas
def capping_outliers(
    *, input_data: pd.DataFrame, variable: str, lower_limit: float, upper_limit: float
) -> pd.DataFrame:
    """
    Capping outliers in the data set.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - variable (str): The numerical variable.
    - lower_limit (float): The lower limit.
    - upper_limit (float): The upper limit.

    Returns:
    - pd.DataFrame: A pandas frame contains data without outliers.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(variable, str):
        raise ValueError("variable should be a strings.")
    if not isinstance(lower_limit, float) or not isinstance(upper_limit, float):
        raise ValueError("Both lower_limit and upper_limit should be float.")

    # Make a copy of a part of the input data wich contains
    # only numerical variable with missing values
    data = input_data.copy()

    # Cap variables
    # data[variable].clip(lower=lower_limit, upper=upper_limit, inplace=True)
    data[variable] = data[variable].clip(
        lower=lower_limit, upper=upper_limit, inplace=True
    )

    # data without outliers for variable
    data_capping = data

    # Return result pandas dataframe
    return data_capping


# function to create new features that capture information about
# presence or abscence of Outliers in data set
# suitable for skewed variable
# normaly fold = 1.5, but for extrem skewed variable fold = 3.0
def add_outliers_indicator_with_IQR(
    *, input_data: pd.DataFrame, num_var_list: list, fold: float
) -> pd.DataFrame:
    """
    Add outliers indicator to data set.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - num_var_list (list): The numerical variable list for study.
    - fold (float): The fold for limit.

    Returns:
    - pd.DataFrame: A data frame contains outliers indicator.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be a list.")
    if not isinstance(fold, float):
        raise ValueError("fold should be float.")

    # Make a copy of input data
    data = input_data.copy()

    # initialize list
    lower_limit_list = []
    upper_limit_list = []

    for var in num_var_list:
        # find the limit
        lower_limit, upper_limit = find_limits_with_IQR(
            input_data=data, variable=var, fold=1.5
        )
        # Create lower_limit and upper_limit list
        lower_limit_list.append(lower_limit)
        upper_limit_list.append(upper_limit)

    # Create a list to store the new Series: for min_variable_list
    new_columns = []

    # Loop through each min variable
    for count, var in enumerate(num_var_list):
        # Create a Series based on the conditions
        new_column = pd.Series(
            np.where(
                (data[var] < lower_limit_list[count])
                | (data[var] > upper_limit_list[count]),
                1,
                0,
            ),
            index=data.index,  # Preserve original index
            name=f"{var}_outliers",  # Name of the new column
        )
        # Append the new Series to the list
        new_columns.append(new_column)

    # Concatenate the list of Series along the columns axis
    data = pd.concat([data] + new_columns, axis=1)

    # Return result as dataframe
    return data


# function to create new features that capture information about
# presence or abscence of Outliers in data set
# suitable for variable with normaly distribution
# normaly fold = 3
# function to find limits of outliers with Normal distribution method
def add_outliers_indicator_with_Gaussian(
    *, input_data: pd.DataFrame, num_var_list: list, fold: float
) -> pd.DataFrame:
    """
    Add outliers indicator to data set.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - num_var_list (list): The numerical variable list for study.
    - fold (float): The fold for limit.

    Returns:
    - pd.DataFrame: A data frame contains outliers indicator.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be a list.")
    if not isinstance(fold, float):
        raise ValueError("fold should be float.")

    # Make a copy of input data
    data = input_data.copy()

    # initialize list
    lower_limit_list = []
    upper_limit_list = []

    for var in num_var_list:
        # find the limit
        lower_limit, upper_limit = find_limits_with_normal_distribution(
            input_data=data, variable=var, fold=1.5
        )
        # Create lower_limit and upper_limit list
        lower_limit_list.append(lower_limit)
        upper_limit_list.append(upper_limit)

    # Create a list to store the new Series: for min_variable_list
    new_columns = []

    # Loop through each min variable
    for count, var in enumerate(num_var_list):
        # Create a Series based on the conditions
        new_column = pd.Series(
            np.where(
                (data[var] < lower_limit_list[count])
                | (data[var] > upper_limit_list[count]),
                1,
                0,
            ),
            index=data.index,  # Preserve original index
            name=f"{var}_outliers",  # Name of the new column
        )
        # Append the new Series to the list
        new_columns.append(new_column)

    # Concatenate the list of Series along the columns axis
    data = pd.concat([data] + new_columns, axis=1)

    # Return result as dataframe
    return data


# function to create new features that capture information about
# presence or abscence of Outliers in data set
# suitable for variable with normaly distribution
# normaly fold = 3
# function to find limits of outliers with Normal distribution method
def add_outliers_indicator_with_Quantiles(
    *, input_data: pd.DataFrame, num_var_list: list
) -> pd.DataFrame:
    """
    Add outliers indicator to data set.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - num_var_list (list): The numerical variable list for study.

    Returns:
    - pd.DataFrame: A data frame contains outliers indicator.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be a list.")

    # Make a copy of input data
    data = input_data.copy()

    # initialize list
    lower_limit_list = []
    upper_limit_list = []

    for var in num_var_list:
        # find the limit
        lower_limit, upper_limit = find_limits_with_quantile(
            input_data=data, variable=var
        )
        # Create lower_limit and upper_limit list
        lower_limit_list.append(lower_limit)
        upper_limit_list.append(upper_limit)

    # Create a list to store the new Series: for min_variable_list
    new_columns = []

    # Loop through each min variable
    for count, var in enumerate(num_var_list):
        # Create a Series based on the conditions
        new_column = pd.Series(
            np.where(
                (data[var] < lower_limit_list[count])
                | (data[var] > upper_limit_list[count]),
                1,
                0,
            ),
            index=data.index,  # Preserve original index
            name=f"{var}_outliers",  # Name of the new column
        )
        # Append the new Series to the list
        new_columns.append(new_column)

    # Concatenate the list of Series along the columns axis
    data = pd.concat([data] + new_columns, axis=1)

    # Return result as dataframe
    return data


# function to plot box plot and histogramme for numerical variable
def plot_boxplot_and_hist(*, data: pd.DataFrame, variable: str) -> None:
    """
    Capping outliers in the data set.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - variable (str): The numerical variable.

    Returns:
    - None: None value.
    """
    # Check the type of variables passed in the function
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data should be pandas dataframe.")
    if not isinstance(variable, str):
        raise ValueError("variable should be a strings.")

    # creating a figure composed of two matplotlib.Axes
    # objects (ax_box and ax_hist)

    f, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, gridspec_kw={"height_ratios": (0.50, 0.85)}
    )

    # assigning a graph to each ax
    sns.boxplot(x=data[variable], ax=ax_box)
    sns.histplot(data=data, x=variable, ax=ax_hist)

    # Remove x axis name for the boxplot
    ax_box.set(xlabel="")
    plt.title(variable)
    plt.show()

    # Return result as None
    return None


# function to generate new variable based on somme
def variable_generated_by_somme_of_two_columns(
    *, input_data: pd.DataFrame, var_list: list
) -> pd.DataFrame:
    """
    Generate new columns based on somme of existant columns.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - var_list (list): The list of numerical variable use to make the somme.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new feature .
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(var_list, list):
        raise ValueError("var_list should be a list of strings.")

    # Make a copy of a part of the input data wich contains
    # only categorical variable with missing values
    new_data = input_data.copy()

    # Initialize
    duplicate_list = []

    # Generate somme column
    if var_list:
        for var1 in var_list:
            for var2 in var_list:
                if var2 not in duplicate_list and var2 != var1:
                    new_var_name = f"{var1}_som_{var2}"
                    new_series = pd.Series(
                        pd.to_numeric(new_data[var1].copy())
                        + pd.to_numeric(new_data[var2].copy()),
                        name=new_var_name,
                    )
                    new_series_frame = pd.DataFrame(new_series, columns=[new_var_name])

                    # update new data frame
                    new_data = pd.concat([new_data.copy(), new_series_frame], axis=1)

            # lits to avoid redundant somme like var1*var2 and var2*var1
            duplicate_list.append(var1)

    # Return result as dataframe
    return new_data


# function to generate new variable based on product
def variable_generated_by_product_of_two_columns(
    *, input_data: pd.DataFrame, var_list: list
) -> pd.DataFrame:
    """
    Generate new columns based on product of existant columns.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - var_list (list): The list of numerical variable use to make the product.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new feature .
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(var_list, list):
        raise ValueError("var_list should be a list of strings.")

    # Make a copy of a part of the input data wich contains
    # only categorical variable with missing values
    new_data = input_data.copy()

    # Initialize
    duplicate_list = []

    # Generate product column
    if var_list:
        for var1 in var_list:
            for var2 in var_list:
                if var2 not in duplicate_list and var2 != var1:
                    new_var_name = f"{var1}_prod_{var2}"
                    new_series = pd.Series(
                        pd.to_numeric(new_data[var1].copy())
                        * pd.to_numeric(new_data[var2].copy()),
                        name=new_var_name,
                    )
                    new_series_frame = pd.DataFrame(new_series, columns=[new_var_name])

                    # update new data frame
                    new_data = pd.concat([new_data.copy(), new_series_frame], axis=1)

            # lits to avoid redundant product like var1*var2 and var2*var1
            duplicate_list.append(var1)

    # Return result as dataframe
    return new_data


# function to generate new variable based on ration
def variable_generated_by_ratio_of_two_columns(
    *, input_data: pd.DataFrame, var_list: list
) -> pd.DataFrame:
    """
    Generate new columns based on ratio of existant columns.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - var_list (list): The list of numerical variable use to make the ratio.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new feature .
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(var_list, list):
        raise ValueError("var_list should be a list of strings.")

    # Make a copy of a part of the input data wich contains
    # only categorical variable with missing values
    new_data = input_data.copy()

    # Generate ratio column
    if var_list:
        for var1 in var_list:
            for var2 in var_list:
                if not (new_data[var2] == 0).any() and var2 != var1:
                    new_var_name = f"{var1}_div_{var2}"
                    new_series = pd.Series(
                        pd.to_numeric(new_data[var1].copy())
                        / pd.to_numeric(new_data[var2].copy()),
                        name=new_var_name,
                    )
                    new_series_frame = pd.DataFrame(new_series, columns=[new_var_name])

                    # update new data frame
                    new_data = pd.concat([new_data.copy(), new_series_frame], axis=1)

    # Return result as dataframe
    return new_data


# Only for linear regression problem
# function to check linear and non-linear relationship between features ans target
def linear_and_nonlinear_relationship_between_features_and_target(
    *,
    input_data: pd.DataFrame,
    feature_var_list: list,
    target_var: str,
    threshold: float,
) -> tuple:
    """
    Check linear and non linear relationship between feature in the data set.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - feature_var_list (list): The list of feature variable.
    - target_var (str): The target variable.
    - threshold (float): The linearity and non-linearity threshold.

    Returns:
    - tuple: A tuple of list.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(feature_var_list, list):
        raise ValueError("feature_var_list should be a list of strings.")
    if not isinstance(target_var, str):
        raise ValueError("featuretarget_var should be a strings.")
    if not isinstance(threshold, float):
        raise ValueError("threshold should be a float.")

    # # import library and set imputer
    # from feature_engine.imputation import MeanMedianImputer
    # imputer = MeanMedianImputer(
    #     imputation_method='median',
    #     variables=feature_var_list
    # )

    # # Make a copy of a part of the input data
    # new_data = imputer.fit_transform(input_data.copy())
    # new_data = input_data.copy()

    # Initialize
    linear_var_list = []
    linear_pvalue_list = []
    linear_statistic_list = []
    no_linear_var_list = []
    no_linear_pvalue_list = []
    no_linear_statistic_list = []
    nonlinear_var_list = []
    nonlinear_pvalue_list = []
    nonlinear_statistic_list = []
    no_nonlinear_var_list = []
    no_nonlinear_pvalue_list = []
    no_nonlinear_statistic_list = []

    # Generate product column
    for var in feature_var_list:
        # data to process
        new_data = pd.DataFrame()
        new_data = input_data.copy()[[var, target_var]].dropna()

        # linearity test
        pr_result = scipy.stats.pearsonr(new_data[var], new_data[target_var])
        if abs(pr_result.statistic) >= threshold:
            linear_var_list.append(var)
            linear_pvalue_list.append(pr_result.pvalue)
            linear_statistic_list.append(pr_result.statistic)
        else:
            no_linear_var_list.append(var)
            no_linear_pvalue_list.append(pr_result.pvalue)
            no_linear_statistic_list.append(pr_result.statistic)

        # non-linearity test
        sp_result = scipy.stats.spearmanr(new_data[var], new_data[target_var])
        if abs(sp_result.statistic) >= threshold:
            nonlinear_var_list.append(var)
            nonlinear_pvalue_list.append(sp_result.pvalue)
            nonlinear_statistic_list.append(pr_result.statistic)
        else:
            no_nonlinear_var_list.append(var)
            no_nonlinear_pvalue_list.append(sp_result.pvalue)
            no_nonlinear_statistic_list.append(pr_result.statistic)

    # Initialize
    law_variation_with_target_list = []
    high_variation_with_target_list = []

    for var in feature_var_list:
        if var in no_linear_var_list and var in no_nonlinear_var_list:
            law_variation_with_target_list.append(var)
        else:
            high_variation_with_target_list.append(var)

    # Return result as tuple
    return (
        linear_var_list,
        linear_pvalue_list,
        linear_statistic_list,
        nonlinear_var_list,
        nonlinear_pvalue_list,
        nonlinear_statistic_list,
        no_linear_var_list,
        no_linear_pvalue_list,
        no_linear_statistic_list,
        no_nonlinear_var_list,
        no_nonlinear_pvalue_list,
        no_nonlinear_statistic_list,
        law_variation_with_target_list,
        high_variation_with_target_list,
    )


# Only for linear regression problem
# function to check linear and non-linear relationship between features ans target
def multicolinearity_between_features_by_VIF_method(
    *, input_data: pd.DataFrame, feature_var_list: list, threshold: float
) -> tuple:
    """
    Check multicolinearity between feature in the data set.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - feature_var_list (list): The list of feature variable.
    - threshold (float): The multicolinearity threshold.

    Returns:
    - tuple: A tuple of frame and list.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(feature_var_list, list):
        raise ValueError("feature_var_list should be a list of strings.")
    if not isinstance(threshold, float):
        raise ValueError("threshold should be a float.")

    # Make a copy of a part of the input data
    new_data = input_data[feature_var_list].copy()

    # Initialize
    var_with_high_vif_list = []
    value_with_high_vif_list = []
    var_with_less_vif_list = []
    value_with_less_vif_list = []

    vif = pd.DataFrame()
    vif["features"] = new_data.columns
    vif["VIF Factor"] = [
        variance_inflation_factor(new_data.values, i) for i in range(new_data.shape[1])
    ]
    vif.round(1)

    # Append into list features witch have high multicoliearity
    for i in range(vif.shape[0]):
        if vif.copy().loc[i, "VIF Factor"] >= threshold:
            var_with_high_vif_list.append(vif["features"].values.tolist()[i])
            value_with_high_vif_list.append(vif.copy().loc[i, "VIF Factor"])
        else:
            var_with_less_vif_list.append(vif["features"].values.tolist()[i])
            value_with_less_vif_list.append(vif.copy().loc[i, "VIF Factor"])

    # Return result as tuple
    return (
        vif.sort_values(by="VIF Factor", ascending=False),
        var_with_high_vif_list,
        value_with_high_vif_list,
        var_with_less_vif_list,
        value_with_less_vif_list,
    )


# function to check correlation between features and target
def multicolinearity_between_features_by_correlation_method(
    *, input_data: pd.DataFrame, feature_var_list: list, threshold: float
) -> pd.DataFrame:
    """
    Generate new columns based on product of existant columns.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - feature_var_list (list): The list of feature variable.
    - threshold (float): The multicolinearity threshold.

    Returns:
    - pd.DataFrame: A pandas frame contains initial feature and new feature .
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(feature_var_list, list):
        raise ValueError("feature_var_list should be a list of strings.")
    if not isinstance(threshold, float):
        raise ValueError("threshold should be a float.")

    # Make a copy of a part of the input data
    new_data = input_data[feature_var_list].copy()

    # Initialize
    var_with_high_corr_list = []
    value_with_high_corr_list = []
    high_corr_not_duplicated_list = []
    var_with_less_corr_list = []
    value_with_less_corr_list = ["None"]

    # create correlation matrix
    corr_matrix = new_data.corr()

    # find the feature with higher correlation
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                if colname not in high_corr_not_duplicated_list:
                    value_with_high_corr_list.append(corr_matrix.iloc[i, j])
                    var_with_high_corr_list.append(colname)
                    high_corr_not_duplicated_list.append(colname)

    # find the feature with less correlation
    for feature in feature_var_list:
        if feature not in var_with_high_corr_list:
            var_with_less_corr_list.append(feature)

    # Return result as tuple
    return (
        var_with_high_corr_list,
        value_with_high_corr_list,
        var_with_less_corr_list,
        value_with_less_corr_list,
    )


# Only for linear regression problem
# function to get multicolinearity of variable
def get_multicolinearity_of_variable(
    *, input_data: pd.DataFrame, num_var_list: list, linearity_threshold: float
) -> list:
    """
    Get multicolinearity between features.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - num_var_list (list): The list of numerical variables.
    - linearity_threshold (float): The threshold for linearity of variables.

    Returns:
    - list: A list of tuple contains variables witch have high linear correlation.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(linearity_threshold, float):
        raise ValueError("linearity_threshold should be a float.")
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be a list of strings.")

    # Make a copy of a part of the input data wich contains only
    # categorical variable with missing values
    data = input_data[num_var_list].copy()

    # list of tuple contains variables
    linear_variables = []

    # the default correlation method of pandas.corr is pearson
    # I include it anyways for the demo
    corrmat = data.corr(method="pearson")

    # Find indices where values are higher than 0.8
    indices = (
        (abs(corrmat) > linearity_threshold) & (abs(corrmat) != 1)
    ).values.nonzero()

    # Get corresponding row and column labels
    rows = corrmat.index[indices[0]]
    columns = corrmat.columns[indices[1]]

    # Print tuples of row and column corresponding to values higher than 0.8
    for row, column in zip(rows, columns):
        linear_variables.append((row, column))

    # remove a redondant tuple
    unique_tuples = set()
    non_redundant_tuples = []

    for tup in linear_variables:
        # Convert tuple to frozenset to handle unordered comparison
        ftup = frozenset(tup)
        if ftup not in unique_tuples:
            # If tuple is not already present, add it to unique_tuples set
            unique_tuples.add(ftup)
            non_redundant_tuples.append(tup)

    # Return result as list of tuple
    return non_redundant_tuples


# Only for linear regression problem
# function to get linearity relationship between feature and target
def get_regression_linearity_between_feature_and_target(
    *,
    input_data: pd.DataFrame,
    target: str,
    num_var_list: list,
    linearity_threshold: float,
) -> pd.DataFrame:
    """
    Check the linear relationship between feature and target.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - target (str): The target variable.
    - num_var_list (list): The list of numerical variables.
    - linearity_threshold (float): The threshold for linearity of variables.

    Returns:
    - list: A list of tuple contains variables witch have high linear correlation.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(target, str):
        raise ValueError("target should be a string.")
    if not isinstance(linearity_threshold, float):
        raise ValueError("linearity_threshold should be a float.")
    if not isinstance(num_var_list, list):
        raise ValueError("num_var_list should be a list of strings.")

    # Make a copy of a part of the input data wich contains only
    # categorical variable with missing values
    data = pd.concat([input_data[num_var_list], input_data[target]], axis=1).copy()

    # list of tuple contains variables
    linear_variables = []

    # the default correlation method of pandas.corr is pearson
    # I include it anyways for the demo
    corrmat = data.corr(method="pearson")[target]

    # Find indices where values are higher than 0.8
    indices = (
        (abs(corrmat) > linearity_threshold) & (abs(corrmat) != 1)
    ).values.nonzero()

    # Get corresponding row and column labels
    rows = corrmat.index[indices[0]]
    # columns = corrmat.columns[indices[1]]

    # Print tuples of row and column corresponding to values higher than 0.8
    # for row, column in zip(rows, columns):
    #     linear_variables.append((row, column))
    for row in rows:
        linear_variables.append(row)

    # remove a redondant tuple
    unique_tuples = set()
    non_redundant_tuples = []

    for tup in linear_variables:
        # Convert tuple to frozenset to handle unordered comparison
        ftup = frozenset(tup)
        if ftup not in unique_tuples:
            # If tuple is not already present, add it to unique_tuples set
            unique_tuples.add(ftup)
            non_redundant_tuples.append(tup)

    # Return result as tuple
    return non_redundant_tuples


# function to detect non rare Labels
def find_non_rare_labels(
    *, dataframe: pd.DataFrame, var: str, rare_perc: float
) -> list:
    """
    Find non rare labels.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame.
    - var (str): The categorical variable to analyze.
    - rare_perc (float): The threshold percentage for considering a label as rare.

    Returns:
    - list: A list of non rare labels.
    """
    # Check the type of variables passed in the function
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe should be pandas dataframe.")
    if not isinstance(var, str):
        raise ValueError("var should be strings.")
    if not isinstance(rare_perc, float):
        raise ValueError("rare_perc should be float.")

    # Make a copy of the dataframe
    data = dataframe.copy()

    # Determine the % of observations per category
    temp = data.groupby([var])[var].count() / len(data)

    # find non rare categories
    non_rare = [x for x in temp.loc[temp > rare_perc].index.values]

    # return result as list
    return non_rare


# function to detect non Rare Labels
def analyse_non_rare_labels(
    *, dataframe: pd.DataFrame, var: str, rare_perc: float
) -> list:
    """
    Analyze rare labels in a categorical variable based on the specified percentage threshold.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - var (str): The categorical variable to analyze.
    - rare_perc (float): The threshold percentage for considering a label as rare.

    Returns:
    - pd.Series: A Series containing the names of rare categories.
    """
    # Check the type of variables passed in the function
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe should be pandas dataframe.")
    if not isinstance(var, str):
        raise ValueError("var should be strings.")
    if not isinstance(rare_perc, float):
        raise ValueError("rare_perc should be float.")

    # Make a copy of the dataframe
    df = dataframe.copy()

    # Determine the % of observations per category
    category_percentages = df.groupby(var)[var].count() / len(df)

    # Return names of categories that are rare
    rare_categories = category_percentages[category_percentages > rare_perc].index

    # return result as list
    return rare_categories


# function to detect Rare Labels
def analyse_rare_labels(*, dataframe: pd.DataFrame, var: str, rare_perc: float) -> list:
    """
    Analyze rare labels in a categorical variable based on the specified percentage threshold.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - var (str): The categorical variable to analyze.
    - rare_perc (float): The threshold percentage for considering a label as rare.

    Returns:
    - pd.Series: A Series containing the names of rare categories.
    """
    # Check the type of variables passed in the function
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe should be pandas dataframe.")
    if not isinstance(var, str):
        raise ValueError("var should be strings.")
    if not isinstance(rare_perc, float):
        raise ValueError("rare_perc should be float.")

    # Make a copy of the dataframe
    df = dataframe.copy()

    # Determine the % of observations per category
    category_percentages = df.groupby(var)[var].count() / len(df)

    # Return names of categories that are rare
    rare_categories = category_percentages[category_percentages <= rare_perc].index

    # return result as list
    return rare_categories


# function to collect categorical variables with Rare labels in dataset
def get_categorical_variable_with_Rare_Label(
    *, input_data: pd.DataFrame, target: str, rare_perc: float
) -> tuple:
    """
    Identify categorical variables with rare labels in a DataFrame.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - target (str): The target variable.
    - rare_perc (float): The threshold percentage for considering a label as rare.

    Returns:
    - tuple: A list tuple's of categorical variable names with rare labels.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(target, str):
        raise ValueError("target should be strings.")
    if not isinstance(rare_perc, float):
        raise ValueError("rare_perc should be float.")

    # Make a copy of the input data
    data = input_data.copy()

    # Get the categorical variables list
    cat_var_list = [
        var for var in data.columns if data[var].dtype == "O" and var != target
    ]

    # Identify categorical variables with rare labels using list comprehension
    cat_var_rare_labels: list = []
    cat_var_rare_labels_to_regroup: list = []
    cat_var_rare_labels_non_regroup: list = []
    cat_var_non_rare_labels: list = []
    rare_categories: list = []
    non_rare_categories: list = []

    for var in cat_var_list:
        rare_categories = analyse_rare_labels(
            dataframe=data, var=var, rare_perc=rare_perc
        )
        non_rare_categories = analyse_non_rare_labels(
            dataframe=data, var=var, rare_perc=rare_perc
        )

        # Get variable contains Rare labels
        if not rare_categories:
            cat_var_rare_labels.append(var)

        # Get variable contains Rare labels non re-group
        if not rare_categories and len(rare_categories) == 1:
            cat_var_rare_labels_non_regroup.append(var)

        # Get variable contains Rare labels to re-group
        if not rare_categories and len(rare_categories) > 1:
            cat_var_rare_labels_to_regroup.append(var)

        # Get variable contains non Rare labels
        if not non_rare_categories:
            cat_var_non_rare_labels.append(var)

    # return result as list
    return (
        cat_var_rare_labels,
        cat_var_rare_labels_non_regroup,
        cat_var_rare_labels_to_regroup,
        cat_var_non_rare_labels,
    )


# function to re-group rare labels into "Rare"
def rare_encoding_with_Pandas(
    *, dataframe: pd.DataFrame, target: str, var: str, rare_perc: float
) -> pd.DataFrame:
    """
    rare labels re-grouping into "Rare".

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame.
    - target (str): The target variable.
    - var (str): The categorical variable to analyze.
    - rare_perc (float): The threshold percentage for considering a label as rare.

    Returns:
    - pd.DataFrame: A dataframe containing the rare categories re-group into "Rare" label.
    """
    # Check the type of variables passed in the function
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe should be pandas dataframe.")
    if not isinstance(var, str) or not isinstance(target, str):
        raise ValueError("Both target and var should be strings.")
    if not isinstance(rare_perc, float):
        raise ValueError("rare_perc should be float.")

    # Make a copy of the dataframe
    data = dataframe.copy()

    # find the most frequent category
    frequent_cat = find_non_rare_labels(dataframe=data, var=var, rare_perc=rare_perc)
    # frequent_cat = analyse_non_rare_labels(dataframe=data, var=var, rare_perc=rare_perc)

    # variables contains rare labels in dataset
    cat_var_rare = get_categorical_variable_with_Rare_Label(
        input_data=data, target=target, rare_perc=0.05
    )

    # re-group rare labels
    # data[var] = np.where(
    #     data[var].isin(frequent_cat), data[var], "Rare"
    # )
    data[var] = np.where(
        (data[var].isin(frequent_cat)) | (var in cat_var_rare[1]), data[var], "Rare"
    )

    # return result as dataframe
    return data


# Define a function to detect outliers in a column
def detect_outliers_for_normal_distribution_with_norm_dist_method(column, fold=1.5):
    """
    Detect outliers in a numeric column using the normaly distribution method.

    Parameters:
    - column (array-like): Numeric array or pandas Series representing the column.
    - fold : coeeficient.

    Returns:
    - array-like: Boolean array indicating the positions of outliers.
    """
    # Check if the column is numeric
    if not np.issubdtype(column.dtype, np.number):
        raise ValueError("Input column should be numeric.")
    if not isinstance(fold, float):
        raise ValueError("variable should be float.")

    # Calculate lower and upper bounds
    lower_bound = column.mean() - (column.std() * fold)
    upper_bound = column.mean() + (column.std() * fold)

    # Identify outliers
    outliers = (column < lower_bound) | (column > upper_bound)

    # Return results as column
    return outliers


# Define a function to detect outliers in a column
def detect_outliers_for_normal_distribution_with_quantile_method(column):
    """
    Detect outliers in a numeric column using the quantile method.

    Parameters:
    - column (array-like): Numeric array or pandas Series representing the column.

    Returns:
    - array-like: Boolean array indicating the positions of outliers.
    """
    # Check if the column is numeric
    if not np.issubdtype(column.dtype, np.number):
        raise ValueError("Input column should be numeric.")

    # Calculate lower and upper bounds
    lower_bound = column.quantile(0.05)
    upper_bound = column.quantile(0.95)

    # Identify outliers
    outliers = (column < lower_bound) | (column > upper_bound)

    # Return results as column
    return outliers


# Define a function to detect outliers in a column
def detect_outliers_columns(column):
    """
    Detect outliers in a numeric column using the Interquartile Range (IQR) method.

    Parameters:
    - column (array-like): Numeric array or pandas Series representing the column.

    Returns:
    - array-like: Boolean array indicating the positions of outliers.
    """
    # Check if the column is numeric
    if not np.issubdtype(column.dtype, np.number):
        raise ValueError("Input column should be numeric.")

    # Calculate quartiles and IQR
    Q1 = np.percentile(column, 25)
    Q3 = np.percentile(column, 75)
    IQR = Q3 - Q1

    # Calculate lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = (column < lower_bound) | (column > upper_bound)

    # Return results as column
    return outliers


# Define a function to detect outliers in a column
def detect_outliers_for_skewed_distribution_with_IQR_method(column, fold=1.5):
    """
    Detect outliers in a numeric column using the Interquartile Range (IQR) method.

    Parameters:
    - column (array-like): Numeric array or pandas Series representing the column.
    - fold : IQR coeeficient.

    Returns:
    - array-like: Boolean array indicating the positions of outliers.
    """
    # Check if the column is numeric
    if not np.issubdtype(column.dtype, np.number):
        raise ValueError("Input column should be numeric.")
    if not isinstance(fold, float):
        raise ValueError("variable should be float.")

    # Calculate quartiles and IQR
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1

    # Calculate lower and upper bounds
    lower_bound = Q1 - fold * IQR
    upper_bound = Q3 + fold * IQR

    # Identify outliers
    outliers = (column < lower_bound) | (column > upper_bound)

    # Return results as column
    return outliers


# function to create dictionnary with unique value and associated number
# for categorical variables
def get_unique_and_numerical_dictionnary_for_categorical_variable(
    *, input_data: pd.DataFrame, target: str
) -> tuple:
    """
    Identify categorical variables in a DataFrame
    and create dictionnary contains unique value and associated number.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame.
    - target (str): The target variable.

    Returns:
    - tuple: A tuple of categorical variable names, unique value and associated number dictionnary.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("input_data should be pandas dataframe.")
    if not isinstance(target, str):
        raise ValueError("target should be strings.")

    # Make a copy of the input data
    data = input_data.copy()

    # Get categorical features
    categorical_feature = [var for var in data.columns.tolist() if data[var].dtype == "O"]

    # Initialize dictionnary
    uniq_val_dict: dict = {}
    uniq_val_inv_dict: dict = {}
    num_to_cat_dict: dict = {}
    cat_to_num_dict: dict = {}

    # Get unique value disctionnary
    for var in categorical_feature:
        # Initialize dictionnary
        uniq_val_list: list = []

        # Get unique value list and dictionnary
        uniq_val_list = data[var].unique().tolist()

        # Get dictionnary to make association between number and categorical unique values
        for count, uniqval in enumerate(uniq_val_list):
            uniq_val_dict[uniqval] = count
            uniq_val_inv_dict[count] = uniqval

        # Get final dictionnary for all categorical variable
        cat_to_num_dict[var] = uniq_val_dict
        num_to_cat_dict[var] = uniq_val_inv_dict

    # Return results as dictionnary tuple's
    return (cat_to_num_dict, num_to_cat_dict)
