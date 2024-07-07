# import library and module
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             make_scorer, mean_squared_error, r2_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

# for reproducibility, split size
seed = 0
split_size = 0.3
n_fold = 5


# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# # FOR DATA EXPLORATORY AND ANALYSIS AND ALL OF REST
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Replace all missing value in the data set by nan
class ReplaceMissingValueByNanTransform(BaseEstimator, TransformerMixin):
    def __init__(self, missing_values_list):
        # Check the input of function
        if not isinstance(missing_values_list, list):
            raise ValueError("variable_list should be a list")

        self.missing_values_list = missing_values_list

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline

        # Assign variable to cast
        self.miss_val_list = self.missing_values_list

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Generate the data frame to return
        for missing_value in self.miss_val_list:
            X = X.replace(missing_value, np.nan)

        # return result as data frame
        return X


# Casting certain numerical variables as float
class CastingCertainNumericalVariableAsFloatTransform(BaseEstimator, TransformerMixin):
    def __init__(self, casted_variable_list):
        # Check the input of function
        if not isinstance(casted_variable_list, list):
            raise ValueError("variable_list should be a list")

        self.casted_variable_list = casted_variable_list

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Generate the data frame to return
        for feature in self.casted_variable_list:
            X[feature] = X[feature].astype("float")

        # return result as data frame
        return X


# Assign the right type to all variables
class AssignRightTypeToAllVariableTransform(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_variable_list, categorical_variable_list):
        # Check the input of function
        if not isinstance(numerical_variable_list, list):
            raise ValueError("numerical_variable_list should be a list")
        if not isinstance(categorical_variable_list, list):
            raise ValueError("categorical_variable_list should be a list")

        self.numerical_variable_list = numerical_variable_list
        self.categorical_variable_list = categorical_variable_list

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # Assign variable to cast
        self.num_var_list = self.numerical_variable_list
        self.cat_var_list = self.categorical_variable_list

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Generate the data frame to return
        for feature in self.num_var_list:
            X[feature] = X[feature].astype("float")
        for feature in self.cat_var_list:
            X[feature] = X[feature].astype("category")

        # return result as data frame
        return X


# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# # FEATURES AND ROWS CREATION
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Generation of new column in data set based on product
class AddArtificialNaNRowsToTrainSetTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fit_perform = []

    def add_artificial_NaN_rows_to_train_set(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> pd.DataFrame:
        """
        Add artifician Nan rows to train set.

        Parameters:
        - X_train (pd.DataFrame): The X_train DataFrame.
        - y_train (pd.Series): y_train series.

        Returns:
        - tuple: A tuple containing X_train with artificial NaN rows
          and y_train with corresponding classes.
        """
        # Check the type of variables passed in the function
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train should be pandas dataframe.")
        if not isinstance(y_train, pd.Series):
            raise ValueError("y_train should be pandas series.")

        # Make a copy of the input data
        X_train_new = X_train.copy()
        y_train_new = y_train.copy()

        # Add Nan rows to train set
        for uniq_target in y_train_new.unique().tolist():
            List = [[np.nan for i in range(len(X_train_new.columns.values.tolist()))]]
            new_frame = pd.DataFrame(List, columns=X_train_new.columns.values.tolist())
            new_series = pd.Series([uniq_target])
            X_train_new = X_train_new._append(new_frame, ignore_index=False)
            y_train_new = y_train_new._append(new_series, ignore_index=False)

            # Reset the index of the appended DataFrame and set it to start
            # from the next index after the last index of df1
            X_train_new = X_train_new.reset_index(drop=True)
            next_index = len(X_train_new)
            new_frame = new_frame.reset_index(drop=True)
            new_frame.index += next_index

            # Reset the index of the appended DataFrame and set it to start
            # from the next index after the last index of df1
            y_train_new = y_train_new.reset_index(drop=True)
            next_index = len(y_train_new)
            new_series = new_series.reset_index(drop=True)
            new_series.index += next_index

        # Return result as tuple
        return X_train_new, y_train_new

    def fit(self, X, y):
        # we need this step to fit the sklearn pipeline
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Generate the product variable
        self.X_train_new, self.y_train_new = self.add_artificial_NaN_rows_to_train_set(
            X_train=X, y_train=y
        )
        self.fit_perform.append("True")

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Generate the artificial NaN rows
        if not self.fit_perform:
            X = X
        else:
            X = self.X_train_new

        # return result as data frame
        return X


# Generation of new column in data set based on product
class GenerateProductVariableTransform(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer

    def __init__(self, variable_list):
        # Check the input of function
        if not isinstance(variable_list, list):
            raise ValueError("variable_list should be a list")

        self.variable_list = variable_list

    def get_columns_for_variable_generated(self, X: pd.DataFrame) -> tuple:
        """
        Generate new columns based on product of existant columns.

        Parameters:
        - input_data (pd.DataFrame): The input DataFrame.

        Returns:
        - tuple: A tuple of variable and variable name list .
        """
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # Make a copy of a part of the input data wich contains
        # only categorical variable with missing values
        # new_data = X.copy()

        # Initialize list
        list_of_tuple_var_fit = []
        list_of_new_var_name = []
        duplicate_list = []

        # Generate ratio column
        for var1 in self.variable_list:
            for var2 in self.variable_list:
                if var2 not in duplicate_list and var2 != var1:
                    new_var_name = f"{var1}_prod_{var2}"
                    list_of_tuple_var_fit.append((var1, var2))
                    list_of_new_var_name.append(new_var_name)

            # lits to avoid redundant somme like var1*var2 and var2*var1
            duplicate_list.append(var1)

        # Return result as tuple
        return list_of_tuple_var_fit, list_of_new_var_name

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Generate the product variable
        self.list_of_tuple_var_fit, self.list_of_new_var_name = (
            self.get_columns_for_variable_generated(X=X)
        )

        # return result as self
        return self

    def variable_generated_by_product_of_two_columns(
        self, X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate new columns based on product of existant columns.

        Parameters:
        - input_data (pd.DataFrame): The input DataFrame.

        Returns:
        - pd.DataFrame: A pandas frame contains initial feature and new feature .
        """
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # Make a copy of a part of the input data wich contains
        # only categorical variable with missing values
        new_data = X.copy()

        # Generate product column
        for count, var in enumerate(self.list_of_tuple_var_fit):
            new_series = pd.Series(
                pd.to_numeric(new_data[var[0]].copy())
                * pd.to_numeric(new_data[var[1]].copy()),
                name=self.list_of_new_var_name[count],
            )
            new_series_frame = pd.DataFrame(
                new_series, columns=[self.list_of_new_var_name[count]]
            )

            # update new data frame
            new_data = pd.concat([new_data.copy(), new_series_frame], axis=1)

        # Return result as data frame
        return new_data

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Generate the data frame to return
        X = self.variable_generated_by_product_of_two_columns(X=X)

        # return result as data frame
        return X


# Generation of new column in data set based on somme
class GenerateSommeVariableTransform(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer

    def __init__(self, variable_list):
        # Check the input of function
        if not isinstance(variable_list, list):
            raise ValueError("variable_list should be a list")

        self.variable_list = variable_list

    def get_columns_for_variable_generated(self, X: pd.DataFrame) -> tuple:
        """
        Generate new columns based on product of existant columns.

        Parameters:
        - input_data (pd.DataFrame): The input DataFrame.

        Returns:
        - tuple: A tuple of variable and variable name list .
        """
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # Make a copy of a part of the input data wich contains
        # only categorical variable with missing values
        # new_data = X.copy()

        # Initialize list
        list_of_tuple_var_fit = []
        list_of_new_var_name = []
        duplicate_list = []

        # Generate ratio column
        for var1 in self.variable_list:
            for var2 in self.variable_list:
                if var2 not in duplicate_list and var2 != var1:
                    new_var_name = f"{var1}_som_{var2}"
                    list_of_tuple_var_fit.append((var1, var2))
                    list_of_new_var_name.append(new_var_name)

            # lits to avoid redundant somme like var1*var2 and var2*var1
            duplicate_list.append(var1)

        # Return result as tuple
        return list_of_tuple_var_fit, list_of_new_var_name

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Generate the somme variable
        self.list_of_tuple_var_fit, self.list_of_new_var_name = (
            self.get_columns_for_variable_generated(X=X)
        )

        # return result as self
        return self

    def variable_generated_by_somme_of_two_columns(
        self, X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate new columns based on somme of existant columns.

        Parameters:
        - input_data (pd.DataFrame): The input DataFrame.

        Returns:
        - pd.DataFrame: A pandas frame contains initial feature and new feature .
        """
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # Make a copy of a part of the input data wich contains
        # only categorical variable with missing values
        new_data = X.copy()

        # Generate somme column
        for count, var in enumerate(self.list_of_tuple_var_fit):
            new_series = pd.Series(
                pd.to_numeric(new_data[var[0]].copy())
                + pd.to_numeric(new_data[var[1]].copy()),
                name=self.list_of_new_var_name[count],
            )
            new_series_frame = pd.DataFrame(
                new_series, columns=[self.list_of_new_var_name[count]]
            )

            # update new data frame
            new_data = pd.concat([new_data.copy(), new_series_frame], axis=1)

        # Return result as data frame
        return new_data

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Generate the data frame to return
        X = self.variable_generated_by_somme_of_two_columns(X=X)

        # return result as data frame
        return X


# Generation of new column in data set based on ratio
class GenerateRatioVariableTransform(BaseEstimator, TransformerMixin):
    def __init__(self, variable_list):
        # Check the input of function
        if not isinstance(variable_list, list):
            raise ValueError("variable_list should be a list")

        self.variable_list = variable_list

    def get_columns_for_variable_generated(self, X: pd.DataFrame) -> tuple:
        """
        Generate new columns based on product of existant columns.

        Parameters:
        - input_data (pd.DataFrame): The input DataFrame.

        Returns:
        - tuple: A tuple of variable and variable name list .
        """
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # Make a copy of a part of the input data wich contains
        # only categorical variable with missing values
        new_data = X.copy()

        # Initialize list
        list_of_tuple_var_fit = []
        list_of_new_var_name = []

        # Generate ratio column
        for var1 in self.variable_list:
            for var2 in self.variable_list:
                if not (new_data[var2] == 0).any() and var2 != var1:
                    new_var_name = f"{var1}_div_{var2}"
                    list_of_tuple_var_fit.append((var1, var2))
                    list_of_new_var_name.append(new_var_name)

        # Return result as tuple
        return list_of_tuple_var_fit, list_of_new_var_name

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Generate the ratio variable
        self.list_of_tuple_var_fit, self.list_of_new_var_name = (
            self.get_columns_for_variable_generated(X=X)
        )

        # return result as self
        return self

    def variable_generated_by_ratio_of_two_columns(
        self, X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate new columns based on product of existant columns.

        Parameters:
        - input_data (pd.DataFrame): The input DataFrame.

        Returns:
        - pd.DataFrame: A pandas frame contains initial feature and new feature .
        """
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # Make a copy of a part of the input data wich contains
        # only categorical variable with missing values
        new_data = X.copy()

        # Generate ratio column
        for count, var in enumerate(self.list_of_tuple_var_fit):
            new_series = pd.Series(
                pd.to_numeric(new_data[var[0]].copy())
                / pd.to_numeric(new_data[var[1]].copy()),
                name=self.list_of_new_var_name[count],
            )
            new_series_frame = pd.DataFrame(
                new_series, columns=[self.list_of_new_var_name[count]]
            )

            # update new data frame
            new_data = pd.concat([new_data.copy(), new_series_frame], axis=1)

        # Return result as data frame
        return new_data

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Generate the data frame to return
        X = self.variable_generated_by_ratio_of_two_columns(X=X)

        # return result as data frame
        return X


# Temporal elapsed time transformer
class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables, reference_variable):
        # Check the input of function
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.reference_variable = reference_variable

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Generate the data frame to return
        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]

        # return result as data frame
        return X


# for mapping categorical variable like quality, ... : when we have mapping dictionary
class Mapper_Dict(BaseEstimator, TransformerMixin):
    def __init__(self, variables, mappings):
        # Check the input of function
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        if not isinstance(mappings, dict):
            raise ValueError("mappings should be a dictionnary")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        # we need the fit statement to accommodate the sklearn pipeline
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # Make a copy of original data set, so that we do not over-write the original data frame
        X = X.copy()

        # Generate the data frame to return
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings)

        # return result as data frame
        return X


# for mapping categorical variable like quality, ... : when we have mapping dictionary
class Mapper(BaseEstimator, TransformerMixin):
    def __init__(self, variables, mappings):
        # Check the input of function
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        if not isinstance(mappings, dict):
            raise ValueError("mappings should be a dictionnary")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        # we need the fit statement to accommodate the sklearn pipeline
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # Make a copy of original data set, so that we do not over-write the original data frame
        X = X.copy()

        # Generate the data frame to return
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings[feature])

        # return result as data frame
        return X


# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# # IMPUTATION TECHNICS OF MISSING VALUES
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# mean imputation per group
class MeanImputationPerGroupTransform(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer

    def __init__(self, variables, categorical_variable_for_grouping):
        # Check the input of function
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        if not isinstance(categorical_variable_for_grouping, str):
            raise ValueError("categorical_variable_for_grouping should be a strings")

        self.variables = variables
        self.categorical_variable_for_grouping = categorical_variable_for_grouping

    def fit(self, X, y=None):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Create imputation dictionary
        self.imputation_dict = {}
        for i in X[self.categorical_variable_for_grouping].unique():
            self.imputation_dict[i] = (
                X[X[self.categorical_variable_for_grouping] == i][self.variables]
                .mean()
                .to_dict()
            )

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Replace missing data
        for i in self.imputation_dict.keys():
            X[X[self.categorical_variable_for_grouping] == i] = X[
                X[self.categorical_variable_for_grouping] == i
            ].fillna(self.imputation_dict[i])

        # return result as data frame
        return X


# median imputation per group
class MedianImputationPerGroupTransform(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer

    def __init__(self, variables, categorical_variable_for_grouping):
        # Check the input of function
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        if not isinstance(categorical_variable_for_grouping, str):
            raise ValueError("categorical_variable_for_grouping should be a strings")

        self.variables = variables
        self.categorical_variable_for_grouping = categorical_variable_for_grouping

    def fit(self, X, y=None):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Create imputation dictionary
        self.imputation_dict = {}
        for i in X[self.categorical_variable_for_grouping].unique():
            self.imputation_dict[i] = (
                X[X[self.categorical_variable_for_grouping] == i][self.variables]
                .median()
                .to_dict()
            )

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Replace missing data
        for i in self.imputation_dict.keys():
            X[X[self.categorical_variable_for_grouping] == i] = X[
                X[self.categorical_variable_for_grouping] == i
            ].fillna(self.imputation_dict[i])

        # return result as data frame
        return X


# grouping variable for mean/median imputation per group
class GroupingVariableTransform(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer
    def __init__(self, variables, categorical_variable_for_grouping):
        # Check the input of function
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        if not isinstance(categorical_variable_for_grouping, str):
            raise ValueError("categorical_variable_for_grouping should be a strings")

        self.variables = variables
        self.categorical_variable_for_grouping = categorical_variable_for_grouping

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        for missing_variable in self.variables:
            # create new variable and append it into new list
            new_var_name = f"{missing_variable}_groupvar"
            X[new_var_name] = X[missing_variable]

            # Create a grouping variable
            for i, labels in enumerate(
                X[self.categorical_variable_for_grouping].unique()
            ):
                # re-group variable to imput based on grouping variable
                X[new_var_name] = np.where(
                    X[self.categorical_variable_for_grouping].isin([labels]),
                    i,
                    X[missing_variable],
                )

        # return result as data frame
        return X


# Define class to have type for imputer
class imputer_type:
    # Set imputer
    imputer_bayes = IterativeImputer(
        estimator=BayesianRidge(), max_iter=10, random_state=seed
    )


# grouping variable for mean/median imputation per group
class MICEImputationTransform(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer
    def __init__(self, variables, imputer):
        # Check the input of function
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        # if not isinstance(imputer, imputer_type):
        #     raise ValueError("imputer should be sklearn.impute._iterative.IterativeImputer type")

        self.variables = variables
        self.imputer = imputer

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X_variables = X[self.variables].copy()

        # fit imputer
        self.imputer.fit(X_variables)

        # return result as data frame
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()
        X_variables = X[self.variables].copy()

        # imput NaN values
        X_ndarray = self.imputer.transform(X_variables)

        # creating the dataframe
        X_frame = pd.DataFrame(data=X_ndarray, columns=self.variables)

        # concatenate dataframe horizontaly
        X = pd.concat([X.drop(self.variables, axis=1), X_frame], axis=1)

        # return result as data frame
        return X


# for imputation of missing numerical variables (replaced by the mean or median or ...)
class MeanImputer(BaseEstimator, TransformerMixin):
    """Numerical missing value imputer."""

    def __init__(self, variables):
        # Check the input of function
        self.imputer_dict_ = None
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        self.variables = variables

    def fit(self, X, y=None):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # persist mean values in a dictionary
        self.imputer_dict_ = X[self.variables].mean().to_dict()

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")
        assert isinstance(self.imputer_dict_, dict)

        # Make a copy of original data set
        X = X.copy()

        # Generate the data frame to return
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)

        # return result as data frame
        return X


# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# # NUMERICAL DATA TRANSFORMATION
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Apply QuantileTransformer() method to transform data
class QuantileTransformerTransform(BaseEstimator, TransformerMixin):
    def __init__(self, variable_list):
        # Check the input of function
        if not isinstance(variable_list, list):
            raise ValueError("variable_list should be a list")

        self.variable_list = variable_list

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Define transformer
        self.scaler = QuantileTransformer(
            output_distribution="normal", random_state=seed
        )

        # Apply transformation fit to data
        self.scaler.fit(X[self.variable_list])

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Apply transformation transform to data
        X[self.variable_list] = self.scaler.transform(X[self.variable_list])

        # return result as data frame
        return X


# Add IQR Outliers indicator to data
class AddIqrOutliersIndicatorTransform(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer

    def __init__(self, variable_list, fold):
        # Check the input of function
        if not isinstance(variable_list, list):
            raise ValueError("variable_list should be a list")

        self.variable_list = variable_list
        self.fold = fold

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # initialize list
        self.lower_limit_list = []
        self.upper_limit_list = []

        for var in self.variable_list:
            # To calcul the IQR
            IQR = X[var].quantile(0.75) - X[var].quantile(0.25)

            # Calcul lower_limit and upper_limit
            self.lower_limit_list.append(X[var].quantile(0.25) - (IQR * self.fold))
            self.upper_limit_list.append(X[var].quantile(0.75) + (IQR * self.fold))

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Create a list to store the new Series
        new_columns = []

        # Loop through each variable
        for count, var in enumerate(self.variable_list):
            # Create a Series based on the conditions
            new_column = pd.Series(
                np.where(
                    (X[var] < self.lower_limit_list[count])
                    | (X[var] > self.upper_limit_list[count]),
                    1,
                    0,
                ),
                index=X.index,  # Preserve original index
                name=f"{var}_outliers",  # Name of the new column
            )
            # Append the new Series to the list
            new_columns.append(new_column)

        # Concatenate the list of Series along the columns axis
        X = pd.concat([X] + new_columns, axis=1)

        # return result as data frame
        return X


# Add Gaussian Outliers indicator to data
class AddGaussianOutliersIndicatorTransform(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer

    def __init__(self, variable_list, fold):
        # Check the input of function
        if not isinstance(variable_list, list):
            raise ValueError("variable_list should be a list")

        self.variable_list = variable_list
        self.fold = fold

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # initialize list
        self.lower_limit_list = []
        self.upper_limit_list = []

        for var in self.variable_list:
            # Calcul lower_limit and upper_limit
            self.lower_limit_list.append(X[var].mean() - (X[var].std() * self.fold))
            self.upper_limit_list.append(X[var].mean() + (X[var].std() * self.fold))

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Create a list to store the new Series
        new_columns = []

        # Loop through each variable
        for count, var in enumerate(self.variable_list):
            # Create a Series based on the conditions
            new_column = pd.Series(
                np.where(
                    (X[var] < self.lower_limit_list[count])
                    | (X[var] > self.upper_limit_list[count]),
                    1,
                    0,
                ),
                index=X.index,  # Preserve original index
                name=f"{var}_outliers",  # Name of the new column
            )
            # Append the new Series to the list
            new_columns.append(new_column)

        # Concatenate the list of Series along the columns axis
        X = pd.concat([X] + new_columns, axis=1)

        # return result as data frame
        return X


# Add Quantiles Outliers indicator to data
class AddQuantilesOutliersIndicatorTransform(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer

    def __init__(self, variable_list):
        # Check the input of function
        if not isinstance(variable_list, list):
            raise ValueError("variable_list should be a list")

        self.variable_list = variable_list

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # initialize list
        self.lower_limit_list = []
        self.upper_limit_list = []

        for var in self.variable_list:
            # Calcul lower_limit and upper_limit
            self.lower_limit_list.append(X[var].quantile(0.05))
            self.upper_limit_list.append(X[var].quantile(0.95))

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Create a list to store the new Series
        new_columns = []

        # Loop through each variable
        for count, var in enumerate(self.variable_list):
            # Create a Series based on the conditions
            new_column = pd.Series(
                np.where(
                    (X[var] < self.lower_limit_list[count])
                    | (X[var] > self.upper_limit_list[count]),
                    1,
                    0,
                ),
                index=X.index,  # Preserve original index
                name=f"{var}_outliers",  # Name of the new column
            )
            # Append the new Series to the list
            new_columns.append(new_column)

        # Concatenate the list of Series along the columns axis
        X = pd.concat([X] + new_columns, axis=1)

        # return result as data frame
        return X


# Add Both ends Arbitrary Outliers indicator to data
class AddArbitraryOutliersIndicatorTransform(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer

    def __init__(self, max_capping_dict, min_capping_dict):
        # Check the input of function
        if not isinstance(max_capping_dict, dict) or not isinstance(
            min_capping_dict, dict
        ):
            raise ValueError(
                "Both max_capping_dict and min_capping_dict should be a dictionnary"
            )
        if not list(min_capping_dict.keys()) == list(max_capping_dict.keys()):
            raise ValueError(
                "Both max_capping_dict and min_capping_dict should be the same variable"
            )

        self.max_capping_dict = max_capping_dict
        self.min_capping_dict = min_capping_dict

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # initialize list
        self.variable_list = []
        self.lower_limit_list = []
        self.upper_limit_list = []

        # Calcul variable_list, lower_limit and upper_limit
        self.variable_list = list(self.max_capping_dict.keys())
        self.lower_limit_list = list(self.min_capping_dict.values())
        self.upper_limit_list = list(self.max_capping_dict.values())

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Create a list to store the new Series: for min_variable_list
        new_columns = []

        # Loop through each min variable
        for count, var in enumerate(self.variable_list):
            # Create a Series based on the conditions
            new_column = pd.Series(
                np.where(
                    (X[var] < self.lower_limit_list[count])
                    | (X[var] > self.upper_limit_list[count]),
                    1,
                    0,
                ),
                index=X.index,  # Preserve original index
                name=f"{var}_outliers",  # Name of the new column
            )
            # Append the new Series to the list
            new_columns.append(new_column)

        # Concatenate the list of Series along the columns axis
        X = pd.concat([X] + new_columns, axis=1)

        # return result as data frame
        return X


# Add Maximum Arbitrary Outliers indicator to data
class AddMaximumArbitraryOutliersIndicatorTransform(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer

    def __init__(self, max_capping_dict, min_capping_dict=None):
        # Check the input of function
        if not isinstance(max_capping_dict, dict):
            raise ValueError("max_capping_dict should be a dictionnary")
        if min_capping_dict is not None:
            raise ValueError("min_capping_dict should be equal to None")

        self.max_capping_dict = max_capping_dict
        self.min_capping_dict = min_capping_dict

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # initialize list
        self.variable_list = []
        self.upper_limit_list = []

        # Calcul variable_list, lower_limit and upper_limit
        self.variable_list = list(self.max_capping_dict.keys())
        self.upper_limit_list = list(self.max_capping_dict.values())

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Create a list to store the new Series: for min_variable_list
        new_columns = []

        # Loop through each min variable
        for count, var in enumerate(self.variable_list):
            # Create a Series based on the conditions
            new_column = pd.Series(
                np.where((X[var] > self.upper_limit_list[count]), 1, 0),
                index=X.index,  # Preserve original index
                name=f"{var}_outliers",  # Name of the new column
            )
            # Append the new Series to the list
            new_columns.append(new_column)

        # Concatenate the list of Series along the columns axis
        X = pd.concat([X] + new_columns, axis=1)

        # return result as data frame
        return X


# Add Minimum Arbitrary Outliers indicator to data
class AddMinimumArbitraryOutliersIndicatorTransform(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer

    def __init__(self, min_capping_dict, max_capping_dict=None):
        # Check the input of function
        if not isinstance(min_capping_dict, dict):
            raise ValueError("min_capping_dict should be a dictionnary")
        if max_capping_dict is not None:
            raise ValueError("max_capping_dict should be equal to None")

        self.max_capping_dict = max_capping_dict
        self.min_capping_dict = min_capping_dict

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # initialize list
        self.variable_list = []
        self.lower_limit_list = []
        self.upper_limit_list = []

        # Calcul variable_list, lower_limit and upper_limit
        self.variable_list = list(self.min_capping_dict.keys())
        self.lower_limit_list = list(self.min_capping_dict.values())

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Create a list to store the new Series: for min_variable_list
        new_columns = []

        # Loop through each min variable
        for count, var in enumerate(self.variable_list):
            # Create a Series based on the conditions
            new_column = pd.Series(
                np.where((X[var] < self.lower_limit_list[count]), 1, 0),
                index=X.index,  # Preserve original index
                name=f"{var}_outliers",  # Name of the new column
            )
            # Append the new Series to the list
            new_columns.append(new_column)

        # Concatenate the list of Series along the columns axis
        X = pd.concat([X] + new_columns, axis=1)

        # return result as data frame
        return X


# Create new features that capture presence or absence of Outliers in data set
class OutliersFeatureCreation(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer

    def __init__(self, outliers_num_vars_list):
        # Check the input of function
        if not isinstance(outliers_num_vars_list, list):
            raise ValueError("outliers_num_vars should be a list")

        self.outliers_num_vars_list = outliers_num_vars_list

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # capture the outliers and create the new feature
        for var in self.outliers_num_vars_list:
            # Identify outliers using IQR
            Q1 = np.percentile(X[var], 25)
            Q3 = np.percentile(X[var], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (X[var] < lower_bound) | (X[var] > upper_bound)

            # add outliers indicator for each column with outliers data
            X[var + "_outliers"] = np.where(outliers, 1, 0)

        # return result as data frame
        return X


# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# # CATEGORICAL VARIABLES ENCODING
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# for encoding categorical variable using Count Encoder
class CountEncoderTransform(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self, variables):
        # Check the input of function
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.count_encoder_dict_ = None

    def fit(self, X, y):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Initialize count dictiionnary
        self.count_encoder_dict_ = {}

        # let's obtain the counts for each one of the labels in the variable
        for var in self.variables:
            self.count_encoder_dict_[var] = X[var].value_counts().to_dict()

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")
        assert isinstance(self.count_encoder_dict_, dict)

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Generate the data frame to return
        for feature in self.variables:
            X[feature] = X[feature].map(self.count_encoder_dict_[feature])

        # return result as dataa frame
        return X


# for encoding categorical variable using Frequency Encoder
class FrequencyEncoderTransform(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self, variables):
        # Check the input of function
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.frequency_encoder_dict_ = None

    def fit(self, X, y):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Initialize frequency dictiionnary
        self.frequency_encoder_dict_ = {}

        # let's obtain the frequency for each one of the labels in the variable
        for var in self.variables:
            self.frequency_encoder_dict_[var] = (
                X[var].value_counts(normalize=True)
            ).to_dict()

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")
        assert isinstance(self.frequency_encoder_dict_, dict)

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Generate the data frame to return
        for feature in self.variables:
            X[feature] = X[feature].map(self.frequency_encoder_dict_[feature])

        # return as result as data frame
        return X


# for encoding Rare labels (categorical variable)
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Groups infrequent categories into a single string"""

    def __init__(self, variables, tol=0.05):
        # Check the input of function
        self.encoder_dict_ = None
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.tol = tol
        self.variables = variables

    def fit(self, X, y=None):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            tup = pd.Series(X[var].value_counts(normalize=True))
            # frequent labels:
            self.encoder_dict_[var] = list(tup[tup >= self.tol].index)

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")
        assert isinstance(self.encoder_dict_, dict)

        # Make a copy of original data set
        X = X.copy()

        # Generate the data frame to return
        for feature in self.variables:
            X[feature] = np.where(
                X[feature].isin(self.encoder_dict_[feature]), X[feature], "Rare"
            )

        # return result as data frame
        return X


# for encoding categorical variable
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self, variables):
        # Check the input of function
        self.encoder_dict_ = None
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X, y=None):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # concatenate and rename column name of target
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ["target"]

        # persist transforming dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            tup = temp.groupby([var])["target"].mean().sort_values(ascending=True).index
            self.encoder_dict_[var] = {k: i for i, k in enumerate(tup, 0)}

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")
        assert isinstance(self.encoder_dict_, dict)

        # Make a copy of original data set
        X = X.copy()

        # Generate the data frame to return
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        # return result as data frame
        return X


# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# # MULTIVARIATE OUTLIERS DETECTING WITH PyOD LIBRARY
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Detecte multivariate outliers in data set with PyOD library
class DetecteMultivariateOutliersWithPyODTransform(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        # variable_list: list,
        model_name: str,
        model,
    ):
        # Check the type of variables passed in the function
        # if not isinstance(variable_list, list):
        #     raise ValueError("variable_list should be a list")
        if not isinstance(model_name, str):
            raise ValueError("model_name should be string.")

        # self.variable_list = variable_list
        self.model_name = model_name
        self.model = model

    # function to train Doc2Vec Embedding Transformers model
    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        """
        Generate multivariate outliers indicator feature in data set.

        Parameters:
        - input_data (pd.DataFrame): The input data frame.

        Returns:
        - self: .
        """
        # Check the type of variables passed in the function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be dataframe.")

        # Make a copy of the input data
        X = X.copy()

        # Set the outlier detection
        self.outliers_detector_name = self.model_name
        self.outliers_detector = self.model

        # train IForest detector
        self.outliers_detector.fit(X)

        # return result as self
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # get the prediction label and outlier scores of the training data
        # outlier labels (0 or 1)
        inliers_outliers_pred = self.outliers_detector.predict(X)

        # Create encoded senteces dataframe
        inliers_outliers_frame = pd.DataFrame(inliers_outliers_pred)

        # creating the dataframe
        inliers_outliers_frame.columns = [
            f"{self.outliers_detector_name}_multi_var_outliers_indic"
        ]

        # have data frame with same index values before concatenate
        # get dictionnary to use for replacing index in data set
        X_index = X.index.tolist()
        iof_index = inliers_outliers_frame.index.tolist()
        index_values_dict: dict = {}
        for count in range(len(X_index)):
            index_values_dict[iof_index[count]] = X_index[count]

        # replace inliers_outliers_frame index by X index
        inliers_outliers_frame.rename(index=index_values_dict, inplace=True)

        # concatenate dataframe horizontaly
        concat_data_frame = pd.concat([X, inliers_outliers_frame], axis=1)

        # Generate the new data frame with existant columns and outliers and inliers indicator
        X = concat_data_frame.copy()

        # return result as data frame
        return X


# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# # MULTIVARIATE OUTLIERS DETECTING WITH PyOD LIBRARY
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Select feature needed for predict step
class SelectFeatureTransform(BaseEstimator, TransformerMixin):
    def __init__(self, selected_variables):
        # Check the input of function
        if not isinstance(selected_variables, list):
            raise ValueError("selected_variables should be a list")

        self.selected_variables = selected_variables

    def fit(self, X, y=None):
        # we need the fit statement to accommodate the sklearn pipeline
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # return result as self
        return self

    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # Make a copy of original data set, so that we do not over-write the original data frame
        X = X.copy()

        # Select feature needed
        X = X[self.selected_variables]

        # return result as data frame
        return X


# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# # STACKING ESTIMATOR BUILDING
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# First stage of stacking estimator
class StackingEstimatorFirstStageTransform(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer

    def __init__(self, variables, estimators, estimators_names, estimators_thresholds):
        # Check the input of function
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        if not isinstance(estimators, list):
            raise ValueError("estimators should be a list")
        if not isinstance(estimators_names, list):
            raise ValueError("estimators_names should be a list")
        if not isinstance(estimators_thresholds, list):
            raise ValueError("estimators_thresholds should be a list")

        self.variables = variables
        self.estimators = estimators
        self.estimators_names = estimators_names
        self.estimators_thresholds = estimators_thresholds

    # model performance for customer metric
    def model_performance_for_custom_metric(
        self, y_true: pd.Series, y_pred: pd.Series, metric: str, **kwargs
    ) -> float:
        """
        Calculation of roc auc metric.

        Parameters:
        - y_true (pd.Series): The true series .
        - y_pred (pd.Series): The predicted series.
        - metric (str): The metric to use.

        Returns:
        - float: The metric calculated.
        """
        # Check the type of variables passed in the function
        if not isinstance(y_true, pd.Series):
            raise ValueError("y_true should be a pandas series.")
        if not isinstance(y_pred, pd.Series):
            raise ValueError("y_pred should be a pandas series.")
        if not isinstance(metric, str):
            raise ValueError("metric should be string.")

        # Classification metric
        if metric == "roc_auc_score":
            metric_score = roc_auc_score(y_true, y_pred)
        if metric == "accuracy_score":
            metric_score = accuracy_score(y_true, y_pred)
        if metric == "balanced_accuracy_score":
            metric_score = balanced_accuracy_score(y_true, y_pred)

        # Regression metric
        if metric == "r2_score":
            metric_score = r2_score(y_true, y_pred)
        if metric == "mean_squared_error":
            metric_score = mean_squared_error(y_true, y_pred)

        # return result as float
        return metric_score

    # custom metric
    def custom_metric_rocauc(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """
        Calculation of roc auc metric.

        Parameters:
        - y_true (pd.Series): The true series .
        - y_pred (pd.Series): The predicted series.

        Returns:
        - float: The metric calculated.
        """
        # # Check the type of variables passed in the function
        # if not isinstance(y_true, pd.Series):
        #     raise ValueError("y_true should be a pandas series.")
        # if not isinstance(y_pred, pd.Series):
        #     raise ValueError("y_pred should be a pandas series.")

        # Metric based on roc auc
        metric_score = roc_auc_score(y_true, y_pred)

        # return result as float
        return metric_score

    # custom metric
    def custom_metric_balanced_accuracy(
        self, y_true: pd.Series, y_pred: pd.Series
    ) -> float:
        """
        Calculation of roc auc metric.

        Parameters:
        - y_true (pd.Series): The true series .
        - y_pred (pd.Series): The predicted series.

        Returns:
        - float: The metric calculated.
        """
        # # Check the type of variables passed in the function
        # if not isinstance(y_true, pd.Series):
        #     raise ValueError("y_true should be a pandas series.")
        # if not isinstance(y_pred, pd.Series):
        #     raise ValueError("y_pred should be a pandas series.")

        # Metric based on balanced accuracy
        metric_score = balanced_accuracy_score(y_true, y_pred)

        # return result as float
        return metric_score

    # define the cross validation function for ann model
    def manual_cross_validation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        fold_method: str,
        n_fold: int,
        random_state: int,
        epochs: int,
        batch_size: int,
        verbosity: int,
        model,
    ) -> tuple:
        """
        Generate the architecture of artificial neural network.

        Parameters:
        - X_train (pd.DataFrame): The train data set.
        - y_train (pd.Series): The target associated to train data set.
        - fold_method (str): The fold method to use for split data for cross validation.
        - n_fold (int): The cross-validation fold.
        - random_state (integer): The random state number for the reproductibility.
        - activation_output (str): The activation function for output layers.
        - kernel_initializer (str): The kernel initializer.
        - epochs (int): The number of iteration to find the parameters of model when hyperparameters
            are fixed.
        - batch_size (int): The number of data to fit.
        - verbosity (int): The verbosity needed (0 or 1).
        - model (...): The model to use for cross validation.

        Returns:
        - tuple: A tuple.
        """
        # Check the type of variables passed in the function
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train should be pandas dataframe.")
        if not isinstance(y_train, pd.Series):
            raise ValueError("y_train should be a pandas series.")
        if not isinstance(fold_method, str):
            raise ValueError("fold_method should be string.")
        if not isinstance(n_fold, int):
            raise ValueError("n_fold should be integer.")
        if not isinstance(random_state, int):
            raise ValueError("random_state should be integer.")
        if not isinstance(epochs, int):
            raise ValueError("epochs should be an integer.")
        if not isinstance(batch_size, int):
            raise ValueError("batch_size should be integer.")
        if not isinstance(verbosity, int):
            raise ValueError("verbosity should be integer.")

        # Define per-fold score containers list
        metric_per_fold: list = []
        loss_per_fold: list = []

        # Create fold Cross Validator object.
        if fold_method == "KFold":
            split_method = KFold(n_splits=n_fold, shuffle=True)
        if fold_method == "StratifiedKFold":
            split_method = StratifiedKFold(
                n_splits=n_fold, shuffle=True, random_state=random_state
            )

        for train_index, test_index in split_method.split(X_train, y_train):
            X_train_fold, X_test_fold = (
                X_train.iloc[train_index],
                X_train.iloc[test_index],
            )
            y_train_fold, y_test_fold = (
                y_train.iloc[train_index],
                y_train.iloc[test_index],
            )

            # Train model
            # maximum of batch_size = y_train.shape[0]
            model.fit(
                X_train_fold,
                y_train_fold,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbosity,
            )

            # Generate generalization metrics
            scores = model.evaluate(X_test_fold, y_test_fold, verbose=verbosity)
            loss_per_fold.append(scores[0])
            metric_per_fold.append(scores[1])

        # Return the result as tuple
        return (
            model,
            np.mean(metric_per_fold),
            np.std(metric_per_fold),
            np.mean(loss_per_fold),
            np.std(loss_per_fold),
        )

    # Function for cross validation on train step
    def cross_validation_function(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        metric: str,
        random_state: int,
        n_fold: int,
        estimator,
    ):
        """
        Plot Feature Importance based on Permutation Importance on test set.

        Parameters:
        - X_train (pd.DataFrame): The train data set.
        - y_train (pd.Series): The target associated to train data set.
        - metric (str): The metric want to use.
        - random_state (integer): The random state number.
        - n_fold (int): The number for splitting train set in cross validation procedure.
        - estimator (): The model to use.

        Returns:
        - tuple: A tuple of list contains feature to select and to remove.
        """
        # Check the type of variables passed in the function
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train should be pandas dataframe.")
        if not isinstance(y_train, pd.Series):
            raise ValueError("y_train should be a pandas series.")
        if not isinstance(metric, str):
            raise ValueError("metric should be string.")
        if not isinstance(random_state, int):
            raise ValueError("random_state should be an integer.")
        if not isinstance(n_fold, int):
            raise ValueError("n_fold should be integer.")
        # if not isinstance(estimator, ...):
        #     raise ValueError("estimator should be a ....")

        # Make a copy of a part of the input data
        X = X_train.copy()
        y = y_train.copy()

        # Create the transformers and estimators
        model_to_train = estimator

        # Make pipeline with transformer and estimators ==> pipe
        pipe = Pipeline([("estimator", model_to_train)])

        # Create hyper parameter dictionnary for GreedSearchCV,
        # put in this dictionnary parameter of estimator you want to find the best ==> param_grid
        param_grid: dict = {}

        # Specification of the splitting method and it's parameter
        # cv = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

        # # custom metric
        # custom_scoring = make_scorer(
        #         self.model_performance_for_custom_metric, greater_is_better=True
        #     )

        # custom metric
        if metric == "roc_auc_score":
            custom_scoring = make_scorer(
                self.custom_metric_rocauc,
                greater_is_better=True,
                response_method="predict_proba",
            )
        if metric == "balanced_accuracy_score":
            custom_scoring = make_scorer(
                self.custom_metric_balanced_accuracy, greater_is_better=True
            )
        scoring = {"customer_metric": custom_scoring}

        # Make GridSearch with param_grid
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            refit="customer_metric",
        )

        # Make traning with grid.fit
        grid_search.fit(X, y)

        # write best model and it parameters
        best_model_trained = grid_search.best_estimator_.named_steps["estimator"]
        # best_model_one_feature_param = grid_search.best_params_

        # return the estimator fitted
        return best_model_trained

    # function for fit method
    def fit(self, X, y=None):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Train the estimator
        self.estimators_trained: list = []
        for count, estimator in enumerate(self.estimators):
            # train estimator
            if self.estimators_thresholds[count] is not None:
                # define parameters
                metric = "roc_auc_score"
                n_fold = 5

                # train model with cross validation
                etstimator_trained = self.cross_validation_function(
                    X_train=X[self.variables[count]],
                    y_train=y,
                    metric=metric,
                    random_state=seed,
                    n_fold=n_fold,
                    estimator=estimator,
                )
            else:
                # define parameters
                metric = "balanced_accuracy_score"
                n_fold = 5

                # train model with cross validation
                etstimator_trained = self.cross_validation_function(
                    X_train=X[self.variables[count]],
                    y_train=y,
                    metric=metric,
                    random_state=seed,
                    n_fold=n_fold,
                    estimator=estimator,
                )

            # append estimator trained in list
            self.estimators_trained.append(etstimator_trained)

        # return result as self
        return self

    # wrapper to controll Discrimination threshold
    def custom_predict(
        self, estimator_trained, frame_to_predict: pd.DataFrame, threshold: float
    ) -> tuple:
        """
        Wrapper to controll the class prediction with discrimination threshold.

        Parameters:
        - estimator_trained (Pipeline or model): The estimator already trained.
        - frame_to_predict (pd.DataFrame): The data set to predict.
        - threshold (float): The discrimination threshold to use for class prediction.

        Returns:
        - tuple: A tuple.
        """
        # Check the type of variables passed in the function
        # if not isinstance(estimator_trained, Pipeline):
        #     raise ValueError("estimator_trained should be Pipeline.")
        if not isinstance(frame_to_predict, pd.DataFrame):
            raise ValueError("frame_to_predict should be pandas dataframe.")
        if not isinstance(threshold, float):
            raise ValueError("threshold should be a float.")

        # get predict probability for class 1 (always minority class)
        proba_prediction = estimator_trained.predict_proba(frame_to_predict)[:, 1]

        # calculate probability and predict class with threshold
        probs = proba_prediction
        preds = (probs > threshold).astype(int)

        # return results as tuple
        return probs, preds

    # function for transform method
    def transform(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Predict with etimator trained
        output_dict: dict = {}
        for count, estimator in enumerate(self.estimators_trained):
            if self.estimators_thresholds[count] is not None:
                y_prob, y_pred = self.custom_predict(
                    estimator_trained=estimator,
                    frame_to_predict=X[self.variables[count]],
                    threshold=float(self.estimators_thresholds[count][0]),
                )
                y_pred_list = pd.Series(y_pred).tolist()
                output_dict[self.estimators_names[count]] = y_pred_list
            else:
                y_pred = estimator.predict(X[self.variables[count]])
                y_pred_list = pd.Series(y_pred).tolist()
                output_dict[self.estimators_names[count]] = y_pred_list

        # Create output dataframe
        output_frame = pd.DataFrame(data=output_dict)

        # return result as data frame
        return output_frame


# Second stage of stacking estimator
class StackingEstimatorSecondStageTransform(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer

    def __init__(
        self, final_estimators, final_estimators_names, final_estimators_thresholds
    ):
        # Check the input of function
        if not isinstance(final_estimators_names, list):
            raise ValueError("final_estimators_names should be a list")
        if not isinstance(final_estimators_thresholds, list):
            raise ValueError("final_estimators_thresholds should be a list")

        self.final_estimators = final_estimators
        self.final_estimators_names = final_estimators_names
        self.final_estimators_thresholds = final_estimators_thresholds

    def fit(self, X, y=None):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # # Train the estimator
        # self.final_estimators.fit(X, y)

        # return result as self
        return self

    # wrapper to controll Discrimination threshold
    def custom_predict(
        self, estimator_trained, frame_to_predict: pd.DataFrame, threshold: float
    ) -> tuple:
        """
        Wrapper to controll the class prediction with discrimination threshold.

        Parameters:
        - estimator_trained (Pipeline or model): The estimator already trained.
        - frame_to_predict (pd.DataFrame): The data set to predict.
        - threshold (float): The discrimination threshold to use for class prediction.

        Returns:
        - tuple: A tuple.
        """
        # Check the type of variables passed in the function
        # if not isinstance(estimator_trained, Pipeline):
        #     raise ValueError("estimator_trained should be Pipeline.")
        if not isinstance(frame_to_predict, pd.DataFrame):
            raise ValueError("frame_to_predict should be pandas dataframe.")
        if not isinstance(threshold, float):
            raise ValueError("threshold should be a float.")

        # get predict probability for class 1 (always minority class)
        proba_prediction = estimator_trained.predict_proba(frame_to_predict)[:, 1]

        # calculate probability and predict class with threshold
        probs = proba_prediction
        preds = (probs > threshold).astype(int)

        # return results as tuple
        return probs, preds

    def predict(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        for count, estimator in enumerate(self.final_estimators):
            if self.final_estimators_thresholds[count] is not None:
                y_prob, y_pred = self.custom_predict(
                    estimator_trained=estimator,
                    frame_to_predict=X,
                    threshold=float(self.final_estimators_thresholds[count][0]),
                )
            else:
                y_pred = estimator.predict(X)

        # return result array
        return y_pred

    def predict_proba(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        for count, estimator in enumerate(self.final_estimators):
            if self.final_estimators_thresholds[count] is not None:
                y_prob, y_pred = self.custom_predict(
                    estimator_trained=estimator,
                    frame_to_predict=X,
                    threshold=float(self.final_estimators_thresholds[count][0]),
                )

                # return result as tuple
                return y_prob
            else:
                raise ValueError(
                    f"{self.final_estimators_names[count]} has no attribute 'predict_proba'"
                )


# Final estimator for pipeline
class FinalEstimatorTransform(BaseEstimator, TransformerMixin):
    # Temporal elapsed time transformer

    def __init__(self, variables, estimators, estimators_names, estimators_thresholds):
        # Check the input of function
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        if not isinstance(estimators, list):
            raise ValueError("estimators should be a list")
        if not isinstance(estimators_names, list):
            raise ValueError("estimators_names should be a list")
        if not isinstance(estimators_thresholds, list):
            raise ValueError("estimators_thresholds should be a list")

        self.variables = variables
        self.estimators = estimators
        self.estimators_names = estimators_names
        self.estimators_thresholds = estimators_thresholds

    # model performance for customer metric
    def model_performance_for_custom_metric(
        self, y_true: pd.Series, y_pred: pd.Series, metric: str, **kwargs
    ) -> float:
        """
        Calculation of roc auc metric.

        Parameters:
        - y_true (pd.Series): The true series .
        - y_pred (pd.Series): The predicted series.
        - metric (str): The metric to use.

        Returns:
        - float: The metric calculated.
        """
        # Check the type of variables passed in the function
        if not isinstance(y_true, pd.Series):
            raise ValueError("y_true should be a pandas series.")
        if not isinstance(y_pred, pd.Series):
            raise ValueError("y_pred should be a pandas series.")
        if not isinstance(metric, str):
            raise ValueError("metric should be string.")

        # Classification metric
        if metric == "roc_auc_score":
            metric_score = roc_auc_score(y_true, y_pred)
        if metric == "accuracy_score":
            metric_score = accuracy_score(y_true, y_pred)
        if metric == "balanced_accuracy_score":
            metric_score = balanced_accuracy_score(y_true, y_pred)

        # Regression metric
        if metric == "r2_score":
            metric_score = r2_score(y_true, y_pred)
        if metric == "mean_squared_error":
            metric_score = mean_squared_error(y_true, y_pred)

        # return result as float
        return metric_score

    # custom metric
    def custom_metric_rocauc(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """
        Calculation of roc auc metric.

        Parameters:
        - y_true (pd.Series): The true series .
        - y_pred (pd.Series): The predicted series.

        Returns:
        - float: The metric calculated.
        """
        # # Check the type of variables passed in the function
        # if not isinstance(y_true, pd.Series):
        #     raise ValueError("y_true should be a pandas series.")
        # if not isinstance(y_pred, pd.Series):
        #     raise ValueError("y_pred should be a pandas series.")

        # Metric based on roc auc
        metric_score = roc_auc_score(y_true, y_pred)

        # return result as float
        return metric_score

    # custom metric
    def custom_metric_balanced_accuracy(
        self, y_true: pd.Series, y_pred: pd.Series
    ) -> float:
        """
        Calculation of roc auc metric.

        Parameters:
        - y_true (pd.Series): The true series .
        - y_pred (pd.Series): The predicted series.

        Returns:
        - float: The metric calculated.
        """
        # # Check the type of variables passed in the function
        # if not isinstance(y_true, pd.Series):
        #     raise ValueError("y_true should be a pandas series.")
        # if not isinstance(y_pred, pd.Series):
        #     raise ValueError("y_pred should be a pandas series.")

        # Metric based on balanced accuracy
        metric_score = balanced_accuracy_score(y_true, y_pred)

        # return result as float
        return metric_score

    # define the cross validation function for ann model
    def manual_cross_validation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        fold_method: str,
        n_fold: int,
        random_state: int,
        epochs: int,
        batch_size: int,
        verbosity: int,
        model,
    ) -> tuple:
        """
        Generate the architecture of artificial neural network.

        Parameters:
        - X_train (pd.DataFrame): The train data set.
        - y_train (pd.Series): The target associated to train data set.
        - fold_method (str): The fold method to use for split data for cross validation.
        - n_fold (int): The cross-validation fold.
        - random_state (integer): The random state number for the reproductibility.
        - activation_output (str): The activation function for output layers.
        - kernel_initializer (str): The kernel initializer.
        - epochs (int): The number of iteration to find the parameters of model when hyperparameters
            are fixed.
        - batch_size (int): The number of data to fit.
        - verbosity (int): The verbosity needed (0 or 1).
        - model (...): The model to use for cross validation.

        Returns:
        - tuple: A tuple.
        """
        # Check the type of variables passed in the function
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train should be pandas dataframe.")
        if not isinstance(y_train, pd.Series):
            raise ValueError("y_train should be a pandas series.")
        if not isinstance(fold_method, str):
            raise ValueError("fold_method should be string.")
        if not isinstance(n_fold, int):
            raise ValueError("n_fold should be integer.")
        if not isinstance(random_state, int):
            raise ValueError("random_state should be integer.")
        if not isinstance(epochs, int):
            raise ValueError("epochs should be an integer.")
        if not isinstance(batch_size, int):
            raise ValueError("batch_size should be integer.")
        if not isinstance(verbosity, int):
            raise ValueError("verbosity should be integer.")

        # Define per-fold score containers list
        metric_per_fold: list = []
        loss_per_fold: list = []

        # Create fold Cross Validator object.
        if fold_method == "KFold":
            split_method = KFold(n_splits=n_fold, shuffle=True)
        if fold_method == "StratifiedKFold":
            split_method = StratifiedKFold(
                n_splits=n_fold, shuffle=True, random_state=random_state
            )

        for train_index, test_index in split_method.split(X_train, y_train):
            X_train_fold, X_test_fold = (
                X_train.iloc[train_index],
                X_train.iloc[test_index],
            )
            y_train_fold, y_test_fold = (
                y_train.iloc[train_index],
                y_train.iloc[test_index],
            )

            # Train model
            # maximum of batch_size = y_train.shape[0]
            model.fit(
                X_train_fold,
                y_train_fold,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbosity,
            )

            # Generate generalization metrics
            scores = model.evaluate(X_test_fold, y_test_fold, verbose=verbosity)
            loss_per_fold.append(scores[0])
            metric_per_fold.append(scores[1])

        # Return the result as tuple
        return (
            model,
            np.mean(metric_per_fold),
            np.std(metric_per_fold),
            np.mean(loss_per_fold),
            np.std(loss_per_fold),
        )

    # Function for cross validation on train step
    def cross_validation_function(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        metric: str,
        random_state: int,
        n_fold: int,
        estimator,
    ):
        """
        Plot Feature Importance based on Permutation Importance on test set.

        Parameters:
        - X_train (pd.DataFrame): The train data set.
        - y_train (pd.Series): The target associated to train data set.
        - metric (str): The metric want to use.
        - random_state (integer): The random state number.
        - n_fold (int): The number for splitting train set in cross validation procedure.
        - estimator (): The model to use.

        Returns:
        - tuple: A tuple of list contains feature to select and to remove.
        """
        # Check the type of variables passed in the function
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train should be pandas dataframe.")
        if not isinstance(y_train, pd.Series):
            raise ValueError("y_train should be a pandas series.")
        if not isinstance(metric, str):
            raise ValueError("metric should be string.")
        if not isinstance(random_state, int):
            raise ValueError("random_state should be an integer.")
        if not isinstance(n_fold, int):
            raise ValueError("n_fold should be integer.")
        # if not isinstance(estimator, ...):
        #     raise ValueError("estimator should be a ....")

        # Make a copy of a part of the input data
        X = X_train.copy()
        y = y_train.copy()

        # Create the transformers and estimators
        model_to_train = estimator

        # Make pipeline with transformer and estimators ==> pipe
        pipe = Pipeline([("estimator", model_to_train)])

        # Create hyper parameter dictionnary for GreedSearchCV,
        # put in this dictionnary parameter of estimator you want to find the best ==> param_grid
        param_grid: dict = {}

        # Specification of the splitting method and it's parameter
        # cv = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

        # # custom metric
        # custom_scoring = make_scorer(
        #         self.model_performance_for_custom_metric, greater_is_better=True
        #     )

        # custom metric
        if metric == "roc_auc_score":
            custom_scoring = make_scorer(
                self.custom_metric_rocauc,
                greater_is_better=True,
                response_method="predict_proba",
            )
        if metric == "balanced_accuracy_score":
            custom_scoring = make_scorer(
                self.custom_metric_balanced_accuracy, greater_is_better=True
            )
        scoring = {"customer_metric": custom_scoring}

        # Make GridSearch with param_grid
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            refit="customer_metric",
        )

        # Make traning with grid.fit
        grid_search.fit(X, y)

        # write best model and it parameters
        best_model_trained = grid_search.best_estimator_.named_steps["estimator"]
        # best_model_one_feature_param = grid_search.best_params_

        # return the estimator fitted
        return best_model_trained

    # function for fit method
    def fit(self, X, y=None):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Train the estimator
        self.estimators_trained: list = []
        for count, estimator in enumerate(self.estimators):
            # train estimator
            if self.estimators_thresholds[count] is not None:
                # define parameters
                metric = "roc_auc_score"
                n_fold = 5

                # train model with cross validation
                etstimator_trained = self.cross_validation_function(
                    X_train=X[self.variables[count]],
                    y_train=y,
                    metric=metric,
                    random_state=seed,
                    n_fold=n_fold,
                    estimator=estimator,
                )
            else:
                # define parameters
                metric = "balanced_accuracy_score"
                n_fold = 5

                # train model with cross validation
                etstimator_trained = self.cross_validation_function(
                    X_train=X[self.variables[count]],
                    y_train=y,
                    metric=metric,
                    random_state=seed,
                    n_fold=n_fold,
                    estimator=estimator,
                )

            # append estimator trained in list
            self.estimators_trained.append(etstimator_trained)

        # return result as self
        return self

    # wrapper to controll Discrimination threshold
    def custom_predict(
        self, estimator_trained, frame_to_predict: pd.DataFrame, threshold: float
    ) -> tuple:
        """
        Wrapper to controll the class prediction with discrimination threshold.

        Parameters:
        - estimator_trained (Pipeline or model): The estimator already trained.
        - frame_to_predict (pd.DataFrame): The data set to predict.
        - threshold (float): The discrimination threshold to use for class prediction.

        Returns:
        - tuple: A tuple.
        """
        # Check the type of variables passed in the function
        # if not isinstance(estimator_trained, Pipeline):
        #     raise ValueError("estimator_trained should be Pipeline.")
        if not isinstance(frame_to_predict, pd.DataFrame):
            raise ValueError("frame_to_predict should be pandas dataframe.")
        if not isinstance(threshold, float):
            raise ValueError("threshold should be a float.")

        # get predict probability for class 1 (always minority class)
        proba_prediction = estimator_trained.predict_proba(frame_to_predict)[:, 1]

        # calculate probability and predict class with threshold
        probs = proba_prediction
        preds = (probs > threshold).astype(int)

        # return results as tuple
        return probs, preds

    # function for pedict method
    def predict(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Predict with etimator trained
        for count, estimator in enumerate(self.estimators_trained):
            if self.estimators_thresholds[count] is not None:
                y_prob, y_pred = self.custom_predict(
                    estimator_trained=estimator,
                    frame_to_predict=X[self.variables[count]],
                    threshold=float(self.estimators_thresholds[count][0]),
                )
            else:
                y_pred = estimator.predict(X[self.variables[count]])

        # return result as nd.array
        return y_pred

    # function for predict proba method
    def predict_proba(self, X):
        # Check the input of function
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")

        # so that we do not over-write the original dataframe
        X = X.copy()

        # Predict with etimator trained
        for count, estimator in enumerate(self.estimators_trained):
            if self.estimators_thresholds[count] is not None:
                y_prob, y_pred = self.custom_predict(
                    estimator_trained=estimator,
                    frame_to_predict=X[self.variables[count]],
                    threshold=float(self.estimators_thresholds[count][0]),
                )
            else:
                raise ValueError(
                    f"{self.estimators_names[count]} has no attribute predict_proba"
                )

        # return result as data frame
        return y_prob
