# import library and module
import os
import pickle
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
import string

# for reproducibility, split size
seed = 0
split_size = 0.3

# make alphabet list
alphabet = list(string.ascii_uppercase)
unknow = ["unknown"]


# function to convert dataframe into csv file
# IMPORTANT: Cache the function to prevent computation on every rerun
@st.cache_data
def convert_frame_into_csv(*, input_frame: pd.DataFrame):
    """
    Convert dataframe into csv file.

    Parameters:
    - input_frame (pd.DataFrame): The input dataframe.

    Returns:
    - csv file: csv file.
    """
    # Check the type of variables passed in the function
    if not isinstance(input_frame, pd.DataFrame):
        raise ValueError("input_frame should be dataframe.")

    # return result as csv file
    return input_frame.to_csv().encode("utf-8")


# function to load model from folder
# IMPORTANT: Cache the function to prevent computation on every rerun
@st.cache_data
def load_model_from_directory(*, directory: str, file_name: str) -> None:
    """
    Load or persisted model.

    Parameters:
    - model (list): The model to be saved.
    - directory (str): The path to the directory where the file is saved.
    - file_name (str): The name of the file.

    Returns:
    - the model saved.
    """

    # check the type of variable witch will be pass in the function
    if not isinstance(directory, str) or not isinstance(file_name, str):
        raise ValueError("Both directory and file_name should be a string type")

    # Get the model from folder
    file_path = Path(directory) / file_name
    trained_model = joblib.load(filename=file_path)

    # return result as model
    return trained_model


# function to load pipeline
# IMPORTANT: Cache the function to prevent computation on every rerun
@st.cache_data
def load_pipeline(*, directory: Path, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    # check the type of variable witch will be pass in the function
    if not isinstance(directory, Path):
        raise ValueError('variable passed in directory should be a Path (directory of folder) type')
    if not isinstance(file_name, str):
        raise ValueError('variable passed in file_name should be a string type')

    file_path = directory / file_name
    trained_pipeline = joblib.load(filename=file_path)
    return trained_pipeline


# function to loading file from directory path
# IMPORTANT: Cache the function to prevent computation on every rerun
@st.cache_data
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


# load all neccessary variable
# about dictionnary
cat_feat_uniqval_dict = get_variable_list_by_loading_file(
    file_path="./variables_needed/",
    file_name="production_categorical_features_unique_value.pkl"
)


# function for set trip information
# Function to set sidebar of app
def user_manual_input() -> pd.DataFrame:
    # side bar on frontend app to enter user manual inputs
    title_attrib = st.sidebar.selectbox(
        "title",
        tuple(unknow+cat_feat_uniqval_dict["title"])
    )
    sex_attrib = st.sidebar.selectbox(
        "sex",
        tuple(unknow+cat_feat_uniqval_dict["sex"])
    )
    age_attrib = st.sidebar.slider(
        label="age",
        min_value=0, max_value=100,
        value=29
    )
    ticket_attrib = st.sidebar.selectbox(
        "ticket",
        tuple(unknow+cat_feat_uniqval_dict["ticket"])
    )
    fare_attrib = st.sidebar.slider(
        label="fare",
        min_value=0.0, max_value=512.3292,
        value=3.0
    )
    cabin_attrib = st.sidebar.selectbox(
        "cabin",
        tuple(unknow+alphabet+cat_feat_uniqval_dict["cabin"])
    )
    pclass_attrib = st.sidebar.slider(
        label="pclass",
        min_value=0, max_value=3,
        value=3
    )
    embarked_attrib = st.sidebar.selectbox(
        "embarked",
        tuple(unknow+cat_feat_uniqval_dict["embarked"])
    )
    sibsp_attrib = st.sidebar.slider(
        label="sibsp",
        min_value=0, max_value=8,
        value=7
    )
    parch_attrib = st.sidebar.slider(
        label="parch",
        min_value=0, max_value=9,
        value=7
    )
    home_attrib = st.sidebar.selectbox(
        "home",
        tuple(unknow+cat_feat_uniqval_dict["home"])
    )
    destination_attrib = st.sidebar.selectbox(
        "destination",
        tuple(unknow+cat_feat_uniqval_dict["destination"])
    )

    # dictionnary of manual users input data
    data = {
        "title": title_attrib,
        "sex": sex_attrib,
        "age": age_attrib,
        "ticket": ticket_attrib,
        "fare": fare_attrib,
        "cabin": cabin_attrib,
        "pclass": pclass_attrib,
        "embarked": embarked_attrib,
        "sibsp": sibsp_attrib,
        "parch": parch_attrib,
        "home": home_attrib,
        "destination": destination_attrib,
    }

    # dataframe of manual users input data
    data_df = pd.DataFrame(data, index=[0])

    # return result as dataframe
    return data_df


# function for prediction
def download_csv_file(*, input_frame: pd.DataFrame) -> None:
    """
    Prediction with user input.

    Parameters:
    - input_frame (DataFrame): The input dataframe.

    Returns:
    - None
    """
    # check the type of variable witch will be pass in the function
    if not isinstance(input_frame, pd.DataFrame):
        raise ValueError(
            "input_frame should be a dataframe"
        )

    # make a copy of input dataframe
    input_frame_c = input_frame.copy()

    # test if dataframe is not empty
    if input_frame_c.shape != (0, 0):
        # convert frame into csv file
        csv_file = convert_frame_into_csv(input_frame=input_frame_c)

        # make space between element on stremlit
        # st.write('#')
        st.write('--')

        # if submit_button:
        # button to download csv file if needed
        st.download_button(
            label="Download response",
            data=csv_file,
            file_name="trip_information_and_response.csv",
            mime="text/csv",
        )
    else:
        st.write("The input dataframe is empty.")

    # return result as None
    return None
