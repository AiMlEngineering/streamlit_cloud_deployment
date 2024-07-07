# import library and module
import streamlit as st
import os
import typing as t
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline


# function save model as pickle file into dedicated folder
# IMPORTANT: Cache the function to prevent computation on every rerun
@st.cache_data
def save_model_into_directory(*, model, file_path: str, file_name: str) -> None:
    """
    Save a model to a file using pickle.

    Parameters:
    - model (list): The model to be saved.
    - file_path (str): The path to the directory where the file will be saved.
    - file_name (str): The name of the file.

    Returns:
    - None
    """
    # Check the type of variables passed in the function
    if not isinstance(file_path, str) or not isinstance(file_name, str):
        raise ValueError("Both file_path and file_name should be strings.")

    # Save the model to a file using pickle
    file_full_path = Path(file_path) / file_name
    try:
        with open(file_full_path, "wb") as file:
            joblib.dump(model, file)
    except (FileNotFoundError, PermissionError) as e:
        raise IOError(f"Error saving the file {file_full_path}: {e}")

    # Return results as None
    return None


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


# function to save pipeline
# IMPORTANT: Cache the function to prevent computation on every rerun
@st.cache_data
def save_pipeline(
    *,
    pipeline_save_dir_path: str,
    pipeline_save_file: str,
    pipeline_to_persist: Pipeline,
) -> None:
    """
    Persist pipeline using pickle.

    Parameters:
    - pipeline_save_dir_path (str): The path to the directory where the file will be saved.
    - pipeline_save_file (str): The name of the file.
    - pipeline_to_persist (Pipeline): The pipeline to be saved.

    Returns:
    - None
    """

    # check the type of variable witch will be pass in the function
    if not isinstance(pipeline_save_dir_path, str):
        raise ValueError(
            "variable passed in pipeline_save_dir_path should be a string type"
        )
    if not isinstance(pipeline_save_file, str):
        raise ValueError(
            "variable passed in pipeline_save_file should be a string type"
        )
    if not isinstance(pipeline_to_persist, Pipeline):
        raise ValueError(
            "variable passed in pipeline_to_persist should be a Pipeline type"
        )

    # Prepare versioned save file name
    save_file_name = f"{pipeline_save_file}.pkl"
    save_path = Path(pipeline_save_dir_path) / save_file_name

    # # remove the old model and save the new model to a file using pickle
    # remove_old_pipelines(
    #     directory=pipeline_save_dir_path, files_to_keep=[save_file_name]
    # )
    joblib.dump(pipeline_to_persist, save_path)

    # return result as None
    return None


# function to remove pipeline
# IMPORTANT: Cache the function to prevent computation on every rerun
@st.cache_data
def remove_old_pipelines(*, directory: str, files_to_keep: t.List[str]) -> None:
    """
    Remove pipeline.

    Parameters:
    - directory (str): The path to the directory where the file will be saved.
    - files_to_keep List[str]: The name of the file to keep.

    Returns:
    - None
    """

    # check the type of variable witch will be pass in the function
    if not isinstance(directory, str):
        raise ValueError("variable passed in directory should be a string type")
    if not isinstance(files_to_keep, list):
        raise ValueError(
            "variable passed in files_to_keep should be a list (of string) type"
        )

    # Remove the old model file
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in Path(directory).iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

    # return result as None
    return None


# function to load pipeline from folder
# IMPORTANT: Cache the function to prevent computation on every rerun
@st.cache_data
def load_pipeline(*, directory: str, file_name: str) -> Pipeline:
    """
    Load a pipeline.

    Parameters:
    - directory (str): The path to the directory where the file is saved.
    - file_name (str): The name of the file.

    Returns:
    - the pipeline saved.
    """

    # check the type of variable witch will be pass in the function
    if not isinstance(directory, str):
        raise ValueError("variable passed in directory should be a string type")
    if not isinstance(file_name, str):
        raise ValueError("variable passed in file_name should be a string type")

    # Load the model saved
    file_path = Path(directory) / file_name
    trained_model = joblib.load(filename=file_path)

    # return the model saved
    return trained_model
