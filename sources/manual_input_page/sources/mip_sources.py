# import library and module
from pathlib import Path

import pandas as pd
import streamlit as st

from sources.streamlit_cloud_deployement.sources import \
    streamlit_cloud_deployement as st_cloud_dep_src

# ===== SET THE CONFIGURATION OF THE INTERFACE OF DASHBOARD =====
# all streamlit icon or emoji shortcodes supported by Streamlit can be found in this link :
# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

# load the model want to use
production_ml_model = st_cloud_dep_src.load_pipeline(
    directory=Path('./ml_model/'),
    file_name='production_prediction_pipeline_v0.pkl'
)

# load threshold value for the model use to prediction
mlp_classifier_thresholds = st_cloud_dep_src.get_variable_list_by_loading_file(
    file_path="./variables_needed/",
    file_name="threshold_mlflow_bagg_tuned_tpe_mlpclassifier_stacking_tuned_tpe_class_1.pkl",
)


# function for prediction
# IMPORTANT: Decorate the function with @st.experimental_fragment will run only this function
# when user make modification on frontend app
@st.experimental_fragment
def prediction(*, user_manual_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Prediction with user input.

    Parameters:
    - user_manual_frame (DataFrame): The dataframe of user input.

    Returns:
    - pd.DataFrame
    """
    # check the type of variable witch will be pass in the function
    if not isinstance(user_manual_frame, pd.DataFrame):
        raise ValueError(
            "user_manual_frame should be a dataframe"
        )

    # Treatment of case when user recorded manually the features
    if user_manual_frame is not None:
        # Get user input data
        user_manual_input_dataframe = user_manual_frame

        # Apply ML model to get prediction and probability of class appartenance
        class_proba_ = production_ml_model.predict_proba(user_manual_input_dataframe)
        class_ = production_ml_model.predict(user_manual_input_dataframe)

        # save prediction values in dictionnary
        results: dict = {}
        results["probability"] = class_proba_
        results["class"] = class_

        if class_[0] == 0:
            results["surivival"] = ["no"]
            results["perccentage"] = [(1-class_proba_[0])*100]
        else:
            results["surivival"] = ["yes"]
            results["perccentage"] = [class_proba_[0]*100]

        # Convert input data of model from list to dataframe
        prob_pred_df = pd.DataFrame.from_dict(
            results["perccentage"]
        )
        prob_pred_df.columns = ['perccentage']

        # Convert prediction from list to dataframe
        class_pred_df = pd.DataFrame.from_dict(
            results["surivival"]
        )
        class_pred_df.columns = ['surivival']

        # Concatenate of inlet feature and prediction
        feature_and_pred_frame = pd.concat(
            [user_manual_input_dataframe, class_pred_df, prob_pred_df],
            axis=1
        )

        # survived or not pecentage chance
        survived_percentage = class_proba_[0]*100
        not_survived_percentage = (1 - class_proba_[0])*100

        # printing probability and prediction
        if class_proba_ > mlp_classifier_thresholds:
            st.write(
                f"Based on my analyze, I'm certain at {survived_percentage:.2f} % that \
                    you'll survive to titanic disaster"
            )
        else:
            st.write(
                f"Based on my analyze, I'm certain at {not_survived_percentage:.2f} % that \
                    you won't survive to titanic disaster"
            )

        # make space between element on stremlit
        st.write('--')

        # printing resume of user input data, probability and prediction
        with st.form(key="my_form"):
            submit_button = st.form_submit_button(label="Display Trip and Predicted Information")

        if submit_button:
            st.write(feature_and_pred_frame)

            # # Download trip imformation and prediction
            # button to download csv file if needed
            st_cloud_dep_src.download_csv_file(input_frame=feature_and_pred_frame)
    else:
        st.write("Please enter a trip information before get prediction.")

    # return result as dataframe
    return feature_and_pred_frame
