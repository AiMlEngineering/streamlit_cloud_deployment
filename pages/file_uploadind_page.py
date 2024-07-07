# import library and module
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from sources.file_uploadind_page.sources import fip_sources as fip_src
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


# function for streamlit app frontend
def app():
    # set the title of the page icon
    st.subheader(
        'Titanic Disaster Survivor Prediction'
    )

    # formulate text to display
    text_1 = "This application is used as a front-end prototype for deployment of machine"
    text_2 = " learning model in production."
    text_3 = " It is not recommended to use this app in real production world."
    final_text = text_1+text_2+text_3

    # Display the split text
    st.text_area(
        label='#',
        value=final_text,
        height=100
    )

    # make space between element on stremlit
    # st.write('#')
    st.write('--')

    # Data uploaded by user
    user_upload_file = st.file_uploader(
        ":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"])
    )

    # Initialize dataframe
    userdata_pred_prob = None
    csv_file = None

    # Convert data recorded manually by user into frame
    if user_upload_file is not None:
        # make space between element on stremlit
        # st.write('#')
        st.write('--')

        with st.form(key="form"):
            # if user_upload_file is not None:
            # Convert data uploaded by user into dataframe
            user_upload_file_frame = pd.read_csv(user_upload_file, encoding="ISO-8859-1")
            submit_button = st.form_submit_button(
                label="Submit trip information and Get prediction"
            )

        if submit_button:
            # make space between element on stremlit
            # st.write('#')
            st.write('--')

            # get prediction
            userdata_pred_prob = fip_src.prediction(user_upload_frame=user_upload_file_frame)
            csv_file = st_cloud_dep_src.convert_frame_into_csv(input_frame=userdata_pred_prob)

            # make space between element on stremlit
            # st.write('#')
            st.write('--')

    # Download trip imformation and prediction
    if csv_file is not None:
        # button to download csv file if needed
        st.download_button(
            label="Download response",
            data=csv_file,
            file_name="trip_information_and_response.csv",
            mime="text/csv",
        )
