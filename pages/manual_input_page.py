# import library and module
import sys
from pathlib import Path

import streamlit as st

from sources.manual_input_page.sources import mip_sources as mip_src
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

    # Display Simple decription of app goal's
    st.text_area(
        label='#',
        value=final_text,
        height=100
    )

    # make space between element on stremlit
    # st.write('#')
    st.write('--')

    # App sidebar to enter manually features
    st.sidebar.header("Enter a trip information")

    # enter trip information
    user_manual_input_frame = st_cloud_dep_src.user_manual_input()

    # Convert data recorded manually by user into frame
    if user_manual_input_frame is not None:
        with st.form(key="form"):
            submit_button = st.form_submit_button(
                label="Submit trip information and Get prediction"
            )

        if submit_button:
            # get prediction
            mip_src.prediction(user_manual_frame=user_manual_input_frame)

    else:
        st.write("Please enter a trip information before get prediction.")
