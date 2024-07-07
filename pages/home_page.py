# import library and module
import streamlit as st


# for page on
def app():
    st.subheader('titanic disaster survivor prediction app')

    # Display Simple decription of app goal's
    st.text_area(
        label='#',
        value='This application will be continually updated with new features',
        height=100
    )
