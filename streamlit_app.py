# import library and module
import sys
from pathlib import Path

import streamlit as st

from pages import home_page, manual_input_page, file_uploadind_page


# create multi app pages
# function to run app page
def run_multi_app_apge():
    # manage dynimically many page
    # define all pages
    p1 = st.Page(
        "./pages/home_page.py",
        title="home",
        icon=":material/home:",
    )
    p2 = st.Page(
        "./pages/manual_input_page.py",
        title="manual",
        icon=":material/query_stats:",
    )
    p3 = st.Page(
        "./pages/file_uploadind_page.py",
        title="file",
        icon=":material/query_stats:",
    )

    # Install Multipage
    pg = st.navigation(
        {
            "General": [p1],
            "Upload data": [p2, p3],
        }
    )

    # run only manual input app page
    if pg == p1:
        home_page.app()
    # run only manual input app page
    if pg == p2:
        manual_input_page.app()
    # run only file uploading app page
    if pg == p3:
        file_uploadind_page.app()

    # Run pages
    pg.run()


# run the main app page
if __name__ == "__main__":
    run_multi_app_apge()
