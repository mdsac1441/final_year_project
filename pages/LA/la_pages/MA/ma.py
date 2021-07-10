import streamlit as st
from .ma_pages.EDA import eda as eda
from .ma_pages.MT import mt as mt


PAGES = {
    "Exploratory Data Analysis": eda,
    "Model Trainning": mt,

}


def write():
    selection = st.sidebar.selectbox(
        "", list(PAGES.keys()))
    page = PAGES[selection]

    with st.spinner(f"Loading  {selection} ..."):
        page.write()


if __name__ == "__main__":
    write()
