import streamlit as st
from .ip_pages.MP import mp as mp
from .ip_pages.MA import ma as ma


PAGES = {
    "Model Prediction": mp,
    "Model Analysis": ma,

}


def write():
    selection = st.sidebar.radio(
        "What you Want ?", list(PAGES.keys()))
    page = PAGES[selection]

    with st.spinner(f"Loading  {selection} ..."):
        page.write()


if __name__ == "__main__":
    write()
