import streamlit as st
from .la_pages.MP import mp as mp
from .la_pages.MA import ma as ma


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
