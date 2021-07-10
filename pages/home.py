"""Home page shown when the user enters the application"""
import streamlit as st
from PIL import Image
import os


def write():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image = Image.open((os.path.join(BASE_DIR, "college.png")))
    st.image(image, width=700)

    st.markdown("<h3 style='text-align: center; color: #F63366;'>Machine Learning Based Web Application</h3>",
                unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color: ;'>Under the Guidance of</h2>",
                unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #F63366;'>Ms. Neha Kashyap</h4>",
                unsafe_allow_html=True)
    """Used to write the page in the app.py file"""
    st.header("**Train and Predict Various Real-World Problems**")
    with st.spinner("Loading Home ..."):
        st.write(
            """
This application provides
- A List of Various  AI Problem  for Real-Time **Applications**.
- Model Prediction.
- Exploratory Data Analysis.
- Model Trainning.



    """
        )


if __name__ == "__main__":
    write()
