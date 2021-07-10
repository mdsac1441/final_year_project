import streamlit as st
from PIL import Image
import os


def write():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image = Image.open((os.path.join(BASE_DIR, "contact.jpg")))
    st.image(image, width=700)

    st.markdown("<h3 style='text-align: center; color: #F63366;'>We’d love to hear from you.</h1>",
                unsafe_allow_html=True)
    st.write(
        """
        We’re located in India because there’s nowhere else like it.
        Being around this many vibrant humans is our source of constant inspiration, what’s yours?


    """
    )

    col1, col2, col3 = st.beta_columns(3)

    with col1:
        st.info(":e-mail: ahmed@gmail.com")

    with col2:
        st.info("📞 102287911")

    with col3:
        st.info("📍  Greater Noida,India")
