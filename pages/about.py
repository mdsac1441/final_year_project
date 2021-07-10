import streamlit as st
from PIL import Image
import os


def write():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image = Image.open((os.path.join(BASE_DIR, "about.jpg")))
    st.image(image, width=700)

    st.markdown("<h2 style='text-align: center; color: #F63366;'>Our Core Values.</h2>",
                unsafe_allow_html=True)
    st.write(
        """
         Our founding principles are at the core of our culture and guide every decision we make.
    """
    )

    st.markdown("<h2 style='text-align: center; color: #F63366;'>Teamwork.</h2>",
                unsafe_allow_html=True)
    st.write(
        """
         Collaborate. Exceed expectations. Build relationships as partners.
            Help every individual reach their full potential.
    """
    )

    st.markdown("<h2 style='text-align: center; color:#F63366;'>Our Team Leads</h2>",
                unsafe_allow_html=True)

    st.write(
        """
        Meet Our Exceptionally Talented Team Of Lazy Natives.\n\n

    """
    )

    col1, col2, col3, col4 = st.beta_columns(4)

    with col1:
        image = Image.open((os.path.join(BASE_DIR, "huzaib.jpeg")))
        st.image(image, width=100)

        st.text("- Huzaib Umar\n-Founder & Programmer")

    with col2:
        image = Image.open((os.path.join(BASE_DIR, "danish.jpg")))
        st.image(image, width=120)

        st.text("- Danish Faraz\n-CEO &Project Manager")

    with col3:
        image = Image.open((os.path.join(BASE_DIR, "aman.jpeg")))
        st.image(image, width=100)

        st.text("- Aman Kumar Sharma\n-Frontend Developer")

    with col4:
        image = Image.open((os.path.join(BASE_DIR, "ahmed.jpg")))
        st.image(image, width=120)

        st.text("- Md Sharique Ahmed\n-Gareeb Engineer")

    st.markdown("<h2 style='text-align: center; color: #F63366;'>Proactive.</h2>",
                unsafe_allow_html=True)
    st.write(
        """
        
       Share success. Advocate for clients. Take initiative. Innovate.
    """
    )

    st.markdown("<h2 style='text-align: center; color: #F63366;'>Ownership.</h2>",
                unsafe_allow_html=True)
    st.write(
        """
        Take interest. Earn respect. Analyze actions to improve.
      Deliver quality. Take pride in our work.
    """
    )
