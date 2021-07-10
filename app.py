import streamlit as st
import pages.home
import pages.IP.ip as ip
import pages.LA.la as la
from pages import about, contact
import os
from PIL import Image

PAGES = {
    "Home": pages.home,
    "Loan Approval": la,
    "Income Prediction": ip,
    "About Us": about,
    "Contact Us": contact
}


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image = Image.open((os.path.join(BASE_DIR, "ai.jpg")))
    st.sidebar.image(image, width=300)
    st.sidebar.title("Artificial Intelligence")
    selection = st.sidebar.selectbox("Pages", list(PAGES.keys()))
    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        page.write()

    st.sidebar.title("Contribute")
    st.sidebar.info(
        "This is a College project and you are very welcome to **contribute** your awesome "
        "comments, questions, resources and apps as "
        "[issues](https://github.com/mdsac1441/final_year_project/issues) of or "
        "[pull requests](https://github.com/mdsac1441/final_year_project/pulls) "
        "to the [source code](https://github.com/mdsac1441/final_year_project). "
    )

    st.sidebar.title("About")
    st.sidebar.info(
        """
            This app is maintained by Md Sharique Ahmed and his team. You can know more about me at
            [Linkedin.com](https://www.linkedin.com/in/md-sharique-ahmed1441/).
    """
    )


if __name__ == "__main__":
    main()
