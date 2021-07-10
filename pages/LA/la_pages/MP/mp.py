import streamlit as st
import pickle
import os


# this is the main function in which we define our webpage


# model = pickle.load(open(
#     r'E:\Deploy_ML\streamlit\project6\pages\LA\la_pages\MP\saved_model\loan.pkl', 'rb'))


def write():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = pickle.load(open((os.path.join(BASE_DIR, "saved_model/loan.pkl")), "rb"))

    # front end elements of the web page
    html_temp = """
    <div style ="background-color:#F63366;padding:15px">
    <h2 style ="text-align:center;"> Loan Approval Prediction</h2>
    </div>
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    col1, col2, col3 = st.beta_columns(3)

    with col1:

        # Full Name
        f_name = st.text_input('Full Name')

        # For gender
        gen_display = ('Male', 'Female')
        gen_options = list(range(len(gen_display)))
        gen = st.selectbox("Gender", gen_options,
                           format_func=lambda x: gen_display[x])

        # For Marital Status
        mar_display = ('No', 'Yes')
        mar_options = list(range(len(mar_display)))
        mar = st.selectbox("Marital Status", mar_options,
                           format_func=lambda x: mar_display[x])

        # No of dependets
        dep_display = ('No', 'One', 'Two', 'More than Two')
        dep_options = list(range(len(dep_display)))
        dep = st.selectbox("Dependents",  dep_options,
                           format_func=lambda x: dep_display[x])
    with col2:

        # For edu
        edu_display = ('Not Graduate', 'Graduate')
        edu_options = list(range(len(edu_display)))
        edu = st.selectbox("Education", edu_options,
                           format_func=lambda x: edu_display[x])

        # For emp status
        emp_display = ('Job', 'Business')
        emp_options = list(range(len(emp_display)))
        emp = st.selectbox("Employment Status", emp_options,
                           format_func=lambda x: emp_display[x])

        # For Property status
        prop_display = ('Rural', 'Semi-Urban', 'Urban')
        prop_options = list(range(len(prop_display)))
        prop = st.selectbox("Property Area", prop_options,
                            format_func=lambda x: prop_display[x])

        # For Credit Score
        cred_display = ('Between 300 to 500', 'Above 500')
        cred_options = list(range(len(cred_display)))
        cred = st.selectbox("Credit Score", cred_options,
                            format_func=lambda x: cred_display[x])
    with col3:

        # Applicant Monthly Income
        mon_income = st.number_input("Applicant's Monthly Income($)", value=0)

        # Co-Applicant Monthly Income
        co_mon_income = st.number_input(
            "Co-Applicant's Monthly Income($)", value=0)

        # Loan AMount
        loan_amt = st.number_input("Loan Amount", value=0)

        # loan duration
        dur_display = ['2 Month', '6 Month', '8 Month', '1 Year', '16 Month']
        dur_options = range(len(dur_display))
        dur = st.selectbox("Loan Duration", dur_options,
                           format_func=lambda x: dur_display[x])

    if st.button("Predict"):
        duration = 0
        if dur == 0:
            duration = 60
        if dur == 1:
            duration = 180
        if dur == 2:
            duration = 240
        if dur == 3:
            duration = 360
        if dur == 4:
            duration = 480
        features = [[gen, mar, dep, edu, emp, mon_income,
                     co_mon_income, loan_amt, duration, cred, prop]]
        print(features)
        prediction = model.predict(features)
        lc = [str(i) for i in prediction]
        ans = int("".join(lc))
        if ans == 0:
            st.error(
                'Sorry ☹️' + f_name.title() + " \n\n You are not Eligible for Loan Approval   "
            )
        else:
            st.success(
                'Congratulations !  🥳 ' + f_name.title() + "  \n\n You are Eligible for Loan Approval "
            )


if __name__ == '__main__':
    write()
