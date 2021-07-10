import streamlit as st
import pickle
import numpy as np
import os


# @st.cache(suppress_st_warning=True)
# def income_predict(input_data):
#     to_predict = np.array(input_data).reshape(1, 12)
#     # loading the trained model
#     model = pickle.load(open(
#         r"E:\Deploy_ML\streamlit\project6\pages\IP\ip_pages\MP\saved_model\income.pickle", "rb"))

#     pred = model.predict(to_predict)
#     return pred[0]

# model = pickle.load(open(
#     r"E:\Deploy_ML\streamlit\project6\pages\IP\ip_pages\MP\saved_model\income.pickle", "rb"))


def write():
        
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = pickle.load(open((os.path.join(BASE_DIR, "saved_model/income.pickle")), "rb"))
    # front end elements of the web page
    html_temp = """
    <div style ="background-color:#F63366;padding:15px">
    <h2 style ="text-align:center;">Income Prediction Form</h2>
    </div>
     """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    # following lines create boxes in which user can enter data required to make prediction

    col1, col2, col3 = st.beta_columns(3)

    with col1:

        # Age
        Age = st.number_input("Age ", value=0)

        # Working Class
        w_class_display = ("Federal-gov", "Local-gov", "Never-worked",
                           "Private", "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay")
        w_class_options = list(range(len(w_class_display)))
        Working_Class = st.selectbox(
            'Working Class', w_class_options, format_func=lambda x: w_class_display[x])

        # Education
        Edu_display = ("10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th",
                       "9th", "Assoc-acdm", "Assoc-voc", "Bachelors", "Doctorate", "HS-grad", "Masters", "Preschool", "Prof-school", "16 - Some-college")
        Edu_options = list(range(len(Edu_display)))
        Education = st.selectbox('Education', Edu_options,
                                 format_func=lambda x: Edu_display[x])

        # Marital Status
        ms_display = ("married", "non married", "divorced")
        ms_options = list(range(len(ms_display)))
        Marital_Status = st.selectbox(
            'Marital Status', ms_options, format_func=lambda x: ms_display[x])

    with col2:

        # Occupation
        op_display = ("Tech-support", "Sales", "Farming-fishing")
        op_options = list(range(len(op_display)))
        Occupation = st.selectbox('Occupation', op_options,
                                  format_func=lambda x: op_display[x])

        # Relationship
        rel_display = ("Husband", "Wife", "Own-child",
                       "Not-in-family", "Other-relative", "Unmarried")
        rel_options = list(range(len(rel_display)))
        Relationship = st.selectbox(
            'Relationship', rel_options, format_func=lambda x: rel_display[x])

        # Race
        race_display = ("Amer Indian Eskimo", "Black",
                        "White", "Asian Pac Islander", "Other")
        race_options = list(range(len(race_display)))
        Race = st.selectbox('Race', race_options,
                            format_func=lambda x: race_display[x])

        # Gender
        g_display = ("Male", "Female")
        g_options = list(range(len(g_display)))
        Gender = st.selectbox('Gender', g_options,
                              format_func=lambda x: g_display[x])

    with col3:

        #  Capital Gain
        Capital_Gain = st.number_input(
            " Capital Gain  btw:[0-99999] ", min_value=0, max_value=99999,  value=0)

        #  Capital Loss
        Capital_Loss = st.number_input(
            " Capital Loss  btw:[0-4356]", min_value=0, max_value=4356, value=0)

        # Hours per Week
        Hours_per_Week = st.number_input(
            "Hours per Week  btw:[1-99]", min_value=1, max_value=4356, value=1)

        # Country
        country_display = ("India", "United States", "Japan", "China",
                           "Greece", "Germany", "France", "England", "Cuba", "Canada", "Iran")
        country_options = list(range(len(country_display)))
        Native_Country = st.selectbox(
            'Native Country', country_options, format_func=lambda x: country_display[x])

    input_data = [Age, Working_Class, Education, Marital_Status,
                  Occupation, Relationship, Race, Gender, Capital_Gain, Capital_Loss, Hours_per_Week, Native_Country]
    print(input_data)
    input_data = list(map(int, input_data))
    to_predict = np.array(input_data).reshape(1, 12)
    print(to_predict)
    # result = income_predict(input_data)

    prediction = model.predict(to_predict)
    lc = [str(i) for i in prediction]
    result = int("".join(lc))

    if st.button("Predict"):
        if int(result) == 1:
            st.success("Income more than 50K")
        else:
            st.success("Income less than 50K")


if __name__ == "__main__":
    write()
