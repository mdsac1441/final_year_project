import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy
import os
# Data Viz Pkgs
# import matplotlib
# matplotlib.use('Agg')  # To Prevent Errors

st.set_option('deprecation.showPyplotGlobalUse', False)

def write():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # front end elements of the web page
    html_temp = """
    <div style ="background-color:#F63366;padding:15px;margin:15px">
    <h2 style ="text-align:center;">Exploratory Data Analysis For Income Prediction </h2>
    </div> 
     """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv((os.path.join(BASE_DIR, "adult.csv")))
        return data

    df = load_data()

    # Show Dataset
    if st.checkbox("Show DataSet"):
        number = st.number_input("Number of Rows to View", value=0)
        st.dataframe(df.head(number))

    # Show Column Names
    if st.button("Columns Names"):
        st.write(df.columns)

    # Show Shape of Dataset
    if st.checkbox("Shape of Dataset"):
        st.write(df.shape)
        data_dim = st.radio("Show Dimension by", ("Rows", "Columns"))
        if data_dim == 'Rows':
            st.text("Number of  Rows")
            st.write(df.shape[0])
        elif data_dim == 'Columns':
            st.text("Number of Columns")
            st.write(df.shape[1])

    # Show Columns By Selection
    if st.checkbox("Select Columns To Show"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect('Select', all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

    # Datatypes
    if st.button("Data Types"):
        st.write(df.dtypes)

    # Value Counts
    if st.button("Value Counts"):
        st.text("Value Counts By Target/Class")
        st.write(df.iloc[:, -1].value_counts())

    # Summary
    if st.checkbox("Show Summary of Dataset"):
        st.write(df.describe())

    st.sidebar.subheader("Data Visualization")

    # Show Correlation Plots
    # Matplotlib Plot
    if st.sidebar.checkbox("Correlation Plot [Matplotlib]"):
        plt.matshow(df.corr())
        st.pyplot()

    # Seaborn Plot
    if st.sidebar.checkbox("Correlation Plot with Annotation [Seaborn]"):
        st.write(sns.heatmap(df.corr(), annot=True))
        st.pyplot()

    # Counts Plots
    if st.sidebar.checkbox("Plot of Value Counts"):
        st.text("Value Counts By Target/Class")

        all_columns_names = df.columns.tolist()
        primary_col = st.sidebar.selectbox(
            'Select Primary Column To Group By', all_columns_names)
        selected_column_names = st.sidebar.multiselect(
            'Select Columns', all_columns_names)
        if st.sidebar.button("Plot"):
            st.text("Generating Plot for: {} and {}".format(
                primary_col, selected_column_names))
            if selected_column_names:
                vc_plot = df.groupby(primary_col)[
                    selected_column_names].count()
            else:
                vc_plot = df.iloc[:, -1].value_counts()
            st.write(vc_plot.plot(kind='bar'))
            st.pyplot()

    # Pie Plot
    if st.sidebar.checkbox("Pie Plot"):
        all_columns_names = df.columns.tolist()
        # st.info("Please Choose Target Column")
        # int_column =  st.selectbox('Select Int Columns For Pie Plot',all_columns_names)
        if st.sidebar.button("Generate Pie Plot"):
            # cust_values = df[int_column].value_counts()
            # st.write(cust_values.plot.pie(autopct="%1.1f%%"))
            st.write(df.iloc[:, -1].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()

    # # Barh Plot
    # if st.checkbox("BarH Plot"):
    #     all_columns_names = df.columns.tolist()
    #     st.info("Please Choose the X and Y Column")
    #     x_column = st.selectbox(
    #         'Select X Columns For Barh Plot', all_columns_names)
    #     y_column = st.selectbox(
    #         'Select Y Columns For Barh Plot', all_columns_names)
    #     barh_plot = df.plot.barh(x=x_column, y=y_column, figsize=(10, 10))
    #     if st.button("Generate Barh Plot"):
    #         st.write(barh_plot)
    #         st.pyplot()

    # Custom Plots
    st.sidebar.subheader("Customizable Plots")
    all_columns_names = df.columns.tolist()
    type_of_plot = st.sidebar.selectbox("Select the Type of Plot", [
        "area", "bar", "line", "hist", "box", "kde"])
    selected_column_names = st.sidebar.multiselect(
        'Select Columns To Plot', all_columns_names)
    # plot_fig_height = st.number_input("Choose Fig Size For Height",10,50)
    # plot_fig_width = st.number_input("Choose Fig Size For Width",10,50)
    # plot_fig_size =(plot_fig_height,plot_fig_width)
    cust_target = df.iloc[:, -1].name

    if st.sidebar.button("Generate Plot"):
        st.success("Generating A Customizable Plot of: {} for :: {}".format(
            type_of_plot, selected_column_names))
        # Plot By Streamlit
        if type_of_plot == 'area':
            cust_data = df[selected_column_names]
            st.area_chart(cust_data)
        elif type_of_plot == 'bar':
            cust_data = df[selected_column_names]
            st.bar_chart(cust_data)
        elif type_of_plot == 'line':
            cust_data = df[selected_column_names]
            st.line_chart(cust_data)
        elif type_of_plot == 'hist':
            custom_plot = df[selected_column_names].plot(
                kind=type_of_plot, bins=2)
            st.write(custom_plot)
            st.pyplot()
        elif type_of_plot == 'box':
            custom_plot = df[selected_column_names].plot(kind=type_of_plot)
            st.write(custom_plot)
            st.pyplot()
        elif type_of_plot == 'kde':
            custom_plot = df[selected_column_names].plot(kind=type_of_plot)
            st.write(custom_plot)
            st.pyplot()
        else:
            cust_plot = df[selected_column_names].plot(kind=type_of_plot)
            st.write(cust_plot)
            st.pyplot()

    st.info("Check the missing Values in data")

    if st.checkbox("Null Value in Each Columns of Dataset "):
        yo = df.isnull().sum()
        st.write(yo)

    if st.checkbox("Total Null value in Dataset "):
        yo = df.isnull().sum().sum()
        st.write(yo)

    st.info("Our Features and Target")
    if st.checkbox("Show Features"):
        all_features = df.iloc[:, 0:-1]
        st.text('Features Names:: {}'.format(all_features.columns[0:-1]))
        st.dataframe(all_features.head(10))

    if st.checkbox("Show Target"):
        all_target = df.iloc[:, -1]
        st.text('Target/Class Name:: {}'.format(all_target.name))
        st.dataframe(all_target.head(10))


if __name__ == '__main__':
    write()
