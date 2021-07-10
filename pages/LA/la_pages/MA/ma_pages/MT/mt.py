import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

import os


def write():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # front end elements of the web page
    html_temp = """
    <div style ="background-color:#F63366;padding:15px;margin:15px">
    <h2 style ="text-align:center;">Model Training For Loan Approval </h2>
    </div> 
     """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv((os.path.join(BASE_DIR, "bankloan.csv")))
        data = data.drop('Loan_ID', axis=1)
        # Handle Missing Values
        data.Credit_History.fillna(np.random.randint(0, 2), inplace=True)
        data.Married.fillna(np.random.randint(0, 2), inplace=True)
        data.LoanAmount.fillna(data.LoanAmount.median(), inplace=True)
        data.Loan_Amount_Term.fillna(
            data.Loan_Amount_Term.mean(), inplace=True)
        data.Gender.fillna(np.random.randint(0, 2), inplace=True)
        data.Dependents.fillna(data.Dependents.median(), inplace=True)
        data.Self_Employed.fillna(np.random.randint(0, 2), inplace=True)

        return data

    @st.cache(persist=True)
    def split(df):
        pre_x = df.iloc[:, :-1]
        pre_y = df.iloc[0:, -1]
        # Handle Label Data
        x = pd.get_dummies(pre_x)
        y = pre_y.map(dict(Y=1, N=0))
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=10)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test,
                                  display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    df = load_data()
    class_names = ["Approved", 'Rejected']

    x_train, x_test, y_train, y_test = split(df)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",
                                      ("XGBClassifier", "Support Vector Machine (SVM)", "Logistic Regression", "Random Forest", "GradientBoostingClassifier",
                                       "K-Nearest Neighbor", "Decision Tree", "Neural Network"))

    if classifier == 'XGBClassifier':
        st.sidebar.subheader("Hyperparameters Tuning")
        learning_rate = st.sidebar.number_input("learning_rate", 0.05, 0.3, step=.05,
                                                key='learning_rate')
        min_child_weigh = st.sidebar.number_input("min_child_weigh", 1, 7, step=1,
                                                  key='min_child_weigh')

        gamma = st.sidebar.number_input("gamma", 0.0, 0.4, step=0.1,
                                        key='gamma')
        colsample_bytree = st.sidebar.number_input("colsample_bytree", 0.3, 0.7, step=0.1,
                                                   key='colsample_bytree')

        n_estimators = st.sidebar.number_input("n_estimators", 100, 5000, step=10,
                                               key='n_estimators')
        max_depth = st.sidebar.number_input(
            "The maximum depth of the tree", 3, 10, step=1, key='max_depth')

        metrics = st.multiselect("What metrics to plot?",
                                 ('Confusion Matrix', 'ROC Curve',
                                  'Precision-Recall Curve'))

        if st.button("Classify", key='classify'):
            st.success("XGBClassifier Results")
            model = XGBClassifier(learning_rate=learning_rate, min_child_weigh=min_child_weigh, gamma=gamma,
                                  n_estimators=n_estimators, max_depth=max_depth, colsample_bytree=colsample_bytree)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Hyperparameters Tuning")
        # choose parameters
        C = st.sidebar.number_input("C (Regularization parameter)",
                                    0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio(
            "Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.multiselect("What metrics to plot?",
                                 ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.button("Classify", key='classify'):
            st.success("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Hyperparameters Tuning")
        C = st.sidebar.number_input(
            "C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider(
            "Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.multiselect("What metrics to plot?",
                                 ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.button("Classify", key='classify'):
            st.success("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Hyperparameters Tuning")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10,
                                               key='n_estimators')
        max_depth = st.sidebar.number_input(
            "The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio(
            "Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        metrics = st.multiselect("What metrics to plot?",
                                 ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.button("Classify", key='classify'):
            st.success("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap,
                                           n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'GradientBoostingClassifier':
        st.sidebar.subheader("Hyperparameters Tuning")
        n_estimators = st.sidebar.number_input("n_estimators", 100, 5000, step=10,
                                               key='n_estimators')
        max_depth = st.sidebar.number_input(
            "The maximum depth of the tree", 3, 15, step=1, key='max_depth')

        metrics = st.multiselect("What metrics to plot?",
                                 ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.button("Classify", key='classify'):
            st.success("GradientBoostingClassifier Results")
            model = GradientBoostingClassifier(
                n_estimators=n_estimators, max_depth=max_depth)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'K-Nearest Neighbor':
        st.sidebar.subheader("Hyperparameters Tuning")
        criterion = st.sidebar.selectbox("criterion", ("gini", "entropy"))
        n_neighbors = st.sidebar.number_input("Number of neighbors, K", 1, 428, step=1,
                                              key='n_neighbors')
        metrics = st.multiselect("What metrics to plot?",
                                 ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.button("Classify", key='classify'):
            st.success("K-Nearest Neighbor Results")
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Decision Tree':
        st.sidebar.subheader("Hyperparameters Tuning")
        max_depth = st.sidebar.number_input(
            "The maximum depth of the tree", 2, 16, step=2, key='max_depth')
        max_leaf_nodes = st.sidebar.number_input("Maximum leaf node", 2, 20, step=1,
                                                 key='max_leaf_nodes')
        metrics = st.multiselect("What metrics to plot?",
                                 ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.button("Classify", key='classify'):
            st.success("Decision Tree Results")
            model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Neural Network':
        st.sidebar.subheader("Hyperparameters Tuning")
        solver = st.sidebar.radio(
            "Solver", ("lbfgs", "sgd", "adam"), key='solver')
        alpha = st.sidebar.number_input(
            "Regularization parameter", 0.000001, 10.0000, key='alpha')
        metrics = st.multiselect("What metrics to plot?",
                                 ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.button("Classify", key='classify'):
            st.success("Neural Network Results")
            scaler = StandardScaler()
            scaler.fit(x_train)
            X_train = scaler.transform(x_train)
            X_test = scaler.transform(x_test)
            model = MLPClassifier(solver=solver, alpha=alpha,
                                  hidden_layer_sizes=(5, 2), random_state=1)
            model.fit(X_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Adult Income Data Set (Classification)")
        st.write(df)


if __name__ == '__main__':
    write()
