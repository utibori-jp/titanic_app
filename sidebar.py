import streamlit as st
import numpy as np

def check_data(dataset):
    st.sidebar.markdown("### Select the variable you wish to check")
    variables = []
    with st.sidebar:
        variables = st.sidebar.multiselect("variables", dataset.column_list)

    max_n_rows = np.arange(4, dataset.train_data.shape[0])
    with st.sidebar:
        n_rows = st.selectbox(
            "Num of rows",
            max_n_rows,
        )
    return variables, n_rows

def select_features(dataset):
    st.sidebar.markdown("### Select features")
    # itemはPair plotを使わないと定義されないと出る問題がある。
    item = st.sidebar.multiselect("items", dataset.column_list_processed)
    hue = "temp"
    if item:
        hue = item[0]
    return hue, item

def choose_algorithm(dataset):
    st.sidebar.markdown("### Choose algorithm for classification")
    with st.sidebar:
        select_algorithm = st.radio(
            "Choose algorithm for classification",
            [
                "SVC",
                "LogisticRegression",
                "RandomForestClassifier",
                "KNeighborsClassifier",
                "DecisionTreeClassifier",
                "XGBClassifier"
            ],
            key = "classification",
        )
    return select_algorithm

def sidebar(dataset):
    variables, n_rows = check_data(dataset)
    hue, item = select_features(dataset)
    algorithm = choose_algorithm(dataset)

    sidebar_info = {
        "variables": variables,
        "n_rows": n_rows,
        "item": item,
        "hue":hue,
        "algorithm":algorithm
    }

    return sidebar_info
