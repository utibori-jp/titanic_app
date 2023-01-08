import streamlit as st
import seaborn as sns
from models import get_predict_score

def display_data(dataset, variables, n_rows):
    st.markdown("## Display train data")
    if not variables:
        st.write("No variable is selected.")
    else:
        st.write(dataset.train_data[variables].head(n_rows))
        st.write(dataset.train_data[variables].describe())

def visualize_features(dataset, item, hue):
    st.markdown("## Visualize features")
    execute_pairplot = st.button("Pair plot drawing")
    if execute_pairplot:
        if item:
            df_sns = dataset.train_data_for_ML[item]
            df_sns["hue"] = dataset.train_data_for_ML[hue]
            fig = sns.pairplot(df_sns, hue = "hue")
            st.pyplot(fig)
        else:
            st.write("no variable is choosen.")

def prediction(dataset, algorithm):
    st.markdown("## Prediction")
    execute = st.button("emplement")
    score_dict = get_predict_score(
        dataset.train_data_for_ML,
        dataset.test_data_for_ML,
        dataset.y_test,
        algorithm)

    if execute:
        st.markdown("### Result")
        # st.write(score_dict)
        st.write("train score:" + str(score_dict["train_score"]))
        st.write("test score:" + str(score_dict["test_score"]))

def main_field(dataset, sidebar_info):
    variables = sidebar_info["variables"]
    n_rows = sidebar_info["n_rows"]
    item = sidebar_info["item"]
    hue = sidebar_info["hue"]
    algorithm = sidebar_info["algorithm"]

    display_data(dataset, variables, n_rows)
    visualize_features(dataset, item, hue)
    prediction(dataset, algorithm)
