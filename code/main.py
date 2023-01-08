from dataset import DataSet
from sidebar import sidebar
import field
import streamlit as st

st.title("Machine Learning App with Titanic")
dataset = DataSet()
sidebar_info = sidebar(dataset)
field.main_field(dataset, sidebar_info)
