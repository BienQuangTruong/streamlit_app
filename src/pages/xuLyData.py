import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def write():
    st.title(' PHÂN TÍCH DỮ LIỆU ')

    uploaded_file = st.file_uploader("Chọn tập dữ liệu CSV từ máy của bạn")

    @st.cache
    def load_data():
        data = pd.read_csv(uploaded_file, low_memory=False)
        return data

    st.set_option('deprecation.showPyplotGlobalUse', False)
    def raw_data(data):
        name_columns = list(data.columns)
        for item in name_columns:
            datas = data[item]
            plt.title("Histogram")
            plt.xlabel(item)
            plt.ylabel("Frequency")
            plt.hist(datas,10)
            st.pyplot()
    
    if uploaded_file is not None:
        data_load_state = st.text('Loading data...')
        data = load_data()
        data_load_state.text("Done! (using st.cache)")
        raw_data(data)