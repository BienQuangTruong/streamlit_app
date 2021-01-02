import base64
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def write():
    st.title('TIỀN XỬ LÝ DỮ LIỆU')

    uploaded_file = st.file_uploader("Chọn tập dữ liệu CSV từ máy của bạn")
    names = st.sidebar.selectbox("Chọn tiền xử lý dữ liệu", ("PCA", "Normalize"))

    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

        return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    @st.cache
    def load_data():
        datas = pd.read_csv(uploaded_file, low_memory=False)
        return datas

    st.set_option('deprecation.showPyplotGlobalUse', False)
    def tienXuLy(datas):
        name_columns = list(datas.columns)
        X = datas.reindex(columns = name_columns[:-1])
        y = datas.reindex(columns = name_columns[-1:])
        X = StandardScaler().fit_transform(X)
        if names == "PCA":
            K = st.sidebar.slider('K', 1, 10)
            pca = PCA(n_components=K)
            pca.fit(X)
            principalComponents = pca.fit_transform(X)
            cum_explained_var = []
            for i in range(0, len(pca.explained_variance_ratio_)):
                if i == 0:
                    cum_explained_var.append(pca.explained_variance_ratio_[i])
                else:
                    cum_explained_var.append(pca.explained_variance_ratio_[i] + cum_explained_var[i-1])
            df_cumulative = pd.DataFrame(cum_explained_var, columns=['Cumulative (%)'])
            st.write("Cumulative",df_cumulative)
            # st.write("Cumulative",cum_explained_var)
            principalDf = pd.DataFrame(data = principalComponents)
            finalDf = pd.concat([principalDf, y], axis = 1)
            st.write("Data is PCA",finalDf)
            if st.button('Download Dataframe as CSV'):
                tmp_download_link = download_link(finalDf, 'PCA.csv', 'Click here to download your data!')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
        elif names == "Normalize":
            min_max_scaler = MinMaxScaler()
            data_scaler = min_max_scaler.fit_transform(X)
            df = pd.DataFrame(data_scaler, columns=name_columns[:-1])
            finalDf = pd.concat([df, y], axis = 1)
            st.write("Data is Normalize",finalDf)
            if st.button('Download Dataframe as CSV'):
                tmp_download_link = download_link(finalDf, 'NORMALIZE.csv', 'Click here to download your data!')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
            
    if uploaded_file is not None:
        datas = load_data()
        tienXuLy(datas)