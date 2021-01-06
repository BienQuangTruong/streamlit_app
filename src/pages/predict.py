import pickle
import base64
from sklearn.svm import SVR
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def write():

    st.title(" DỰ ĐOÁN (PREDICT) ")

    ###########################################
    uploaded_file = st.file_uploader("Chọn tập dữ liệu CSV từ máy của bạn")

    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

        return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, low_memory=False)

        # Normalize
        sc = StandardScaler()
        x_test = sc.fit_transform(df)

        #load model
        clf = pickle.load(open('classifier_models.pkl', 'rb'))

        pred_clf = clf.predict(x_test)

        df['tasty'] = pred_clf

        # st.write("Sau khi classifier: ",df)

        # Normalize
        sc1 = StandardScaler()
        reg_test = sc1.fit_transform(df)

        #load model
        reg = pickle.load(open('svr_model.pkl', 'rb'))

        y_pred_regressor = reg.predict(reg_test)

        df['quality'] = y_pred_regressor

        st.write('Đã dự đoán xong !!!', df)

        if st.button('Download Dataframe as CSV'):
            tmp_download_link = download_link(df, 'predict.csv', 'Click here to download your data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)