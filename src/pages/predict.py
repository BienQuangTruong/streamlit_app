import pickle
import base64
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def write():

    st.title(" DỰ ĐOÁN (PREDICT) ")

    ###########################################
    uploaded_model = st.file_uploader("Chọn model từ máy của bạn")
    uploaded_file = st.file_uploader("Chọn tập dữ liệu CSV từ máy của bạn")

    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

        return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    if st.button('Run predict'):
        if uploaded_file is not None and uploaded_model is not None:
            df = pd.read_csv(uploaded_file, low_memory=False)
            model = pickle.load(uploaded_model)
            
            # Normalize
            sc = StandardScaler()
            x_test = sc.fit_transform(df)

            #load model
            # clf = pickle.load(open('classifier_models.pkl', 'rb'))

            pred_clf = model.predict(x_test)

            bad = []
            good = []
            for i in pred_clf:
                if i == 0:
                    bad.append(i)
                else:
                    good.append(i)

            df['tasty'] = ['Bad' if x == 0 else 'Good' for x in pred_clf]

            st.write('Đã đánh giá tự động xong !!!', df)
            st.write('Bad: ', len(bad))
            st.write('Good: ', len(good))
            


            if st.button('Download Dataframe as CSV'):
                tmp_download_link = download_link(df, 'predict.csv', 'Click here to download your data!')
                st.markdown(tmp_download_link, unsafe_allow_html=True)