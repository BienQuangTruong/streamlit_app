import streamlit as st 
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cross_decomposition import PLSRegression
import os
from io import StringIO

def write():

    st.title(" THUẬT TOÁN HỒI QUY (REGRESSION) ")

    ###########################################
    regressor_name = st.sidebar.selectbox("Select Regressor", ("PLS", "SVR", "MLP Regressor"))
    uploaded_file = st.file_uploader("Chọn tập dữ liệu CSV từ máy của bạn")

    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        # uploaded_file.seek(0)
        dataframe = pd.read_csv(uploaded_file, low_memory=False)
        st.write(dataframe)
        y = dataframe['quality'].values
        X = dataframe.values[:, 0:-1]
        dataset_name = {'data': (X), 'target': (y)}

        def get_dataset(dataset_name):
            data = dataset_name
            X = data['data']
            y = data['target']
            return X, y

        X, y = get_dataset(dataset_name)
        st.write("Shape of dataset", X.shape)
        st.write("number of classes", len(np.unique(y)))

        # def add_parameter_ui_reg(reg_name):
        #     params = dict()
        #     if reg_name == 'MLP_Regressor':
        #         M = st.sidebar.slider('M', 1, 100)
        #         params['M'] = M
        #     elif reg_name == 'SVR':
        #         C = st.sidebar.slider('C', 0.01, 10.0)
        #         params['C'] = C
        #     return params
        # params_regression = add_parameter_ui_reg(regressor_name)

        # Set up regression
        def get_regression(reg_name):
            reg = None
            if reg_name == 'MLP Regressor':
                reg = MLPRegressor()
            elif reg_name == 'PLS':
                reg = PLSRegression()
            else:
                reg = SVR()
            return reg

        reg = get_regression(regressor_name)

        #### REGRESSION ####

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

        reg.fit(X_train, y_train)
        y_pred_regressor = reg.predict(X_test)

        r2 = r2_score(y_test, y_pred_regressor)
        mse = mean_squared_error(y_test, y_pred_regressor)

        st.write(f'Regressior = {regressor_name}')
        st.write(f'R2 =', r2)
        st.write(f'MSE =', mse)

        #### PLOT DATASET ####
        # Project the data onto the 2 primary principal components
        pca = PCA(2)
        X_projected = pca.fit_transform(X)

        x1 = X_projected[:, 0]
        x2 = X_projected[:, 1]

        fig = plt.figure()
        plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar()

        #plt.show()
        st.pyplot(fig)