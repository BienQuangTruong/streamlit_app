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
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn import neighbors
import pickle

def write():

    st.title(" THUẬT TOÁN HỒI QUY (REGRESSION) ")

    ###########################################
    regressor_name = st.sidebar.selectbox("Select Regressor", ("PLS", "SVR", "MLP Regressor", "KNN"))
    uploaded_file = st.file_uploader("Chọn tập dữ liệu CSV từ máy của bạn")

    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        # uploaded_file.seek(0)
        dataframe = pd.read_csv(uploaded_file, low_memory=False)
        z = np.abs(stats.zscore(dataframe))

        data = dataframe[(z < 3).all(axis=1)]
        data['tasty'] = [0 if x < 6 else 1 for x in data['quality']]

        df = data[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","tasty","quality"]]

        st.write(df)

        # st.write(dataframe)
        y = df['quality'].values
        X = df.values[:, 0:-1]
        dataset_name = {'data': (X), 'target': (y)}

        def get_dataset(dataset_name):
            data = dataset_name
            X = data['data']
            y = data['target']
            return X, y

        X, y = get_dataset(dataset_name)
        st.write("Shape of dataset", X.shape)
        st.write("number of classes", len(np.unique(y)))

        # Set up regression
        def get_regression(reg_name):
            reg = None
            if reg_name == 'MLP Regressor':
                reg = MLPRegressor()
            elif reg_name == 'PLS':
                reg = PLSRegression()
            elif reg_name == 'KNN':
                reg = neighbors.KNeighborsRegressor()
            else:
                reg = SVR()
            return reg

        reg = get_regression(regressor_name)

        #### REGRESSION ####

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        print(x_test)
        # Normalize
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        print(x_test)
        reg.fit(x_train, y_train)

        # save the model to disk
        filename = 'svr_model.pkl'
        pickle.dump(reg, open(filename, 'wb'))

        # x_tests = x_test[:, :-1]

        dfs = pd.DataFrame(x_test)
        dfs.to_csv('test.csv', index = False)

        y_pred_regressor = reg.predict(x_test)

        r2 = r2_score(y_test, y_pred_regressor)
        mse = mean_squared_error(y_test, y_pred_regressor)

        st.write(f'Regressior = {regressor_name}')
        st.write(f'R2 =', r2)
        st.write(f'MSE =', mse)

        # from sklearn.model_selection import cross_val_score
        # accuracies = cross_val_score(estimator = reg, X = x_train, y = y_train, cv = 20)
        # st.write("Accuracy: {:.2f} %".format(accuracies.mean()*100))
        # st.write("Accuracy: {:.2f} %".format(accuracies.std()*100))

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