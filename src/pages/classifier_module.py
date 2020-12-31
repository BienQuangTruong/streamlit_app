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

    st.title(" THUẬT TOÁN PHÂN LỚP (CLASSIFIER) ")

    ###########################################
    uploaded_file = st.file_uploader("Chọn tập dữ liệu CSV từ máy của bạn")
    classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "MLP Classifier", "SVM"))

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
        print(X)
        print(y)
        st.write("Shape of dataset", X.shape)
        st.write("number of classes", len(np.unique(y)))

        # def add_parameter_ui(clf_name):
        #     params = dict()
        #     if clf_name == 'SVM':
        #         C = st.sidebar.slider('C', 0.01, 10.0)
        #         params['C'] = C
        #     elif  clf_name == 'MLP_Classifier':
        #         M = st.sidebar.slider('M', 1, 100)
        #         params['M'] = M
        #     else:
        #         K = st.sidebar.slider('K', 1, 15)
        #         params['K'] = K
        #     return params

        # params = add_parameter_ui(classifier_name)

        # Set up classifier
        def get_classifier(clf_name):
            clf = None
            if clf_name == 'SVM':
                clf = SVC()
            elif clf_name == 'MLP Classifier':
                clf = MLPClassifier()
            else:
                clf = KNeighborsClassifier()
            return clf

        clf = get_classifier(classifier_name)

        #### CLASSIFICATION ####

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.write(f'Classifier = {classifier_name}')
        st.write(f'Accuracy =', acc)

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

        st.pyplot(fig)