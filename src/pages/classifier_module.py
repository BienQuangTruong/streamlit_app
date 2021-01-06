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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import pickle

def write():

    st.title(" THUẬT TOÁN PHÂN LỚP (CLASSIFIER) ")

    ###########################################
    uploaded_file = st.file_uploader("Chọn tập dữ liệu CSV từ máy của bạn")
    classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "MLP Classifier", "SVM"))

    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        # uploaded_file.seek(0)
        wine = pd.read_csv(uploaded_file, low_memory=False)
    
        #z-core
        z = np.abs(stats.zscore(wine))

        wine = wine[(z < 3).all(axis=1)]
        # Thực hiện phân loại nhị phân cho biến phản hồi.
        # Phân chia rượu ngon và dở bằng cách đưa ra giới hạn chất lượng
        bins = (2, 6, 8)
        group_names = ['bad', 'good']
        wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)

        # Bây giờ, hãy gán nhãn cho biến chất lượng của chúng tôi
        label_quality = LabelEncoder()

        #Bad becomes 0 and good becomes 1 
        wine['quality'] = label_quality.fit_transform(wine['quality'])
        st.write(wine)

        y = wine['quality'].values
        X = wine.values[:, 0:-1]
        dataset_name = {'data': (X), 'target': (y)}
            
        def get_dataset(dataset_name):
            data = dataset_name
            X = data['data']
            y = data['target']
            return X, y

        X, y = get_dataset(dataset_name)
        st.write("Shape of dataset", X.shape)
        st.write("number of classes", len(np.unique(y)))

        # Set up classifier
        def get_classifier(clf_name):
            clf = None
            if clf_name == 'SVM':
                clf = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')
            elif clf_name == 'MLP Classifier':
                clf = MLPClassifier()
            else:
                clf = KNeighborsClassifier()
            return clf

        clf = get_classifier(classifier_name)

        #### CLASSIFICATION ####

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Normalize
        sc = StandardScaler()

        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.write(f'Classifier = {classifier_name}')
        st.write(f'Accuracy =', acc)

        # save the model to disk
        filename = 'classifier_models.pkl'
        pickle.dump(clf, open(filename, 'wb'))


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