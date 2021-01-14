import streamlit as st 
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
import os
from io import StringIO
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from scipy import stats
from sklearn.model_selection import StratifiedKFold
import pickle
import base64

def write():

    st.title(" THUẬT TOÁN PHÂN LỚP (CLASSIFIER) ")

    ###########################################
    uploaded_file = st.file_uploader("Chọn tập dữ liệu CSV từ máy của bạn")
    classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "MLP Classifier", "SVM", "RandomForest"))

    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        # uploaded_file.seek(0)
        wine = pd.read_csv(uploaded_file, low_memory=False)
    
        #z-core
        z = np.abs(stats.zscore(wine))

        wine = wine[(z < 3).all(axis=1)]
        wine['quality'] = [0 if x < 7 else 1 for x in wine['quality']]
        # wine['quality'] = [0 if x < 5 else 2 if x > 6 else 1 for x in wine['quality']]
        
        st.write(wine)

        # y = wine['quality'].values
        # X = wine.values[:, 0:-1]

        X = wine.drop(['quality'], axis = 1)
        y = wine['quality']

        #Normalize
        X = StandardScaler().fit_transform(X)
        # dataset_name = {'data': (X), 'target': (y)}
            
        # def get_dataset(dataset_name):
        #     data = dataset_name
        #     X = data['data']
        #     y = data['target']
        #     return X, y

        # X, y = get_dataset(dataset_name)
        st.write("Shape of dataset", X.shape)
        # strtifiedKFold
        skf = StratifiedKFold(n_splits=5)
        # Set up classifier
        def get_classifier(clf_name):
            if clf_name == 'SVM':
                svc = SVC()
                grid_params = {'C': [0.1, 0.3, 1, 3, 10], 'gamma': [0.1, 0.3, 1, 3, 10], 'kernel': ['rbf', 'sigmoid']}
                svm_gs = GridSearchCV(estimator=svc, param_grid=grid_params, scoring='accuracy', cv=skf)
                return svm_gs
            elif clf_name == 'MLP Classifier':
                mlp = MLPClassifier()
                grid_params = {
                    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
                    'activation': ['tanh', 'relu'],
                    'solver': ['sgd', 'adam'],
                    'alpha': [0.0001, 0.05],
                    'learning_rate': ['constant','adaptive'],
                }
                # from sklearn.model_selection import GridSearchCV
                mlp_gs = GridSearchCV(mlp, param_grid=grid_params, n_jobs=-1, cv=skf)
                return mlp_gs
            elif clf_name == 'RandomForest':
                rfc = RandomForestClassifier(random_state=2018, oob_score=True)
                grid_params = {"n_estimators": [50, 100, 150, 200, 250],
                                'min_samples_leaf': [1, 2, 4]}
                # grid_params = {"bootstrap":[True, False], "max_depth": list(range(2,10,1)), "min_samples_leaf": list(range(5,20,1))}
                rfc_gs = GridSearchCV(rfc, param_grid=grid_params, scoring='accuracy', cv=skf)
                
                return rfc_gs
            else:
                knn = KNeighborsClassifier()
                grid_params = { 'n_neighbors' : [5,7,9,11,13,15],
                                'weights' : ['uniform','distance'],
                                'metric' : ['minkowski','euclidean','manhattan']}
                knn_gs = GridSearchCV(knn, param_grid=grid_params, scoring='accuracy', cv=skf)
                return knn_gs
            return

        clf = get_classifier(classifier_name)

        #### CLASSIFICATION ####

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf.fit(X_train, y_train)
        st.write('Best Score: ', clf.best_score_)
        
        st.write('Best Params: ', clf.best_params_)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        report = classification_report(y_test,y_pred, output_dict=True)

        df_report = pd.DataFrame(report).transpose()

        st.write(f'Classifier = {classifier_name}')

        st.write(df_report)

        st.write(f'The {classifier_name} model accuracy on Test data is ', acc)

        clf_confusion = confusion_matrix(y_test,  y_pred)
        st.write("Confusion matrix: ", clf_confusion)

        # if st.button('Save Model'):
        #     # save the model to disk
        #     filename = 'classifier_models.pkl'
        #     pickle.dump(clf, open(filename, 'wb'))
        #     st.write('Saved !!')

        def download_model(model):
            output_model = pickle.dumps(model)
            b64 = base64.b64encode(output_model).decode()
            href = f'<a href="data:file/output_model;base64,{b64}" download="myfile.pkl">Download Trained Model .pkl File</a>'
            st.markdown(href, unsafe_allow_html=True)

        if st.button('Download Trained Model'):
            tmp_download_link = download_model(clf, '')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

        # #### PLOT DATASET ####
        # # Project the data onto the 2 primary principal components
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