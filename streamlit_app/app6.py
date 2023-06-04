import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.preprocessing import *
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris, load_diabetes
import pickle
import base64
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Machine Learning App', layout='wide', initial_sidebar_state='auto')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.header('Machine Learning App')
st.markdown("""
In this implementation, the **Random Forest Classifier** algorithm is used to build the model.
""")

# Model building
def build_model(df):
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    
    st.markdown("***1.2. Data Splitting***")
    st.write("Dataset Shape")
    st.info(X.shape)
    
    st.markdown("***1.3 Variable Details***")
    st.write("X variable")
    st.info(list(X.columns))
    st.write("Y variable")
    st.info(Y.name)
    
    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, random_state=seed)
    rf = RandomForestClassifier(n_estimators=parameter_n_estimators, max_depth=parameter_max_depth, 
    random_state=seed,
    criterion=parameter_criterion,
    min_samples_split=parameter_min_samples_split,
    min_samples_leaf=parameter_min_samples_leaf,
    bootstrap=parameter_bootstrap,
    oob_score=parameter_oob_score,
    n_jobs=parameter_n_jobs,)
    rf.fit(X_train, Y_train)
    
    st.subheader('2. Model Performance')
    
    st.markdown("***2.1. Training Set***")
    Y_pred_train = rf.predict(X_train)
    st.write("Confusion Matrix")
    cmp = ConfusionMatrixDisplay(confusion_matrix(Y_train, Y_pred_train), display_labels=rf.classes_)
    fig, ax = plt.subplots(figsize=(5,5))
    cmp.plot(ax=ax)
    st.pyplot(fig)
    st.write("Accuracy Score")
    st.info(round(accuracy_score(Y_train, Y_pred_train),5))
    st.write("R2 Score")
    st.info(round(r2_score(Y_train, Y_pred_train), 5))
    st.write("Explained Variance Score")
    st.info(round(explained_variance_score(Y_train, Y_pred_train), 5))
    st.write("Mean squared error")
    st.info(round(mean_squared_error(Y_train, Y_pred_train),5))
    
    # Download trained model
    st.markdown("***2.2. Download Trained Model***")
    st.write("You can download the trained model in pickle format by clicking the button below.")
    def download_model(model):
        output_model = pickle.dumps(model)
        b64 = base64.b64encode(output_model).decode()
        href = f'<a href="data:file/pkl;base64,{b64}" download="model.pkl">Download Trained Model .pkl File</a>'
        return href
    st.markdown(download_model(rf), unsafe_allow_html=True)
        

# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)""")
    
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    
with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_depth = st.sidebar.slider('Max depth (max_depth)', 1, 100, 10, 1)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)
    
with st.sidebar.subheader('2.2. General Parameters'):
    seed = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['gini', 'entropy'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the generalization accuracy (odiabetesob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])
    
    
# Main panel
st.subheader('1. Dataset')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("A **preview** of the dataset is shown below.")
    st.write(df.head(5))
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # iris dataset
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        Y = pd.Series(iris.target, name='response')
        df = pd.concat([X, Y], axis=1)
        st.markdown("The **iris** dataset is used as the example.")
        st.write(df.head(5))
        build_model(df)
        

    