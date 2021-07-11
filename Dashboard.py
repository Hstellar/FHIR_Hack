import pandas as pd
import streamlit as st
import os
import pickle
import shap
import xgboost
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 

def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass
anemia_features = ['DR1DRSTZ','DRD350A','DRD350C','DRD350D','DRD350E','DRD350G','DRD350I','DRD360','DRD370C','DRD370G','DRD370H','DRD370I','DRD370K','DRD370L','DRD370N',
 'DRD370R','DRD370T','DRD370U','Gender','Age','Tot_family_income','Tot_no_fam_members','Cancer','BMI','Educationlevel','Ethinicity','Breast_fed',
 'GlycoHemoglobin']

###read csv

x_test = pd.read_csv('x_test_anemia.csv')
# seed = 195
# test_size = 0.3
# X_train, X_test, y_train, y_test = train_test_split(df_anemia[anemia_features], df_anemia['Anemia'], test_size=test_size, random_state=seed)

##########Streamlit dashboard
desc = "Using trained model we predict anemia for demonstration"

st.title('Predictive analytics on diseases considering diet intake')
st.write(desc)

Gender = st.number_input('Gender', min_value=1, max_value=2, value=1)
Age = st.number_input('Age', )
Weight = st.number_input('Weight')
Tot_family_income = st.number_input('Tot_family_income')
Tot_no_fam_members = st.number_input('Tot_no_fam_members')
Hemoglobin =st.number_input('Hemoglobin', value=8)
Cancer = st.number_input('Cancer', value=2)
BMI = st.number_input('BMI')
Educationlevel = st.number_input('Educationlevel')
Ethinicity = st.number_input('Ethinicity')
Breast_fed = st.number_input('Breast_fed',value=2)
GlycoHemoglobin = st.number_input('GlycoHemoglobin')
DR1DRSTZ = st.number_input('Dietary recall status',value=2)
DRD350A = st.number_input('Clams eaten during past 30 days',value=2)
DRD350C = st.number_input('Crayfish eaten during past 30 days',value=2)
DRD350D = st.number_input('Lobsters eaten during past 30 days',value=2)
DRD350E = st.number_input('Mussels eaten during past 30 days',value=2)
DRD350G = st.number_input('Scallops eaten during past 30 days',value=2)
DRD350I = st.number_input('Other shellfish eaten during past 30 days',value=2)
DRD350J = st.number_input('Other unknown shellfish eaten during past 30 days',value=2)
DRD350K = st.number_input('Refused on shellfish eaten during past 30 days',value=2)
DRD360 = st.number_input('Fish eaten during past 30 days',value=2)
DRD370C = st.number_input('Bass eaten during past 30 days',value=2)
DRD370G = st.number_input('Haddock eaten during past 30 days',value=2)
DRD370H = st.number_input('Mackerel  eaten during past 30 days',value=2)
DRD370I = st.number_input('Perch  eaten during past 30 days',value=2)
DRD370J = st.number_input('Pike  eaten during past 30 days',value=2)
DRD370K = st.number_input('Pollock  eaten during past 30 days',value=2)
DRD370L = st.number_input('Porgy eaten during past 30 days',value=2)
DRD370N = st.number_input('Sardines  eaten during past 30 days',value=2)
DRD370O = st.number_input('sea bass eaten during past 30 days',value=2)
DRD370P = st.number_input('Shark  eaten during past 30 days',value=2)
DRD370Q = st.number_input('Swordfish  eaten during past 30 days',value=2)
DRD370R = st.number_input('Trout eaten during past 30 days',value=2)
DRD370S =st.number_input('Walleye  eaten during past 30 days',value=2)
DRD370T = st.number_input('Other fish eaten during past 30 days',value=2)
DRD370U =st.number_input('Other unknown eaten during past 30 days',value=2)
DRD370V =st.number_input('Refused to eat fish during past 30 days',value=2)

# X_test = pd.DataFrame(data = [DR1DRSTZ,DRD350A,DRD350C,DRD350D,DRD350E,DRD350G,DRD350I,DRD360,DRD370C,DRD370G,DRD370H,DRD370I,DRD370K,DRD370L,DRD370N,
#  DRD370R,DRD370T,DRD370U,Gender,Age,Tot_family_income,Tot_no_fam_members,Cancer,BMI,Educationlevel,Ethinicity,Breast_fed,
#  GlycoHemoglobin],columns = anemia_features)
X_test = pd.DataFrame([[DR1DRSTZ],[DRD350A],[DRD350C],[DRD350D],[DRD350E],[DRD350G],[DRD350I],[DRD360],[DRD370C],[DRD370G],[DRD370H],[DRD370I],[DRD370K],[DRD370L],[DRD370N],
 [DRD370R],[DRD370T],[DRD370U],[Gender],[Age],[Tot_family_income],[Tot_no_fam_members],[Cancer],[BMI],[Educationlevel],[Ethinicity],[Breast_fed],
 [GlycoHemoglobin]]).T
X_test.columns = anemia_features
pkl_filename = "Pickle_anemia_Model.pkl"
with open(pkl_filename, 'rb') as file:
    anemia_model = pickle.load(file)
#anemia_model = joblib.load(pkl_filename)
if st.button('Generate prediction for anemia'):
    predict = anemia_model.predict(X_test)
    if predict==0:
        st.write('No anemia')
    else:
        st.write('Risk of anemia')
## Feature importance of model
shap_values = shap.TreeExplainer(anemia_model).shap_values(x_test)
st.set_option('deprecation.showPyplotGlobalUse', False)
fig, axes = plt.subplots()
shap.summary_plot(shap_values, x_test, max_display=10)
st.pyplot(fig)

###Feature importance of model cholerstrol level
components.html(
    """
<div style="font-family:Helvetica Neue; font-size:20px; line-height:1.7; color:slategray;">
<br>Built model(Random forest) on demographic + diet variable count variables where 41 features where used <br>
<br>Below is the graph for model interpretation which is known as SHAP value plot.<br>
SHAP value refers to the contribution of a feature value to a prediction.<br>
The larger the SHAP value magnitude, the more important the driver is.<br>
</div>
<br>
""",
    height=450,
)
pkl_filename = "Pickle_cholestrol_level.pkl"
with open(pkl_filename, 'rb') as file:
    rf = pickle.load(file)
    
df = pd.read_csv('x_test_chol.csv')
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(df)
st.set_option('deprecation.showPyplotGlobalUse', False)
fig, axes = plt.subplots()
shap.summary_plot(shap_values, df, max_display=10)
st.pyplot(fig)
# fig, axes = plt.subplots()
# shap.force_plot(explainer.expected_value, shap_values[0,:], df[0,:100])
# st.pyplot(fig)
