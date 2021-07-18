import pandas as pd
import streamlit as st
import os
import pickle
import shap
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
desc = "Using trained model we predict risk of anemia and cardiovascular disease and for demonstration"

st.title('Predictive analytics on diseases considering diet intake')
st.write(desc)

Gender = st.number_input('Gender(1- Male, 2-Female)', min_value=1, max_value=2, value=1)
Age = st.number_input('Age' )
Weight = st.number_input('Weight')
Tot_family_income = st.number_input('Tot_family_income')
Tot_no_fam_members = st.number_input('Tot_no_fam_members')
Hemoglobin =st.number_input('Hemoglobin', value=8)
Cancer = st.number_input('Cancer', value=2)
BMI = st.number_input('BMI')
BMXWAIST = st.number_input('Waist Circumference')
CDQ008 = st.number_input("Severe pain in chest more than half hour(1-yes, 2-No, 9 - Don't know )")
Educationlevel = st.number_input('Educationlevel')
Ethinicity = st.number_input('Ethinicity')
Breast_fed = st.number_input("Breast_fed(1- yes, 2-no 9-Don't know)",value=2)
GlycoHemoglobin = st.number_input('GlycoHemoglobin')
MCQ160D = st.number_input("Ever told you had angina/angina pectoris(1- yes, 2-no 9-Don't know)", value=1)
SMQ020 = st.number_input("Smoked at least 100 cigarettes in life(1- yes, 2-no 9-Don't know)", value=1)
DIQ010 = st.number_input("Doctor told you have diabetes(1-yes, 2-No, 3-Borderline, 7-Refused, 9-Don't know)")
DR1LANG = st.number_input('Language(1- English, 2 - Spanish, 3-English and Spanish, 4-other, 5- Asian LAnguages, 6-Asian Languages & English)')
DR1HELPD = st.number_input('Helped in responding for this interview')
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
DRD370A = st.number_input('Breaded fish product eaten during past 30 days',value=2)
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
DRD370F = st.number_input('flatfish eaten during past 30 days',value=2)
DRDINT = st.number_input('Indicates whether the sample person has intake data for one or two days.',value=2)
DR1MNRSP = st.number_input('Who was the main respondent for this interview?(1-individual, 2-mother, 3-father, 4-wife, 5-husband, 6-daughter, 7-son, 8-Grandparent, 9-Friend,partner, non relative, 10-Translator, 11-child care, 12-other relative)', value=2)
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
        
X_test = pd.DataFrame([[Age],[Gender],[DR1DRSTZ],[Breast_fed],[DRDINT],[DR1LANG], [DR1MNRSP], [DR1HELPD], [DRD350C], [DRD350E],
                       [DRD350G], [DRD350I], [DRD350J], [DRD360], [DRD370A], [DRD370C], [DRD370F], [DRD370G], [DRD370H], [DRD370I],
                      [DRD370J],[DRD370K], [DRD370L], [DRD370O], [DRD370P], [DRD370Q], [DRD370R], [DRD370S], [BMI], [BMXWAIST],
                      [CDQ008], [MCQ160D], [SMQ020], [DIQ010]]).T
X_test.columns = ['RIDAGEYR', 'RIAGENDR', 'DR1DRSTZ', 'DRABF', 'DRDINT', 'DR1LANG',
       'DR1MNRSP', 'DR1HELPD', 'DRD350C', 'DRD350E', 'DRD350G', 'DRD350I',
       'DRD350J', 'DRD360', 'DRD370A', 'DRD370C', 'DRD370F', 'DRD370G',
       'DRD370H', 'DRD370I', 'DRD370J', 'DRD370K', 'DRD370L', 'DRD370O',
       'DRD370P', 'DRD370Q', 'DRD370R', 'DRD370S', 'BMXBMI',
       'BMXWAIST', 'CDQ008', 'MCQ160D', 'SMQ020', 'DIQ010']
categorical_columns = ['Gender', 'Dietary recall status', 'angina', 'Smoked', 'diabetes']
numeric_columns = ['Age',
 'Breast-fed infant',
 'days of intake',
 'Language',
 'DR1MNRSP',
 'Helped in responding for this interview',
 'Crayfish',
 'Mussels',
 'Scallops',
 'shellfish',
 'Other unknown shellfish',
 'Fish',
 'Breaded fish products',
 'Basss',
 'Flatfish',
 'Haddock',
 'Mackerel',
 'Perch',
 'Pike',
 'Pollock',
 'Porgy',
 'Sea bass',
 'Shark',
 'Swordfish',
 'Trout',
 'Walleye',
 'Body Mass Index',
 'Waist Circumference',
 'Severe pain in chest']
X_test = X_test.rename(columns = {
'RIDAGEYR' : 'Age',
'RIAGENDR' : 'Gender',
'DR1DRSTZ' : 'Dietary recall status',
'DRABF'    : 'Breast-fed infant',
'DRDINT'   : 'days of intake',
'DR1LANG'  : 'Language',
'DR1HELPD' : 'Helped in responding for this interview',
'DRD350C'  : 'Crayfish',
'DRD350E'  : 'Mussels',
'DRD350G'  : 'Scallops',
'DRD350I'  : 'shellfish',
'DRD350J'  : 'Other unknown shellfish',
'DRD360'   : 'Fish',
'DRD370A'  : 'Breaded fish products',
'DRD370C'  : 'Basss',
'DRD370F'  : 'Flatfish',
'DRD370G'  : 'Haddock',
'DRD370H'  : 'Mackerel',
'DRD370I'  : 'Perch',
'DRD370J'  : 'Pike',
'DRD370K'  : 'Pollock',
'DRD370L'  : 'Porgy',
'DRD370O'  : 'Sea bass',
'DRD370P'  : 'Shark',
'DRD370Q'  : 'Swordfish',
'DRD370R'  : 'Trout',
'DRD370S'  : 'Walleye',
'BMXBMI'   : 'Body Mass Index',
'BMXWAIST' : 'Waist Circumference',
'CDQ008'   : 'Severe pain in chest',
'MCQ160D'  : 'angina',
'SMQ020'   : 'Smoked',
'DIQ010'   : 'diabetes',
})
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', dtype=np.int))
])

## Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])
# pkl_filename = "encoder.pickle"
# with open(pkl_filename, 'rb') as file:
#     preprocessor = pickle.load(file)
x_test = preprocessor.fit_transform(X_test) 
feature_names = list(preprocessor.named_transformers_['cat'].named_steps['onehot'] \
                            .get_feature_names(input_features=categorical_columns))
feature_names = feature_names + numeric_columns

pkl_filename = "Pickle_cholestrol_level.pkl"
with open(pkl_filename, 'rb') as file:
    rf = pickle.load(file)
if st.button('Generate prediction for Cholestrol'):
    predict = rf.predict(x_test)
    if predict==0:
        st.write('No Risk')
    else:
        st.write('Risk of Cardiovascular disease')
# ## Feature importance of model
# shap_values = shap.TreeExplainer(anemia_model).shap_values(x_test)
# st.set_option('deprecation.showPyplotGlobalUse', False)
# fig, axes = plt.subplots()
# shap.summary_plot(shap_values, x_test, max_display=10)
# st.pyplot(fig)

# ###Feature importance of model cholerstrol level
# components.html(
#     """
# <div style="font-family:Helvetica Neue; font-size:20px; line-height:1.7; color:slategray;">
# <br>Built model(Random forest) on demographic + diet variable count variables where 41 features where used <br>
# <br>Below is the graph for model interpretation which is known as SHAP value plot.<br>
# SHAP value refers to the contribution of a feature value to a prediction.<br>
# The larger the SHAP value magnitude, the more important the driver is.<br>
# </div>
# <br>
# """,
#     height=450,
# )
# pkl_filename = "Pickle_cholestrol_level.pkl"
# with open(pkl_filename, 'rb') as file:
#     rf = pickle.load(file)
    
# df = pd.read_csv('x_test_chol.csv')
# explainer = shap.TreeExplainer(rf)
# shap_values = explainer.shap_values(df)
# st.set_option('deprecation.showPyplotGlobalUse', False)
# fig, axes = plt.subplots()
# shap.summary_plot(shap_values, df, max_display=10)
# st.pyplot(fig)
