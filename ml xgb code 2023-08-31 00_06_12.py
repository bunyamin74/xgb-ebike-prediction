# Databricks notebook source
pip install xgboost

# COMMAND ----------

pip install cufflinks

# COMMAND ----------

pip install catboost

# COMMAND ----------

pip install optuna

# COMMAND ----------

pip install plotly

# COMMAND ----------

pip install missingno

# COMMAND ----------

pip install joblib

# COMMAND ----------

pip install pickle

# COMMAND ----------

# MAGIC %run "/Users/benjaminisac74@gmail.com/kullanışlı fonksiyonlar ve ebike/functions_lib.py"

# COMMAND ----------

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# # model selection & validation
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, cross_val_predict, train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_validate

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

## Pandas need cufflinks to link with plotly and add the iplot method:
## plotly and cufflinks
import plotly 
import plotly.express as px
import cufflinks as cf #cufflink connects plotly with pandas to create graphs and charts of dataframes directly
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import iplot
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

## regression/prediction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor, plot_importance

import optuna

# COMMAND ----------

file_location = "/FileStore/tables/01_df_eBike_neu_UVP_januar_2023_clean__3_.csv"
file_type = "csv"

# CSV options
infer_schema = "True"
first_row_is_header = "True"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
          .option("inferSchema", infer_schema) \
          .option("header", first_row_is_header) \
          .option("sep", delimiter) \
          .load(file_location).toPandas()

# COMMAND ----------

#datasetine genel bir bakış ve ttekrarlı değerlerin tekilleştirilmesi
first_looking(df)
duplicate_values(df)

# COMMAND ----------

file_location = "/FileStore/tables/01_df_eBike_neu_UVP_januar_2023_clean__3_.csv"
file_type = "csv"

# CSV options
infer_schema = "True"
first_row_is_header = "True"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
          .option("inferSchema", infer_schema) \
          .option("header", first_row_is_header) \
          .option("sep", delimiter) \
          .load(file_location).toPandas()

first_looking(df)
duplicate_values(df)

df.drop(columns=["dämpfer", "kassette"], inplace=True)

target = "uvp-€"

X = df[['kategorie',
        'rahmenmaterial',
        'gänge',
        'gabel_federweg-mm',
        'akkukapazität-wh',
        'hersteller']] # .drop(target, axis=1)


y = df[target]

X_train, X_test, y_train, y_test  = train_test_split(X, y, 
                                                     test_size=0.2, 
                                                     random_state=42, 
                                                     shuffle=True)

shape_control(df, X_train, y_train, X_test, y_test)

numerics = X.select_dtypes(include="number").astype("float64")
categorics = X.select_dtypes(include=["object", "category", "bool"])

numeric_transformer = Pipeline([('Scaler', StandardScaler()), 
                                ('R_Scaler', RobustScaler()),
                                ('Pw_Scaler', PowerTransformer())])

categorical_transformer = Pipeline([('OHE', OneHotEncoder(handle_unknown="ignore"))])

transformer = ColumnTransformer([('numeric', numeric_transformer, numerics.columns),
                                 ('categoric', categorical_transformer, categorics.columns)])

pipeline_model_xgb = Pipeline([('transform', transformer), ('prediction', XGBRegressor())])    

pipeline_model_xgb.fit(X_train, y_train)

y_pred = pipeline_model_xgb.predict(X_test)

def eval_metric(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"r2 score: {r2.round(2)}")
    return r2, mae, mse, rmse

r2, mae, mse, rmse = eval_metric(y_test, y_pred)
print("****************************************************************************************************")
# print(f"Inputs: {df[['gabel_federweg_mm', 'lenkerbreite_mm', 'akkukapazität_wh', 'unterstützung_%']].columns.tolist()}")
print(f"Target: {target}")
print("****************************************************************************************************")
print(f"{pipeline_model_xgb[-1]} >> r2: {r2}, MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
print("****************************************************************************************************")


######################################
transformed_features_Xtrain = transformer.fit_transform(X_train)
transformed_features_Xtest = transformer.transform(X_test)

# Get the column names of the transformed features
transformed_feature_names_Xtrain = numerics.columns.tolist() + transformer.transformers_[1][1]['OHE'].get_feature_names(categorics.columns).tolist()

# Convert the sparse matrix to a dense matrix
transformed_features_dense_Xtrain = transformed_features_Xtrain.toarray()

# Create a DataFrame with the transformed features and their names
transformed_df_Xtrain = pd.DataFrame(transformed_features_dense_Xtrain, columns=transformed_feature_names_Xtrain)

# Display the transformed DataFrame
print(transformed_df_Xtrain.head(2))


# Get the column names of the transformed features
transformed_feature_names_Xtest = numerics.columns.tolist() + transformer.transformers_[1][1]['OHE'].get_feature_names(categorics.columns).tolist()

# Convert the sparse matrix to a dense matrix
transformed_features_dense_Xtest = transformed_features_Xtest.toarray()

# Create a DataFrame with the transformed features and their names
transformed_df_Xtest = pd.DataFrame(transformed_features_dense_Xtest, columns=transformed_feature_names_Xtest)

# Display the transformed DataFrame
print(transformed_df_Xtest.head(2))
######################################

import matplotlib.pyplot as plt

# Assuming your pipeline has a RandomForestRegressor as the final step
xgb_regressor = pipeline_model_xgb.named_steps['prediction']

# Get feature importances from the trained RandomForestRegressor
feature_importances = xgb_regressor.feature_importances_

# Create a DataFrame to associate feature names with their importances
feature_importance_df = pd.DataFrame({
    'Feature': transformed_df_Xtrain.columns,
    'Importance': feature_importances
})

# Sort the features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


 

# COMMAND ----------

list(feature_importance_df['Feature'][0:12])

# COMMAND ----------

def objective(trial):
    params = {"n_estimators": trial.suggest_int("n_estimators",10, 1000, 10),
              "max_depth": trial.suggest_int("max_depth", 2, 16),
              "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
              "subsample": trial.suggest_float("subsample", 0.2, 1),
              "num_leaves": trial.suggest_int("num_leaves", 10, 200, 10),
              "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, 100),
              "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
              "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
              "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
              "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.1),
              "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
              "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1)
             }

    numeric_transformer = Pipeline([('Scaler', StandardScaler()), 
                                    ('R_Scaler', RobustScaler()),
                                    ('Pw_Scaler', PowerTransformer())])
    
    categorical_transformer = Pipeline([('OHE', OneHotEncoder(handle_unknown="ignore"))])
    
    transformer = ColumnTransformer([('numeric', numeric_transformer, numerics.columns),
                                     ('categoric', categorical_transformer, categorics.columns)])
    
    pipeline_model = Pipeline([('transform', transformer),
                               ('prediction', XGBRegressor(**params, random_state=42))])
  
    pipeline_model.fit(X_train, y_train)
    y_pred = pipeline_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective, n_trials=20)

study = study_xgb
print()
print(f'R2: {study.best_value}')
print(f'Best parameter: {study.best_params}')

# COMMAND ----------

XBG_best_parameters = []
study = study_xgb
print("R2      :", round(study.best_value, 3), "Best Parameters:", study.best_params)
XBG_best_parameters .append(study.best_params)

# COMMAND ----------

study.best_params

# COMMAND ----------

study = study_xgb

tuned_model_xgb = Pipeline([('transform', transformer),
                        ('prediction', XGBRegressor(**study.best_params))])

tuned_model_xgb.fit(X_train, y_train)
y_pred = tuned_model_xgb.predict(X_test)
r2 = r2_score(y_test, y_pred)
r2

# COMMAND ----------

import joblib
joblib.dump(tuned_model_xgb, open('final_model_xgb', 'wb'))

# COMMAND ----------

X_test.iloc[[2]]

# COMMAND ----------

y_test.iloc[[2]]

# COMMAND ----------

rf_deploy =joblib.load(open('final_model_xgb', 'rb'))
y_pred = rf_deploy.predict(X_test.iloc[[2]])
y_pred

# COMMAND ----------

pip install streamlit

# COMMAND ----------

import streamlit as st
import joblib
import pandas as pd


st.sidebar.title('E-Bike Price Prediction')
html_temp = """
<div style="background-color:orange;padding:10px">
<h2 style="color:white;text-align:center;">Coding Book ML Solutions</h2>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)


gabel_federweg_mm=st.sidebar.selectbox("Please provide the travel length of the fork in millimeters.",( 30,
                                                                                                        35,
                                                                                                        50,
                                                                                                        60,
                                                                                                        63,
                                                                                                        65,
                                                                                                        70,
                                                                                                        75,
                                                                                                        80,
                                                                                                        100,
                                                                                                        110,
                                                                                                        120,
                                                                                                        130,
                                                                                                        140,
                                                                                                        150,
                                                                                                        160,
                                                                                                        170,
                                                                                                        180,
                                                                                                        200))
akkukapazität_wh=st.sidebar.selectbox("What is the watt-hour capacity of your electric bike's battery?", (  248,
                                                                                                            250,
                                                                                                            252,
                                                                                                            300,
                                                                                                            320,
                                                                                                            324,
                                                                                                            360,
                                                                                                            396,
                                                                                                            400,
                                                                                                            410,
                                                                                                            416,
                                                                                                            418,
                                                                                                            420,
                                                                                                            430,
                                                                                                            446,
                                                                                                            460,
                                                                                                            474,
                                                                                                            500,
                                                                                                            504,
                                                                                                            508,
                                                                                                            520,
                                                                                                            522,
                                                                                                            530,
                                                                                                            540,
                                                                                                            545,
                                                                                                            555,
                                                                                                            558,
                                                                                                            562,
                                                                                                            600,
                                                                                                            601,
                                                                                                            603,
                                                                                                            604,
                                                                                                            612,
                                                                                                            621,
                                                                                                            625,
                                                                                                            630,
                                                                                                            650,
                                                                                                            670,
                                                                                                            691,
                                                                                                            700,
                                                                                                            710,
                                                                                                            720,
                                                                                                            750,
                                                                                                            850))
rahmenmaterial=st.sidebar.radio("What material was used for the frame of your bicycle (e.g., aluminum, carbon, steel)?",('Aluminium',
                                                                                                                         'Carbon',
                                                                                                                         'Aluminium-Carbon',
                                                                                                                         'Diamant',
                                                                                                                         'Aluminium-Stahl'))
#sattel=st.sidebar.selectbox("What type of saddle do you prefer? (e.g., mountain bike saddle, road bike saddle, city bike saddle)", ('Selle Bassano Feel GT'))
gänge=st.sidebar.radio("Please specify the number of gears on your bicycle.",(0,3,5,7,8,9,10,11,12,14,20,22,24,27,30))
#bremse_vorne=st.sidebar.selectbox("Which brand of braking system located at the front of your bicycle (e.g., MAGURA HS-11 , Shimano MT-200).", ('Shimano MT-200'))
#schaltwerk=st.sidebar.selectbox("Which rear derailleur is installed on your bicycle? (e.g., Shimano Deore, SRAM GX)", ('Shimano ', 'Shimano Deore))
kategorie=st.sidebar.selectbox("To which category does your bicycle belong? (e.g., mountain bike, road bike, electric bike)", ('Trekking', 
                                                                                                                               'City',
                                                                                                                               'MTB_Hardtail',
                                                                                                                               'MTB_Fully'))
hersteller=st.sidebar.selectbox("Who is the manufacturer of your bicycle?", (   'Kalkhoff',
                                                                                'CUBE',
                                                                                'Haibike',
                                                                                'Hercules',
                                                                                'Winora',
                                                                                'SCOTT',
                                                                                'corratec',
                                                                                'Diamant',
                                                                                'GHOST',
                                                                                'Specialized',
                                                                                'Cannondale',
                                                                                'Canyon'))

rf_model=joblib.load('final_model_xgb')


my_dict = {
    "gabel_federweg_mm": gabel_federweg_mm,
    "akkukapazität_wh": akkukapazität_wh,
    "rahmenmaterial": rahmenmaterial,
    'gänge': gänge,
    "kategorie": kategorie,
    "hersteller": hersteller    
}

df = pd.DataFrame.from_dict([my_dict])


st.header("The configuration of your e-bike is below")
st.table(df)


st.subheader("Press predict if configuration is okay")

if st.button("Predict"):
    prediction = rf_deploy.predict(df)
    st.success("The estimated price of your e-bike is €{}. ".format(int(prediction[0])))


# COMMAND ----------


