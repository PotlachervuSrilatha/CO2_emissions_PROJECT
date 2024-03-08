#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries-----------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load the vehicle dataset
df = pd.read_csv("co2_emissions.csv")

# Drop rows with natural gas as fuel type
fuel_type_mapping = {"Z": "Premium Gasoline","X": "Regular Gasoline","D": "Diesel","E": "Ethanol(E85)","N": "Natural Gas"}
df["fuel_type"] = df["fuel_type"].map(fuel_type_mapping)
df_natural = df[~df["fuel_type"].str.contains("Natural Gas")].reset_index(drop=True)

# Remove outliers from the data
df_new = df_natural[['engine_size', 'cylinders', 'fuel_consumption_comb(l/100km)', 'co2_emissions', 'make', 'model']]
df_new_model = df_new[(np.abs(stats.zscore(df_new.drop(columns=['make', 'model']))) < 1.9).all(axis=1)]

# Encode 'make' and 'model' columns using LabelEncoder
le_make = LabelEncoder()
le_model = LabelEncoder()
df_new_model['make_encoded'] = le_make.fit_transform(df_new_model['make'])
df_new_model['model_encoded'] = le_model.fit_transform(df_new_model['model'])

# Model-----------------------------------------------------------------------------------------------------
st.title('CO2 Emission Prediction')
st.write('Enter the vehicle specifications to predict CO2 emissions.')

# Create a dropdown with unique values of 'make'
unique_makes = df_new_model['make'].unique()
selected_make = st.selectbox('make', unique_makes)

# Filter the dataframe based on selected 'make' for model multiselect
filtered_df = df_new_model[df_new_model['make'] == selected_make]

# Create a multiselect with unique values of 'model' for the selected 'make'
unique_models = filtered_df['model'].unique()
selected_models = st.multiselect('model', unique_models)

# Filter the dataframe based on selected 'model(s)'
if selected_models:
    filtered_df = filtered_df[filtered_df['model'].isin(selected_models)]

# Prepare the data for modeling
X = filtered_df.drop(columns=['co2_emissions', 'make', 'model'])
X['model_encoded'] = le_model.transform(filtered_df['model'])  # Include 'model_encoded' in the input data
y = filtered_df['co2_emissions']

# Train the random forest regression model
model = RandomForestRegressor().fit(X, y)

# Input fields for user
engine_size = st.number_input('engine_size', step=0.1, format="%.1f")
cylinders = st.number_input('cylinders', min_value=2, max_value=16, step=1)
fuel_consumption = st.number_input('fuel_consumption_comb(l/100km)', step=0.1, format="%.1f")

# Encode 'make' and 'model' inputs using LabelEncoder
make_encoded = le_make.transform([selected_make])[0]
models_encoded = le_model.transform(selected_models) if selected_models else le_model.transform(unique_models)

# Create a copy of X for prediction with input data
input_data = X.copy()
input_data.loc[:, 'engine_size'] = engine_size
input_data.loc[:, 'cylinders'] = cylinders
input_data.loc[:, 'fuel_consumption_comb(l/100km)'] = fuel_consumption
input_data.loc[:, 'make_encoded'] = make_encoded
input_data.loc[:, 'model_encoded'] = models_encoded[0]  # Assign the first value as default if no model is selected

# Predict CO2 emissions
predicted_co2 = model.predict(input_data)

# Display the prediction
st.write(f'Predicted CO2 Emissions: {predicted_co2[0]:.2f} g/km')







# In[ ]:




