import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Set page configuration
st.set_page_config(page_title="Climate-Resilient Agriculture", layout="wide")

# Mock dataset for demonstration
data = {
    "Region": ["Region A", "Region B", "Region C", "Region D", "Region E"],
    "Temperature (°C)": [24, 28, 15, 18, 32],
    "Rainfall (mm)": [1200, 800, 1500, 1100, 700],
    "Soil Type": ["Loamy", "Sandy", "Clay", "Silty", "Loamy"],
    "Crop": ["Rice", "Millets", "Wheat", "Barley", "Maize"],
    "Farming Method": ["Flood Irrigation", "Dryland Farming", "Terrace Farming", "Crop Rotation", "Drip Irrigation"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Title and description
st.title("Optimizing Climate-Resilient Agricultural Practices Using AI")
st.markdown(
    """
    With the increasing impact of climate change, traditional agricultural practices are becoming less sustainable and prone to failure. 
    This AI-driven system predicts the most sustainable crops and farming method for your region.
    """
)

# Sidebar inputs
st.sidebar.header("Input Parameters")
temperature = st.sidebar.slider("Average Temperature (°C)", 10, 40, 25)
rainfall = st.sidebar.slider("Annual Rainfall (mm)", 500, 2000, 1200)
soil_type = st.sidebar.selectbox("Soil Type", df["Soil Type"].unique())

# Machine Learning Model
st.header("AI-Driven Crop and Farming Method Prediction")
st.write("This system uses a machine learning model to recommend the most suitable crop and farming method for your region.")

# Prepare data for machine learning
df_encoded = df.copy()
df_encoded["Soil Type"] = df_encoded["Soil Type"].astype("category")  # Explicitly convert to category type
soil_type_mapping = dict(enumerate(df_encoded["Soil Type"].cat.categories))

X = df_encoded[["Temperature (°C)", "Rainfall (mm)", "Soil Type"]]
X["Soil Type"] = X["Soil Type"].cat.codes  # Encode categorical data
y = df_encoded[["Crop", "Farming Method"]]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
crop_model = RandomForestClassifier(random_state=42)
method_model = RandomForestClassifier(random_state=42)
crop_model.fit(X_train, y_train["Crop"])
method_model.fit(X_train, y_train["Farming Method"])

# Predict based on user input
user_input = pd.DataFrame(
    [[temperature, rainfall, list(soil_type_mapping.keys())[list(soil_type_mapping.values()).index(soil_type)]]],
    columns=["Temperature (°C)", "Rainfall (mm)", "Soil Type"]
)
crop_prediction = crop_model.predict(user_input)
method_prediction = method_model.predict(user_input)

st.subheader(f"Recommended Crop: **{crop_prediction[0]}**")
st.subheader(f"Recommended Farming Method: **{method_prediction[0]}**")

# Model performance
st.subheader("Model Performance")
crop_y_pred = crop_model.predict(X_test)
method_y_pred = method_model.predict(X_test)

st.text("Crop Prediction Performance:")
st.text(classification_report(y_test["Crop"], crop_y_pred))

st.text("Farming Method Prediction Performance:")
st.text(classification_report(y_test["Farming Method"], method_y_pred))

# Data Visualization
st.subheader("Crop and Farming Method Recommendations")
fig = px.bar(df, x="Region", y="Rainfall (mm)", color="Crop", barmode="group", title="Rainfall and Crop Recommendations")
st.plotly_chart(fig, use_container_width=True)

# Dataset preview
st.markdown("---")
st.subheader("Dataset Preview")
st.dataframe(df)

# Feature importance visualization
st.subheader("Feature Importance for Crop Prediction")
crop_importance = pd.Series(crop_model.feature_importances_, index=X.columns)
fig_crop = px.bar(crop_importance, x=crop_importance.index, y=crop_importance.values, title="Feature Importance for Crops", labels={"y": "Importance", "x": "Feature"})
st.plotly_chart(fig_crop, use_container_width=True)

st.subheader("Feature Importance for Farming Method Prediction")
method_importance = pd.Series(method_model.feature_importances_, index=X.columns)
fig_method = px.bar(method_importance, x=method_importance.index, y=method_importance.values, title="Feature Importance for Farming Methods", labels={"y": "Importance", "x": "Feature"})
st.plotly_chart(fig_method, use_container_width=True)

# Additional insights
st.markdown("---")
st.subheader("About Climate-Resilient Agriculture")
st.markdown(
    """
    Climate-resilient agriculture focuses on adapting farming practices to changing environmental conditions. 
    This involves selecting crops and techniques that can thrive in local conditions while promoting sustainability and food security.
    """
)

# Contact information
st.markdown("---")
st.write("Developed by: Your Name")
st.markdown(
    '[![GitHub](https://img.shields.io/badge/GitHub-Profile-black)](https://github.com/yourusername) '
    '[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/yourusername/)'
)
