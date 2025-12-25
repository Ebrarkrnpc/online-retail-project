# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 14:39:14 2025

@author: Dell
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


st.set_page_config(page_title="Online Retail Dashboard", layout="wide")

@st.cache_data
def load_data():
 
 url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    
 
 df = pd.read_excel(url, engine='openpyxl')
 df.dropna(inplace=True)
 df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
 df['Revenue'] = df['Quantity'] * df['UnitPrice'] 
 return df



st.title("Programming for Data Science - Final Project")
st.write("Online Retail Dataset Analysis Dashboard")


try:
    df = load_data()
except FileNotFoundError:
    st.error("Error loading data.Please check your connection or file path.")
    st.stop()

st.header("1.Dataset Overview")
col1, col2 = st.columns(2)

with col1:
    st.subheader("First 10 Rows (Interactive)")
    st.dataframe(df.head(10))
    
with col2:
    st.subheader("Structural Information")
    st.write(f"**Number of Observations:** {df.shape[0]}")
    st.write(f"**Number of Variables:** {df.shape[1]}")
    st.write("**Data Types:**")
    st.write(df.dtypes.astype(str))

st.divider()



st.sidebar.title("Dashboard Controls")
cat_columns = df.select_dtypes(include=['object']).columns.tolist()
selected_cat = st.sidebar.selectbox("Select Categorical Variable for Pie Chart", cat_columns)


st.header("2. Categorical Distribution Analysis")
fig_pie = px.pie(df.head(1000), names=selected_cat, title=f"Distribution of {selected_cat}(Top 1000 records)")
st.plotly_chart(fig_pie)


st.header("3. Top 10 Countries by Transactions")
top_countries = df['Country'].value_counts().head(10)
st.bar_chart(top_countries) 


st.header("4.Numeric Data Visualizations ")
col3, col4 = st.columns(2)

with col3:
    st.subheader("Quantity vs UnitPrice (Scatter Plot)")
    
    fig_scatter = px.scatter(df.sample(500), x="Quantity", y="UnitPrice", color="Country")
    st.plotly_chart(fig_scatter)
    
    
with col4:
    st.subheader("Revenue Distribution (Histogram)")
    fig_hist = px.histogram(df[df['Revenue'] < 200], x="Revenue", nbins=50, title="Revenue Distribution (Filtered)")
    st.plotly_chart(fig_hist)
    
    
st.header("5. Transactions Over Time")

df_time = df.set_index('InvoiceDate').resample('M').size().reset_index(name='Transaction Count')
fig_line = px.line(df_time, x='InvoiceDate', y='Transaction Count', title="Monthly Total Transactions")
st.plotly_chart(fig_line)

st.divider()


st.header("6. PCA (Principal Component Analysis)")

features = ['Quantity', 'UnitPrice', 'Revenue']
x = df[features].dropna()
x_scaled = StandardScaler().fit_transform(x)


# 2 Bileşenli PCA
pca = PCA(n_components=2)
components = pca.fit_transform(x_scaled)
pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
pca_df['Country'] = df['Country'].values


top_5_countries = df['Country'].value_counts().head(5).index
pca_df['Color_Group'] = pca_df['Country'].apply(lambda x: x if x in top_5_countries else 'Other')


fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Color_Group', title="2D PCA Scatter Plot")
st.plotly_chart(fig_pca)




st.header("7. Machine Learning: Revenue Prediction")

if st.button("Train Model (Random Forest)"): 
    with st.spinner("Training model, please wait..."): 
       
        X_ml = df[['Quantity', 'UnitPrice']]
        y_ml = df['Revenue']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_ml, y_ml)
        
        score = model.score(X_ml, y_ml)
        st.success(f"Model Trained Successfully! R2 Score: {score:.4f}") 
        
      
        importances = pd.Series(model.feature_importances_, index=['Quantity', 'UnitPrice'])
        st.write("Feature Importance Levels:")
        st.bar_chart(importances)


with st.expander("Project Notes"):
    st.write("This application was developed as a final project for the Programming for Data Science course.")






# Yan Panel Başlığı
st.sidebar.title("Analysis Settings")


categorical_columns = ["Country", "StockCode"]
selected_category = st.sidebar.selectbox("Please select a category:", categorical_columns)

st.write(f"### Analysis Based on {selected_category}")


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    

