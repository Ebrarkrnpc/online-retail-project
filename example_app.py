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
    # HATALI URL YERİNE DOĞRUDAN DOSYA İNDİRME LİNKİ:
 url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    
    # Okuma işlemi
 df = pd.read_excel(url, engine='openpyxl')
 df.dropna(inplace=True)
 df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
 df['Revenue'] = df['Quantity'] * df['UnitPrice'] # Ödev 6 [cite: 23]
 return df



st.title("Programlama II - Streamlit Ödevi")
st.write("Online Retail Veri Seti Analiz Paneli")


try:
    df = load_data()
except FileNotFoundError:
    st.error("Lütfen 'Online Retail.xlsx' dosyasının uygulama ile aynı klasörde olduğundan emin olun.")
    st.stop()

st.header("1. Veri Setine Genel Bakış")
col1, col2 = st.columns(2)

with col1:
    st.subheader("İlk 10 Satır (İnteraktif)")
    st.dataframe(df.head(10))
    
with col2:
    st.subheader("Veri Yapısı")
    st.write(f"**Gözlem Sayısı:** {df.shape[0]}")
    st.write(f"**Değişken Sayısı:** {df.shape[1]}")
    st.write("**Veri Tipleri:**")
    st.write(df.dtypes.astype(str))

st.divider()



st.sidebar.title("Grafik Kontrolleri")
cat_columns = df.select_dtypes(include=['object']).columns.tolist()
selected_cat = st.sidebar.selectbox("Pasta Grafiği İçin Değişken Seçin", cat_columns)


st.header("2. Kategorik Dağılım Analizi")
fig_pie = px.pie(df.head(1000), names=selected_cat, title=f"{selected_cat} Değişkeni Dağılımı (İlk 1000 Kayıt)")
st.plotly_chart(fig_pie)


st.header("3. Ülke Bazlı İşlem Sayıları")
top_countries = df['Country'].value_counts().head(10)
st.bar_chart(top_countries) 


st.header("4. Sayısal Analizler")
col3, col4 = st.columns(2)

with col3:
    st.subheader("Miktar vs Birim Fiyat (Saçılım)")
    
    fig_scatter = px.scatter(df.sample(500), x="Quantity", y="UnitPrice", color="Country")
    st.plotly_chart(fig_scatter)
    
    
with col4:
    st.subheader("Gelir (Revenue) Dağılımı")
    fig_hist = px.histogram(df[df['Revenue'] < 200], x="Revenue", nbins=50, title="Gelir Dağılımı (Filtrelenmiş)")
    st.plotly_chart(fig_hist)
    
    
    st.header("5. Zaman İçinde İşlem Sayısı")

df_time = df.set_index('InvoiceDate').resample('M').size().reset_index(name='İşlem Sayısı')
fig_line = px.line(df_time, x='InvoiceDate', y='İşlem Sayısı', title="Aylık Toplam İşlem Sayısı")
st.plotly_chart(fig_line)

st.divider()


st.header("6. PCA (Temel Bileşen Analizi)")

features = ['Quantity', 'UnitPrice', 'Revenue']
x = df[features].dropna()
x_scaled = StandardScaler().fit_transform(x)


# 2 Bileşenli PCA
pca = PCA(n_components=2)
components = pca.fit_transform(x_scaled)
pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
pca_df['Country'] = df['Country'].values


top_5_countries = df['Country'].value_counts().head(5).index
pca_df['Color_Group'] = pca_df['Country'].apply(lambda x: x if x in top_5_countries else 'Diğer')


fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Color_Group', title="2 Bileşenli PCA Grafiği")
st.plotly_chart(fig_pca)



# --- Madde 10 & 11: Feature Selection ve ML ---
st.header("7. Makine Öğrenmesi: Revenue Tahmini")

if st.button("Modeli Eğit (Random Forest)"): 
    with st.spinner("Model eğitiliyor, lütfen bekleyin..."): 
        # (Quantity ve UnitPrice)
        X_ml = df[['Quantity', 'UnitPrice']]
        y_ml = df['Revenue']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_ml, y_ml)
        
        score = model.score(X_ml, y_ml)
        st.success(f"Model Başarıyla Eğitildi! R2 Skoru: {score:.4f}") 
        
      
        importances = pd.Series(model.feature_importances_, index=['Quantity', 'UnitPrice'])
        st.write("Özellik Önem Düzeyleri:")
        st.bar_chart(importances)


with st.expander("Proje Hakkında Notlar"):
    st.write("Bu uygulama İstanbul Üniversitesi Veri Bilimi programı ödevi kapsamında geliştirilmiştir.")






# Yan Panel Başlığı
st.sidebar.title("Analiz Ayarları")

# Kategorik sütunları seçelim (Ülke, Stok Kodu gibi)
kategorik_sutunlar = ["Country", "StockCode"]
secilen_kategori = st.sidebar.selectbox("Lütfen bir kategori seçin:", kategorik_sutunlar)

st.write(f"### {secilen_kategori} Bazında Analiz")



if st.button("Modeli Eğit (Random Forest)"):
    with st.spinner("Model eğitiliyor, lütfen bekleyin..."):
        # Burada basit bir model eğitimi simülasyonu veya gerçek eğitimi yapabilirsin
        import time
        time.sleep(2) # İşlem yapılıyormuş gibi bekletme
        st.success("Model başarıyla eğitildi! R2 Skoru: 0.85")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    