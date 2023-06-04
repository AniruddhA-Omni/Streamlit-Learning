import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf
import streamlit as st

st.title('Stock Price App')
st.markdown("""
This app retrieves stock closing price and volume of Google!
""")

@st.cache_data
def load_data(url):
    data = pd.read_html(url, header=0)
    data =data[0]
    return data

df = load_data('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

sector_unique = sorted(df["GICS Sector"].unique())

sector = df.groupby("GICS Sector")

data = yf.download(tickers = list(df.Symbol), period = 'ytd', interval = '1d', group_by = 'ticker', auto_adjust = True, prepost = True, threads = True, proxy = None)

df2 = pd.DataFrame(data['ABT'].Close)
df2["Data"] = df2.index

def price_plot(symbol):
    df2 = pd.DataFrame(data[symbol].Close)
    df2["Data"] = df.index
    plt.fill_between(df2.Data, df2.Close, color = 'skyblue', alpha = 0.3)
    plt.plot(df2.Data, df2.Close, color = 'skyblue', alpha = 0.8)
    plt.xticks(rotation = 90)
    return st.pyplot()

st.sidebar.header('User Input Features')

selected_sector = st.sidebar.multiselect('Sector', sector_unique, sector_unique)

df_selected_sector = df[(df["GICS Sector"].isin(selected_sector))]

st.header('Display Companies in Selected Sector')
st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(df_selected_sector.shape[1]) + ' columns.')
st.dataframe(df_selected_sector)

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="stock.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

num_company = st.sidebar.slider('Number of Companies', 1, 20)

if st.button("Show Plots"):
    st.header("Stock Closing Price")
    for i in list(df_selected_sector.Symbol)[:num_company]:
        price_plot(i)