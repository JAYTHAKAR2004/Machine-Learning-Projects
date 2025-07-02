import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

st.set_page_config(page_title="Stock Price Prediction", layout="wide")
stock=st.text_input("Enter Stock Ticker", "GOOG")
from datetime import datetime, timedelta
end=datetime.now()
start=datetime(end.year-20, end.month, end.day)
google_data= yf.download(stock, start=start, end=end)
model= load_model("Latest_Stock_price_model.keras")
st.subheader("Stock Price Prediction")
st.write(google_data)
splitting_len= int(len(google_data)*0.7)
x_test=pd.DataFrame(google_data.Close[splitting_len:])
def plot_graph(figsize,values,full_data):
    fig= plt.figure(figsize=figsize)
    plt.plot(values,'Orange' )
    plt.plot(full_data.Close, 'blue')
    return fig
st.subheader("Original Close Price And MA for 250 days")
google_data['MA_for_250_days']=google_data.Close.rolling(250).mean() 
st.pyplot(plot_graph((20, 5), google_data.MA_for_250_days, google_data))

st.subheader("Original Close Price And MA for 200 days")
google_data['MA_for_200_days']=google_data.Close.rolling(200).mean() 
st.pyplot(plot_graph((20, 5), google_data.MA_for_200_days, google_data))

st.subheader("Original Close Price And MA for 100 days")
google_data['MA_for_100_days']=google_data.Close.rolling(100).mean() 
st.pyplot(plot_graph((20, 5), google_data.MA_for_100_days, google_data))












from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0,1))
scaled_data= scaler.fit_transform(google_data.Close.values.reshape(-1,1))
x_data= []
y_data= []

for i in range(250, len(scaled_data)):
    x_data.append(scaled_data[i-250:i, 0])
    y_data.append(scaled_data[i, 0])

x_data, y_data = np.array(x_data), np.array(y_data)
predictions=model.predict(x_data)
inv_pre=scaler.inverse_transform(predictions.reshape(-1, 1))
inv_y_test=scaler.inverse_transform(y_data.reshape(-1, 1))
plotting_data=pd.DataFrame(
    {
        'original_test_data':inv_y_test.reshape(-1),
        'prediction':inv_predictions.reshape(-1)
    },
    index=google_data.index[splitting_len+100:]
)
plotting_data.head()

st.write("Predicted vs Original Stock Price")
st.subheader("Predicted vs Original Stock Price")
st.pyplot(plot_graph((20, 5), plotting_data.prediction, plotting_data))
