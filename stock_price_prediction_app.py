import streamlit as st
import datetime
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the models and scalers
models = {}
scalers = {}
tickers = ['AAPL', 'MSFT', 'JNJ', 'JPM', 'PG', 'XOM', 'NVDA', 'PFE', 'KO', 'TSLA']
company_names = {
    'AAPL': 'Apple (AAPL)',
    'MSFT': 'Microsoft (MSFT)',
    'JNJ': 'Johnson & Johnson (JNJ)',
    'JPM': 'JPMorgan Chase (JPM)',
    'PG': 'Procter & Gamble (PG)',
    'XOM': 'Exxon Mobil (XOM)',
    'NVDA': 'NVIDIA (NVDA)',
    'PFE': 'Pfizer (PFE)',
    'KO': 'Coca-Cola (KO)',
    'TSLA': 'Tesla (TSLA)'
}

for ticker in tickers:
    models[ticker] = load_model(f'Models/{ticker}_model.h5')
    with open(f'Scalers/{ticker}_scaler.pkl', 'rb') as f:
        scalers[ticker] = pickle.load(f)

# Function to get prediction
def predict_stock_price(ticker, start_date, end_date):
    model = models[ticker]
    scaler = scalers[ticker]
    
    # Calculate the number of days for prediction
    num_days = (end_date - start_date).days

    # Create an initial dummy data array for the past 90 days (look-back period)
    dummy_data = np.zeros((90, 1))  # 90 days of initial data
    
    # Scale the initial dummy data
    scaled_data = scaler.transform(dummy_data)
    
    predictions = []
    date_range = pd.date_range(start=start_date, periods=num_days + 1)
    
    for _ in range(num_days + 1):
        X_pred = np.reshape(scaled_data[-90:], (1, 90, 1))
        pred_price = model.predict(X_pred)
        pred_price_unscaled = scaler.inverse_transform(pred_price)
        
        predictions.append(pred_price_unscaled[0, 0])
        
        # Append the predicted price to the scaled data for the next prediction
        scaled_data = np.append(scaled_data, scaler.transform(pred_price_unscaled), axis=0)

    # Create a DataFrame for the results
    results = pd.DataFrame({
        'Date': date_range,
        'Predicted Close Price': predictions
    })

    return results

# Streamlit app
st.title('Stock Price Prediction App')

# Dropdown for selecting company
selected_ticker = st.selectbox('Select Company', [company_names[ticker] for ticker in tickers])

# Date input for start and end dates
start_date = st.date_input('Start Date', datetime.date.today())
end_date = st.date_input('End Date', datetime.date.today() + datetime.timedelta(days=30))

# Check date constraints
if start_date > end_date:
    st.error('Error: End Date must fall after Start Date.')
    st.stop()

# Button to predict
if st.button('Predict'):
    selected_ticker_code = [key for key, value in company_names.items() if value == selected_ticker][0]
    if (end_date - datetime.date.today()).days > 365:
        proceed = st.radio('Warning: Predictions are recommended within a year for better accuracy. Do you want to proceed?', ('Cancel', 'Proceed'))
        if proceed == 'Cancel':
            st.stop()
        elif proceed == 'Proceed':
            predictions = predict_stock_price(selected_ticker_code, start_date, end_date)
            fig = px.line(predictions, x='Date', y='Predicted Close Price', title=f'Predicted Stock Prices for {selected_ticker}')
            fig.update_traces(mode='lines+markers', hovertemplate='Date: %{x}<br>Price: %{y:.2f}')
            st.plotly_chart(fig)
            st.write(predictions)
    else:
        predictions = predict_stock_price(selected_ticker_code, start_date, end_date)
        fig = px.line(predictions, x='Date', y='Predicted Close Price', title=f'Predicted Stock Prices for {selected_ticker}')
        fig.update_traces(mode='lines+markers', hovertemplate='Date: %{x}<br>Price: %{y:.2f}')
        st.plotly_chart(fig)
        st.write(predictions)
