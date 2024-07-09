import streamlit as st
import datetime
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# List of tickers and company names
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

# Load historical data for all tickers
historical_data = {}
for ticker in tickers:
    historical_data[ticker] = pd.read_csv(f'data/{ticker}_historical_data.csv', index_col='Date', parse_dates=True)

# Function to get prediction
def predict_stock_price(ticker, end_date):
    # Load the model and scaler for the selected ticker
    model = load_model(f'Models/{ticker}_model.h5')
    with open(f'Scalers/{ticker}_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Get today's date
    start_date = datetime.datetime.today().date()
    required_lookback_days = 90
    
    # Ensure 90 days of data
    hist_data = historical_data[ticker].loc[:start_date]['Close']
    if len(hist_data) < required_lookback_days:
        raise ValueError(f"Not enough historical data to meet the {required_lookback_days}-day lookback period.")
    historical_prices = hist_data.tail(required_lookback_days).values.reshape(-1, 1)

    # Scale the historical data
    scaled_data = scaler.transform(historical_prices)
    
    # Prepare data for batch prediction
    X_pred = np.reshape(scaled_data[-required_lookback_days:], (1, required_lookback_days, 1))
    date_range = pd.date_range(start=start_date, end=end_date)
    n_days = len(date_range)
    
    predictions = np.zeros(n_days)
    for i in range(n_days):
        pred_price = model.predict(X_pred, verbose=0)
        pred_price_unscaled = scaler.inverse_transform(pred_price)
        predictions[i] = pred_price_unscaled[0, 0]
        
        # Append the predicted price to the scaled data for the next batch prediction
        scaled_data = np.append(scaled_data[1:], scaler.transform(pred_price_unscaled), axis=0)
        X_pred = np.reshape(scaled_data[-required_lookback_days:], (1, required_lookback_days, 1))

    # Create a DataFrame for the results
    results = pd.DataFrame({
        'Date': date_range,
        'Predicted Close Price': predictions
    })

    # Calculate price changes
    results['Predicted Price Change'] = results['Predicted Close Price'].diff().fillna(0)
    results['Predicted Price Change (%)'] = results['Predicted Close Price'].pct_change().fillna(0) * 100
    results['Predicted Price Change'] = results.apply(
        lambda row: f"${row['Predicted Price Change']:.2f} ({row['Predicted Price Change (%)']:.2f}%)", axis=1)

    # Calculate overall changes from the first predicted value
    first_value = results['Predicted Close Price'].iloc[0]
    results['Overall Change'] = results['Predicted Close Price'].apply(
        lambda x: f"${x - first_value:.2f} ({((x - first_value) / first_value) * 100:.2f}%)"
    )

    results.drop(columns=['Predicted Price Change (%)'], inplace=True)
    
    return results

# Streamlit app
st.title('Stock Price Prediction App')

# Dropdown for selecting company
selected_ticker = st.selectbox('Select Company', [company_names[ticker] for ticker in tickers])

# Use today's date as the start date
start_date = datetime.date.today()
st.write(f"Start Date: {start_date}")

# Date input for end date
end_date = st.date_input('End Date', start_date + datetime.timedelta(days=30))

# Check date constraints
if start_date > end_date:
    st.error('Error: End Date must fall after Start Date.')
    st.stop()

# Button to predict
if st.button('Predict'):
    selected_ticker_code = [key for key, value in company_names.items() if value == selected_ticker][0]
    if (end_date - datetime.date.today()).days > 365:
        proceed = st.radio('Warning: Predictions are recommended within a year for better accuracy. Are you sure that you want to proceed?', ('Cancel', 'Proceed'))
        if proceed == 'Cancel':
            st.stop()
        elif proceed == 'Proceed':
            predictions = predict_stock_price(selected_ticker_code, end_date)
            fig = go.Figure()

            # Add traces for positive and negative changes
            fig.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Predicted Close Price'],
                mode='lines+markers',
                name='Predicted Close Price',
                marker=dict(
                    color=['green' if x >= 0 else 'red' for x in predictions['Predicted Close Price'].diff().fillna(0)],
                    size=8,
                    line=dict(width=1)
                )
            ))

            fig.update_layout(
                title=f'Predicted Stock Prices for {selected_ticker}',
                xaxis_title='Date',
                yaxis_title='Predicted Close Price',
                template='plotly_dark'
            )

            st.plotly_chart(fig)
            st.write(f"Predicted Stock Prices for {selected_ticker}")
            st.write(predictions)
    else:
        predictions = predict_stock_price(selected_ticker_code, end_date)
        fig = go.Figure()

        # Add traces for positive and negative changes
        fig.add_trace(go.Scatter(
            x=predictions['Date'],
            y=predictions['Predicted Close Price'],
            mode='lines+markers',
            name='Predicted Close Price',
            marker=dict(
                color=['green' if x >= 0 else 'red' for x in predictions['Predicted Close Price'].diff().fillna(0)],
                size=8,
                line=dict(width=1)
            )
        ))

        fig.update_layout(
            title=f'Predicted Stock Prices for {selected_ticker}',
            xaxis_title='Date',
            yaxis_title='Predicted Close Price',
            template='plotly_dark'
        )

        st.plotly_chart(fig)
        st.write(f"Predicted Stock Prices for {selected_ticker}")
        st.write(predictions)




