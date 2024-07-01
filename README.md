# Predicting Stock Prices Using Machine Learning

## Summary
The purpose of this project is to predict stock prices for specific companies or a set of companies using historical stock data and related financial indicators. By leveraging machine learning techniques, particularly Long Short-Term Memory (LSTM) networks, we aim to provide investors with a tool to make informed decisions about buying or selling stocks. This project involves extracting, cleaning, and transforming historical stock data, training machine learning models, and developing an interactive application for users to predict stock prices for up to one year in advance.

## Functionality
### Long Short-Term Memory (LSTM) Networks
Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) that is well-suited for sequence prediction problems. It is chosen for this project due to its ability to remember long-term dependencies and handle the vanishing gradient problem, which is crucial for predicting stock prices based on historical data. LSTMs have three types of gates (input, forget, and output) that regulate the flow of information, making them powerful for time series prediction.

### Main Functions
1. **Data Extraction and Cleaning**
    - **Purpose**: Extract historical stock data from sources such as Yahoo Finance API, clean and transform the data for model training.
    - **Key Points**:
        - Merges data from multiple sources.
        - Converts column names to common terms for ease of understanding.
        - Displays data before and after cleaning for transparency.

2. **Model Training and Evaluation**
    - **Purpose**: Initialize, train, and evaluate LSTM models for predicting stock prices.
    - **Key Points**:
        - Utilizes LSTM's ability to handle sequential data for accurate predictions.
        - Demonstrates meaningful predictive power with at least 75% classification accuracy or 0.80 R-squared.
        - Documents the model optimization and evaluation process.
        - Displays overall model performance.

3. **Interactive Prediction App**
    - **Purpose**: Provide an interactive interface for users to input specific dates and parameters to see predicted stock prices.
    - **Key Points**:
        - Allows predictions for any given stock for up to 1 year in advance.
        - Enhances user experience by making predictions easily accessible.
        - Saves predictions for future reference.

## Data Sources
- **Yahoo Finance API**: Supplies historical stock data and additional financial indicators.

## Individual Functions

### `main.ipynb`

1. **extract_data()**
    - **What it is**: Function to extract historical stock data from Yahoo Finance.
    - **Why it is in the code**: Essential for obtaining the raw data needed for training the LSTM model.
    - **Additional Point**: Handles API requests and data formatting.

2. **clean_data()**
    - **What it is**: Function to clean and preprocess the extracted stock data.
    - **Why it is in the code**: Ensures data quality and consistency for accurate model training.
    - **Additional Point**: Converts column names to user-friendly terms and handles missing values.

3. **transform_data()**
    - **What it is**: Function to transform the cleaned data into a format suitable for LSTM model training.
    - **Why it is in the code**: Prepares data by scaling and reshaping it for time series analysis.
    - **Additional Point**: Splits data into training and testing sets.

4. **initialize_model()**
    - **What it is**: Function to initialize the LSTM model with specified parameters.
    - **Why it is in the code**: Sets up the model architecture for training.
    - **Additional Point**: Defines the layers and configurations of the LSTM network.

5. **train_model()**
    - **What it is**: Function to train the LSTM model on the prepared data.
    - **Why it is in the code**: Crucial for developing the predictive capabilities of the model.
    - **Additional Point**: Tracks model performance metrics during training.

6. **evaluate_model()**
    - **What it is**: Function to evaluate the performance of the trained LSTM model.
    - **Why it is in the code**: Measures the accuracy and reliability of the model's predictions.
    - **Additional Point**: Generates performance reports and visualizations.

### `stock_price_prediction_app.py`
This script focuses on creating an interactive application for predicting future stock prices. It uses several key functions to load the pre-trained LSTM model, make predictions based on user inputs, and visualize the predicted stock prices. The application enhances user experience by providing an easy-to-use interface and clear visual representations of the predictions.

The script starts by loading the pre-trained LSTM model using the `load_model()` function, ensuring that the model is ready for making predictions. The `predict_stock_prices()` function takes user inputs, such as the stock ticker and the desired prediction date range, and uses the LSTM model to generate future stock price predictions. These predictions are then plotted alongside historical data using the `plot_predictions()` function, providing a visual comparison that helps users understand the trends and forecasts. Finally, the `run_app()` function initializes and runs the interactive application, handling user inputs and displaying the predictions.

### Individual Functions

1. **load_model()**
    - **What it is**: Function to load the pre-trained LSTM model.
    - **Why it is in the code**: Enables the app to use the trained model for making predictions.
    - **Additional Point**: Handles model serialization and deserialization.

2. **predict_stock_prices()**
    - **What it is**: Function to predict future stock prices based on user inputs.
    - **Why it is in the code**: Provides the core functionality of the interactive app.
    - **Additional Point**: Utilizes the LSTM model to generate predictions for up to one year in advance.

3. **plot_predictions()**
    - **What it is**: Function to plot the predicted stock prices along with historical data.
    - **Why it is in the code**: Visualizes the predictions for better user understanding.
    - **Additional Point**: Enhances the user interface by providing clear and informative charts.

4. **run_app()**
    - **What it is**: Function to run the interactive stock price prediction app.
    - **Why it is in the code**: Initializes and launches the app for user interaction.
    - **Additional Point**: Handles user inputs and displays predictions in a user-friendly manner.

## Project Findings
### Pros
- **High Predictive Power**: The LSTM models demonstrated high accuracy in predicting stock prices.
- **User-Friendly Application**: The interactive app allows users to easily access predictions.
- **Comprehensive Data Handling**: The project handles data extraction, cleaning, and merging from multiple sources efficiently.

### Cons
- **Data Quality**: The accuracy of predictions is highly dependent on the quality of the input data.
- **Model Complexity**: LSTM networks are complex and require significant computational resources for training.

### Initial Perceptions and Findings
- **Effectiveness of LSTM**: LSTM networks proved effective in capturing the temporal dependencies in stock price data, resulting in accurate predictions.
- **Data Integration**: Merging data from multiple sources enhanced the robustness of the dataset, leading to better model performance.
- **User Engagement**: The interactive app was well-received, making it easier for users to explore stock price predictions.

## Real-World Application
This stock price prediction tool can be utilized by various professionals in the finance and investment sectors. Financial analysts can use the tool to forecast stock trends and provide insights to their clients. Portfolio managers can leverage the predictions to optimize their investment strategies and balance risks. Individual investors can use the app to make informed decisions about buying or selling stocks based on predicted future prices. Additionally, the tool can be beneficial for academic researchers studying market behavior and testing new financial theories. Overall, this project aims to enhance decision-making processes and contribute to more strategic investment planning.

## Instructions for Running the Code
1. **Data Extraction and Cleaning**: Run the `main.ipynb` notebook to extract, clean, and transform the data.
2. **Model Training and Evaluation**: Execute the Python script provided to train and evaluate the LSTM models.
3. **Interactive Prediction App**: Use the `stock_price_prediction_app.py` to launch the interactive app for making stock price predictions.

### Additional Notes
- Ensure all necessary libraries are installed before running the scripts.
- Configure API keys for data extraction from Yahoo Finance.
- Follow the detailed comments and pseudocode in the provided scripts for further understanding and customization.

By following these instructions and utilizing the provided code, users can effectively predict stock prices and make informed investment decisions based on historical data and advanced machine learning techniques.
