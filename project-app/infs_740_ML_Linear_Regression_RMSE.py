import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

def predict_next_closing_price_symbol():
  # Fetch stock data
  ticker_symbol = 'AAPL'  ##use any symbol
  ticker_symbol = 'GOOGL'
  data = yf.download(ticker_symbol, start='2020-01-01', end='2025-03-31')

  # Prepare data
  data = pd.DataFrame(data)  # ensure DataFrame format
  #print(data.columns)
  data['Prev Close'] = data['Close'].shift(1)
  data.dropna(inplace=True) # Remove any rows with NaN values
  X = data[['Prev Close']]
  y = data['Close']
  data.sort_index(inplace=True)

  # Split the data into train and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Model training using LinearRegression
  model = LinearRegression()
  model.fit(X_train, y_train)

  # Predicting the stock prices
  predictions = model.predict(X_test)
  predictions = pd.Series(predictions.flatten(), index=X_test.index) # Convert predictions to a pandas Series with proper index

  # Calculate the RMSE
  '''
  The root-mean-square deviation (RMSD) or root-mean-square error (RMSE)
  is one of frequently used measures of the differences between true or predicted values on the one hand and observed values or an estimator on the other.
  '''
  rmse = np.sqrt(mean_squared_error(y_test, predictions))
  print("Root Mean Squared Error:", rmse)


  # plt.figure(figsize=(10, 6))
  # plt.plot(y_test.index, y_test, label='Actual')
  # plt.plot(predictions.index, predictions, label='Predicted', linestyle='--')
  #
  # plt.title(f'Actual vs Predicted Stock Prices for {ticker_symbol}')
  # plt.xlabel('Date')
  # plt.ylabel('Stock Price')
  # plt.legend()
  # plt.grid(True)
  # plt.show()

  # Plotting using Seaborn and Matplotlib
  results = pd.DataFrame({
    'Date': y_test.index,
    'Actual': np.array(y_test).flatten(),
    'Predicted': np.array(predictions).flatten()
  })
  results_melted = results.melt(id_vars='Date', value_vars=['Actual', 'Predicted'],
                                var_name='Legend', value_name='Stock Price')
  plt.figure(figsize=(12, 6))
  sns.lineplot(data=results_melted, x='Date', y='Stock Price', hue='Legend')
  plt.title(f'Actual vs Predicted Stock Prices for {ticker_symbol}')
  plt.xlabel('Date')
  plt.ylabel('Stock Price')
  plt.grid(True)
  #plt.legend(title='Legend', labels=['Actual', 'Predicted'])
  plt.show()


if __name__=='__main__':
  predict_next_closing_price_symbol()
