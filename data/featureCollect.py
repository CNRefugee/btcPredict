# features to learn
# 1.MA (moving average)
# 2.MACD (Moving Average Convergence Divergence)
# 3.RSI (Relative Strength Index)
# 4.Bollinger Bands (BOLL)

from common.constants import BTC, ETH, BNB
from dataSpyder import fileProcessor
from dataSpyder.fileProcessor import read_data_file
import pandas as pd

symbol = BTC
DateInterval = '4h'  # 4h or multiple of 4h
filepath = './processedData/' + symbol + '.pkl'

df = read_data_file(filepath)
start = df.index[0]
end = df.index[-1]
date_range = pd.date_range(start=start, end=end, freq=DateInterval)
df = df.reindex(date_range)

# 1.EMA Exponential Moving Averages
df['EMA_7'] = df['c'].ewm(span=7, adjust=False).mean()
df['EMA_25'] = df['c'].ewm(span=25, adjust=False).mean()
df['EMA_99'] = df['c'].ewm(span=99, adjust=False).mean()

# 2.MACD related
# Calculate the short-term exponential moving average (EMA)
ShortEMA = df['c'].ewm(span=12, adjust=False).mean()  # Fast moving average
# Calculate the long-term exponential moving average (EMA)
LongEMA = df['c'].ewm(span=26, adjust=False).mean()  # Slow moving average

# Calculate the Moving Average Convergence Divergence (MACD)
df['MACD'] = ShortEMA - LongEMA

# Calculate the signal line
df['Signal_Line'] = df.MACD.ewm(span=9, adjust=False).mean()

# Create a feature for the MACD crossing above the signal line
df['MACD_above_Signal'] = (df['MACD'] > df['Signal_Line']).astype(int)

# Create a feature for the difference between MACD and the signal line
df['MACD_diff_Signal'] = df['MACD'] - df['Signal_Line']

# 3. RSI
delta = df['c'].diff(1)
gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# 4.Bollinger Bands
period = 20
multiplier = 2
df['Middle_Band'] = df['c'].rolling(window=period, min_periods=1).mean()
df['STD'] = df['c'].rolling(window=period, min_periods=1).std()
df['Upper_Band'] = df['Middle_Band'] + (df['STD'] * multiplier)
df['Lower_Band'] = df['Middle_Band'] - (df['STD'] * multiplier)

df = df[20:]
path = './dataWithCalculatedFeatures/' + symbol + '_' + DateInterval
fileProcessor.save_dataframe_to_pickle(df, path + '.pkl')
fileProcessor.save_dataframe_to_csv(df, path + '.csv')
