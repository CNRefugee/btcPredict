# keep the column of
# timestamp, o,h,l,c,v,qav,num_trades,taker_quote_vol,ignore
from common.constants import BTC, ETH, BNB
from dataSpyder import fileProcessor
from dataSpyder.fileProcessor import read_data_file
import pandas as pd

symbol = BNB
suffix = '.pkl'
filepath = './rawData/' + symbol + suffix
df = read_data_file(filepath)

# Define the start and end of your series
start = df.index[0]
end = df.index[-1]
print(len(df.index))  # Total number of rows
print(len(df.index.unique()))  # Number of unique index values
df = df[~df.index.duplicated(keep='first')]

# Create a date range
date_range = pd.date_range(start=start, end=end, freq='4h')

# Reindex the DataFrame
df = df.reindex(date_range)
df = df[['o', 'h', 'l', 'c', 'v', 'qav', 'num_trades']]

# Select only numeric columns for interpolation
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.infer_objects()
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].interpolate(method='linear')

# 选取的列: 开盘时间open_time，开盘o，最高h，最低l，收盘c，交易量v，qav交易的usdt数量，成交数num_trades
path = './processedData/' + symbol
fileProcessor.save_dataframe_to_pickle(df, path + '.pkl')
fileProcessor.save_dataframe_to_csv(df, path + '.csv')

