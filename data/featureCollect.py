# features to learn
# 1.MA (moving average)
# 2.MACD (Moving Average Convergence Divergence)
# 3.RSI (Relative Strength Index)
# 4.Bollinger Bands (BOLL)
from common.constants import BTC
from dataSpyder.fileProcessor import read_data_file

symbol = BTC
filepath = './processedData/' + symbol + '.pkl'

df = read_data_file(filepath)

print(1)
