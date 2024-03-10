import requests  # for making http requests to binance
import json  # for parsing what binance sends back to us
import pandas as pd  # for storing and manipulating the data we get back
import numpy as np  # numerical python, i usually need this somewhere
# and so i import by habit nowadays

import matplotlib.pyplot as plt  # for charts and such

import datetime as dt  # for dealing with times
import matplotlib.pyplot as plt

root_url = 'https://api.binance.com/api/v3/klines'

# startTime is the timestamp of the start time, this function returns the data of the appointed trading set
# symbol stands for the trading set, for instance: ETHUSDT stands for the data of eth/usdt, and you can input BTCUSDT to obtain the data of btc/usdt
def get_bars(symbol, interval='1d', startTime='1546300800000', endTime='1808356428000'):
    url = root_url + '?symbol=' + symbol + '&interval=' + interval + '&startTime=' + startTime
    data = json.loads(requests.get(url).text)
    df = pd.DataFrame(data)
    df.columns = ['open_time',
                  'o', 'h', 'l', 'c', 'v',
                  'close_time', 'qav', 'num_trades',
                  'taker_base_vol', 'taker_quote_vol', 'ignore']
    df.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in df.close_time]
    return df


ethusdt = get_bars('ETHUSDT')
ethusdt = ethusdt['c'].astype('float')
# ethusdt.plot(figsize=(16,9))
plt.plot(ethusdt)
plt.show()
# The server's response to your POST request
