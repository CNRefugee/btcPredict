import time

import requests  # for making http requests to binance
import json  # for parsing what binance sends back to us
import pandas as pd  # for storing and manipulating the data we get back
from datetime import datetime, timedelta
import datetime as dt  # for dealing with times
import dataSaver

root_url = 'https://api.binance.com/api/v3/klines'


# startTime is the timestamp of the start time, this function returns the data of the appointed trading set
# symbol stands for the trading set, for instance: ETHUSDT stands for the data of eth/usdt, and you can input BTCUSDT to obtain the data of btc/usdt
def get_bars(symbol, interval='4h', startTime='1546300800000', endTime='1808356428000'):
    url = root_url + '?symbol=' + symbol + '&interval=' + interval + '&startTime=' + startTime + '&endTime=' + endTime
    data = json.loads(requests.get(url).text)
    df = pd.DataFrame(data)
    df.columns = ['open_time',
                  'o', 'h', 'l', 'c', 'v',
                  'close_time', 'qav', 'num_trades',
                  'taker_base_vol', 'taker_quote_vol', 'ignore']
    df.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in df.close_time]
    return df


def get_dataset(start_date, end_date, symbol, interval, time_interval):
    return dataset


start_date = datetime(2017, 8, 22)
end_date = datetime(2024, 3, 9)
symbol = 'BNBUSDT'
interval = '4h'
millisecond = 1000
time_interval = timedelta(days=80)
dataset = None
start_timestamp = None
end_timestamp = None
current_date = start_date

while current_date <= end_date:
    start_timestamp = int(datetime.timestamp(current_date) * millisecond)
    end_timestamp = int(datetime.timestamp(current_date + time_interval - timedelta(seconds=1)) * millisecond)
    raw_data = get_bars(symbol=symbol, interval=interval, startTime=str(start_timestamp),
                        endTime=str(end_timestamp))
    if dataset is None:
        dataset = raw_data
    else:
        dataset = pd.concat([dataset, raw_data], ignore_index=False)
    # start = current_date.strftime("%Y-%m-%d")
    # end = (current_date + time_interval).strftime("%Y-%m-%d")
    # print('fetch data from ' + start + ' to ' + end)
    current_date += time_interval

current_date -= time_interval
start_timestamp = int(datetime.timestamp(current_date) * millisecond)
end_timestamp = int(datetime.timestamp(datetime.now()) * millisecond)
raw_data = get_bars(symbol=symbol, interval=interval, startTime=str(start_timestamp),
                    endTime=str(end_timestamp))
if dataset is None:
    dataset = raw_data
else:
    dataset = pd.concat([dataset, raw_data], ignore_index=False)

filename = symbol
dataSaver.save_dataframe_to_pickle(dataset, filename)
dataSaver.save_dataframe_to_csv(dataset, filename)
