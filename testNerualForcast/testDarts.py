from common.constants import BTC
from dataSpyder.fileProcessor import read_data_file
from darts.models import TransformerModel
from darts.metrics import mape
from darts import TimeSeries
import pandas as pd
from pytorch_lightning import Trainer

# Define the columns you want to predict
target_columns = ['o', 'h', 'l', 'c']

# Include other features as exogenous variables if needed for the model
exogenous_columns = ['v', 'qav', 'num_trades', 'EMA_7', 'EMA_25', 'EMA_99',
                     'MACD', 'Signal_Line', 'MACD_above_Signal', 'MACD_diff_Signal',
                     'RSI', 'Middle_Band', 'STD', 'Upper_Band', 'Lower_Band']

# Define the Transformer model with specific chunk lengths
model = TransformerModel(
    input_chunk_length=180,  # Lookback period
    output_chunk_length=24,  # Forecast horizon
    nhead=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=512,
    dropout=0.1,
    activation="relu"
)

# Fit the model on your time series data
# model.fit(series, verbose=True)

symbol = BTC
date_interval = '_4h'
filepath = '../data/dataWithCalculatedFeatures/' + symbol + date_interval + '.pkl'

df = read_data_file(filepath)

# Assuming 'df' is your DataFrame
# Ensure your DataFrame has a datetime index
if not isinstance(df.index, pd.DatetimeIndex):
    df['date_column'] = pd.to_datetime(df.index)  # convert your date column to datetime if it's not
    df.set_index('date_column', inplace=True)

# # 定义一个 Trainer 实例
# trainer = Trainer(
#     max_epochs=10,  # 最大训练轮数
#     gpus=1,  # 使用的 GPU 数量，0 表示仅使用 CPU
#     fast_dev_run=False,  # 快速运行一次训练、验证和测试循环（用于调试）
#     progress_bar_refresh_rate=20,  # 进度条更新频率
#     # 更多参数可以根据需要设置
# )

# df 是包含所有数据的 DataFrame
# total_length = len(df)
# train_end = int(total_length * 0.7)
# val_end = int(total_length * 0.85)
#
# # 按时间顺序分割数据集
# df_train = df.iloc[:train_end]
# df_val = df.iloc[train_end:val_end]
# df_test = df.iloc[val_end:]
#
# # 对于 TimeSeries 对象，您可以直接使用 from_dataframe 方法来创建
# series_train = TimeSeries.from_dataframe(df_train[target_columns])
# past_covariates_train = TimeSeries.from_dataframe(df_train[exogenous_columns])
#
# series_val = TimeSeries.from_dataframe(df_val[target_columns])
# past_covariates_val = TimeSeries.from_dataframe(df_val[exogenous_columns])
#
# series_test = TimeSeries.from_dataframe(df_test[target_columns])
# # 如果测试集也需要 past_covariates，可以相应地创建
# past_covariates_test = TimeSeries.from_dataframe(df_test[exogenous_columns])
multivariate_series = TimeSeries.from_dataframe(df[target_columns+exogenous_columns], freq='4H')

model.fit(series=multivariate_series,
          # past_covariates=past_covariates_train,
          # val_series=series_val,
          # val_past_covariates=past_covariates_val,
          # epochs=100,
          verbose=True,
          num_loader_workers=4)


# Calculate forecast horizon based on your time frequency
# For a 24-hour forecast in a 4-hourly series, forecast_horizon would be 6
# forecast_horizon = 24  # 24 hours / 4 hours per period
#
# 执行 backtest，评估模型在历史数据上的性能
backtest_result = model.backtest(
    series=series_train.concatenate(series_val),
    past_covariates=past_covariates_train.concatenate(past_covariates_val),
    start=pd.Timestamp(df_val.index[0]),  # 从验证集开始的时间点进行 backtesting
    forecast_horizon=1,  # 预测未来一个时间步长，根据实际需求调整
    stride=1,  # 每次移动一个时间步长，根据实际需求调整
    retrain=False,  # 如果为 True，则每次预测后重新训练模型
    verbose=True
)

model.predict()

MAPE MAE RMSE

# # 评估预测性能，例如使用 MAPE


# print(mape(series_val, backtest_result))
