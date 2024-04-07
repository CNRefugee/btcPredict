
import pandas as pd
from neuralforecast.models import VanillaTransformer, Informer, Autoformer, FEDformer, PatchTST
from neuralforecast.core import NeuralForecast
import matplotlib.pyplot as plt
from neuralforecast.losses.numpy import mae, rmse, mse
from common.constants import BTC
from dataSpyder.fileProcessor import read_data_file

# Load your time series data
# data = ...

# Example of data preparation
# Let's assume 'data' is a pandas DataFrame with 'time' and 'value' columns
symbol = BTC
filepath = '../data/dataWithCalculatedFeatures/' + symbol + '.pkl'

df = read_data_file(filepath)

# h: 预测时域（forecast horizon） - 这是您想要预测未来的时间步长数量。例如，如果您想要预测未来24小时的数据，h 应该设置为24。
horizon = 12
# input_size: 输入序列的长度 - 这是模型一次性看到的历史数据点的数量，用于基于这些数据点来做出未来的预测。
input_size = 180
# max_steps: 训练步骤的最大数量 - 这是训练过程中迭代优化模型权重的次数。
train_steps = 50
# val_check_steps: 验证检查步骤 - 这是模型在每次训练过程中经过多少步后执行验证集上的性能评估。
check_steps = 10
# early_stop_patience_steps: 早停耐心步骤 - 这是在验证集上性能没有改进时，模型继续训练的步骤数量，在此之后训练将停止。
#
# scaler_type: 标准化类型 - 这指定了用于输入数据的标准化方法。例如，'standard' 代表标准标准化，它会从数据中减去均值并除以标准差。

models = [VanillaTransformer(h=horizon,
                             input_size=input_size,
                             max_steps=train_steps,
                             val_check_steps=check_steps,
                             early_stop_patience_steps=3,
                             scaler_type='standard'),
          Informer(h=horizon,  # Forecasting horizon
                   input_size=input_size,  # Input size
                   max_steps=train_steps,  # Number of training iterations
                   val_check_steps=check_steps,  # Compute validation loss every 100 steps
                   early_stop_patience_steps=3,  # Number of validation iterations before early stopping
                   scaler_type='standard'),  # Stop training if validation loss does not improve
          FEDformer(h=horizon,
                     input_size=input_size,
                     max_steps=train_steps,
                     val_check_steps=check_steps,
                     early_stop_patience_steps=3),
          Autoformer(h=horizon,
                     input_size=input_size,
                     max_steps=train_steps,
                     val_check_steps=check_steps,
                     early_stop_patience_steps=3),
          PatchTST(h=horizon,
                   input_size=input_size,
                   max_steps=train_steps,
                   val_check_steps=check_steps,
                   early_stop_patience_steps=3),
          ]


nf = NeuralForecast(
    models=models,
    freq='B')

df = df.rename(columns={'c': 'y'})
df['unique_id'] = 1
df['ds'] = df.index
df['ds'] = pd.to_datetime(df['ds'])
df = df[['unique_id', 'ds', 'y']]

Y_hat_df = nf.cross_validation(df=df,
                               val_size=100,
                               test_size=100,
                               n_windows=None)


Y_plot = Y_hat_df
cutoffs = Y_hat_df['cutoff'].unique()[::horizon]
Y_plot = Y_plot[Y_hat_df['cutoff'].isin(cutoffs)]


plt.figure(figsize=(20, 5))
plt.plot(Y_plot['ds'], Y_plot['y'], label='True')
for model in models:
    plt.plot(Y_plot['ds'], Y_plot[model], label=model)
    rmse_value = rmse(Y_hat_df['y'], Y_hat_df[model])
    mae_value = mae(Y_hat_df['y'], Y_hat_df[model])
    mse_value = mse(Y_hat_df['y'], Y_hat_df[model])
    print(f'{model}: rmse {rmse_value:.4f} mae {mae_value:.4f} mse {mse_value:.4f}')

plt.xlabel('Datestamp')
plt.ylabel('Close')
plt.grid()
plt.legend()
plt.show()