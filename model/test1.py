import csv
import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

from common.constants import BTC
from dataSpyder.fileProcessor import read_data_file

torch.manual_seed(0)
np.random.seed(0)

feature_columns = ['o', 'h', 'l', 'c', 'v', 'qav', 'num_trades',
                   'EMA_7', 'EMA_25', 'EMA_99',
                   'MACD', 'Signal_Line', 'MACD_above_Signal', 'MACD_diff_Signal',
                   'RSI',
                   'Middle_Band', 'STD', 'Upper_Band', 'Lower_Band'
                   ]
# output targets columns
target_columns = ['o', 'h', 'l', 'c', 'v', 'qav', 'num_trades']
input_dim = len(feature_columns)
input_window = 180
output_window = 1
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

scaler = MinMaxScaler(feature_range=(0, 1))
# 暂时选取0号站台的流入量
# train_data = data[:0.8*len(data)]
# test_data = data[0.8*len(data):]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, input_dim, d_model=128, num_layers=2, dropout=0.1):
        super(TransAm, self).__init__()
        self.input_dim = input_dim
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, input_dim)
        self.fc = nn.Linear(input_dim, d_model)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.fc(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + output_window:i + tw + output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


def get_data():
    symbol = BTC
    filepath = '../data/dataWithCalculatedFeatures/' + symbol + '.pkl'

    df = read_data_file(filepath)
    df = df.to_numpy()
    train_samples = int(0.7 * len(df))
    train_data = df[:train_samples]
    test_data = df[train_samples:]

    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]

    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]

    return train_sequence.to(device), test_data.to(device)

def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    input = torch.squeeze(input, dim=2)
    target = torch.squeeze(target, dim=2)
    return input, target


def train(train_data):
    model.train()

    for batch_index, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        start_time = time.time()
        total_loss = 0
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        targets = targets[:, :, :7]
        output = model(data)[:, :, :7]
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch_index % log_interval == 0 and batch_index > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f} '
                  .format(epoch, batch_index, len(train_data) // batch_size, scheduler.get_lr()[0],
                          elapsed * 1000 / log_interval, cur_loss))


def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
    return total_loss / len(data_source)

def evaluate_and_plot(model, data_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            features, targets = features.to(device), targets.to(device)
            output = model(features, src_mask)
            predictions.extend(output[:, -1, :7].cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)

    print(f'MSE: {mse}, MAE: {mae}, RMSE: {rmse}')

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(actuals, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
def plot_and_loss(eval_model, data_source, epoch):
    eval_model.eval()
    total_loss = 0.
    predictions = torch.zeros(len(data_source)-1, input_dim).to(device)
    real_flow = torch.zeros(len(data_source)-1, input_dim).to(device)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            output = eval_model(data)
            total_loss += criterion(output, target).item()
            predictions[i] = output[-1, 0].cpu()
            real_flow[i] = target[-1, 0].cpu()


    predictions = predictions.cpu()
    real_flow = real_flow.cpu()
    plt.plot(predictions[:200, 0], color="red", label="predictions")
    plt.plot(real_flow[:200, 0], color="blue", label="real")
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.legend(loc='upper right')
    plt.savefig('../graph/multi_dim_81/transformer-epoch%d.png' % epoch)
    plt.close()
    mae = torch.mean(torch.abs(predictions - real_flow))
    mse = torch.mean((predictions - real_flow) ** 2)
    rmse = torch.sqrt(torch.mean((predictions - real_flow) ** 2))
    with open('../indexes/multi_dim.csv', mode='a', newline='') as file:
        # Create a writer object
        writer = csv.writer(file)

        # Write a new row of data
        writer.writerow([epoch, mae, mse, rmse])
    return total_loss / i


train_data, val_data = get_data()
model = TransAm(input_dim=input_dim).to(device)
criterion = nn.MSELoss()
lr = 0.05
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
epochs = 200

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)
    #
    # if (epoch % 10 is 0):
    #     val_loss = plot_and_loss(model, val_data, epoch)
    # else:
    #     val_loss = evaluate(model, val_data)
    #
    # print('-' * 89)
    # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} '.format(epoch, (
    #         time.time() - epoch_start_time), val_loss))
    # print('-' * 89)
    scheduler.step()

def plot_and_loss_1(eval_model, data_source, test_len):
    eval_model.eval()
    total_loss = 0.
    predictions = torch.zeros(test_len-1, input_dim).to(device)
    real_flow = torch.zeros(test_len-1, input_dim).to(device)
    with torch.no_grad():
        for i in range(0, test_len - 1):
            data, target = get_batch(data_source, i, 1)
            output = eval_model(data)
            total_loss += criterion(output, target).item()
            predictions[i] = output[-1, 0]
            real_flow[i] = target[-1, 0].cpu()


    predictions = np.round(predictions.cpu().numpy()).astype(np.float32)
    predictions[predictions < 0] = 0
    real_flow = real_flow.cpu().numpy().astype(np.float32)
    nonzero_indices = real_flow.nonzero()  # 获取非零元素的索引
    real_flow = real_flow[nonzero_indices]
    predictions = predictions[nonzero_indices]
    mae = mean_absolute_error(predictions, real_flow)
    mape = mean_absolute_percentage_error(real_flow, predictions)
    rmse = math.sqrt(mean_squared_error(predictions, real_flow))
    outcome = {}
    outcome['predictions'] = predictions
    outcome['real_flow'] = real_flow
    # predict_pkl = 'predict_outcome.pickle'
    # with open(predict_pkl, 'wb') as f:
    #     pickle.dump(outcome, f)
    with open('../test/tsfm_only.csv', mode='a', newline='') as file:
        # Create a writer object
        writer = csv.writer(file)

        # Write a new row of data
        writer.writerow([mae, mape, rmse])
torch.save(model.state_dict(), "../model/transformer_only.pth")

for i in range(2, 8):
    plot_and_loss_1(model, val_data, i)