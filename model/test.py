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
from torch.utils.data import DataLoader, Dataset

from common.constants import BTC
from dataSpyder.fileProcessor import read_data_file

torch.manual_seed(0)
np.random.seed(0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

scaler = MinMaxScaler(feature_range=(0, 1))


feature_columns = ['o', 'h', 'l', 'c', 'v', 'qav', 'num_trades',
                   'EMA_7', 'EMA_25', 'EMA_99',
                   'MACD', 'Signal_Line', 'MACD_above_Signal', 'MACD_diff_Signal',
                   'RSI',
                   'Middle_Band', 'STD', 'Upper_Band', 'Lower_Band'
                   ]
# output targets columns
target_columns = ['o', 'h', 'l', 'c', 'v', 'qav', 'num_trades']
input_dim = len(feature_columns)
input_window = 20
output_window = 1
batch_size = 64

class StockDataset(Dataset):
    def __init__(self, dataframe, sequence_length):
        self.dataframe = dataframe
        self.sequence_length = sequence_length
        self.features = dataframe[feature_columns].values
        self.targets = dataframe[target_columns].values

    def __len__(self):
        return len(self.dataframe) - self.sequence_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx:idx + self.sequence_length], dtype=torch.float),
            torch.tensor(self.targets[idx + 1:idx + self.sequence_length + 1], dtype=torch.float)  # Next day's close price
        )


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
    def __init__(self, input_dim, d_model=256, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.input_dim = input_dim
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=16, dropout=dropout)
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


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    input = torch.squeeze(input, dim=2)
    target = torch.squeeze(target, dim=2)
    return input, target


symbol = BTC
filepath = '../data/dataWithCalculatedFeatures/' + symbol + '.pkl'

df = read_data_file(filepath)
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.15)
test_size = len(df) - (train_size + val_size)

train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
test_df = df[train_size + val_size:]

sequence_length = 60  # Adjust based on your needs
train_dataset = StockDataset(train_df, sequence_length)
val_dataset = StockDataset(val_df, sequence_length)
test_dataset = StockDataset(test_df, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def train(train_loader):
    model.train()
    for batch, (data, targets) in enumerate(train_loader):
        start_time = time.time()
        data = data.permute(1, 0, 2)
        targets = targets.permute(1, 0, 2)
        total_loss = 0
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output[:, :, :7]
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_loader) / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f} '
                  .format(epoch, batch, len(train_loader) // batch_size, scheduler.get_lr()[0],
                          elapsed * 1000 / log_interval, cur_loss))



model = TransAm(input_dim=input_dim).to(device)
criterion = nn.MSELoss()
lr = 0.005
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
epochs = 200

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_loader)
    scheduler.step()
