import csv
import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GATConv,GATv2Conv, GCNConv
from torch_geometric.utils import dense_to_sparse
import pandas as pd

import utils

torch.manual_seed(0)
np.random.seed(0)

nodes_num = 81
node_dim = 2
input_dim = nodes_num * node_dim
input_window = 20
output_window = 1
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
road_map = np.array(utils.load_pickle('../data/road_map.pkl'))
adj_matrix_dense = torch.tensor(road_map, dtype=torch.float32)
scaler = MinMaxScaler(feature_range=(0, 1))
edge_index, _ = dense_to_sparse(adj_matrix_dense)
edge_index = edge_index.to(device)


class GATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=1):
        super(GATLayer, self).__init__()
        self.gat = GATv2Conv(input_dim, output_dim, heads=num_heads)

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        return x

class SimpleGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleGCNLayer, self).__init__()
        self.gcn = GCNConv(input_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        return x


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


class GCN_Transformer(nn.Module):
    def __init__(self, input_dim, adj_matrix, d_model=256, num_layers=1, dropout=0.1):
        super(GCN_Transformer, self).__init__()
        self.input_dim = input_dim
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=16, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, input_dim)
        self.gat_layer = GATLayer(d_model, d_model)
        self.gcn_layer = SimpleGCNLayer(node_dim, node_dim)
        self.adj_matrix = adj_matrix
        self.fc = nn.Linear(input_dim, d_model)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        global device
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        for t in range(src.size(0)):
            for batch in range(src.size(1)):
                feature_matrix = src[t, batch]
                in_flow = feature_matrix[:nodes_num]
                out_flow = feature_matrix[nodes_num:]
                gcn_input = torch.cat([in_flow.unsqueeze(1), out_flow.unsqueeze(1)], dim=1)
                gcn_input = gcn_input.to(device)
                gcn_output = self.gcn_layer(gcn_input, self.adj_matrix)
                gcn_output = gcn_output.transpose(0, 1).contiguous().view(-1)
                src[t, batch] = gcn_output
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
    series = utils.get_raw_data()
    series = series.transpose(0,2,1)
    series = series.reshape((3600, -1))
    train_samples = int(0.8 * len(series))
    train_data = series[:train_samples]
    test_data = series[train_samples:]

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
        output = model(data)
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

def MAPE(y_true, y_pred):
    epsilon = 1e-7  # 添加一个小的常数，避免除以零的错误
    diff = torch.abs((y_true - y_pred) / (y_true + epsilon))
    mape_value = 100.0 * torch.mean(diff)
    return mape_value

def test(eval_model, data_source, offset):
    eval_model.eval()
    total_loss = 0.
    predictions = torch.zeros(offset - 1, input_dim).to(device)
    real_flow = torch.zeros(offset - 1, input_dim).to(device)
    with torch.no_grad():
        for i in range(0, offset - 1):
            data, target = get_batch(data_source, i, 1)
            output = eval_model(data)
            total_loss += criterion(output, target).item()
            predictions[i] = output[-1, 0].cpu()
            real_flow[i] = target[-1, 0].cpu()

    predictions = predictions.cpu()
    real_flow = real_flow.cpu()
    # plt.plot(predictions[:offset, 0], color="red", label="predictions")
    # plt.plot(real_flow[:offset, 0], color="blue", label="real")
    # plt.grid(True, which='both')
    # plt.axhline(y=0, color='k')
    # plt.legend(loc='upper right')
    # plt.plot()
    # plt.savefig('../graph/gcn_pe/transformer-%d0 minutes.png' % offset)
    # plt.close()
    mae = torch.mean(torch.abs(predictions - real_flow))
    mape = MAPE(real_flow, predictions)
    # mse = torch.mean((predictions - real_flow) ** 2)
    rmse = torch.sqrt(torch.mean((predictions - real_flow) ** 2))
    with open('../indexes/test/Gcn_transform.csv', mode='a', newline='') as file:
        # Create a writer object
        writer = csv.writer(file)

        # Write a new row of data
        writer.writerow([offset, mae, mape, rmse])
    # return total_loss / i

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
    plt.savefig('../graph/gcn_pe/transformer-epoch%d.png' % epoch)
    plt.close()
    mae = torch.mean(torch.abs(predictions - real_flow))
    mse = torch.mean((predictions - real_flow) ** 2)
    rmse = torch.sqrt(torch.mean((predictions - real_flow) ** 2))
    with open('../indexes/Gcn_transform.csv', mode='a', newline='') as file:
        # Create a writer object
        writer = csv.writer(file)

        # Write a new row of data
        writer.writerow([epoch, mae, mse, rmse])
    return total_loss / i


train_data, val_data = get_data()
model = GCN_Transformer(input_dim=input_dim, adj_matrix=edge_index).to(device)
criterion = nn.MSELoss()
lr = 0.005
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
epochs = 70

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)
    if (epoch % 10 is 0):
        val_loss = plot_and_loss(model, val_data, epoch)
    else:
        val_loss = evaluate(model, val_data)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} '.format(epoch, (
            time.time() - epoch_start_time), val_loss))
    print('-' * 89)
    scheduler.step()

torch.save(model.state_dict(), "../model/gcn_transformer.pth")

for i in range(2,8):
    val_loss = test(model, val_data, i)