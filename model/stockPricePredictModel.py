import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from common.constants import BNB, BTC
from dataSpyder.fileProcessor import read_data_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training features
feature_columns = ['o', 'h', 'l', 'c', 'v', 'qav', 'num_trades',
                   'EMA_7', 'EMA_25', 'EMA_99',
                   'MACD', 'Signal_Line', 'MACD_above_Signal', 'MACD_diff_Signal',
                   'RSI',
                   'Middle_Band', 'STD', 'Upper_Band', 'Lower_Band'
                   ]
# targets
target_columns = ['o', 'h', 'l', 'c', 'v', 'qav', 'num_trades']


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
            torch.tensor(self.targets[idx + self.sequence_length], dtype=torch.float)  # Next day's close price
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term_exp = -(math.log(10000.0) / d_model)
        # Adjusting for even and odd d_model values
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * div_term_exp)

        # Ensure the expansion fits both even and odd d_model
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            # d_model is even
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # d_model is odd, adjust div_term to fit the dimension
            pe[:, 1::2] = torch.cos(position * torch.exp(torch.arange(1, d_model, 2).float() * div_term_exp))

        pe = pe.unsqueeze(0)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def generate_square_subsequent_mask(seq_len):
    """Generates a mask to prevent attention to future positions."""
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    return mask


class StockPriceTransformer(nn.Module):
    def __init__(self, feature_size, num_layers, nhead, target_size, dim_feedforward=512, dropout=0.1, max_len=5000):
        super(StockPriceTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.feature_size = feature_size
        self.target_size = target_size
        # Ensure d_model is divisible by nhead
        adjusted_feature_size = nhead * (
                feature_size // nhead + (feature_size % nhead > 0))  # Adjusted to be divisible by nhead
        self.input_adjustment = nn.Linear(feature_size, adjusted_feature_size)  # Adjust feature_size if necessary

        self.pos_encoder = PositionalEncoding(d_model=adjusted_feature_size, dropout=dropout, max_len=max_len)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=adjusted_feature_size, nhead=nhead,
                                                                    dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(adjusted_feature_size, self.target_size)

    def forward(self, src, src_mask):
        src = src.permute(1, 0, 2)
        src = self.input_adjustment(src) * math.sqrt(self.feature_size)  # Adjust input dimensions
        src = self.pos_encoder(src)
        out = self.transformer_encoder(src, src_mask)
        out = self.linear(out)
        out = out.permute(1, 0, 2)
        return out


# class StockPriceTransformer(nn.Module):
#     def __init__(self, feature_size, num_layers, nhead, dim_feedforward=512, dropout=0.1, max_len=5000):
#         super(StockPriceTransformer, self).__init__()
#         self.pos_encoder = PositionalEncoding(d_model=feature_size, dropout=dropout, max_len=max_len)
#         self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead,
#                                                                     dim_feedforward=dim_feedforward, dropout=dropout)
#         self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
#         self.linear = nn.Linear(feature_size, len(target_columns))  # Adjusting output size
#
#     def forward(self, src):
#         global device
#         if self.src_mask is None or self.src_mask.size(0) != len(src):
#             device = src.device
#             mask = self._generate_square_subsequent_mask(len(src)).to(device)
#             self.src_mask = mask
#         src = self.pos_encoder(src)
#         out = self.transformer_encoder(src)
#         out = self.linear(out[-1])
#         return out
#
#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask


def evaluate_and_plot(model, data_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            output = model(features, src_mask)
            predictions.extend(output.cpu().numpy())
            actuals.extend(targets[:, -1, :].cpu().numpy())

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


batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

src_mask = generate_square_subsequent_mask(sequence_length).to(device)

model = StockPriceTransformer(feature_size=len(feature_columns), target_size=len(target_columns), num_layers=3,
                              nhead=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_function = nn.MSELoss()

epochs = 50  # Or however many you feel are necessary

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        output = model(data, src_mask)
        last_output = output[:, -1, :]
        loss = loss_function(last_output, targets)  # Assuming you want to predict the last step
        loss.backward()
        optimizer.step()
        tmp = loss.item()
        total_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

    # if epoch % 10 == 9:  # Every 10 epochs
    #     print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')
    #     torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')
    #     # Evaluation part goes here (covered in the next step)
    #     evaluate_and_plot(model, val_loader)
