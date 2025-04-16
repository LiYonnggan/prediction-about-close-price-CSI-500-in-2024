import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# some code of transformer was ingored, because it is too long
# you can find it in the original repo, like github
from model.positional_encoding import PositionalEncoding
from model.models import  EncoderDecoder, Encoder, Decoder, TransformerLayer, Generator
from model.mha import MultiHeadAttention
from model.ffn import FeedForward      

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class TransformerTimeSeries(nn.Module):
    def __init__(self, n_vocab, d_model=128, n_layers=3, n_heads=8, dropout=0.1, d_ff=256):
        super().__init__()
        self_attn = MultiHeadAttention(n_heads, d_model, dropout_prob=dropout)
        src_attn = MultiHeadAttention(n_heads, d_model, dropout_prob=dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout=dropout)
        transformer_layer = TransformerLayer(
            d_model=d_model,
            self_attn=self_attn,
            src_attn=src_attn,
            feed_forward=feed_forward,
            dropout_prob=dropout
        )
        encoder = Encoder(transformer_layer, n_layers)
        decoder = Decoder(transformer_layer, n_layers)
        src_embed = nn.Sequential(
            nn.Linear(n_vocab, d_model),
            PositionalEncoding(d_model, dropout)
        )
        tgt_embed = nn.Sequential(
            nn.Linear(n_vocab, d_model),
            PositionalEncoding(d_model, dropout)
        )
        generator = Generator(1, d_model)
        self.model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.model(src, tgt, src_mask, tgt_mask)


# Main function to run the model
# the value of some parameters are set to default values, you can change them according to your needs
# at the same time, you can plt some figures to see the performance of the model
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_excel("new.xlsx")

    y = data['close'].values
    X = data.drop(['close'], axis=1).values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).reshape(-1)

    seq_len = 32
    step = 1

    def create_sequences(X, y, seq_len, step=1):
        X_seq = []
        y_seq = []
        for i in range(0, len(X) - seq_len, step):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)

    X, Y = create_sequences(X, y, seq_len, step)

    n_vocab = X.shape[2]
    model = TransformerTimeSeries(n_vocab).to(device)

    dataset = TensorDataset(X.to(device), Y.to(device))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4) 
    
    num_epochs = 100
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data, data, None, None)
            output = output[:, -1, :]
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data, data, None, None)
                output = output[:, -1, :]
                loss = criterion(output.squeeze(), target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    torch.save(model.state_dict(), 'CNN+HyperLSTM+Transformer-XL.pth')