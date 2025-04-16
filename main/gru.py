import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from typing import Optional, Tuple
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_lin = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.hidden_lin = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.n_lin = nn.Linear(input_size, hidden_size, bias=False)
    def forward(self, x: torch.Tensor, h: torch.Tensor):
        gates = self.input_lin(x) + self.hidden_lin(h)
        r, z, _ = gates.chunk(3, dim=-1)

        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(self.n_lin(x) + r * h)

        h_next = (1 - z) * n + z * h

        return h_next

class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList([GRUCell(input_size, hidden_size)] +
                                   [GRUCell(hidden_size, hidden_size) for _ in range(n_layers - 1)])

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        n_steps, batch_size = x.shape[:2]

        if state is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
        else:
            h = list(torch.unbind(state))

        out = []
        for t in range(n_steps):
            inp = x[t]
            for layer in range(self.n_layers):
                h[layer] = self.cells[layer](inp, h[layer])
                inp = h[layer]
            out.append(h[-1])

        out = torch.stack(out)
        h = torch.stack(h)

        return out, h
    
class MAINGRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout, output_size):
        super(MAINGRU, self).__init__()
        self.gru = GRU(input_size, hidden_size, n_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, state=None):
        x, state = self.gru(x, state)
        x = self.dropout(x)
        x = self.fc(x)
        return x, state


# Main function to run the model
# the value of some parameters are set to default values, you can change them according to your needs
# at the same time, you can plt some figures to see the performance of the model
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_excel("dataset.xlsx")

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

    dataset = TensorDataset(X.to(device), Y.to(device))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = MAINGRU(
        input_size=X.shape[2],
        hidden_size=128,
        n_layers=2,
        dropout=0.5,
        output_size=1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) 
    
    num_epochs = 100
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
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
                output, _ = model(data)
                output = output[:, -1, :]
                loss = criterion(output.squeeze(), target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')