import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from typing import Any, List, TypeVar, Iterator, Iterable, Generic, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# some code of transformer was ingored, because it is too long
# you can find it in the original repo, like github
from transformers.model.models import clone_module_list
from transformers.xl.relative_mha import RelativeMultiHeadAttention
from transformers.model.ffn import FeedForward, Module       

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class LSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, layer_norm: bool = False):
        super().__init__()
        self.hidden_lin = nn.Linear(hidden_size, 4 * hidden_size)
        self.input_lin = nn.Linear(input_size, 4 * hidden_size, bias=False)
        if layer_norm:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
            self.layer_norm_c = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.layer_norm_c = nn.Identity()

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        ifgo = self.hidden_lin(h) + self.input_lin(x)
        ifgo = ifgo.chunk(4, dim=-1)
        ifgo = [self.layer_norm[i](ifgo[i]) for i in range(4)]

        i, f, g, o = ifgo
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))

        return h_next, c_next


class HyperLSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, hyper_size: int, n_z: int):
        super().__init__()

        self.hyper = LSTMCell(hidden_size + input_size, hyper_size, layer_norm=True)
        self.z_h = nn.Linear(hyper_size, 4 * n_z)
        self.z_x = nn.Linear(hyper_size, 4 * n_z)
        self.z_b = nn.Linear(hyper_size, 4 * n_z, bias=False)
        d_h = [nn.Linear(n_z, hidden_size, bias=False) for _ in range(4)]
        self.d_h = nn.ModuleList(d_h)
        d_x = [nn.Linear(n_z, hidden_size, bias=False) for _ in range(4)]
        self.d_x = nn.ModuleList(d_x)
        d_b = [nn.Linear(n_z, hidden_size) for _ in range(4)]
        self.d_b = nn.ModuleList(d_b)
        self.w_h = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, hidden_size)) for _ in range(4)])
        self.w_x = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, input_size)) for _ in range(4)])
        self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
        self.layer_norm_c = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor,
                h: torch.Tensor, c: torch.Tensor,
                h_hat: torch.Tensor, c_hat: torch.Tensor):
        x_hat = torch.cat((h, x), dim=-1)
        h_hat, c_hat = self.hyper(x_hat, h_hat, c_hat)
        z_h = self.z_h(h_hat).chunk(4, dim=-1)
        z_x = self.z_x(h_hat).chunk(4, dim=-1)
        z_b = self.z_b(h_hat).chunk(4, dim=-1)

        ifgo = []
        for i in range(4):
            d_h = self.d_h[i](z_h[i])
            d_x = self.d_x[i](z_x[i])

            y = d_h * torch.einsum('ij,bj->bi', self.w_h[i], h) + \
                d_x * torch.einsum('ij,bj->bi', self.w_x[i], x) + \
                self.d_b[i](z_b[i])

            ifgo.append(self.layer_norm[i](y))

        i, f, g, o = ifgo
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))

        return h_next, c_next, h_hat, c_hat


class HyperLSTM(Module):
    def __init__(self, input_size: int, hidden_size: int, hyper_size: int, n_z: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.hyper_size = hyper_size

        self.cells = nn.ModuleList([HyperLSTMCell(input_size, hidden_size, hyper_size, n_z)] +
                                   [HyperLSTMCell(hidden_size, hidden_size, hyper_size, n_z) for _ in
                                    range(n_layers - 1)])

    def forward(self, x: torch.Tensor,
                state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None):
        n_steps, batch_size = x.shape[:2]

        if state is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            h_hat = [x.new_zeros(batch_size, self.hyper_size) for _ in range(self.n_layers)]
            c_hat = [x.new_zeros(batch_size, self.hyper_size) for _ in range(self.n_layers)]
        else:
            (h, c, h_hat, c_hat) = state
            h, c = list(torch.unbind(h)), list(torch.unbind(c))
            h_hat, c_hat = list(torch.unbind(h_hat)), list(torch.unbind(c_hat))

        out = []
        for t in range(n_steps):
            inp = x[t]
            for layer in range(self.n_layers):
                h[layer], c[layer], h_hat[layer], c_hat[layer] = \
                    self.cells[layer](inp, h[layer], c[layer], h_hat[layer], c_hat[layer])
                inp = h[layer]
            out.append(h[-1])

        out = torch.stack(out)
        h = torch.stack(h)
        c = torch.stack(c)
        h_hat = torch.stack(h_hat)
        c_hat = torch.stack(c_hat)

        return out, (h, c, h_hat, c_hat)

class TransformerXLLayer(Module):
    def __init__(self, *,
                 d_model: int,
                 self_attn: RelativeMultiHeadAttention,
                 feed_forward: FeedForward,
                 dropout_prob: float):
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])
        
    def forward(self, *,
                x: torch.Tensor,
                mem: Optional[torch.Tensor],
                mask: torch.Tensor):
        z = self.norm_self_attn(x)
        if mem is not None:
            mem = self.norm_self_attn(mem)
            m_z = torch.cat((mem, z), dim=0)
        else:
            m_z = z
        self_attn = self.self_attn(query=z, key=m_z, value=m_z, mask=mask)
        x = x + self.dropout(self_attn)
        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)
        return x
    
class TransformerXL(Module):
    def __init__(self, layer: TransformerXLLayer, n_layers: int):
        super().__init__()
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm([layer.size])
        
    def forward(self, x: torch.Tensor, mem: List[torch.Tensor], mask: torch.Tensor):
        new_mem = []
        for i, layer in enumerate(self.layers):
            new_mem.append(x.detach())
            m = mem[i] if mem else None
            x = layer(x=x, mem=m, mask=mask)
        return self.norm(x), new_mem
    
class CNNHyperLSTMTransformerXL(nn.Module):
    def __init__(self, cnn_params: dict, hyperlstm_params: dict, transformerxl_params: dict, output_size: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=cnn_params['in_channels'],
                      out_channels=cnn_params['out_channels'],
                      kernel_size=cnn_params['kernel_size']),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=cnn_params['pool_kernel_size']),
        )
        self.hyperlstm = HyperLSTM(
            input_size=hyperlstm_params['input_size'],
            hidden_size=hyperlstm_params['hidden_size'],
            hyper_size=hyperlstm_params['hyper_size'],
            n_z=hyperlstm_params['n_z'],
            n_layers=hyperlstm_params['n_layers']
        )
        self.transformerxl = TransformerXL(
            layer=transformerxl_params['layer'],
            n_layers=transformerxl_params['n_layers']
        )
        self.fc = nn.Linear(transformerxl_params['d_model'], output_size)
    
    def forward(self, x: torch.Tensor, 
                hyperlstm_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None, 
                transformerxl_mem: Optional[List[torch.Tensor]] = None, 
                mask: Optional[torch.Tensor] = None):
        x = self.cnn(x)
        x = x.permute(2, 0, 1)
        x, hyperlstm_state = self.hyperlstm(x, hyperlstm_state)
        x, transformerxl_mem = self.transformerxl(x, transformerxl_mem, mask)
        x = x.permute(1, 0, 2)
        output = self.fc(x)
        return output, hyperlstm_state, transformerxl_mem

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

    cnn_params = {
        'in_channels': 32,
        'out_channels': 16,
        'kernel_size': 3,
        'pool_kernel_size': 2
    }

    hyperlstm_params = {
        'input_size': 16,
        'hidden_size': 128,
        'hyper_size': 64,
        'n_z': 32,
        'n_layers': 2
    }

    transformerxl_layer = TransformerXLLayer(
        d_model=128,
        self_attn=RelativeMultiHeadAttention(heads=4, d_model=128, dropout_prob=0.1),
        feed_forward=FeedForward(d_model=128, d_ff=256, dropout=0.1),
        dropout_prob=0.1
    )

    transformerxl_params = {
        'layer': transformerxl_layer,
        'n_layers': 2,
        'd_model': 128
    }

    model = CNNHyperLSTMTransformerXL(
        cnn_params=cnn_params,
        hyperlstm_params=hyperlstm_params,
        transformerxl_params=transformerxl_params,
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
            output, _, _ = model(data)
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
                output, _, _ = model(data)
                output = output[:, -1, :]
                loss = criterion(output.squeeze(), target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    torch.save(model.state_dict(), 'CNN+HyperLSTM+Transformer-XL.pth')