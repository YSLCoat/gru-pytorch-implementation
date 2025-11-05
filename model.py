import torch
import torch.nn as nn

from gru import GRUCell

class SingleLayerGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gru = GRUCell(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden_state = torch.zeros((x.shape[0], self.hidden_dim), device=x.device)

        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            hidden_state = self.gru(x_t, hidden_state)

        output = self.output_layer(hidden_state)
        return output
    

class MultiLayerGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.gru_layers = nn.ModuleList()
        self.gru_layers.append(GRUCell(input_dim, hidden_dim))

        for _ in range(num_layers-1):
            self.gru_layers.append(GRUCell(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        hidden_states = [torch.zeros((x.shape[0], self.hidden_dim), device=x.device) for _ in range(self.num_layers)]

        for t in range(x.shape[1]):
            x_t = x[:, t, :]

            for layer in range(self.num_layers):
                gru_layer = self.gru_layers[layer]
                previous_hidden_state = hidden_states[layer]

                hidden_state = gru_layer(x_t, previous_hidden_state)
                hidden_states[layer] = hidden_state

                x_t = hidden_state 

        output = self.output_layer(hidden_states[-1])
        return output

