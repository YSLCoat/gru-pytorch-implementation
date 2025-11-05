import torch
import torch.nn as nn


class ResetGate(nn.Module):
    def __init__(self, input_dim, hidden_dim, sigma):
        super().__init__()
        self.W_xr = nn.Parameter(torch.randn(input_dim, hidden_dim)*sigma, requires_grad=True)
        self.W_hr = nn.Parameter(torch.randn(hidden_dim, hidden_dim)*sigma, requires_grad=True)
        self.b_r = nn.Parameter(torch.randn(1, hidden_dim)*sigma, requires_grad=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, previous_hidden_state):
        return self.sigmoid(torch.matmul(x, self.W_xr) + torch.matmul(previous_hidden_state, self.W_hr) + self.b_r)
    

class UpdateGate(nn.Module):
    def __init__(self, input_dim, hidden_dim, sigma):
        super().__init__()
        self.W_xu = nn.Parameter(torch.randn(input_dim, hidden_dim)*sigma, requires_grad=True)
        self.W_hu = nn.Parameter(torch.randn(hidden_dim, hidden_dim)*sigma, requires_grad=True)
        self.b_u = nn.Parameter(torch.randn(1, hidden_dim)*sigma, requires_grad=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, previous_hidden_state):
        return self.sigmoid(torch.matmul(x, self.W_xu) + torch.matmul(previous_hidden_state, self.W_hu) + self.b_u)
    

class CandidateHiddenState(nn.Module):
    def __init__(self, input_dim, hidden_dim, sigma):
        super().__init__()
        self.W_xh = nn.Parameter(torch.randn(input_dim, hidden_dim)*sigma, requires_grad=True)
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim)*sigma, requires_grad=True)
        self.b_h = nn.Parameter(torch.randn(1, hidden_dim)*sigma, requires_grad=True)

        self.tanh = nn.Tanh()

    def forward(self, x, previous_hidden_state, reset_gate_output):
        return self.tanh(torch.matmul(x, self.W_xh) + torch.matmul(reset_gate_output*previous_hidden_state, self.W_hh) + self.b_h)
    

class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, sigma=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.reset_gate = ResetGate(input_dim, hidden_dim, sigma)
        self.update_gate = UpdateGate(input_dim, hidden_dim, sigma)
        self.candiate_hidden_state = CandidateHiddenState(input_dim, hidden_dim, sigma)

    def forward(self, x, previous_hidden_state):
        reset_gate_output = self.reset_gate(x, previous_hidden_state)
        update_gate_output = self.update_gate(x, previous_hidden_state)
        candidate_hidden_state = self.candiate_hidden_state(x, previous_hidden_state, reset_gate_output)

        updated_hidden_state = update_gate_output*previous_hidden_state + (1 - update_gate_output)*candidate_hidden_state
        return updated_hidden_state