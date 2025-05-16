import torch
import torch.nn as nn


class AttentionModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionModule, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # Calculate the attention weights
        print(x.shape)
        attn_weights = self.linear2(torch.tanh(self.linear1(x)))
        print(attn_weights.shape)
        # Normalize the attention weights
        attn_weights = attn_weights.softmax(dim=1)
        print(attn_weights.shape)
        # Multiply each input by its attention weight
        weighted_inputs = x * attn_weights
        # Sum the weighted inputs along the sequence dimension
        print(weighted_inputs.shape)
        output = torch.sum(weighted_inputs, dim=1)
        return output