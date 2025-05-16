import torch
from torch import nn
from .basic_module import BasicModule
from .AttentionModule import AttentionModule

##rnn_dnn变种，增加了注意力区域
class rnn_atb(BasicModule):
    def __init__(self, input_size=50 * 50, hidden1_size=2500, hidden2_size=3000, output_size=2500, dropout_p=0):
        super(rnn_atb, self).__init__()
        self.model_name = 'rnn_atb'
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size

        # self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden1_size, nhead=8), num_layers=6)
        self.AttentionModule = AttentionModule(input_size, 2500)
        self.lstm = nn.LSTM(50, hidden1_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear1 = nn.Linear(1 * hidden1_size, hidden2_size)
        self.linear2 = nn.Linear(hidden2_size, output_size)

    def forward(self, input_seq):
        input_seq = input_seq.reshape(input_seq.shape[0], input_seq.shape[1], -1)
        print(input_seq.shape)
        input_seq = self.AttentionModule(input_seq)
        input_seq = input_seq.view(-1, 50, 50)
        # transformer_out = self.transformer_encoder(input_seq)
        lstm_out1, self.hidden_cell = self.lstm(input_seq)
        lstm_out1 = lstm_out1.squeeze(dim=1)
        lstm_out1 = lstm_out1[:, -1, :]
        lstm_out1 = lstm_out1.view(lstm_out1.size(0), -1)

        lstm_out1 = self.dropout(lstm_out1)
        out2 = self.linear1(lstm_out1)
        out2 = self.dropout(out2)
        predictions = self.linear2(out2)
        predictions = predictions.view(predictions.size(0), 1, 50, 50)
        return predictions