import torch
from torch import nn
from .basic_module import BasicModule

#原始rnn_dnn
class rnnDNN(BasicModule):
    def __init__(self, input_size=50, hidden1_size=500, hidden2_size=200, output_size=2500):
        super(rnnDNN, self).__init__()
        self.model_name = 'rnnDNN'
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size

        self.lstm = nn.LSTM(input_size, hidden1_size)
        self.linear1 = nn.Linear(1 * hidden1_size, hidden2_size)
        self.linear2 = nn.Linear(hidden2_size, output_size)

    def forward(self, input_seq):
        input_seq = input_seq.squeeze()
        #print(input_seq.shape)
        lstm_out1, self.hidden_cell = self.lstm(input_seq)
        lstm_out1 = lstm_out1.squeeze(dim=1)
        #print(lstm_out1.shape)
        lstm_out1 = lstm_out1[:, -1, :]
        lstm_out1 = lstm_out1.view(lstm_out1.size(0), -1)

        out2 = self.linear1(lstm_out1)
        predictions = self.linear2(out2)
        predictions = predictions.view(predictions.size(0), 1, 50, 50)
        return predictions