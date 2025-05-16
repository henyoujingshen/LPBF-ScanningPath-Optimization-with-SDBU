import torch
from torch import nn
from .basic_module import BasicModule

##一个rnn_dnn的变种
class rnnDNN_final(BasicModule):
    def __init__(self, input_size=25, hidden1_size=50, hidden2_size=35, output_size=25):
        super(rnnDNN_final, self).__init__()
        self.model_name = 'rnnDNN_final'
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size

        self.lstm = nn.LSTM(input_size, hidden1_size,bidirectional = True)
        self.linear1 = nn.Linear(25*hidden1_size, 25*hidden2_size)
        self.linear2 = nn.Linear(25*hidden2_size, 25*output_size)

    def forward(self, input_seq):
        input_seq = input_seq.permute(1, 0, 2)

        #input_seq = input_seq.squeeze()
        #print(input_seq.shape)
        lstm_out1, self.hidden_cell = self.lstm(input_seq)
        print(lstm_out1.shape)

        lstm_out1=lstm_out1[:, :, :50]

        lstm_out1 = lstm_out1.permute(1, 0, 2)

        lstm_out1 = lstm_out1.reshape(lstm_out1.size(0), -1)
        print(lstm_out1.shape)
        out2 = self.linear1(lstm_out1)
        predictions = self.linear2(out2)
        predictions = predictions.reshape(lstm_out1.size(0), 25,25)
        # predictions = predictions.view(predictions.size(0), 1,50, 50)
        # predictions = predictions.permute(1, 0, 2, 3)
        #print(predictions.shape)
        return predictions