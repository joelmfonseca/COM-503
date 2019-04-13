import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, batch_size, learning_rate):
        super(LSTM, self).__init__()
        self.input_size = 168
        self.hidden_size = 128
        self.num_layers = 2

        self.cell = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.2,
            batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)

        self.criterion = nn.MSELoss()
        self.criterion_report = nn.L1Loss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate, momentum=0.9)

    def forward(self, input):
        batch_size = input.size(0)
        output, _ = self.cell(input.view(-1, batch_size, self.input_size))
        output = self.fc(output.view(batch_size, self.hidden_size))

        return output
    
    def step(self, input, target, predict=False, scaler=None):
        if not predict:
            self.train()
            self.zero_grad()
            output = self.forward(input)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            return loss.data.item()
        else:
            self.eval()
            output = self.forward(input)
            mse_loss = self.criterion(output, target)
            mae_loss = self.criterion_report(output, target)

            return mse_loss.data.item(), mae_loss.data.item()

    def predict(self, input, look_ahead):
        self.eval()
        predictions = []
        for i in range(look_ahead):
            len_input = input.size(1)
            pred = self.forward(input)
            predictions.append(pred)
            input = input[:, 1:].view(-1, len_input-1)
            input = torch.cat((input, pred), 1).view(-1, len_input, 1)
        
        return torch.cat(predictions, 1)