import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from loader import build_loader
from model import LSTM
from utils import snapshot, plot_prediction

# parameters
test_ratio = 0.13 # to get exactly one test sample based on how we built test samples
batch_size = 10

learning_rate = 0.001
look_back = 168
look_ahead = 574

train_loader, test_loader, scaler = build_loader(test_ratio, look_back, look_ahead, batch_size)
model = LSTM(batch_size, learning_rate)

resume_training = True
if resume_training:
    # load previous model
    checkpoint = torch.load('saved_models/lstm_adam_b10_lb168_model')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
else:
    epoch = 0
    loss = np.inf

train = False
if train:
    best_loss = (loss, epoch)
    patience = 20
    still_learning = True
    while still_learning:

        # train
        losses = []
        for data, target in tqdm((train_loader), leave=False):
            loss = model.step(Variable(data), Variable(target))
            losses.append(loss)

        mean_loss = np.mean(losses)
        print('epoch: {}, loss: {:.5f} ({:.5f},{})'.format(epoch, mean_loss, best_loss[0], epoch-best_loss[1]))

        if mean_loss < best_loss[0]:
            best_loss = (mean_loss, epoch)

            # save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'loss': mean_loss,
            }, 'saved_models/lstm_adam_b10_lb168_model')
        elif epoch - best_loss[1] > patience:
            still_learning = False
        epoch += 1

# get just one sample for prediction
data_pred, target_pred = next(iter(test_loader))
#print('data_pred shape: {}, target_pred shape: {}'.format(data_pred.shape, target_pred.shape))

# prepare data for comparison
pred = model.predict(Variable(data_pred), look_ahead)
target_scaled = scaler.inverse_transform(target_pred[0,:].detach().view(-1, 1).numpy())
pred_scaled = scaler.inverse_transform(pred[0,:].detach().view(-1, 1).numpy())

# plot prediction
plot_prediction(target_scaled, pred_scaled)

# compute MSE and MAE
mse_losses = []
mae_losses = []
for data, target in train_loader:
    mse_loss, mae_loss = model.step(data, target, predict=True, scaler=scaler)
    mse_losses.append(mse_loss)
    mae_losses.append(mae_loss)

mse_train = np.mean(mse_losses)
mae_train = np.mean(mae_losses)

print('Train (log): MSE={:.3f}, MAE={:.3f}'.format(mse_train, mae_train))
print('Test:        MSE={:.3f}, MAE={:.3f}'.format(mean_squared_error(pred_scaled, target_scaled), mean_absolute_error(pred_scaled, target_scaled)))
