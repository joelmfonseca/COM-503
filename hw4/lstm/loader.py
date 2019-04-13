import torch
import torch.utils.data as data

import scipy.io
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils import split_train_test, create_train_samples, create_test_samples

class TrainingSet(data.Dataset):
    def __init__(self, mat, ratio_test, look_back, look_ahead):

        train, _ = split_train_test(mat, ratio_test)
        data, target = create_train_samples(train, look_back)

        self.X = torch.from_numpy(data).float()
        self.Y = torch.from_numpy(target).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

class TestSet(data.Dataset):
    def __init__(self, mat, ratio_test, look_back, look_ahead):

        _, test = split_train_test(mat, ratio_test)
        data, target = create_test_samples(test, look_back, look_ahead)
        
        self.X = torch.from_numpy(data).float()
        self.Y = torch.from_numpy(target).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

def build_loader(ratio_test, look_back, look_ahead, batch_size):

    scaler = MinMaxScaler(feature_range=(0,1))

    mat = scipy.io.loadmat('../PowerValuesOctMay.mat')['MeasurementsTotal']
    mat = np.swapaxes(mat, 0, 1)
    mat = scaler.fit_transform(mat)

    training_set = TrainingSet(mat, ratio_test, look_back, look_ahead) 
    test_set = TestSet(mat, ratio_test, look_back, look_ahead)

    train_loader = data.DataLoader(
        dataset=training_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
        )

    test_loader = data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
        )

    return train_loader, test_loader, scaler