import torch
import numpy as np

import matplotlib.pyplot as plt

def snapshot(saved_model_dir, run_name, state_dict):
	"""Saves the model state."""
	
	# Write the full name
	complete_name = saved_model_dir + run_name
	
	# Save the model
	with open(complete_name + '.pt', 'wb') as f:
		torch.save(state_dict, f)

def create_train_samples(dataset, look_back):
    data = []
    target = []
    size = dataset.shape[0]
    for i in range(size - look_back - 1):
        history = dataset[i:i+look_back, :]
        prediction = dataset[i+1+look_back, :]
        data.append(history)
        target.append(prediction)

    return np.array(data), np.array(target)

def create_test_samples(dataset, look_back, look_ahead):
    data = []
    target = []
    size = dataset.shape[0]
    for i in range(size - look_back - look_ahead):
        history = dataset[i:i+look_back, :]
        prediction = dataset[i+1+look_back:i+1+look_back+look_ahead, :]
        data.append(history)
        target.append(prediction)

    return np.array(data), np.array(target)

def split_train_test(dataset, ratio_test):
    size = dataset.shape[0]
    num_test_samples = int(size*ratio_test)
    train_samples = dataset[:-num_test_samples, :]
    test_samples = dataset[-num_test_samples:, :]
    return train_samples, test_samples

def plot_prediction(target, prediction, num_obs=168):
    fig, ax = plt.subplots(1, 1, figsize=(20,5))
    ax.plot(target[:num_obs], c='b', label='target')
    ax.plot(prediction[:num_obs], c='r', label='prediction')
    #ax.set_title('Energy consumption')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amount')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pred_lstm.png', dpi=300)