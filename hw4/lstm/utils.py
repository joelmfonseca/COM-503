import torch
import numpy as np
from random import sample

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

def plot_prediction(target, predictions, residuals, num_obs=168, show_ci=False):

    def get_ci(prediction, residuals, num_samples=999, level=0.95):
        lower_bound = int(np.floor((num_samples+1)*(1-level)/2))
        upper_bound = int(np.ceil((num_samples+1)*(1-((1-level)/2))))
        predictions = []
        for i in range(num_samples):
            random_residual = sample(list(residuals), 1)[0]
            predictions.append(prediction+random_residual)
        predictions.sort()
    
        return predictions[lower_bound], predictions[upper_bound]

    def get_preds_and_cis(residuals, predictions, num_obs):
        confidence_intervals = []
        for i in range(num_obs):
            conf_inter = get_ci(prediction[i][0], residuals)
            confidence_intervals.append(conf_inter)
        return confidence_intervals

    fig, ax = plt.subplots(1, 1, figsize=(15,5))

    # prepare array for plots
    x = np.arange(num_obs)
    
    if show_ci:
        # get the confidence intervals
        confidence_intervals = get_preds_and_cis(residuals, predictions, num_obs)
        lower_bounds, upper_bounds = list(zip(*confidence_intervals))

        ax.fill_between(x, lower_bounds, upper_bounds, color='r', alpha=0.3, label='95% level')
    
    # plot target prediction
    ax.scatter(x, target[:num_obs], marker='o', facecolors='None', edgecolors='b')
    ax.plot(target[:num_obs], color='b', linewidth=0.5, label='target')
    
    #plot predictions
    ax.plot(predictions[:num_obs], c='r', label='prediction')
    
    # plot details
    ax.set_xlabel('Time')
    ax.set_ylabel('Amount')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pred_lstm.png', dpi=300)
    plt.show()