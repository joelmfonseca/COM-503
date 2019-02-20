import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def get_values(clients, apoints, servers, sciper=217575):
    '''Return the output values of Joe's Shop Simulator.'''
    
    URL='http://tcpip.epfl.ch/output.php'
    params = {'sciper':str(sciper), 'clients':str(clients), 'apoints':str(apoints), 'servers':str(servers)}
    r = requests.post(url=URL, data=params)
    soup = BeautifulSoup(r.text, 'html.parser')

    theta = float(soup.find('td', text=re.compile(r'Theta')).find_next_sibling('td').get_text())
    pps = float(soup.find('td', text=re.compile(r'Packets per second')).find_next_sibling('td').get_text())
    col_prob = float(soup.find('td', text=re.compile(r'Collision probability')).find_next_sibling('td').get_text())
    delay = float(soup.find('td', text=re.compile(r'Delay')).find_next_sibling('td').get_text())

    return theta, pps, col_prob, delay

def q1_plot_variability(num_sim):
    '''Plot variability of output parameters for same input configuration.'''

    list_theta = []
    list_pps = []
    list_prob = []
    list_delay = []
    for i in tqdm(range(num_sim)):
        theta, pps, col_prob, delay = get_values(3,2,1)
        list_theta.append(theta)
        list_pps.append(pps)
        list_prob.append(col_prob)
        list_delay.append(delay)

    fig, axes = plt.subplots(2, 2, figsize=(10,10))
    axes[0][0].set_title('Theta (successful downloads per second)')
    axes[0][0].boxplot(list_theta, notch=True)
    axes[0][0].get_xaxis().set_visible(False)
    axes[0][1].set_title('Packets per second')
    axes[0][1].boxplot(list_pps, notch=True)
    axes[0][1].get_xaxis().set_visible(False)
    axes[1][0].set_title('Collision probability')
    axes[1][0].boxplot(list_prob, notch=True)
    axes[1][0].get_xaxis().set_visible(False)
    axes[1][1].set_title('Delay [s]')
    axes[1][1].boxplot(list_delay, notch=True)
    axes[1][1].get_xaxis().set_visible(False)
    
    plt.tight_layout()
    fig.savefig('q1_variability.png')

def q3_doubling_access_points(power=4):
    '''Plot the effect of doubling the number of access points only.'''

    data = []
    for i in tqdm(range(1,1001)):
        access_points_data = []
        for j in range(power):
            access_points_data.append(get_values(i,1*2**j,1)[0])
        data.append(access_points_data)

    data = np.array(data)

    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    for i in range(power):
        ax.plot(data[:,i], label='AP={}'.format(2**i))
    
    plt.legend()
    plt.xlabel('Number of requests per second')
    plt.ylabel('Theta')
    plt.tight_layout()
    fig.savefig('q3_doubling_ap.png')

if __name__ == '__main__':

    #q1_plot_variability(num_sim=100)
    q3_doubling_access_points()