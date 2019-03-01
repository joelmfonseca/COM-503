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

def q3_doubling_access_points(data, access_points=[1,2,4,8]):
    '''Plot the effect of doubling the number of access points only.'''

    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    for ap in access_points:
        ax.plot(data[:,ap-1,0], label='AP={}'.format(ap))
    
    plt.legend()
    plt.xticks(np.linspace(0,1000,21))
    plt.xlabel('Number of requests per second')
    plt.ylabel('Theta')
    plt.grid()
    plt.tight_layout()
    fig.savefig('q3_doubling_ap.png')

def q2_plot_response_values():

    C= 1000
    AP = 1 
    S = 1

    theta_data = []
    pps_data = []
    col_prob_data  = []
    delay_data = []

    for clients in range(1,1000):
        
        theta, pps, col_prob, delay = get_values(clients, AP, S)

        theta_data.append(theta)
        pps_data.append(pps)
        col_prob_data.append(col_prob)
        delay_data.append(delay)

    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    fig1, ax1 = plt.subplots(1, 1, figsize=(10,5))
    fig2, ax2 = plt.subplots(1, 1, figsize=(10,5))
    fig3, ax3 = plt.subplots(1, 1, figsize=(10,5))
   
    ax.plot(theta_data)
    ax1.plot(pps_data)
    ax2.plot(col_prob_data)
    ax3.plot(delay_data)
    
    plt.legend()
    plt.xlabel('load factor C')
    plt.ylabel('Theta')
    plt.tight_layout()
    fig.savefig('theta.png')

    plt.legend()
    plt.xlabel('load factor C')
    plt.ylabel('pps')
    plt.tight_layout()
    fig1.savefig('pps.png')

    plt.legend()
    plt.xlabel('load factor C')
    plt.ylabel('col_proba')
    plt.tight_layout()
    fig2.savefig('col.png')

    plt.legend()
    plt.xlabel('load factor C')
    plt.ylabel('delay')
    plt.tight_layout()
    fig3.savefig('delay.png')

def q3_doubling_servers(power=4):
    '''Plot the effect of doubling the number of servers only.'''

    data = []
    for i in tqdm(range(1,1001)):
        servers_data = []
        for j in range(power):
            servers_data.append(get_values(i,1,2**j)[0])
        data.append(servers_data)

    data = np.array(data)

    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    for i in range(power):
        ax.plot(data[:,i], label='S={}'.format(2**i))
    
    plt.legend()
    plt.xlabel('Number of requests per second')
    plt.ylabel('Theta')
    plt.tight_layout()
    fig.savefig('q3_doubling_s.png')

def q4_generate_all_data():
    '''Generate all data for once.'''

    data = []
    for i in tqdm(range(1,1001)):
        access_points = []
        for j in tqdm(range(1,11)):
            servers = []
            for k in tqdm(range(1,11)):
                servers.append(get_values(i,j,k)[0])
            access_points.append(servers)
        data.append(access_points)

    data = np.array(data)
    np.save('data.npy', data)

def q4_plot_ap_s(data, access_points, servers):
    '''Plot the configuration passed by argument.'''

    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    for ap in access_points:
        for s in servers:
            ax.plot(data[:, ap-1, s-1], label='AP={}'.format(ap))
    
    plt.legend()
    plt.xlabel('Number of requests per second')
    plt.ylabel('Theta')
    plt.xticks(np.linspace(0,1000,21))
    plt.axvline(x=130, linestyle='--', color='b')
    plt.text(125, 1, 'C=130', ha='right', va='bottom',rotation='vertical', color='b')
    plt.axvline(x=260, linestyle='--', color='b')
    plt.text(255, 1, 'C=260', ha='right', va='bottom',rotation='vertical', color='b')
    plt.axvline(x=390, linestyle='--', color='b')
    plt.text(390, 1, 'C=390', ha='right', va='bottom',rotation='vertical', color='b')
    plt.axvline(x=520, linestyle='--', color='b')
    plt.text(520, 1, 'C=520', ha='right', va='bottom',rotation='vertical', color='b')
    # plt.axvline(x=290, linestyle='--', color='r')
    # plt.text(290, 1, 'C=290', ha='right', va='bottom',rotation='vertical', color='r')
    plt.axvline(x=580, linestyle='--', color='r')
    plt.text(580, 1, 'C=580', ha='right', va='bottom',rotation='vertical', color='r')
    plt.grid()
    plt.tight_layout()
    fig.savefig('q4_analysis_s=2.png')

def q4_generete_engineering_rule(num_sim):
    '''Tentative of linear throughput.'''

    er_data_sim = []
    for n in tqdm(range(num_sim)):
        ap=1
        s=1
        er_data = []
        for i in range(1,1001):
            if i%130 == 0:
                ap += 1
                # print('c={}, ap={}, s={}'.format(i, ap, s))
            if i%290 == 0:
                s += 1
                # print('c={}, ap={}, s={}'.format(i, ap, s))
            er_data.append(get_values(i,ap,s)[0])
        er_data_sim.append(er_data)
    
    er_data_sim = np.array(er_data_sim)
    np.save('er_data_sim.npy', er_data_sim)

def q4_brute_force(num_sim):
    '''Get data with max settings.'''
    er_data_sim = []
    for n in tqdm(range(num_sim)):
        er_data = []
        for i in range(1,1001):
            er_data.append(get_values(i,10,10)[0])
        er_data_sim.append(er_data)
    
    er_data_sim = np.array(er_data_sim)
    np.save('er_brute_force.npy', er_data_sim)

def q4_plot(er_data_sim):
    'Plot the engineering rule simulations.'

    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    mean = np.mean(er_data_sim, axis=0)
    std = np.std(er_data_sim, axis=0)
    ax.plot(mean)
    # ax.plot(np.arange(1,1001), np.arange(1,1001), color='g', alpha=0.7)
    ax.fill_between(np.arange(1,1001), mean+std, mean-std, alpha=0.5)
    plt.xlabel('Number of requests per second')
    plt.ylabel('Theta')
    plt.grid()
    plt.tight_layout()
    fig.savefig('q4_engineering_rule.png')

if __name__ == '__main__':

    # q1_plot_variability(num_sim=100)
    # q2_plot_response_values()
    data = np.load('data.npy')
    q3_doubling_access_points(data)
    # q4_plot_ap_s(data=data, access_points=list(range(1,11)), servers=[2])
    # q4_plot_ap_s(data=data, access_points=[1], servers=list(range(1,11)))
    # q4_engineering_rule(num_sim=100)
    # q4_plot(np.load('er_data_sim.npy'))
    # q4_brute_force(10)
    # q4_plot(np.load('er_brute_force.npy'))