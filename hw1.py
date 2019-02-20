import requests
from bs4 import BeautifulSoup
import re

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
    for i in range(num_sim):
        theta, pps, col_prob, delay = get_values(3,2,1)
        list_theta.append(theta)
        list_pps.append(pps)
        list_prob.append(col_prob)
        list_delay.append(delay)

if __name__ == '__main__':

    #print(get_values(1,1,1))