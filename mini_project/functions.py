import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn import decomposition
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import array
from tqdm import tqdm_notebook as tqdm
from math import pi
import random
random.seed(2019)
from random import sample

# main variables
K_cluster_altitude = 10
K_cluster_wind = 8

def plot_elbow_method(df, search_space=range(2,11)):
   
    X = df.values
    kmeans = [KMeans(i) for i in search_space]
    scores = [kmeans[i].fit(X).inertia_ for i in range(len(kmeans))]

    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    ax.plot(search_space, scores)
    plt.title('Elbow method for optimal K')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distances')
    plt.yticks([])
    plt.grid()
    plt.show()

######################### Faulty Sensor Creation ##############################

def plot_zsbn_sensor(df):
    fig, ax = plt.subplots(figsize=(15,5))
    ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)
    mask = df['LocationName'] == 'ZSBN'
    df[mask]['CO2'].plot(ax=ax, color='b', label='ZSBN measurement')
    plt.axvline(x=pd.to_datetime('2017-10-24 00:00:00'), color='k', linestyle='--', label='2017-10-24 00:00:00')
    plt.legend()
    plt.ylabel('CO2 [ppm]')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig('figures/zsbn_sensor.png', dpi=300)

def generate_faulty_sensors(df, sensors, failure_date=24):
    for sensor in sensors:
        co2_measurments = df[df['LocationName'] == sensor]['CO2'].copy()
        df.loc[df['LocationName'] == sensor,'groundtruth'] = co2_measurments.copy()
        condition = co2_measurments.index.day >= failure_date
        std = np.std(co2_measurments.values)
        drift = random.randint(30, 60)
        sign = random.sample([-1, 1], 1)[0]
        pourcentage = random.uniform(0.2, 0.3)
        co2_measurments.loc[condition] = (co2_measurments.loc[condition] - sign*drift)/std/pourcentage
        df.loc[df['LocationName'] == sensor, 'CO2'] = co2_measurments
        
def plot_faulty_sensor(df, sensor):
    fig, ax = plt.subplots(figsize=(15,5))
    df[df['LocationName'] == sensor][['CO2', 'groundtruth']].plot(ax=ax, color=['g', 'b'])
    plt.title('Example of inaccurate measurement generation for {} sensor'.format(sensor))
    plt.xlabel('Date')
    plt.ylabel('CO2 [ppm]')
    plt.legend(['artifical measurement', 'true measurement'])
    plt.show()

######################### Performance Evaluation ##############################

def compute_regs(df, mask, X_train, y_train, X_test, approach):
    
    models = {'LinearRegression_prediction':LinearRegression(), 'Lasso_prediction':Lasso(),
              'Ridge_prediction':Ridge(), 'ElasticNet_prediction':ElasticNet()}
    
    aic_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        if(approach != 'wt') :
            y_pred = model.predict(X_test)
            y_pred_name = '{}_'.format(approach) + name
            df.loc[mask, y_pred_name] = y_pred
            
            #save number of parameters per model to compute aic later on    
            aic_models[name] =  model.coef_.reshape(1,-1).shape[1]
            
    aic[approach] = aic_models
        
    return models


######################### Regressions ##############################

def naive_regression(df, faulty_sensors, failure_date=24):
    
    features = ['temperature','humidity']
    target = ['CO2']
    sensor_dict_result = {}
    for sensor in faulty_sensors:
        
        write_mask = df['LocationName'] == sensor
        train_mask = write_mask & (df.index.day < failure_date)
        test_mask = write_mask
        res_mask = test_mask & (df.index.day >= failure_date)
        
        X_train = df[train_mask][features].values
        y_train = df[train_mask][target].values
        X_test = df[test_mask][features].values
        
        compute_regs(df, write_mask, X_train, y_train, X_test, 'naive', )
        compute_ci(df, train_mask, res_mask, 'naive', 'LinearRegression')

def za_regression(df, df_altitude, df_metadata, faulty_sensors, failure_date=24):
    
    features = ['temperature', 'humidity']
    for sensor in faulty_sensors:
        
        # aggregate sensors from same zone and cluster altitude
        zone = df_metadata[df_metadata['LocationName'] == sensor]['zone']
        sensors_same_zone = df_metadata[df_metadata['zone'] == zone.values[0]]['LocationName']
        altitude_cluster_id = df_altitude.loc[sensor, 'altitude_cluster']
        sensors_same_altitude_cluster = df_altitude[df_altitude['altitude_cluster'] == altitude_cluster_id].index
        sensors = set(sensors_same_zone).intersection(set(sensors_same_altitude_cluster))
        print(sensor, sensors)
        
        # prepare masks for train and test
        healthy_mask = df['LocationName'].isin(sensors) & ~df['LocationName'].isin(faulty_sensors)
        nonhealthy_mask = df['LocationName'].isin(sensors) & df['LocationName'].isin(faulty_sensors) & (df.index.day < failure_date)
        
        train_mask = healthy_mask | nonhealthy_mask
        test_mask = (df['LocationName'] == sensor)
        write_mask = test_mask
        
        # create train data
        X_train = df.loc[train_mask][features].values
        y_train = df.loc[train_mask][['CO2']].values
        
        # create test data
        X_test = df.loc[test_mask][features].values
        y_test = df.loc[test_mask][['CO2']].values
        
        compute_regs(df, write_mask, X_train, y_train, X_test, 'za')

def wt_regression(df, df_altitude, df_metadata, faulty_sensors, failure_date=24):
    for sensor in faulty_sensors:  
        
        # take sensor from same zone and cluster altitude
        sensor_zone = df_metadata[df_metadata['LocationName'] == sensor]['zone']
        zone_sensors = df_metadata[df_metadata['zone'] == sensor_zone.values[0]]['LocationName']
        sensor_altitude_cluster = df_altitude.loc[sensor, 'altitude_cluster']
        altitude_sensors = df_altitude[df_altitude['altitude_cluster'] == sensor_altitude_cluster].index
        sensors = set(zone_sensors).intersection(set(altitude_sensors))

        # create a model for each wind cluster
        df_for_reg_models = df[(df['LocationName'].isin(sensors)) & (~df['LocationName'].isin(faulty_sensors))]
        if df_for_reg_models.shape[0] == 0:
            df_for_reg_models = df[(df['LocationName'].isin(sensors))]# & (df.index.day < failure_date)]
        
        models = []
        
        models_ridge = []
        models_elastic_net = []
        models_lasso = []
        
        cluster_ids = df_for_reg_models['wind_cluster'].unique().tolist()
        cluster_ids.sort()
        for cluster_id in cluster_ids:
            wind_cluster_data = df_for_reg_models[(df_for_reg_models['wind_cluster'] == cluster_id)]

            X = wind_cluster_data[['temperature', 'humidity', 'time']].values
            y = wind_cluster_data[['CO2']].values
            
            models_ = compute_regs([], [], X, y, [], 'wt')
            
            reg = models_['LinearRegression_prediction']
            reg_ridge = models_['Ridge_prediction']
            reg_elastic = models_['ElasticNet_prediction']
            reg_lasso = models_['Lasso_prediction']
            
            models.append(reg)
            models_ridge.append(reg_ridge)
            models_elastic_net.append(reg_elastic)
            models_lasso.append(reg_lasso)
            
            
        #print('--')
        sensor_data = df[df['LocationName'] == sensor]
        for cluster_id in sensor_data['wind_cluster'].unique().tolist():
            mask = (df['LocationName'] == sensor) & (df['wind_cluster'] == cluster_id)
            
            reg = models[cluster_id]
            reg_ridge = models_ridge[cluster_id]
            reg_elastic = models_elastic_net[cluster_id]
            reg_lasso = models_lasso[cluster_id]

            X = wind_cluster_data[['temperature', 'humidity', 'time']].values
            
            y_pred = reg.predict(X)
            y_pred_ridge = reg_ridge.predict(X)
            y_pred_elastic = reg_elastic.predict(X)
            y_pred_lasso = reg_lasso.predict(X)
            
            df.loc[mask, 'wt_LinearRegression_prediction'] = y_pred
            df.loc[mask, 'wt_Ridge_prediction'] = y_pred_ridge
            df.loc[mask, 'wt_ElasticNet_prediction'] = y_pred_elastic
            df.loc[mask, 'wt_Lasso_prediction'] = y_pred_lasso
            
            print(reg.coef_)
            aic['wt'] = {'LinearRegression_prediction': reg.coef_.shape[1],
                         'Lasso_prediction': reg_lasso.coef_.reshape(1,-1).shape[1],
                         'Ridge_prediction':  reg_ridge.coef_.reshape(1,-1).shape[1],
                         'ElasticNet_prediction':  reg_elastic.coef_.reshape(1,-1).shape[1]}
            
        
        train_mask = (df['LocationName'] == sensor) & (df.index.day < failure_date)
        res_mask = (df['LocationName'] == sensor) & (df.index.day >= failure_date)
        #compute_ci(df, train_mask, res_mask, 'wt', 'LinearRegression')

def brute_force(df, faulty_sensors, failure_date=24):
    
    correct_sensors_mask = (~df['LocationName'].isin(faulty_sensors)) | \
                            (df['LocationName'].isin(faulty_sensors) & (df.index.day < 24))
    features = ['temperature', 'humidity', 'altitude', 'wind_speed']
    target = ['CO2']
    
    X_train = df[correct_sensors_mask][features].values
    y_train = df[correct_sensors_mask][target].values
    
    for sensor in faulty_sensors:
        test_mask = df['LocationName'] == sensor
        write_mask = test_mask
        X_test = df[test_mask][features].values
        
        compute_regs(df, write_mask, X_train, y_train, X_test, 'brute_force')

######################### Confidence Intervals ##############################

def bootstrap(y, residuals):
    residuals = list(residuals)
    ys=[]
    for i in range(1000):
        residual = sample(residuals, 1)[0]
        ys.append(y+residual)
    ys.sort()
    return ys[25], ys[975]

def compute_ci(df, train_mask, res_mask, approach, reg):
    train_residuals = df[train_mask]['groundtruth'] - df[train_mask]['{}_{}_prediction'.format(approach, reg)]
    y_hat = df[res_mask]['{}_{}_prediction'.format(approach, reg)]
    y_hat_bounds = []
    for i in range(y_hat.shape[0]):
        y_hat_bounds.append(bootstrap(y_hat[i], train_residuals))
    
    y_hat_lower_bound, y_hat_upper_bound = zip(*y_hat_bounds)
    df.loc[res_mask, '{}_{}_lower_bound'.format(approach, reg)] = y_hat_lower_bound
    df.loc[res_mask, '{}_{}_upper_bound'.format(approach, reg)] = y_hat_upper_bound

######################### Criterions and Metric ##############################

def AIC(y, y_pred, k):
    
    resid  = np.sum((y-y_pred)**2) 
    return -2*np.log(resid) + 2*k

def BIC(y, y_pred, k):
    
    n = len(y_pred)
    resid  = np.sum((y-y_pred)**2) 
    
    return n*np.log(resid/n) + k*np.log(n)

def print_MSE(df, sensors, approaches, regs, failure_date=24):
    for sensor in sensors:
        mask = (df['LocationName'] == sensor) & (df.index.day >= failure_date)
        print()
        print('{} sensor'.format(sensor))
        print(11*'-')
        for approach in approaches:
            print('*****')
            for reg in regs:
                print('MSE for {} {} prediction: {:.2f}'.format(
                    approach, reg,
                    mean_squared_error(df[mask]['groundtruth'], df[mask]['{}_{}_prediction'.format(approach, reg)]))
                )
                
                """k = aic[approach][reg+"_prediction"]
                print('AIC for {} {} prediction: {:.3f}'.format(
                    approach, reg,
                    AIC(df[mask]['groundtruth'], df[mask]['{}_{}_prediction'.format(approach, reg)],k)
                ))
                
                print('BIC for {} {} prediction: {:.3f}'.format(
                    approach, reg,
                    BIC(df[mask]['groundtruth'], df[mask]['{}_{}_prediction'.format(approach, reg)],k)
                ))
                
                print("-")"""
                
######################### Plot ##############################

def plot_prediction(df, sensor, prediction, reg):
    fig, ax = plt.subplots(figsize=(15,5))
    df[df['LocationName'] == sensor][['CO2', 'groundtruth', '{}_{}_prediction'.format(prediction, reg)]].plot(ax=ax, color=['g', 'b', 'r'])
    plt.title('Prediction using {} {} approach for {} sensor'.format(prediction, reg, sensor))
    plt.xlabel('Date')
    plt.ylabel('CO2 [ppm]')
    plt.legend(['artifical measurement', 'true measurement', '{} {} prediction'.format(prediction, reg)])
    plt.show()
    
def plot_zoomed_prediction(df, sensor, prediction, reg, name=None, ci=False, failure_date=24):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)
    mask = (df['LocationName'] == sensor) & (df.index.day >= failure_date)
    df[mask][['groundtruth', '{}_{}_prediction'.format(prediction, reg)]].plot(ax=ax, color=['b', 'r'])
    if ci:
        x = pd.to_datetime(df[mask].index)
        lower_bounds = df[mask]['{}_{}_lower_bound'.format(prediction, reg)]
        upper_bounds = df[mask]['{}_{}_upper_bound'.format(prediction, reg)]
        p2 = ax.fill_between(x, lower_bounds, upper_bounds, color='r', alpha=0.2, label='95% level')
    #plt.title('Prediction using {} {} approach for {} sensor (zoomed)'.format(prediction, reg, sensor))
    """from matplotlib.dates import MonthLocator, DayLocator, DateFormatter
    ax.xaxis.set_major_locator(DayLocator((24, 25, 26, 27, 28, 29, 30, 31)))
    ax.xaxis.set_major_formatter(DateFormatter("%d"))

    ax.xaxis.set_minor_locator(MonthLocator((10)))
    ax.xaxis.set_minor_formatter(DateFormatter("\n%b\n%Y"))
    plt.xticks(rotation=0)
    
    ax.tick_params(axis="x", which="minor", length=0)
    mticks = ax.get_xticks()
    print(mticks)"""

    #plt.xlabel('Date')
    
    #ax.set_xticks([])
    
    plt.setp(ax.get_xticklabels(), visible=False)
    #plt.xticks([])
    plt.xlabel('')
    plt.ylabel('CO2 [ppm]')
    plt.legend(['true measurement', '{} prediction'.format(prediction), '95% ci'])
    if name is not None:
        plt.tight_layout()
        plt.savefig('figures/{}_ci.png'.format(name), dpi=300)
    plt.show()

def plot_example_faulty_sensor():
    fig, ax = plt.subplots(figsize=(15,5))
    ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)
    df_complete[df_complete['LocationName'] == 'AJGR'][['CO2', 'groundtruth']].plot(ax=ax, alpha=0.5, color=['g', 'b'])
    plt.xlabel('Date')
    plt.ylabel('CO2 [ppm]')
    plt.legend(['artifical measurement', 'true measurement'])
    plt.tight_layout()
    plt.savefig('figures/ajgr_sensor_example.png', dpi=300)

def plot_altitude_cluster_features():
    fig, ax = plt.subplots(figsize=(15,10))
    ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)
    ma.plot(x='median CO2', y='altitude', kind='scatter', color='orange', ax=ax)

    #x,y = ma[['median CO2', 'altitude']].mean()
    #print(x)
    #ax.scatter(x,y, color='r')

    for k, v in ma[['median CO2', 'altitude']].iterrows():
        if k in faulty_sensors:
            ax.annotate(k, v, xytext=(-10,5), textcoords='offset points', fontsize=10)

    #plt.title('Sensor visualisation based on altitude and median CO2 measurement')
    plt.xlabel('Daily median CO2 [ppm]')
    plt.ylabel('Altitude [m]')
    plt.tight_layout()
    plt.savefig('figures/sensor_pos.png', dpi=300)

def plot_wind_cluster_features():
    fig, ax = plt.subplots(figsize=(15,10))
    ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)
    pca_components.plot(x='component_1', y='component_2', kind='scatter', color='m', ax=ax)

    #plt.title('Sensor visualisation based on wind components')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig('figures/wind_components.png', dpi=300)