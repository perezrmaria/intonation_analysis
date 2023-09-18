import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pretty_midi

import tslearn
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import KShape

# Clustering and formatting for tslearn functions
def cluster_and_plot(X, num_clusters=10, max_iter=250, random_state=42, num_samples ='25 to 35 samples'):
    """
    This function takes the data and the number of clusters and returns and plots the clusters.
    """
    #kshape = KShape(n_clusters=num_clusters, max_iter=max_iter, random_state=random_state)
    #kshape.fit(X)
    #y_pred = kshape.predict(X)

    km = TimeSeriesKMeans(n_clusters=num_clusters, verbose=True, random_state=random_state)
    y_pred = km.fit_predict(X)
    
    print("Euclidean k-means")
    plt.figure(figsize=(20, 15))
    for yi in range(num_clusters):
        plt.subplot(5, int(num_clusters/5), yi + 1)
        for xx in X[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(km.cluster_centers_[yi].ravel(), "r-")
        plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                transform=plt.gca().transAxes)
        if yi == 1:
            plt.title("Euclidean $k$-means" + num_samples)
    return y_pred

def cluster_distribution(y_pred):
    """
    This function takes the clusters and returns the cluster distribution.
    """
    cluster_counts = np.bincount(y_pred)
    
    print('\nNumber of clusters: ', len(cluster_counts))
    print('Total number of cases: ',sum(cluster_counts))
    print('Cluster distribution: ', cluster_counts)

def formatting_all_info(all_info_method, max_num_samples=180, min_num_samples=509):    
    """
    This function takes a dictionary with the pitch bend time series for each note and returns a formatted array of pitch bends.
    """
    list_of_lists = []
    list_more_than25 = []
    max_len = 0

    for key, value in all_info_method.items():
        for note, inner_list in value.items():
            if len(inner_list) >= min_num_samples and len(inner_list) <= max_num_samples:
                list_of_lists.append(inner_list)
            else:
                list_more_than25.append(inner_list)
            if len(inner_list) > max_len:
                max_len = len(inner_list)
                max_key = key
                max_value = value

    max_length = max(len(sublist) for sublist in list_of_lists if sublist)
    indices = [i for i, sublist in enumerate(list_of_lists) if len(sublist) == max_length]

    interpolated_time_series = []
    for sublist in list_of_lists:
        original_length = len(sublist)
        new_x = np.linspace(0, 1, max_length)
        interpolated_sublist = []
        for item in sublist:
            interpolated_item = np.array(item, dtype=float)
            interpolated_sublist.append(interpolated_item)
        interpolated_sublist = np.array(interpolated_sublist)
        x = np.linspace(0, 1, original_length)
        
        interpolated = np.interp(new_x, x, interpolated_sublist)
        interpolated_time_series.append(interpolated)

    #reshape the array to match the required input format
    X = np.array(interpolated_time_series)
    print('Shape before formatting',X.shape)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    print('Shape after formatting',X.shape)
    return X

