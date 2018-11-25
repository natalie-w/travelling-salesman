## imports
import pandas as pd
import itertools
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import optimize
import random
import math
from IPython.display import IFrame


## read in data
cities = pd.read_csv('european_cities.csv', delimiter=';')
cities.index = list(cities)


## returns the total length of a path
def path_dist(path):
    dist = 0
    for city in range(len(path) - 1):
        dist += cities.loc[path[city], path[city + 1]]
    dist += cities.loc[path[-1], path[0]]
    return dist


## creates random starting path 
def rand_start(city_list):
    path = city_list
    random.shuffle(path)
    return path


## finds neighbors (neighbors defined as swapping cities in path) and returns best neighboring path
def find_best_neighbor(path):
    curr_dist = path_dist(path)
    curr_path = path
    swappings = itertools.combinations(range(0, len(path)), 2)
    for swap in swappings:
        city_0 = path[swap[0]]
        city_1 = path[swap[1]]
        
        new_path = path.copy()
        new_path[swap[0]] = city_1
        new_path[swap[1]] = city_0
        
        new_dist = path_dist(new_path)
        
        ## if new path is shorter than curr path set curr path as new path and curr dist as new dist
        if new_dist < curr_dist:
            curr_path = new_path
            curr_dist = new_dist
    return curr_path


## implements hill climbing algorithm where neighbors count as current path switching 2 cities
def hill_climber(num_cities):
    subcities = cities.iloc[0:num_cities, 0:num_cities]
    path = rand_start(list(subcities))
    best_neighbor = find_best_neighbor(path)
    ##print(path, best_neighbor, path != best_neighbor)
    while (path != best_neighbor):
        path = best_neighbor
        #print(path_dist(path))
        best_neighbor = find_best_neighbor(path)
        #print(path_dist(best_neighbor))
    
    return path


## hill climbing for first 10 cities
best_path_distances = []

for i in range(20):    
    city_path = hill_climber(10)
    best_path_distances.append(path_dist(city_path))
    
print('best path distance: ', min(best_path_distances))
print('worst path distance: ', max(best_path_distances))
print('mean path distance: ', np.mean(best_path_distances))
print('SD of paths: ', np.std(best_path_distances))


## hill climbing for first 24 cities
best_path_distances = []

for i in range(20):    
    city_path = hill_climber(24)
    best_path_distances.append(path_dist(city_path))
    
print('best path distance: ', min(best_path_distances))
print('worst path distance: ', max(best_path_distances))
print('mean path distance: ', np.mean(best_path_distances))
print('SD of paths: ', np.std(best_path_distances))