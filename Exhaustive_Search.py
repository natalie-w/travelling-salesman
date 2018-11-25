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


## run exhaustive search
tours = pd.DataFrame(columns=['num_cities', 'time', 'tour', 'distance'])

for i in range(6, 11):
    num_cities = i

    start_time = time.time()

    sub_cities = cities.iloc[0:num_cities,0:num_cities]
    possible_paths = [x for x in itertools.permutations(list(sub_cities))]

    min_path = -1
    min_dist = -1

    for path in possible_paths:
        ## only calculate paths starting from the same place to reduce redundancies
        if path[0] != 'Barcelona':
            break;
        
        ## calculate total distances
        dist = 0
        for city in range(len(path)-1):
            dist += sub_cities.loc[path[city], path[city+1]]
        dist += sub_cities.loc[path[-1], path[0]]

        ## only change min_path is distance is shortest or first distance to be calculated
        if min_dist > dist:
            min_path = path
            min_dist = dist

        elif min_dist == -1:
            min_path = path
            min_dist = dist


    runtime = time.time() - start_time

    tours = tours.append(pd.DataFrame([[num_cities, runtime, min_path, min_dist]], columns=['num_cities', 'time', 'tour', 'distance']))

## saving
tours.to_csv('exhaustive_search_city_overview.csv')

##print tours
print(tours)

## experimental data
xdata = np.array([int(c) for c in tours['num_cities']])
ydata = np.array([float(t) for t in tours['time']])

## exponential function to fit
def func(x, a, b, c):
    return a * np.exp(b * x) + c

## fitting function
popt, pcov = scipy.optimize.curve_fit(func, xdata, ydata)

## plotting data
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(xdata, func(xdata, *popt))

plt.title('Num Cities vs Runtime')
plt.xlabel('Num Cities')
plt.ylabel('Runtime (seconds)')

plt.show()

##fig = f.get_figure()
plt.savefig('num_cities_vs_runtime.pdf')

print('estimated time for 24 cities: ' + str(func(24, *popt)) + ' seconds.')