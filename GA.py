
## imports
import pandas as pd
import itertools
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import optimize
import random


## read in data
cities = pd.read_csv('european_cities.csv', delimiter=';')
cities.index = list(cities)


## creates random starting path 
def rand_start(city_list):
    path = city_list
    random.shuffle(path)
    return path


## initialize a random population to start
def rand_pop(num_cities, pop_size):
    subcities = cities.iloc[0:num_cities, 0:num_cities]
    
    pos_cities = list(subcities)
    
    pop = []
    
    for i in range(pop_size):
        r = rand_start(pos_cities)
        pop.append(r.copy())
        
    return pop


## returns the total length of a path
def path_dist(path):
    dist = 0
    for city in range(len(path) - 1):
        dist += cities.loc[path[city], path[city + 1]]
    dist += cities.loc[path[-1], path[0]]
    return dist


## select parents as shortest routes
def select_parents(pop, num_offspring):
    path_lens = {}
    
    for path in pop:
        dist = path_dist(path)
        while dist in path_lens.keys():
            dist += .001
        
        path_lens[dist] = path
    p_keys = list(path_lens.keys())
    p_keys.sort()
    parents = p_keys[0:num_offspring*2]
    return [path_lens[p] for p in parents]


## make offspring form 2 parents
def make_offspring(parents, rand_seg):
    p1 = parents[0]
    p2 = parents[1]
    
    poss_starts = range(len(p1) - rand_seg)
    start = random.sample(poss_starts, 1)[0]
    
    offspring = list(np.zeros(len(p1)))
    
    ## copy randomly selected set from first parent
    for i in range(rand_seg):
        offspring[start+i] = p1[start+i]
        
    ## copy rest from second parent order
    s_order = list(p2[start+rand_seg::] + p2[0:start+rand_seg])
    
    for city in range(len(offspring)):
        if offspring[city] == 0:
            for elm in s_order:
                if elm not in offspring:
                    offspring[city] = elm
                    break;
                    
    return offspring


## make new population
def make_new(pop, num_offspring):
    ## choose parents
    parents = select_parents(pop, num_offspring)
    pos_pairs = random.sample([x for x in itertools.combinations(range(len(parents)), 2)], num_offspring)
    
    ## make offspring
    kids = []
    
    for pair in pos_pairs:
        offspring = make_offspring([parents[pair[0]], parents[pair[1]]], num_offspring//2)
        kids.append(offspring)
        
    return kids


## see how population performs
def performance(population):
    performance = []
    for i in population:
        performance.append(path_dist(i))
    return min(performance)


## perform genetic algo
def genetic_algo(num_cities, pop_size, num_offspring):
    
    ## create randomized starting population
    init_pop = rand_pop(num_cities, pop_size)
    
    ## select parent group
    parents = select_parents(init_pop, num_offspring)
    
    ## make kids from parents
    kids = make_new(parents, num_offspring)

    ## best kids for each run
    best_kid = [performance(init_pop)]
    
    i = 1500
    ## make sure the algo runs at least 100 generations but no more than 1500
    while (performance(parents) != performance(kids) and i > 1  or i > 1400):
        i -= 1
        if pop_size > num_offspring:
            start = time.time()
            parents = select_parents(kids + parents, num_offspring)
            kids = make_new(parents, num_offspring)
            best_kid.append(performance(kids))
        elif pop_size == num_offspring:
            parents = kids
            kids = make_new(parents, num_offspring)
            best_kid.append(performance(kids))
        else:
            parents = select_parents(kids, num_offspring)
            kids = make_new(parents, num_offspring)
            best_kid.append(performance(kids))
            
    return best_kid

## example using GA on 6 cities
start = time.time()
gens = genetic_algo(6, 10, 5)
t = time.time() - start
print('GA for 6 cities')
print('best path: ', min(gens))
print('num gens: ', len(gens))
print('total time: ', t)

## example using GA on 10 cities
start = time.time()
gens = genetic_algo(10, 10, 5)
t = time.time() - start
print('GA for 10 cities')
print('best path: ', min(gens))
print('num gens: ', len(gens))
print('total time: ', t)

## example using GA on 24 cities
start = time.time()
gens = genetic_algo(24, 20, 10)
t = time.time() - start
print('GA for 24 cities')
print('best path: ', min(gens))
print('num gens: ', len(gens))
print('total time: ', t)

## perform GA for given number of cities and create associated data tables and plots
def perform_GA(num_cities):
    pop_sizes = [10, 15, 20]
    
    tens = []
    tens_time = []
    ten_runs = pd.DataFrame(columns = range(100))

    one_hundred = []
    one_hundred_time = []
    one_hundred_runs = pd.DataFrame(columns = range(100))

    two_hundred = []
    two_hundred_time = []
    two_hundred_runs = pd.DataFrame(columns = range(100))
    
    for i in range(20):
        for p in pop_sizes:
            if p == 10:
                start_time = time.time()
                g = genetic_algo(num_cities, p, 10)
                tens.append(min(g))
                tens_time.append(time.time() - start_time)
                ten_runs.loc[i,:] = g[0:100]
                
            elif p == 15:
                start_time = time.time()
                g = genetic_algo(num_cities, p, 10)
                one_hundred.append(min(g))
                one_hundred_time.append(time.time() - start_time)
                one_hundred_runs.loc[i,:] = g[0:100]
                
            else:
                start_time = time.time()
                g = genetic_algo(num_cities, p, 10)
                two_hundred.append(min(g))
                two_hundred_time.append(time.time() - start_time)
                two_hundred_runs.loc[i,:] = g[0:100]
                
    
    stats = pd.DataFrame(columns=['best path', 'worst path', 'mean path', 'SD', 'avg runtime'])
    run_stats = pd.DataFrame(columns=list(range(len(ten_runs))))

    results = [tens, one_hundred, two_hundred]
    times = [tens_time, one_hundred_time, two_hundred_time]
    runs = [ten_runs, one_hundred_runs, two_hundred_runs]

    for i in range(len(pop_sizes)):
        stats.loc[pop_sizes[i], 'best path'] = min(results[i])
        stats.loc[pop_sizes[i], 'worst path'] = max(results[i])
        stats.loc[pop_sizes[i], 'mean path'] = np.mean(results[i])
        stats.loc[pop_sizes[i], 'SD'] = np.std(results[i])
        stats.loc[pop_sizes[i], 'avg runtime'] = np.mean(times[i])

        for j in range(100):
            run_stats.loc[pop_sizes[i], j] = np.mean(runs[i][j])

    ## saving
    stats.to_csv(str(num_cities) + '_city_GA.csv')
    run_stats.to_csv(str(num_cities) + '_city_GA_runs.csv')
    
    ## plotting avg best performing run for each pop size
    f = run_stats.T.plot()
    plt.title('Avg Path Distance for ' + str(num_cities) + ' Cities')
    plt.xlabel('Run #')
    plt.ylabel('Distance')
    plt.legend(title='pop size', fancybox=True)
    plt.show()
    fig = f.get_figure()
    fig.savefig('avg_path_' + str(num_cities) + '_cities.pdf', bbox_inches='tight')
    
    return min(min(tens), min(one_hundred), min(two_hundred))

perform_GA(10)

perform_GA(24)




