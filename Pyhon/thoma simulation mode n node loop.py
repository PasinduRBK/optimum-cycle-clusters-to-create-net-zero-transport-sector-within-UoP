import numpy as np
import random
from operator import add
import matplotlib.pyplot as plt
import time

start_time=time.time()
from scipy.optimize import curve_fit

def generate_random_integers(max_values, target_sum):
    values = [random.randint(0, max_value) if max_value > 0 else 0 for max_value in max_values]

    while sum(values) != target_sum:
        values = [random.randint(0, max_value) if max_value > 0 else 0 for max_value in max_values]
        
    return values
    # print('x')

# L = 1000     # loop count 
# n = 3           # number of nodes
# M = [30, 30, 30] # population at each node
# P = [[0, 0.6, 0.4], [0.7, 0, 0.3], [0.4, 0.6, 0]] # probability of moving between nodes
# travel_time = [[0, 2, 1], [2, 0, 3], [1, 3, 0]]
# Bic = [20, 20, 20] # maximum cycle count at each node
# T = 1000       # time limit
# lambdas = [5, 10, 15] # mean arrival time at each node

# L = 1000    # loop count 
# n = 4           # number of nodes
# M = [30, 20, 30, 30] # population at each node
# P = [[0, 0.3, 0.4, 0.3], [0.4, 0, 0.3, 0.3], [0.2, 0.3, 0, 0.5],[0.4, 0.3, 0.3, 0]] # probability of moving between nodes
# travel_time = [[0, 2, 3, 1], [2, 0, 1, 1], [3, 1, 0, 2],[1, 1, 2, 0]]
# Bic = [20, 30, 20,20] # maximum cycle count at each node
# T = 100       # time limit
# lambdas = [15, 10, 10, 12] # mean arrival time at each node

# L = 100    # loop count 
# n = 8           # number of nodes
# M = [30, 30, 30,30,30,30,30,30] # population at each node
# P = [[0.         , 0.52941176, 0.08823529, 0.11764706, 0.08823529, 0.08823529, 0.05882353, 0.02941176],
#     [0.37037037, 0.         , 0.03703704, 0.11111111, 0.11111111, 0.22222222, 0.11111111, 0.03703704],
#     [0.42857143, 0.42857143, 0.         , 0.14285714, 0.         , 0.         , 0.         , 0.        ],
#     [0.23076923, 0.38461538, 0.         , 0.         , 0.         , 0.         , 0.30769231, 0.07692308],
#     [0.33333333, 0.33333333, 0.33333333, 0.         , 0.         , 0.         , 0.         , 0.        ],
#     [0.25       , 0.75       , 0.         , 0.         , 0.         , 0.         , 0.         , 0.        ],
#     [0.16666667, 0.         , 0.33333333, 0.16666667, 0.         , 0.         , 0.         , 0.33333333],
#     [0.1         , 0.1         , 0.1         , 0.1         , 0.2         , 0.2         , 0.2         , 0.        ]] # probability of moving between nodes
# travel_time = [[0, 1, 1, 1, 1, 1, 1, 1],
#                     [1, 0, 1, 1, 1, 1, 1, 1],
#                     [1, 1, 0, 1, 1, 1, 1, 1],
#                     [1, 1, 1, 0, 1, 1, 1, 1],
#                     [1, 1, 1, 1, 0, 1, 1, 1],
#                     [1, 1, 1, 1, 1, 0, 1, 1],
#                     [1, 1, 1, 1, 1, 1, 0, 1],
#                     [1, 1, 1, 1, 1, 1, 1, 0]]
# Bic = [20, 20, 20,20,20,20,20,20] # maximum cycle count at each node
# T = 10       # time limit
# lambdas = [2, 2 ,2,2,2,2,2,2] # mean arrival time at each node



L = 10   # loop count 
n = 12           # number of nodes
#M = np.array([5,6,5,7,5,7,5,8,7,7,7,3,6,5])# population at each node
M = np.array([21,10,21,61,49,16,21,8,7,40,7,3,6,10])

P = np.array([
 [0.        , 0.49230769, 0.04615385, 0.09230769, 0.04615385, 0.03076923, 0.04615385, 0.07692308, 0.        , 0.03076923, 0.01538462, 0.12307692],
 [0.42592593, 0.        , 0.12962963, 0.07407407, 0.12962963, 0.03703704, 0.14814815, 0.01851852, 0.        , 0.01851852, 0.01851852, 0.        ],
 [0.2       , 0.26666667, 0.        , 0.2       , 0.06666667, 0.06666667, 0.        , 0.06666667, 0.        , 0.13333333, 0.        , 0.        ],
 [0.13793103, 0.44827586, 0.03448276, 0.        , 0.13793103, 0.13793103, 0.03448276, 0.        , 0.        , 0.03448276, 0.03448276, 0.        ],
 [0.07142857, 0.28571429, 0.14285714, 0.28571429, 0.        , 0.        , 0.        , 0.        , 0.14285714, 0.07142857, 0.        , 0.        ],
 [0.14285714, 0.14285714, 0.        , 0.        , 0.14285714, 0.        , 0.14285714, 0.14285714, 0.14285714, 0.        , 0.14285714, 0.        ],
 [0.        , 0.5       , 0.25      , 0.25      , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
 [0.25      , 0.75      , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
 [0.        , 0.        , 0.        , 0.        , 0.57142857, 0.28571429, 0.        , 0.        , 0.        , 0.14285714, 0.        , 0.        ],
 [0.        , 0.5       , 0.1       , 0.1       , 0.        , 0.        , 0.        , 0.        , 0.1       , 0.        , 0.2       , 0.        ],
 [0.        , 0.        , 0.        , 0.5       , 0.        , 0.        , 0.        , 0.        , 0.        , 0.5       , 0.        , 0.        ],
 [1.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ]
])# probability of moving between nodes
travel_time = np.array([[0,4,7,14,10,14,9,12,15,18,8,2],
               [4,0,3,10,6,10,6,9,8,12,14,7],
               [7,3,0,8,9,9,4,7,6,11,12,3],
               [14,10,8,0,5,5,12,15,4,8,6,16],
               [10,6,9,5,0,5,8,10,3,7,8,11],
               [14,10,9,5,5,0,12,14,2,3,4,15],
               [9,6,4,12,8,12,0,3,10,14,15,11],
               [12,9,7,15,10,14,3,0,12,16,17,14],
               [15,8,6,4,3,2,10,12,0,5,6,13],
               [18,12,11,8,7,3,14,16,5,0,7,18],
               [8,14,12,6,8,4,15,17,6,7,0,9],
               [2,7,3,16,11,15,11,14,13,18,9,0]])
        
Bic = np.array([20,20,20,20,20,20,20,20,20,20,20,20])# maximum cycle count at each node
T = 1440       # time limit
lambdas = np.array([2,2,2,2,2,2,2,2,2,2,2,2])# mean arrival time at each node


mean_W=[[0]*L for  _ in range(n)]
mean_C=[[0]*L for  _ in range(n)]
cyc_C =[[0]*L for  _ in range(n)]

for k in range (L):

    randomprob_P = [[] for _ in range(n)]

    cyc_C_list = [[0]*T for _ in range(n)]

    C = [[[0]*T for _ in range(n)] for _ in range(n)] # leving the node

    G = [[[] for _ in range(n)] for _ in range(n)]     # come to the node

    waiting_people = [[0]*T for _ in range(n)]

    arrival_times = [[[0]*T for _ in range(T)] for _ in range(n)]

    remaining_customers_list = [[[0]*T for _ in range(n)] for _ in range(n)]

    for i in range(n):
        T_arr = np.random.poisson(lambdas[i], M[i])
        T_arr[T_arr >= T] = T-1
        
        # Divide T_arr according to probabilities in P
        split_indices = np.cumsum(np.round(np.array(P[i]) * len(T_arr)).astype(int))
        split_T_arr = np.split(T_arr, split_indices)
        
        # Add resulting arrays to G
        for j in range(n):
            G[i][j] = split_T_arr[j].tolist()
            G[i][j].sort()
        cyc_C_list[i][0] = random.randint(1,Bic[i])

    for i in range(n):
        cyc_C[i][k]=cyc_C_list[i][0]
    
    # print(G)


    for t in range(1):
        for i in range(n):
            sum_G = 0
            Cvalue=0
            for j in range(n):
                if i != j:
                    sum_G += G[i][j].count(t)
            if cyc_C_list[i][t] >= sum_G:
                cyc_C_list[i][t] -= sum_G
                for j in range(n):
                    if i != j:
                        C[i][j][t]=G[i][j].count(t)
                        # Calculate arrival time for each customer leaving node i and going to node j at time t
                        arrival_time = t + travel_time[i][j]
                        if arrival_time < T:
                            arrival_times[i][j][arrival_time] += C[i][j][t]

            else:

                max_values = []
                for j in range(n):
                    if i != j:
                        max_values.append(G[i][j].count(t) )
                    else:
                        max_values.append(0)

                max_values = [G[i][j].count(t) if i != j else 0 for j in range(n)]

                values = generate_random_integers(max_values, cyc_C_list[i][t])
                for j, value in enumerate(values):
                    C[i][j][t] = value
                    Cvalue += value
                    # Calculate arrival time for each customer leaving node i and going to node j at time t
                    arrival_time = t + travel_time[i][j]
                    if arrival_time < T:
                        arrival_times[i][j][arrival_time] += C[i][j][t]
                waiting_people[i][t] = sum(G[i][j].count(t) for j in range(n) if i != j)- Cvalue
                for j in range(n):
                    if i != j:
                        remaining_customers_list[i][j][t]=(G[i][j].count(t)-C[i][j][t])
                cyc_C_list[i][t]=0
                # print(max_values)
                

            
    for t in range(1, T):
        for i in range(n):
            sum_G = 0
            sum_C = 0
            sum_R = 0
            Cvalue=0
            sum_C = sum(arrival_times[j][i][t] for j in range(n) if i != j)
            # print(sum_C)
            sum_R = sum(remaining_customers_list[i][j][t-1] for j in range(n) if i != j)
            for j in range(n):
                if i != j:
                    sum_G += G[i][j].count(t)
            if cyc_C_list[i][t-1] + sum_C >= sum_G + sum_R:
                cyc_C_list[i][t] = cyc_C_list[i][t-1]+ sum_C - sum_G-sum_R

                for j in range(n):
                    if i != j:
                        C[i][j][t]=G[i][j].count(t)+ remaining_customers_list[i][j][t-1]
                        # Calculate arrival time for each customer leaving node i and going to node j at time t
                        arrival_time = t + travel_time[i][j]
                        if arrival_time < T:
                            arrival_times[i][j][arrival_time] += C[i][j][t]
            else:

                max_values = []
                for j in range(n):
                    if i != j:
                        max_values.append(G[i][j].count(t) + remaining_customers_list[i][j][t-1])
                    else:
                        max_values.append(0)


                # print(max_values)
                # print(cyc_C_list[i][t-1]+sum_C)
                values = generate_random_integers(max_values, cyc_C_list[i][t-1]+sum_C)
                # print(values)
                for j, value in enumerate(values):
                    C[i][j][t] = value
                    Cvalue +=value
                    remaining_customers_list[i][j][t] = G[i][j].count(t) + remaining_customers_list[i][j][t-1]-value
                    # Calculate arrival time for each customer leaving node i and going to node j at time t
                    arrival_time = t + travel_time[i][j]
                    if arrival_time < T:
                        arrival_times[i][j][arrival_time] += C[i][j][t]
                
                # Calculate the waiting time for all customers that could not be served
                waiting_people[i][t] = sum_G + sum_R - Cvalue
                
    # print(cyc_C_list)
    # print(waiting_time)
    for i in range(n):
        mean_W[i][k]=sum(waiting_people[i])/T
        mean_C[i][k]=sum(cyc_C_list[i])/T
    
     
end_time=time.time()
print(end_time-start_time)

            
mymodel = [0]*n
mymodelU = [0]*n
myline = [0]*n

for i in range(n):
    positive_cyc_C = np.array(cyc_C[i])[np.array(cyc_C[i]) >= 0] # Only include positive values in cyc_C
    positive_mean_W = np.array(mean_W[i])[np.array(cyc_C[i]) >= 0] # Only include corresponding values in mean_W
    mymodel[i] = np.poly1d(np.polyfit(positive_cyc_C, positive_mean_W, 3))
    mymodelU[i] = np.poly1d(np.polyfit(cyc_C[i], mean_C[i], 1))
    myline[i] = np.linspace(1, Bic[i], 100)

for i in range(n):
    plt.figure()
    plt.scatter(cyc_C[i], mean_W[i])
    positive_myline = myline[i][myline[i] >= 0] # Only include positive values in myline
    plt.plot(positive_myline, mymodel[i](positive_myline), color='red')
    plt.xlabel(f'cyc_C{i}')
    plt.ylabel(f'Avg Waiting customers {chr(ord("0")+i)}')
    plt.xlim(left=0) # Set left limit of x-axis to 0
    plt.ylim(bottom=-0.001) # Set bottom limit of y-axis to 0
    plt.grid()

plt.show()

# from scipy.optimize import curve_fit

# define the exponential function
# def func(x, a, b):
#     return a * np.exp(b * x)

# mymodel = [0]*n
# myline = [0]*n

# for i in range(n):
#     # fit the data
#     popt, pcov = curve_fit(func, cyc_C[i], mean_W[i])
#     mymodel[i] = lambda x: func(x, *popt)
#     myline[i] = np.linspace(1, Bic[i], 100)

# for i in range(n):
#     plt.subplot(2, n, i+1)
#     plt.scatter(cyc_C[i], mean_W[i])
#     plt.plot(myline[i], mymodel[i](myline[i]), color='red')
#     plt.xlabel(f'cyc_C{i}')
#     plt.ylabel(f'Avg Waiting customers {chr(ord("0")+i)}')
#     plt.grid()

# plt.show()
