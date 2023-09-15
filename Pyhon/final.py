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

L = 2   # loop count 
n = 12           # number of nodes
M = [219,103,21,61,49,16,21,8,7,40,7,3,6,10] # population at each node

P = [
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
] # probability of moving between nodes
travel_time = [[0,4,7,14,10,14,9,12,15,18,8,2],
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
               [2,7,3,16,11,15,11,14,13,18,9,0]]
        
Bic = [20,20,20,20,20,20,20,20,20,20,20,20] # maximum cycle count at each node
T = 100       # time limit
lambdas = [2,2,2,2,2,2,2,2,2,2,2,2] # mean arrival time at each node


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
                print('a')
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
                print(values)
                for j, value in enumerate(values):
                    print('a1')
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

        

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

mymodel = [0]*n
myline = [0]*n
equations = ['']*n

for i in range(n):
    # Convert cyc_C[i] and mean_W[i] to NumPy arrays
    x = np.array(cyc_C[i]).reshape(-1, 1)
    y = np.array(mean_W[i])
    
    # Generate polynomial features
    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(x)
    
    # Fit a linear regression model to the polynomial features
    reg = LinearRegression().fit(x_poly, y)
    mymodel[i] = lambda x: reg.predict(poly.fit_transform(x.reshape(-1, 1)))
    
    # Generate the equation of the fitted polynomial curve
    coeffs = np.flip(reg.coef_)
    terms = [f'{coeffs[j]:.2f}x^{j}' for j in range(len(coeffs))]
    equation = ' + '.join(terms) + f' + {reg.intercept_:.2f}'
    equations[i] = equation
    
    myline[i] = np.linspace(1, Bic[i], 100)

for i in range(n):
    plt.subplot(1, n, i+1)
    plt.scatter(cyc_C[i], mean_W[i])
    plt.plot(myline[i], mymodel[i](myline[i]), color='red')
    plt.xlabel(f'cyc_C{i}')
    plt.ylabel(f'Avg Waiting customers {chr(ord("0")+i)}')
    plt.title(equations[i])
    plt.grid()

plt.show()
