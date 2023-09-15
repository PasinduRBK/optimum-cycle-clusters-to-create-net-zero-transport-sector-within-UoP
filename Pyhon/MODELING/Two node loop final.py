import numpy as np
import random
from operator import add
import matplotlib.pyplot as plt
import statistics 

# creating null lists
L = 100 
MA = 20
MB = 18
B = 15
T = 60
Ta=[]
Tb=[]
B_A=[0]*L
B_B=[0]*L
Avg_A=[0]*L
Avg_B=[0]*L

for j in range (L):

    Ta=np.random.poisson(6,MA)
    Ta=np.array(Ta)
    Ta.sort()
    time_Ta=Ta.tolist()
    print(time_Ta)

    # print(time_Ta[0])
    Tb=np.random.poisson(6,MB)
    Tb=np.array(Tb)
    Tb.sort()
    time_Tb=Tb.tolist()
    print(time_Tb)


    # cycle number 

    cyc_Ca = random.randint(1,B) 
    print(cyc_Ca)

    cyc_Cb = random.randint(1,B)  
    print(cyc_Cb)

    cyc_Ca_list=[0]*T
    cyc_Cb_list=[0]*T

    Ca=[0]*T
    Cb=[0]*T

    for i in range (1):
        if cyc_Ca > time_Ta.count(i):
            cyc_Ca_list[i] = cyc_Ca_list[i] + cyc_Ca - time_Ta.count(i)
            Ca[i] = time_Ta.count(i)
            W_A=0
        if cyc_Ca <= time_Ta.count(i):
            cyc_Ca_list[i] = 0
            Ca[i]= cyc_Ca
            W_A = time_Ta.count(i)-cyc_Ca

        if cyc_Cb > time_Tb.count(i):
            cyc_Cb_list[i] = cyc_Cb_list[i] + cyc_Cb - time_Tb.count(i)
            Cb[i]= time_Tb.count(i)
            W_B=0
        if cyc_Cb <= time_Tb.count(i):
            cyc_Cb_list[i]= 0
            Cb[i]= cyc_Cb
            W_B = time_Tb.count(i)-cyc_Cb

    for i in range (1,T):
        if cyc_Ca_list[i-1] + Cb[i-2] >= time_Ta.count(i)+ W_A:
            cyc_Ca_list[i] = cyc_Ca_list[i] + cyc_Ca_list[i-1] + Cb[i-2] - time_Ta.count(i)-W_A
            Ca[i] = time_Ta.count(i)+W_A
            W_A =0
        elif cyc_Ca_list[i-1] + Cb[i-2] < time_Ta.count(i)+ W_A:
            cyc_Ca_list[i] = 0
            W_A = time_Ta.count(i) + W_A - cyc_Ca_list[i] - cyc_Ca_list[i-1] - Cb[i-2] 
            Ca[i] =   cyc_Ca_list[i-1] + Cb[i-2] 

        if cyc_Cb_list[i-1] + Ca[i-2] >= time_Tb.count(i) + W_B:
            cyc_Cb_list[i] = cyc_Cb_list[i] + cyc_Cb_list[i-1] + Ca[i-2] - time_Tb.count(i) - W_B
            W_B =0
            Cb[i] = time_Tb.count(i) +W_B
        elif cyc_Cb_list[i-1] + Ca[i-2] < time_Ta.count(i)+ W_B:
            cyc_Cb_list[i] = 0
            W_B = time_Tb.count(i)+ W_B -cyc_Cb_list[i] - cyc_Cb_list[i-1] - Ca[i-2] 
            Cb[i] =  cyc_Cb_list[i-1] + Ca[i-2] 
        
    print (cyc_Ca_list)
    print (cyc_Cb_list)

    waiting_timeA=[0]*T
    waiting_timeB=[0]*T

    for i in range (T):

        if cyc_Ca_list[i] >0:
            waiting_timeA[i]=waiting_timeA[i]
        else:
            waiting_timeA[i]=waiting_timeA[i-1] +1
    print(waiting_timeA)

    for i in range (T):

        if cyc_Cb_list[i] >0:
            waiting_timeB[i]=waiting_timeB[i]
        else:
            waiting_timeB[i]=waiting_timeB[i-1] +1
    print(waiting_timeB)

    mean_W_A=statistics.mean(waiting_timeA)
    mean_W_B=statistics.mean(waiting_timeB)
    print('mean waiting time A:',mean_W_A)
    print('mean waiting time B:',mean_W_B)

    B_A[j] = cyc_Ca
    B_B[j] = cyc_Cb
    Avg_A[j] = mean_W_A
    Avg_B[j] = mean_W_B

mymodelA=np.poly1d(np.polyfit(B_A,Avg_A,4))
mymodelB=np.poly1d(np.polyfit(B_B,Avg_B,4))
myline = np.linspace(1,B,100)


plt.subplot(1,2,1)
plt.scatter(B_A,Avg_A)
plt.plot(myline,mymodelA(myline),color='red')
plt.xlabel('cyc_Ca')
plt.ylabel(' Avg Waiting time A')
plt.grid()

plt.subplot(1,2,2)
plt.scatter(B_B,Avg_B)
plt.plot(myline,mymodelB(myline),color='red')
plt.xlabel('cyc_Cb')
plt.ylabel(' Avg Waiting time B')
plt.grid()
plt.show()








