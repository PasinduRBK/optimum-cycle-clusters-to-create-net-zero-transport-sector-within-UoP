import numpy as np
import random
from operator import add
import matplotlib.pyplot as plt
import statistics 


L = 1000    # loop count 
MA = 200    # population at node A
MB = 200    # population at node B
B = 50      # cycle count
T = 100     # time liit
#creating null lists
Ta=[] # empty list for 
Tb=[]
B_A=[0]*L
B_B=[0]*L
Avg_A=[0]*L
Avg_B=[0]*L
Avg_U_A=[0]*L
Avg_U_B=[0]*L

for j in range (L):

    Ta=np.random.poisson(6,MA)
    Ta=np.array(Ta)
    Ta.sort()
    time_Ta=Ta.tolist()
    #print(time_Ta)

    Tb=np.random.poisson(6,MB)
    Tb=np.array(Tb)
    Tb.sort()
    time_Tb=Tb.tolist()
    #print(time_Tb)


    cyc_Ca = random.randint(1,B) 
   
    # print(cyc_Ca)

    cyc_Cb = random.randint(1,B)
 
    #print(cyc_Cb)

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
            Ca[i] = time_Ta.count(i)+ W_A
            W_A =0
        else:
            cyc_Ca_list[i] = 0
            W_A = time_Ta.count(i) + W_A - cyc_Ca_list[i-1] - Cb[i-2] 
            Ca[i] =   cyc_Ca_list[i-1] + Cb[i-2] 

        if cyc_Cb_list[i-1] + Ca[i-2] >= time_Tb.count(i) + W_B:
            cyc_Cb_list[i] = cyc_Cb_list[i] + cyc_Cb_list[i-1] + Ca[i-2] - time_Tb.count(i) - W_B
            Cb[i] = time_Tb.count(i) +W_B
            W_B =0
        else:
            cyc_Cb_list[i] = 0
            W_B = time_Tb.count(i)+ W_B - cyc_Cb_list[i-1] - Ca[i-2] 
            Cb[i] =  cyc_Cb_list[i-1] + Ca[i-2] 
        
    #print (cyc_Ca_list)
    #print (cyc_Cb_list)

    waiting_timeA=[0]*T
    waiting_timeB=[0]*T
    utilization_A=[0]*T
    utilization_B=[0]*T

    for i in range (T):

        if cyc_Ca_list[i] >0:
            waiting_timeA[i]=waiting_timeA[i]
        else:
            waiting_timeA[i]=waiting_timeA[i-1] +1
    #print(waiting_timeA)

    for i in range (T):

        if cyc_Cb_list[i] >0:
            waiting_timeB[i]=waiting_timeB[i]
        else:
            waiting_timeB[i]=waiting_timeB[i-1] +1
    #print(waiting_timeB)

    for i in range (T):
        if cyc_Ca_list[i] >0:
            utilization_A[i]= utilization_A[i-1]+1
        else:
            utilization_A[i]= utilization_A[i]
    
    for i in range(T):
        if cyc_Cb_list[i]>0:
            utilization_B[i]= utilization_B[i-1]+1
        else:
            utilization_B[i]=utilization_B[i]



    def split_list(lst):
        sublists=[]
        sublist =[]
        for item in lst:
            if item==0:
                if sublist:
                    sublists.append(sublist)
                    sublist=[]
            else:
                sublist.append(item)
        if sublist:
            sublists.append(sublist)
        max_values =[]
        for sublist in sublists:
            max_value=max(sublist)
            max_values.append(max_value)
        return max_values

    max_valuesA=split_list(waiting_timeA)
    max_valuesB=split_list(waiting_timeB)

    max_U_valuesA=split_list(utilization_A)
    max_U_valuesB=split_list(utilization_B)

    mean_W_A= sum(max_valuesA)/len(waiting_timeA)
    mean_W_B= sum(max_valuesB)/len(waiting_timeB)
    #print('mean waiting time A:',mean_W_A)
    #print('mean waiting time B:',mean_W_B)

    mean_U_A= sum(max_U_valuesA)/len(utilization_A)
    mean_U_B= sum(max_U_valuesB)/len(utilization_B)

    B_A[j] = cyc_Ca
    B_B[j] = cyc_Cb
    Avg_A[j] = mean_W_A
    Avg_B[j] = mean_W_B
    Avg_U_A[j]= mean_U_A
    Avg_U_B[j]= mean_U_B


mymodelA=np.poly1d(np.polyfit(B_A,Avg_A,5))
mymodelB=np.poly1d(np.polyfit(B_B,Avg_B,5))
myline = np.linspace(1,B,100)

mymodelUA=np.poly1d(np.polyfit(B_A,Avg_U_A,5))
mymodelUB=np.poly1d(np.polyfit(B_B,Avg_U_B,5))
myline = np.linspace(1,B,100)


plt.subplot(2,2,1)
plt.scatter(B_A,Avg_A)
plt.plot(myline,mymodelA(myline),color='red')
plt.xlabel('cyc_Ca')
plt.ylabel(' Avg Waiting time A')
plt.grid()

plt.subplot(2,2,2)
plt.scatter(B_B,Avg_B)
plt.plot(myline,mymodelB(myline),color='red')
plt.xlabel('cyc_Cb')
plt.ylabel(' Avg Waiting time B')
plt.grid()

plt.subplot(2,2,3)
plt.scatter(B_A,Avg_U_A)
plt.plot(myline,mymodelUA(myline),color='red')
plt.xlabel('cyc_Ca')
plt.ylabel(' Avg Utilization A')
plt.grid()

plt.subplot(2,2,4)
plt.scatter(B_B,Avg_U_B)
plt.plot(myline,mymodelUB(myline),color='red')
plt.xlabel('cyc_Cb')
plt.ylabel(' Avg Utilization B')
plt.grid()


plt.show()








