import numpy as np
#import random
from operator import add
import matplotlib.pyplot as plt
import statistics 

# numbers for time
Ta=[]
Tb=[]
Ta=np.random.poisson(5,16) #array of 16 elements having poisson distribution that lamda is 5
Ta=np.array(Ta)
Ta.sort() #ascending order
time_Ta=Ta.tolist()
print(time_Ta)

# print(time_Ta[0])
Tb=np.random.poisson(6,16) #array of 16 elements having poisson distribution that lamda is 6
Tb=np.array(Tb)
Tb.sort() #ascending order
time_Tb=Tb.tolist() #costomer arrival sequance
print(time_Tb)


# cycle number 

cyc_Ca = 5#random.randint(1,15) 
print(cyc_Ca)

cyc_Cb =5#random.randint(1,15)  
print(cyc_Cb)

#Number of bicycles in clusters
cyc_Ca_list=[0]*16
cyc_Cb_list=[0]*16

#out going bicycles
Ca=[0]*16 
Cb=[0]*16

for i in range (1):
    if cyc_Ca > time_Ta.count(i):
        cyc_Ca_list[i] = cyc_Ca_list[i] + cyc_Ca - time_Ta.count(i)
        Ca[i] = time_Ta.count(i)
        W_A=0 #number of waiting customers
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

for i in range (1,16):
    if cyc_Ca_list[i-1] + Cb[i-2] > time_Ta.count(i)+ W_A:
        cyc_Ca_list[i] = cyc_Ca_list[i] + cyc_Ca_list[i-1] + Cb[i-2] - time_Ta.count(i)-W_A
        W_A =0
        Ca[i] = time_Ta.count(i)+W_A
    elif cyc_Ca_list[i-1] + Cb[i-2] < time_Ta.count(i)+ W_A:
        cyc_Ca_list[i] = 0
        W_A = time_Ta.count(i) + W_A - cyc_Ca_list[i] - cyc_Ca_list[i-1] - Cb[i-2] 
        Ca[i] =   cyc_Ca_list[i-1] + Cb[i-2] 
    else:
        cyc_Ca_list[i]=0
        W_A=0
        Ca[i] = time_Ta.count(i)+W_A

    if cyc_Cb_list[i-1] + Ca[i-2] > time_Tb.count(i) + W_B:
        cyc_Cb_list[i] = cyc_Cb_list[i] + cyc_Cb_list[i-1] + Ca[i-2] - time_Tb.count(i) - W_B
        W_B =0
        Cb[i] = time_Tb.count(i) +W_B
    elif cyc_Cb_list[i-1] + Ca[i-2] < time_Ta.count(i)+ W_B:
        cyc_Cb_list[i] = 0
        W_B = time_Tb.count(i)-cyc_Cb_list[i] - cyc_Cb_list[i-1] - Ca[i-2] 
        Cb[i] =  cyc_Cb_list[i-1] + Ca[i-2] 
    else:
        cyc_Cb_list[i]=0
        W_B=0
        Cb[i] = time_Tb.count(i) +W_B
    




    

print (cyc_Ca_list)
print (cyc_Cb_list)


T =list(range(0,16))

plt.subplot(2,2,1)
plt.plot(T,cyc_Ca_list)
plt.xlabel('Time')
plt.ylabel('Cycles_Ca')
plt.grid()

plt.subplot(2,2,2)
plt.plot(T,cyc_Cb_list)
plt.xlabel('Time')
plt.ylabel('Cycles_Cb')
plt.grid()

waiting_timeA=[0]*16
waiting_timeB=[0]*16

for i in range (16):

    if cyc_Ca_list[i] >0:
        waiting_timeA[i]=waiting_timeA[i]
    else:
        waiting_timeA[i]=waiting_timeA[i-1] +1
print(waiting_timeA)

for i in range (16):

    if cyc_Cb_list[i] >0:
        waiting_timeB[i]=waiting_timeB[i]
    else:
        waiting_timeB[i]=waiting_timeB[i-1] +1
print(waiting_timeB)

mean_W_A=statistics.mean(waiting_timeA)
mean_W_B=statistics.mean(waiting_timeB)
print('mean waiting time A:',mean_W_A)
print('mean waiting time B:',mean_W_B)

plt.subplot(2,2,3)
plt.plot(T,waiting_timeA)
plt.xlabel('Time')
plt.ylabel('Waiting time A')
plt.grid()

plt.subplot(2,2,4)
plt.plot(T,waiting_timeB)
plt.xlabel('Time')
plt.ylabel('Waiting time B')
plt.grid()

plt.show()





