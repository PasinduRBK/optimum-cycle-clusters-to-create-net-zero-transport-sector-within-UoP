import numpy as np
import random
from operator import add
import matplotlib.pyplot as plt
import statistics 


L = 1000        # loop count 
MA = 40         # population at node A
MB = 40         # population at node B
MC = 40         # population at node C
P_AtoB =0.9     # probability AtoB
P_BtoA =0.9     # probability BtoA
P_CtoA =0.1     # Probability CtoA
Bic_A = 15      # maximum cycle count in A
Bic_B = 15      # maximum cycle count in B
Bic_C = 15      # maximun cycle count in C
T = 60          # time limit
lambda_A = 10    # mean arival time at A
lambda_B = 10    # mean arival time at B
lambda_C = 10    # mean arival time at C

#creating null lists

Ta=[]           # empty list for time A
Tb=[]           # empty list for time B
Tc=[]           # empty list for time C

B_A=[0]*L       # null list for cycle A
B_B=[0]*L       # null list for cycle B
B_C=[0]*L       # null list for cycle C

Avg_A=[0]*L     # null list for average time A
Avg_B=[0]*L     # null list for average time B
Avg_C=[0]*L     # null list for average time C

Avg_U_A=[0]*L   # null list for average utilization
Avg_U_B=[0]*L   # null list for average utilization
Avg_U_C=[0]*L   # null list for average utilization

def generate_random_integers(max_value1, max_value2, target_sum):
    x = random.randint(0, max_value1)
    y = random.randint(0, max_value2)

    while x + y != target_sum:
        x = random.randint(0, max_value1)
        y = random.randint(0, max_value2)

    return x, y


for j in range (L):

    randomprob_Pa =[]
    randomprob_Pb =[]
    randomprob_Pc =[]

    cyc_Ca_list=[0]*T
    cyc_Cb_list=[0]*T
    cyc_Cc_list=[0]*T

    Cab=[0]*T
    Cac=[0]*T
    Cba=[0]*T
    Cbc=[0]*T
    Cca=[0]*T
    Ccb=[0]*T

    G_AtoB = []
    G_AtoC = []
    G_BtoA = []
    G_BtoC = []
    G_CtoA = []
    G_CtoB = []

    waiting_timeA=[0]*T
    waiting_timeB=[0]*T
    waiting_timeC=[0]*T

    for i in range (0,MA):
        P = np.random.uniform(0.1, 0.9) 
        P = round(P,2)
        randomprob_Pa.append(P)
    for i in range (0,MB):
        P = np.random.uniform(0.1, 0.9) 
        P = round(P,2)
        randomprob_Pb.append(P)
    for i in range (0,MC):
        P = np.random.uniform(0.1, 0.9) 
        P = round(P,2)
        randomprob_Pc.append(P)
    
    # print(randomprob_Pa)
    # print(randomprob_Pb)
    # print(randomprob_Pc)

    Ta=np.random.poisson(lambda_A,MA)
    Ta=np.array(Ta)
    Ta.sort()
    time_Ta=Ta.tolist()
    #print(time_Ta)


    Tb=np.random.poisson(lambda_B,MB)
    Tb=np.array(Tb)
    Tb.sort()
    time_Tb=Tb.tolist()
    #print(time_Tb)


    Tc=np.random.poisson(lambda_C,MC)
    Tc=np.array(Tc)
    Tc.sort()
    time_Tc=Tc.tolist()
    #print(time_Tc)

    cyc_Ca = random.randint(1,Bic_A) 
    #print(cyc_Ca)

    cyc_Cb = random.randint(1,Bic_B)  
    #print(cyc_Cb)

    cyc_Cc = random.randint(1,Bic_C)
    #print(cyc_Cc)

    for i in range (MA):
        if randomprob_Pa[i]>=P_AtoB:
            G = time_Ta[i]
            G_AtoB.append(G)
        else:
            G = time_Ta[i]
            G_AtoC.append(G)
    for i in range (MB):
        if randomprob_Pb[i]>=P_BtoA:
            G = time_Tb[i]
            G_BtoA.append(G)
        else:
            G = time_Tb[i]
            G_BtoC.append(G)
    for i in range (MC):
        if randomprob_Pc[i]>=P_CtoA:
            G = time_Tc[i]
            G_CtoA.append(G)
        else:
            G = time_Tc[i]
            G_CtoB.append(G)

        

    for i in range (1):
        if cyc_Ca >= G_AtoB.count(i) + G_AtoC.count(i):
            cyc_Ca_list[i] = cyc_Ca_list[i] + cyc_Ca - G_AtoB.count(i) - G_AtoC.count(i)
            Cab[i] = G_AtoB.count(i)
            Cac[i] = G_AtoC.count(i)
            W_AB=0
            W_AC=0
        if cyc_Ca < G_AtoB.count(i) + G_AtoC.count(i):
            cyc_Ca_list[i] = 0
            x, y = generate_random_integers(G_AtoB.count(i), G_AtoC.count(i), cyc_Ca)
            Cab[i]= x
            Cac[i]= y
            W_AB = G_AtoB.count(i) - Cab[i]
            W_AC = G_AtoC.count(i) - Cac[i]
            waiting_timeA[i]= W_AB + W_AC

        if cyc_Cb >= G_BtoA.count(i) + G_BtoC.count(i):
            cyc_Cb_list[i] = cyc_Cb_list[i] + cyc_Cb - G_BtoA.count(i) - G_BtoC.count(i)
            Cba[i] = G_BtoA.count(i)
            Cbc[i] = G_BtoC.count(i)
            W_BA=0
            W_BC=0
        if cyc_Cb < G_BtoA.count(i) + G_BtoC.count(i):
            cyc_Cb_list[i] = 0
            x, y = generate_random_integers(G_BtoA.count(i), G_BtoC.count(i), cyc_Cb)
            Cba[i]= x
            Cbc[i]= y
            W_BA = G_BtoA.count(i) - Cba[i]
            W_BC = G_BtoC.count(i) - Cbc[i]
            waiting_timeB[i]= W_BA + W_BC

        if cyc_Cc >= G_CtoA.count(i) + G_CtoB.count(i):
            cyc_Cc_list[i] = cyc_Cc_list[i] + cyc_Cc - G_CtoA.count(i) - G_CtoB.count(i)
            Cca[i] = G_CtoA.count(i)
            Ccb[i] = G_CtoB.count(i)
            W_CA=0
            W_CB=0
        if cyc_Cc < G_CtoA.count(i) + G_CtoB.count(i):
            cyc_Cc_list[i] = 0
            x, y = generate_random_integers(G_CtoA.count(i), G_CtoB.count(i), cyc_Cc)
            Cca[i]= x
            Ccb[i]= y
            W_CA = G_CtoA.count(i) - Cca[i]
            W_CB = G_CtoB.count(i) - Ccb[i]
            waiting_timeC[i]= W_CA + W_CB


    for i in range (1,T):
        if cyc_Ca_list[i-1] + Cba[i-2] +Cca[i-2]>= G_AtoB.count(i)+ G_AtoC.count(i)+ W_AB + W_AC:
            cyc_Ca_list[i] = cyc_Ca_list[i] + cyc_Ca_list[i-1]  + Cba[i-2] + Cca[i-2] - G_AtoB.count(i)- G_AtoC.count(i)- W_AB - W_AC
            Cab[i] = G_AtoB.count(i)+ W_AB
            Cac[i] = G_AtoC.count(i) + W_AC
            W_AB =0
            W_AC =0
        else:
            cyc_Ca_list[i] = 0
            x, y = generate_random_integers(G_AtoB.count(i)+W_AB, G_AtoC.count(i)+W_AC, cyc_Ca_list[i-1] + Cba[i-2] +Cca[i-2])
            Cab[i] = x
            Cac[i] = y
            W_AB = G_AtoB.count(i) + W_AB - Cab[i]
            W_AC = G_AtoC.count(i) + W_AC - Cac[i]
            waiting_timeA[i]= W_AB + W_AC


        if cyc_Cb_list[i-1] + Cab[i-2] +Ccb[i-2]>= G_BtoA.count(i)+ G_BtoC.count(i)+ W_BA + W_BC:
            cyc_Cb_list[i] = cyc_Cb_list[i] + cyc_Cb_list[i-1]  + Cab[i-2] + Ccb[i-2] - G_BtoA.count(i)- G_BtoC.count(i)- W_BA - W_BC
            Cba[i] = G_BtoA.count(i)+ W_BA
            Cbc[i] = G_BtoC.count(i) + W_BC
            W_BA =0
            W_BC =0
        else:
            cyc_Cb_list[i] = 0
            x, y = generate_random_integers(G_BtoA.count(i)+W_BA, G_BtoC.count(i)+W_BC, cyc_Cb_list[i-1] + Cab[i-2] +Ccb[i-2])
            Cba[i] = x
            Cbc[i] = y
            W_BA = G_BtoA.count(i) + W_BA - Cba[i]
            W_BC = G_BtoC.count(i) + W_BC - Cbc[i]
            waiting_timeB[i]= W_BA + W_BC

        if cyc_Cc_list[i-1] + Cac[i-2] +Cbc[i-2]>= G_CtoA.count(i)+ G_CtoB.count(i)+ W_CA + W_CB:
            cyc_Cc_list[i] = cyc_Cc_list[i] + cyc_Cc_list[i-1]  + Cac[i-2] + Cbc[i-2] - G_CtoA.count(i)- G_CtoB.count(i)- W_CA - W_CB
            Cca[i] = G_CtoA.count(i)+ W_CA
            Ccb[i] = G_CtoB.count(i) + W_CB
            W_CA =0
            W_CB =0
        else:
            cyc_Cc_list[i] = 0
            x, y = generate_random_integers(G_CtoA.count(i)+W_CA, G_CtoB.count(i)+W_CB, cyc_Cc_list[i-1] + Cac[i-2] +Cbc[i-2])
            Cca[i] = x
            Ccb[i] = y
            W_CA = G_CtoA.count(i) + W_CA - Cca[i]
            W_CB = G_CtoB.count(i) + W_CB - Ccb[i]
            waiting_timeC[i]= W_CA + W_CB
        
    # print (cyc_Ca_list)
    # print (cyc_Cb_list)
    # print (cyc_Cc_list)

    mean_W_A=sum(waiting_timeA)/MA
    mean_W_B=sum(waiting_timeB)/MB
    mean_W_C=sum(waiting_timeC)/MC

    # print('mean waiting time A:',mean_W_A)
    # print('mean waiting time B:',mean_W_B)
    # print('mean waiting time c:',mean_W_C)

    mean_U_A = sum(cyc_Ca_list)/T   
    mean_U_B = sum(cyc_Cb_list)/T  
    mean_U_C = sum(cyc_Cc_list)/T

    #print('mean utilization at A:',mean_U_A)
    #print('mean utilization at B:',mean_U_B)
    #print('mean utilization at C:',mean_U_C)

    B_A[j] = cyc_Ca
    B_B[j] = cyc_Cb
    B_C[j] = cyc_Cc

    Avg_A[j] = mean_W_A
    Avg_B[j] = mean_W_B
    Avg_C[j] = mean_W_C

    Avg_U_A[j]= mean_U_A
    Avg_U_B[j]= mean_U_B
    Avg_U_C[j]= mean_U_C

# create bestfit curves

mymodelA=np.poly1d(np.polyfit(B_A,Avg_A,3))
mymodelB=np.poly1d(np.polyfit(B_B,Avg_B,3))
mymodelC=np.poly1d(np.polyfit(B_C,Avg_C,3))
mymodelUA=np.poly1d(np.polyfit(B_A,Avg_U_A,5))
mymodelUB=np.poly1d(np.polyfit(B_B,Avg_U_B,5))
mymodelUC=np.poly1d(np.polyfit(B_C,Avg_U_C,5))

mylineA = np.linspace(1,Bic_A,100)
mylineB = np.linspace(1,Bic_B,100)
mylineC = np.linspace(1,Bic_C,100)


# plot

plt.subplot(2,3,1)
plt.scatter(B_A,Avg_A)
plt.plot(mylineA,mymodelA(mylineA),color='red')
plt.suptitle(f"People A = {MA}, People B = {MB} , People C ={MC} \n Bicycle A = {Bic_A}, Bicycle B = {Bic_B}, Bicycle C = {Bic_C} \n lambda A = {lambda_A}, lambda B = {lambda_B}, lambda ={lambda_C} \n Probability AtoB = {P_AtoB}, probability BtoA = {P_BtoA}, Probability CtoA = {P_CtoA}", fontsize=10, color='red')
plt.xlabel('cyc_Ca')
plt.ylabel(' Avg Waiting time A')
plt.grid()

plt.subplot(2,3,2)
plt.scatter(B_B,Avg_B)
plt.plot(mylineB,mymodelB(mylineB),color='red')
plt.xlabel('cyc_Cb')
plt.ylabel(' Avg Waiting time B')
plt.grid()


plt.subplot(2,3,3)
plt.scatter(B_C,Avg_C)
plt.plot(mylineC,mymodelC(mylineC),color='red')
plt.xlabel('cyc_Cc')
plt.ylabel(' Avg Waiting time C')
plt.grid()

plt.subplot(2,3,4)
plt.scatter(B_A,Avg_U_A)
plt.plot(mylineA,mymodelUA(mylineA),color='red')
plt.xlabel('cyc_Ca')
plt.ylabel(' Avg Utilization A')
plt.grid()

plt.subplot(2,3,5)
plt.scatter(B_B,Avg_U_B)
plt.plot(mylineB,mymodelUB(mylineB),color='red')
plt.xlabel('cyc_Cb')
plt.ylabel(' Avg Utilization B')
plt.grid()

plt.subplot(2,3,6)
plt.scatter(B_C,Avg_U_C)
plt.plot(mylineC,mymodelUC(mylineC),color='red')
plt.xlabel('cyc_Cc')
plt.ylabel(' Avg Utilization C')
plt.grid()

plt.show()



