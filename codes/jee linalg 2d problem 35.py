

import numpy as np
import matplotlib.pyplot as plt

m=2                             #Slope of line through origin
n=np.array([m,-1])              #norm vector of line through origin
n1=np.array([4,3])              #norm vector of 4x+3y=10
n2=np.array([8,6])              #norm vector of 8x+6y+5=0
p=0
p1=10
p2=-5
N1=np.vstack((n,n1))            #vertical stacking n and n1
P1=np.vstack((p,p1))            #vertical stacking n and n2
N2=np.vstack((n,n2))            #vertical stacking p and p1
P2=np.vstack((p,p2))            #vertical stacking p and p2
A=np.matmul(np.linalg.inv(N1),P1)   #point A
B=np.matmul(np.linalg.inv(N2),P2)   #point B
oa=np.linalg.norm(A)                #length of OA
ob=np.linalg.norm(B)                #length of OB
Ratio=oa/ob
print(Ratio)



P=np.array([-1,14/3])               #P and Q are points on the line 4x+3y=10
Q=np.array([13/4,-1])
R=np.array([-1,1/2])                #R and S are points on the line 8x+6y+5=0
S=np.array([1/8,-1])
M=np.array([45/(12*m+16),(45*m)/(12*m+16)])  #M and N are points on the line y=mx
N=np.array([-15/(12*m+16),(-15*m)/(12*m+16)])

len=10
lam= np.linspace(0,1,len)
xPQ = np.zeros((2,len))
xRS= np.zeros((2,len))
xMN = np.zeros((2,len))
# points between PQ,RS,MN are stored in 2-d arrays xPQ,XRS,XMN respectively
for i in range(len):
  temp1 = P + lam[i]*(Q-P)
  xPQ[:,i]= temp1.T
  temp2 = R + lam[i]*(S-R)
  xRS[:,i]= temp2.T
  temp3 = M + lam[i]*(N-M)
  xMN[:,i]= temp3.T

plt.plot(xPQ[0,:],xPQ[1,:],label='$PQ$')
plt.plot(xRS[0,:],xRS[1,:],label='$RS$')
plt.plot(xMN[0,:],xMN[1,:],label='$MN$')
plt.plot(A[0],A[1],'o')    #plotting point A
plt.text(A[0]*(1 + 0.15),A[1]*(1-0.05) ,'A')
plt.plot(B[0],B[1],'o')    #plotting point B
plt.text(B[0]*(1+0.9),B[1]*(1+0.4) ,'B')
plt.plot(0,0,'o')          #plotting origin
plt.text(-0.1,0.2,'O')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.show()
