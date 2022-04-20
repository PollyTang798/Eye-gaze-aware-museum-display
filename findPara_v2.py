import scipy.optimize
import numpy as np
import math
import DLT

g = [[50, 100, 127],
     [100, 150, 202],
     [150, 200, 277],
     [13, 24, 31],
     [34, 23, 39],
     [42, 35, 56],
     [62, 34, 64],
     [43, 24, 45],
     [53, 235, 270]]

t = [[4049.169907, 576.400314],
     [4602.078478, 621.573148],
     [4846.345901, 641.529886],
     [4209.448275, 593.357944],
     [6304.392276, 735.115223], 
     [5922.307579, 738.652925],
     [6871.989268, 810.309099],
     [6850.994416, 815.473325],
     [2926.531489, 494.252034]]
#t = np.array(t)
t3 = [[4049.169907, 576.400314, 1.],
     [4602.078478, 621.573148, 1.],
     [4846.345901, 641.529886, 1.],
     [4209.448275, 593.357944, 1.],
     [6304.392276, 735.115223, 1.],
     [5922.307579, 738.652925, 1.],
     [6871.989268, 810.309099, 1.],
     [6850.994416, 815.473325, 1.],
     [2926.531489, 494.252034, 1.]]

e = [[-5., 50., 50.],
     [-5., 50., 50.],
     [-5., 50., 50.],
     [-5., 50., 50.],
     [-5., 50., 50.],
     [-5., 50., 50.],
     [-5., 50., 50.],
     [-5., 50., 50.],
     [-5., 50., 50.]]


#x0= np.array([10,20,15,5,30,30,10,50,20,15,30,1,1,0])   # randomize initial guess and find which one will achieve the smallest result
#x0= np.array([15,25,15,5,35,35,10,50,20,15,30,1,1,0])
#x0= np.array([37,23,47,16,18,26,5,51,34,1,46,1,1,0])
x0= np.array([5,12,3])

#y = x0.shape
#print(y)
#print(len(g))

"""
alpha = -1*np.dot(l,e)/np.dot(l,g)
q = e + alpha*g
f = np.sum(math.sqrt(np.sum(np.square(t-Hq))))
"""

def f(l):
    print('l = ',l)
    #ltemp = np.array([l[0],l[1],l[2]])
    alpha = -1*np.divide((np.dot(e,l) + 1),np.dot(g,l))   # g,e: Nx4, l: 4x1, alpha: Nx1
    #print('alpha = ',alpha)
    q = e + np.multiply(alpha.reshape((alpha.size, 1)),g)               # q: Nx4
    #print('q = ',q)
    H = DLT.DLT(q,t)
    H = np.reshape(H,(3,4))
    #print('H = ',H)
    #print('H size: ',np.size(H))
    qi = []
    for x in q:
        #print('x: ',x)
        x = np.append(x,[1])
        #print('x: ',x)
        temp = H.dot(x)
        #print('temp_1 = ',temp)
        temp = temp / temp[2]
        #print('temp_2 = ',temp)
        qi.append(temp)
    qi = np.array(qi)
    #print("qi = ",qi)
    f = np.sum(math.sqrt(np.sum((t3-qi)**2)))
    #print("f", f)
    return f

#f = lambda H: t-np.dot(H,e[0] + np.multiply(g[0],(-1*np.dot(e[0],l.T)/np.dot(g[0],l.T))))
result = scipy.optimize.fmin(func=f, x0=x0, maxiter=500)

print('l = ', result)

#lr = np.array([result[11],result[12],result[13],1])
#Hr = np.array([[result[0],result[1],result[2],result[3]],[result[4],result[5],result[6],result[7]],[result[8],result[9],result[10],1]])

#print("Final l = ",lr)
#print("Final H = ",Hr)
