import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def start(N):
  return np.zeros(N)

def zerofun(alpha):
  temp = np.sum(t * alpha)
  return 0

def indicator(alpha, x, t, s):
  index = 0
  index = np.sum(alpha * t * linear_kernel(s,x)) - b
  return index  

def linear_kernel(x1,x2):
  x2_T = np.transpose(x2)
  result = x1 * x2_T
  return result
'''
def polynomial_kernel(x1,x2):
  x1_T = np.transpose(x1)
  result = (x1_T * x2 + 1)**p
  return result

def RBF_kernal(x1,x2,sigma):
  x1_T = np.transpose(x1)
  result = math.exp(-((x1-x2)**2)/2/sigma**2)
  return result
'''
def pre_com(t,x,kernel): 
  P = np.zeros((N,N))
  for i in range (N):
    for j in range (N):
      P[i][j] = t[i] * t[j] * kernel(x[i][:],x[j][:])
  global P
  return P

def objective(alpha):
  temp_vec = alpha
  minimized = 0
  for i in range(N):
    #for j in range(N):
        #k = linear_kernel(inputs[i,:],inputs[j,:])
        #minimized += 1/2*alpha[i]*alpha[j] * t[i] * t[j]\
        #* linear_kernel(inputs[i,:],inputs[j,:])
    minimized -= temp_vec[i]
  return minimized

np.random.seed(100)
classA = np.concatenate((np.random.randn(10,2)*0.2+[1.5,0.5],\
         np.random.randn(10,2)*0.2+[-1.5,0.5]))
classB = np.random.randn(20,2)*0.2+[0.0, -0.5]

inputs = np.concatenate((classA, classB))
targets = np.concatenate((np.ones(classA.shape[0]),\
         -np.ones(classB.shape[0])))

N = inputs.shape[0]

permute = list(range(N))
random.shuffle(permute)
global inputs
inputs = inputs[permute, :]
global t
t = targets[permute]

C = 0.01
alpha = np.array(N)

B = [(0,C) for b in range(N)]
XC = {'type': 'eq', 'fun':zerofun(alpha)}

ret = minimize( objective, start, bounds=B, constraints=XC)
alpha = ret['x']
#note that each training sample will have a corresponding alpha value
print("ret.success: ", ret.success)

non_zero_alpha = []
for non_zero,i in enumerate(ret):
  if non_zero > 10**(-5):
    non_zero_alpha.append(non_zero,i)
print("non_zero_alpha: ",non_zero_alpha)

b = 0
b = np.sum(alpha * t * linear_kernel(s,x)) - t[s] # need to be fixed!

#####plotting
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
plt.axis('equal')
#plt.savefig('svmplot.pdf')
plt.show()

#plotting the decision boundary
xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4 ,4)

grid = np.array([[indicator(x, y) for x in xgrid] for y in ygrid])
plt.contour(xgrid, ygrid, grid, (-1, 0 ,1),\
            colors = ('red', 'black', 'blue'),linewidths=(1,3,1))

def start(N):
  return numpy.zeros(N)

#def zerofun(alpha):
#  temp = np.sum(t * alpha)#  return 0
