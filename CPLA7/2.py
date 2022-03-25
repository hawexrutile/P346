import math
import myLibrary as Lib
import matplotlib.pyplot as plt

def f(x,y,z):
    return 1-x-z-y*(0)
RK4=Lib.rk4_2nd(f,0.05,1,2,0,5)

def g(x):
    return (math.exp(-x)-((x**2)/2)+(2*x))
x=[]
y=[]
n=10000
for i in range (n):
    x.append(-5+i*10/n)
    y.append(g(-5+i*10/n))

plt.plot(RK4[0], RK4[1])
plt.plot(x,y)
plt.title("y vs x")
plt.xlabel("x -->")
plt.ylabel("y -->")
plt.ylim(-5,5)
plt.show()