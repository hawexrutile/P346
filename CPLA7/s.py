import myLibrary as lib
import math
import matplotlib.pyplot as plt
import numpy as np
def f1(x,y,z):#defining the differential function for 1st state
    return -(math.pi**2)*y/4
def f2(x,y,z):#defining the differential function for 2nd state
    return -(math.pi**2)*y

soln1=lib.shoot(f1,0.05,0,0,0,2,-5)#calling the functions
soln2=lib.shoot(f2,0.05,0,0,0,2,-5)
#plotting the output
plt.plot(soln1[0], soln1[1])
plt.title("Psi vs distance")
plt.xlabel("Distance -->")
plt.ylabel("Psi -->")

plt.show()

plt.plot(soln2[0], soln2[1])
plt.title("Psi vs distance")
plt.xlabel("Distance -->")
plt.ylabel("Psi -->")

plt.show()
