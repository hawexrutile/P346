import math
import myLibrary as Lib
import matplotlib.pyplot as plt

def f(x,y,z):
    return -(0.01*(293.15-y))+(0*x)+(0*z)
shoot=Lib.shoot(f,313.15,473.15,0,10,0.01)


# Boundary value problem using shooting method(Guesses taken for dy/dx are 1 and 100)




# Print x at T=100 i.e. x(T=100) =>The length of the rod at which the temperature ,T=100 
print("The length of the rod at which the temperature ,T=100  is {} m.".format(fn.roundoff(x[a],4)))
plt.plot(shoot[0], shoot[1])
plt.title("T as a function of x")
plt.xlabel("x")
plt.ylabel("Temperature T(K)")
plt.show()

########################solution##########################
'''
point of 100 deg C was found manualy as 4.37m from plot
'''

