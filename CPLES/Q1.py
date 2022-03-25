
import math
import myLibrary as Lib
import matplotlib.pyplot as plt

accuracy = 10**(-4)
x = 4.139 # V=RT/P=300*0.0821/5.95=4.139


def f(x):
    return (((5.95)*x**3)-(((5.95*0.05422)+(0.0821*300))*x**2)+6.254*x-(6.254*0.05422))

NR = Lib.newtonRaphson(f, x, accuracy)
print("The root using the Newton Raphson method is:")
print(NR[0])

#########################Solution#####################
'''
The root using the Newton Raphson method is:
3.9299487677815472
'''