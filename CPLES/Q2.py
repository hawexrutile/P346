import math
import myLibrary as Lib
import matplotlib.pyplot as plt

def f(x):
    return (math.exp(-(x**2)/16)/math.sqrt(x**2+1.5**2))

integ=Lib.integ_Simpsons(f,-1,3,12)
print ("The potential at height 1.5m is",integ)

##########################Solution#########################
'''
The potential at height 1.5m is 1.8728487731724819
'''