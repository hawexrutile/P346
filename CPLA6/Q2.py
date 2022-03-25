from math import sqrt
import myLibrary as lib


def f(x):
    return (x*sqrt(1+x))
print(lib.MidPointI(f,1,0,7))
print(lib.TrapezoidalI(f,1,0,10))
print(lib.SimpsonI(f,1,0,2))


#############################################Solution#################################################
'''
Midpoint 
N=7
I=0.643137698010569


Trapezoidal
N=10
I=0.6444300126935936

Simpsons
N=2
I=0.6439505508593788

'''