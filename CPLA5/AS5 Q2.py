import math
import myLibrary as Lib
import matplotlib.pyplot as plt
from tabulate import tabulate

accuracy = 10**(-5)
a = -1
b = 0
x = 0.0


def f(x):
    return (-x - math.cos(x))


bis = Lib.bisection(f, a, b, accuracy)
regF = Lib.regulaFalsi(f, a, b, accuracy)
NR = Lib.newtonRaphson(f, x, accuracy)
print("\n")
print("The root using the Bisection method is:")
print(bis[0])
print("The root using the Regula Falsi method is:")
print(regF[0])
print("The root using the Newton Raphson method is:")
print(NR[0])


# For plotting:
plt.subplot(3, 1, 1)

x_bis = range(1, bis[1]+1)
y_bis = bis[2]

plt.plot(x_bis, y_bis)
plt.title("Bisection")
plt.xlabel("No. of Iterations")
plt.ylabel("|Function Value|")

plt.subplot(3, 1, 2)

x_regF = range(1, regF[1]+1)
y_regF = regF[2]

plt.plot(x_regF, y_regF)
plt.title("Regula Falsi")
plt.xlabel("No. of Iterations")
plt.ylabel("|Function Value|")


plt.subplot(3, 1, 3)

x_NR = range(1, NR[1]+1)
y_NR = NR[2]

plt.plot(x_NR, y_NR)
plt.title("Newton Raphson")
plt.xlabel("No. of Iterations")
plt.ylabel("|Function Value|")

plt.suptitle("Convergence : Comparison")
plt.show()


# For table:

# 1.
print("\nTable showing Bisection root convergence:\n")
l = [[i, bis[3][i-1]] for i in x_bis]
table = tabulate(l, headers=['#Iterations', 'Root'],
                 tablefmt='orgtbl', floatfmt="")
print(table)

# 2.
print("\nTable showing Regula Falsi root convergence:\n")
l = [[i, regF[3][i-1]] for i in x_regF]
table = tabulate(l, headers=['#Iterations', 'Root'],
                 tablefmt='orgtbl', floatfmt="")
print(table)

# 3.
print("\nTable showing Newton Raphson root convergence:\n")
l = [[i, NR[3][i-1]] for i in x_NR]
table = tabulate(l, headers=['#Iterations', 'Root'],
                 tablefmt='orgtbl', floatfmt="")
print(table)


#################################### OUTPUT ####################################
""" 

The root using the Bisection method is:
-0.7390899658203125
The root using the Regula Falsi method is:
-0.7390847824489231
The root using the Newton Raphson method is:
-0.7390851332151704

Table showing Bisection root convergence:

|   #Iterations |                Root |
|---------------+---------------------|
|             1 | -0.5                |
|             2 | -0.75               |
|             3 | -0.625              |
|             4 | -0.6875             |
|             5 | -0.71875            |
|             6 | -0.734375           |
|             7 | -0.7421875          |
|             8 | -0.73828125         |
|             9 | -0.740234375        |
|            10 | -0.7392578125       |
|            11 | -0.73876953125      |
|            12 | -0.739013671875     |
|            13 | -0.7391357421875    |
|            14 | -0.73907470703125   |
|            15 | -0.739105224609375  |
|            16 | -0.7390899658203125 |

Table showing Regula Falsi root convergence:

|   #Iterations |                Root |
|---------------+---------------------|
|             1 | -0.6850733573260451 |
|             2 | -0.736298997613654  |
|             3 | -0.7389453559657132 |
|             4 | -0.7390781308800257 |
|             5 | -0.7390847824489231 |

Table showing Newton Raphson root convergence:

|   #Iterations |                Root |
|---------------+---------------------|
|             1 | -1.0000500024999313 |
|             2 | -0.7503638676089563 |
|             3 | -0.7391126462513903 |
|             4 | -0.7390851327747899 |
|             5 | -0.7390851332151704 |

"""
