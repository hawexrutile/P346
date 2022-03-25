import math
import myLibrary as Lib
import matplotlib.pyplot as plt
from tabulate import tabulate

accuracy = 10**(-5)
a = 1.6
b = 2.4


def f(x):
    return (math.log10(x/2) - math.sin(5*x/2))


bis = Lib.bisection(f, a, b, accuracy)
regF = Lib.regulaFalsi(f, a, b, accuracy)
print("\n")
print("The root using the Bisection method is:")
print(bis[0])
print("The root using the Regula Falsi method is:")
print(regF[0])


# For plotting:
plt.subplot(2, 1, 1)

x_bis = range(1, bis[1]+1)
y_bis = bis[2]

plt.plot(x_bis, y_bis)
plt.title("Bisection")
plt.xlabel("No. of Iterations")
plt.ylabel("|Function Value|")

plt.subplot(2, 1, 2)

x_regF = range(1, regF[1]+1)
y_regF = regF[2]

plt.plot(x_regF, y_regF)
plt.title("Regula Falsi")
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


#################################### OUTPUT ####################################
""" 

The root using the Bisection method is:
2.5559669494628903
The root using the Regula Falsi method is:
2.555965092756274

Table showing Bisection root convergence:

|   #Iterations |               Root |
|---------------+--------------------|
|             1 | 2.5999999999999996 |
|             2 | 2.0999999999999996 |
|             3 | 2.3499999999999996 |
|             4 | 2.4749999999999996 |
|             5 | 2.5374999999999996 |
|             6 | 2.5687499999999996 |
|             7 | 2.5531249999999996 |
|             8 | 2.5609374999999996 |
|             9 | 2.5570312499999996 |
|            10 | 2.5550781249999996 |
|            11 | 2.5560546874999996 |
|            12 | 2.5555664062499996 |
|            13 | 2.5558105468749996 |
|            14 | 2.5559326171874996 |
|            15 | 2.5559936523437496 |
|            16 | 2.5559631347656246 |
|            17 | 2.555978393554687  |
|            18 | 2.555970764160156  |
|            19 | 2.5559669494628903 |

Table showing Regula Falsi root convergence:

|   #Iterations |               Root |
|---------------+--------------------|
|             1 | 3.2159211611751535 |
|             2 | 2.3423647942168495 |
|             3 | 2.6773650463616803 |
|             4 | 2.5566388779855718 |
|             5 | 2.5559489496197685 |
|             6 | 2.5559650941718037 |
|             7 | 2.555965092756274  |

"""
