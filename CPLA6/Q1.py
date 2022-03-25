from math import sqrt
import myLibrary as lib
from tabulate import tabulate

def f(x):
    return (sqrt(1+1/x))

print("\nTable showing N vs Integral in Midpoint Method :\n")
l = [[i, lib.MidPointI(f,4,1,i)] for i in [8,16,24]]
table = tabulate(l, headers=['N', 'Integral'],
                 tablefmt='orgtbl', floatfmt="")
print(table)


########################################Solution#########################################################
'''
Table showing N vs Integral in Midpoint Method :

|   N |          Integral |
|-----+-------------------|
|   8 | 3.618313859329873 |
|  16 | 3.619709761707181 |
|  24 | 3.619972785533525 |
'''