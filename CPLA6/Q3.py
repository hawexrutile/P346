import myLibrary as lib
import matplotlib.pyplot as plt
def f(x):
#     return(x**5)
    return (4/(1+x**2))


x = lib.MonteCarloI(f,1,0,5000)[1]
y = lib.MonteCarloI(f,1,0,5000)[0]
print(lib.MonteCarloI(f,1,0,5000)[2])

plt.plot(x, y)
plt.title("\u03C0 vs Monte Carlo Integration result")
plt.xlabel("Sample size -->")
plt.ylabel("Integration value -->")

plt.show()


############################################Solution###############################################
'''
value of pi :3.14215328448653
'''