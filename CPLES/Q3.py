import myLibrary as Lib
import matplotlib.pyplot as plt
import math
import numpy as np

X, Y = Lib.get_X_and_Y_FromFile("esem_fit1.dat", delimiter='\t')

# (i)


newY = [math.log(i) for i in Y]
A, b, r = Lib.linear_fit(X, newY)

alpha = b
sigma0 = math.exp(A)

# For plotting:
# 100 linearly spaced numbers
x = np.linspace(0, 10, 100)

y = [sigma0*math.exp(alpha*i) for i in x]

# plot the functions
plt.plot(X, Y, 'bo', label='data')
plt.plot(x, y, 'c', label='sigma0*exp(alpha*T)')

plt.title(f"Exponential fitting with Pearson's r={r:.4f}")
plt.xlabel("T")
plt.ylabel("sigma")

plt.legend(loc='upper right')
# show the plot
plt.show()


# (ii)
newX = [math.log(i) for i in X]
newY = [math.log(i) for i in Y]
A, b, r = Lib.linear_fit(newX, newY)

alpha = b
sigma0 = math.exp(A)

# For plotting:
# 100 linearly spaced numbers
x = np.linspace(0, 10, 100)

y = [sigma0*i**(alpha) for i in x]

# plot the functions
plt.plot(X, Y, 'bo', label='data')
plt.plot(x, y, 'c', label='sigma0*T^(alpha)')

plt.title(f"Power fitting with Pearson's r={r:.4f}")
plt.xlabel("T")
plt.ylabel("sigma")

plt.legend(loc='upper right')
# show the plot
plt.show()
