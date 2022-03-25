import random
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import copy
from math import sqrt

# Contents:
# 1. Matrix Operations
# 2. Guass-Jordan
# 3. LU Decomposition
# 4. Root Finding
# 5. Numerical Integration
# 6. Numerical differential equations
# 7. Interpolation and least square fitting


############################### Matrix Operations ###############################


def getMatFromFile(mat_name: str):
    # opening the file containing matrix to read its contents
    with open(f'{mat_name}.csv', newline='') as file:

        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC, delimiter=' ')

        # storing all the rows in an output (2-D)list
        output = []
        for row in reader:
            output.append(row[:])

    return output


def displayMat(mat):
    if mat:
        for rows in mat:
            print([float('{:.2f}'.format(elem)) for elem in rows])
    print()


def nrow(mat):
    return len(mat)  # returns the no. of rows of a matrix


def ncol(mat):
    return len(mat[0])  # returns the no. of columns of a matrix


def matTranspose(mat):
    return [[mat[j][i] for j in range(nrow(mat))] for i in range(ncol(mat))]


def matMult(matA, matB):
    if ncol(matA) == nrow(matB):
        return [[sum(matA[i][k]*matB[k][j] for k in range(ncol(matA)))
                 for j in range(ncol(matB))] for i in range(nrow(matA))]
    else:
        print("\nIndexes do not match for matrix multiplication of %s and %s!" % (
            str(matA), str(matB)))

############################### Guass-Jordan ###############################


def swapRows(A, r, i, ncols):
    temp = [x for x in A[r]]
    A[r] = A[i]
    A[i] = temp


def partialPivot(A, r, nrows, ncols):
    pivot = A[r][r]
    newSwap = 0
    if abs(pivot) > 10 ** (-12):
        return ("success", newSwap)
    else:
        for i in range(r+1, nrows):
            if A[i][r] != 0:
                pivot = A[i][r]
                swapRows(A, r, i, ncols)
                newSwap += 1
                return ("success", newSwap)
                break
    if pivot == 0:
        return ("failed", newSwap)


def guassJordan(mat, nrows, ncols):
    A = mat
    for r in range(nrows):
        x = partialPivot(A, r, nrows, ncols)[0]
        if x == "failed":
            return "No solution exists"
        else:
            for c in range(ncols-1, r-1, -1):
                A[r][c] = A[r][c]/A[r][r]
            for row in range(nrows):
                if row != r and A[row][r] != 0:
                    factor = A[row][r]
                    for c in range(r, ncols):
                        A[row][c] -= factor*A[r][c]
    return A


def solve(mat, nrows, ncols):
    print("\nYour input system of equations in augmented matrix format is:")
    displayMat(mat)
    res = guassJordan(mat, nrows, ncols)
    print("\nYour matrix after GaussJordan Elimination is:")
    displayMat(res)
    if res == "No solution exists":
        print("No solution exists")
    else:
        print("The solutions are:")
        var_holder = {}
        for i in range(nrows):
            var_holder['x_' +
                       str(i+1)] = float('{:.2f}'.format(res[i][ncols-1]))
        locals().update(var_holder)

        for i in range(nrows):
            print(f"x_{i+1} =", vars()[f'x_{i+1}'])


def gjInverse(mat):
    print("\nYour input matrix for finding inverse is:")
    displayMat(mat)
    A = [mat[i]+[1 if i == j else 0 for j in range(len(mat))]
         for i in range(len(mat))]
    print("The augmented matrix is :")
    displayMat(A)
    A = guassJordan(A, len(mat), 2*len(mat))
    print("\nYour matrix after GaussJordan Elimination is:")
    displayMat(A)
    res = [[A[i][j] for j in range(len(A), 2*len(A))] for i in range(len(A))]
    print("Hence, the solution is:")
    displayMat(res)


def gjDeterminant(mat, ncols):
    print("\nYour input matrix for finding determinant is:")
    displayMat(mat)
    A = mat
    swapCount = 0
    for r in range(ncols):
        x = partialPivot(A, r, ncols, ncols)
        swapCount += int(x[1])
        if x[0] == "failed":
            return "No solution exists"
        else:
            for row in range(r+1, ncols):
                if A[row][r] != 0:
                    factor = A[row][r]/A[r][r]
                    for c in range(r, ncols):
                        A[row][c] -= factor*A[r][c]
    k = 1
    for i in range(ncols):
        k *= A[i][i]
    return ((-1) ** swapCount)*k

############################### LU Decomposition ###############################

# Creating augmented matrix


def aug_Mat(A, b):
    for i in range(len(A)):
        for j in range(len(b[i])):
            A[i].append(b[i][j])
    return A


def doolittle(mat: list, b: list, nrow: int):
    Ab = aug_Mat(mat, b)
    ncols = ncol(Ab)
    for i in range(nrow):
        x = partialPivot(Ab, i, nrow, ncols)[0]
        if x == "failed":
            return "No solution exists"
        else:
            for j in range(nrow):
                # Upper Triangular
                if i <= j:
                    s = sum([Ab[i][k]*Ab[k][j] for k in range(i)])
                    Ab[i][j] = Ab[i][j] - s
                # Lower Triangular
                else:
                    s = sum([Ab[i][k]*Ab[k][j] for k in range(j)])
                    Ab[i][j] = float((Ab[i][j] - s) / Ab[j][j])

    return Ab


def crout(mat: list, b: list, nrow: int):
    Ab = aug_Mat(mat, b)
    ncols = ncol(Ab)
    for i in range(nrow):
        x = partialPivot(Ab, i, nrow, ncols)[0]
        if x == "failed":
            return "No solution exists"
        else:
            for j in range(nrow):
                # Lower Triangular
                if i >= j:
                    s = sum([Ab[i][k]*Ab[k][j] for k in range(j)])
                    Ab[i][j] = Ab[i][j] - s

                # Upper Triangular
                else:
                    s = sum([Ab[i][k]*Ab[k][j] for k in range(i)])
                    Ab[i][j] = float((Ab[i][j] - s) / Ab[i][i])

    return Ab


# Combined forward and backward substitution to solve the system or find inverse; in case of doolittle
def substitution_doolittle(Ab):
    Arow = nrow(Ab)
    bcol = ncol(Ab)-Arow
    y = [[0 for y in range(bcol)] for x in range(Arow)]
    for i in range(Arow):
        for j in range(bcol):
            s = sum(Ab[i][k]*y[k][j] for k in range(i))
            y[i][j] = (Ab[i][j+Arow]-s)
    x = [[0 for y in range(bcol)] for x in range(Arow)]
    for i in range(Arow-1, -1, -1):
        for j in range(bcol):
            s = sum(Ab[i][k]*x[k][j] for k in range(i + 1, Arow))
            x[i][j] = (y[i][j]-s)/Ab[i][i]
    return x


# Combined forward and backward substitution to solve the system or find inverse; in case of crout
def substitution_crout(Ab):
    Arow = nrow(Ab)
    bcol = ncol(Ab)-Arow
    y = [[0 for y in range(bcol)] for x in range(Arow)]
    for i in range(Arow):
        for j in range(bcol):
            s = sum(Ab[i][k]*y[k][j] for k in range(i))
            y[i][j] = (Ab[i][j+Arow]-s)/Ab[i][i]
    x = [[0 for y in range(bcol)] for x in range(Arow)]
    for i in range(Arow-1, -1, -1):
        for j in range(bcol):
            s = sum(Ab[i][k]*x[k][j] for k in range(i+1, Arow))
            x[i][j] = (y[i][j]-s)
    return x


def cholesky(mat: list, b: list, nrow: int):
    Ab = aug_Mat(mat, b)
    ncols = ncol(Ab)
    for i in range(nrow):
        x = partialPivot(Ab, i, nrow, ncols)[0]
        if x == "failed":
            return "No solution exists"
        else:
            for j in range(i+1):
                # for diagonals
                if i == j:
                    s = sum([Ab[j][k]**2 for k in range(j)])
                    Ab[j][j] = float(sqrt(Ab[j][j] - s))
                # for non-diagonals
                if i > j:
                    s = sum([Ab[i][k]*Ab[j][k] for k in range(j)])
                    if Ab[j][j] > 0:
                        Ab[i][j] = float((Ab[i][j] - s)/Ab[j][j])
                        Ab[j][i] = Ab[i][j]

    return Ab

# Combined forward and backward substitution to solve the system or find inverse; in case of cholesky


def substitution_cholesky(Ab):
    Arow = nrow(Ab)
    bcol = ncol(Ab)-Arow
    y = [[0 for y in range(bcol)] for x in range(Arow)]
    for i in range(Arow):
        for j in range(bcol):
            f = sum(Ab[i][k]*y[k][j] for k in range(i))
            y[i][j] = (Ab[i][j+Arow]-f)/Ab[i][i]
    x = [[0 for y in range(bcol)] for x in range(Arow)]
    for i in range(Arow-1, -1, -1):
        for j in range(bcol):
            f = sum(Ab[i][k]*x[k][j] for k in range(i+1, Arow))
            x[i][j] = (y[i][j]-f)/Ab[i][i]
    return x

# solve a system of linear equations using the input method given as a parameter(defaulted to doolittle)


def solve_LU(mat: list, b: list, nrow: int, method="doolittle"):
    A = copy.deepcopy(mat)
    if method == "doolittle":
        A = doolittle(A, b, nrow)
        x = [substitution_doolittle(A)[i][0] for i in range(nrow)]
    elif method == "crout":
        A = crout(A, b, nrow)
        x = [substitution_crout(A)[i][0] for i in range(nrow)]
    elif method == "cholesky":
        A = cholesky(A, b, nrow)
        x = [substitution_cholesky(A)[i][0] for i in range(nrow)]
    for i in range(len(x)):
        print("x_"+str(i+1)+" = "+str('{:.2f}'.format(x[i])))

# for finding inverse of a matrix using doolittle's decomposition


def lu_Inverse(A, nrow):
    identity = []
    for i in range(nrow):
        row = [1 if (i == j) else 0 for j in range(nrow)]
        identity.append(row)

    return(substitution_doolittle(doolittle(A, identity, nrow)))


############################### Root Finding ###############################


def bisection(f, a, b, accuracy=10**(-6)):
    if a > b:
        print("\nInput a>b ! Choose a different interval such that a<b.")
        return
    else:
        beta = 1.5
        # Bracketing
        while f(a)*f(b) > 0:
            if abs(f(a)) < abs(f(b)):
                a -= beta*(b-a)
            else:
                b += beta*(b-a)
        #print(a, b)
        flag = 0
        maxiter = 1000
        i = 0
        convergenceFn = []
        convergenceRoot = []
        while flag == 0 and i < maxiter:
            c = (a+b)/2
            if f(a)*f(c) < 0:
                b = c
            else:
                a = c
            fx = f(c)
            convergenceFn.append(abs(fx))
            convergenceRoot.append(c)
            i += 1
            # Checking for convergence:
            if (b-a)/2 < accuracy and abs(fx) < accuracy:
                flag = 1
        return (c, i, convergenceFn, convergenceRoot)


def regulaFalsi(f, a, b, accuracy1=10**(-6), accuracy2=10**(-5)):
    if a > b:
        print("\nInput a>b ! Choose a different interval such that a<b.")
        return
    else:
        beta = 1.5
        # Bracketing
        while f(a)*f(b) > 0:
            if abs(f(a)) < abs(f(b)):
                a -= beta*(b-a)
            else:
                b += beta*(b-a)
        flag = 0
        maxiter = 1000
        i = 0
        convergenceFn = []
        convergenceRoot = []
        c_0 = a
        while flag == 0 and i < maxiter:
            # Using the Regula Falsi formula:
            c = b - ((b-a)*f(b))/(f(b)-f(a))
            if f(a)*f(c) < 0:
                b = c
            else:
                a = c
            fx = f(c)
            convergenceFn.append(abs(fx))
            convergenceRoot.append(c)
            i += 1
            # Checking for convergence:
            if abs(c - c_0) < accuracy1 and abs(fx) < accuracy2:
                flag = 1
            c_0 = c
        return (c, i, convergenceFn, convergenceRoot)


def newtonRaphson(f, x, accuracy=10**(-6)):
    flag = 0
    maxiter = 1000
    i = 0
    convergenceFn = []
    convergenceRoot = []
    h = 10**(-4)
    while flag == 0 and i < maxiter:
        x_0 = x
        Df_x = (f(x+h)-f(x))/h
        x -= f(x)/Df_x
        fx = f(x)
        convergenceFn.append(abs(fx))
        convergenceRoot.append(x)
        i += 1
        # Checking for convergence:
        if abs(x - x_0) < accuracy and abs(fx) < accuracy:
            flag = 1
    return (x, i, convergenceFn, convergenceRoot)


# Deflation of polynomial to one order reduced polynomial using synthetic division:
def deflation(p: list, root, accuracy):
    for i in range(1, len(p)):
        p[i] += p[i-1]*root
    if abs(p[len(p)-1]) < accuracy:
        return(p[0:len(p)-1])
    else:
        print(f"\nWrong root input, {root} given for deflation !")
        return

# Finding a single root using Laguerre's method:


def laguerre(p: list, x, accuracy):
    n = len(p)-1

    # Defining the polynomial function using the coefficients list-p:
    def f(x):
        return sum(p[i]*x**(len(p)-i-1) for i in range(len(p)))

    maxiter = 1000
    iter = 0
    flag = 0
    while flag == 0 and iter < maxiter:
        #print(iter, x)
        x_0 = x
        fx = f(x)
        if fx == 0:
            return x  # return if already converged
        # Finding derivatives and using Laguerre's formula:
        h = 10**(-4)
        Dfx = (f(x+h)-f(x))/h
        DDfx = (f(x+h)+f(x-h)-2*f(x))/2*h
        G = Dfx/fx
        H = G**2-(DDfx/fx)
        sq = sqrt((n-1)*(n*H-G**2))
        den1 = G + sq
        den2 = G - sq
        if abs(den1) > abs(den2):
            a = n/den1
        else:
            a = n/den2
        x -= a
        iter += 1
        # Checking for convergence:
        if abs(x - x_0) < accuracy and abs(f(x)) < accuracy:
            flag = 1
    return newtonRaphson(f, x_0)[0]  # return after polishing by Newton Raphson

# Finding and returning a list of all the roots using Laguerre's method:


def rootsLaguerre(p: list, x, accuracy):
    roots = []
    # The following lines of code will strip the list-p upto the first non-zero term so as to avoid errors later:
    flag = 0
    i = k = 0
    while i < len(p)-1 and flag == 0:
        if p[i] == 0:
            k += 1
        else:
            flag = 1
        i += 1
    p = p[k:len(p)]
    # Find all the roots except the last:
    for i in range(len(p)-2):
        root = laguerre(p, x, accuracy)
        roots.append(root)
        p = deflation(p, roots[len(roots)-1], accuracy)

    # Last root is found from the last monomial based on the sign of coefficient of x
    if p[0] == -1:
        roots.append(p[1])
    else:
        roots.append(-p[1])
    return sorted(roots)


############################### Numerical integration ###############################


def integ_Midpoint(f, a, b, n):
    s = 0
    h = (b-a)/n
    x_n = a+h/2
    for i in range(n):
        s += h*f(x_n)
        x_n = x_n + h
    return s


def integ_Trapezoidal(f, a, b, n):
    s = 0
    h = (b-a)/n
    x_n = a
    for i in range(n):
        s += h*(f(x_n)+f(x_n + h))/2
        x_n = x_n + h
    return s


def integ_Simpsons(f, a, b, n):
    s = 0
    h = (b-a)/n
    x_n = a
    for i in range(n):
        s += h*(f(x_n)+4*f(x_n + h/2)+f(x_n + h))/6
        x_n = x_n + h
    return s


def integ_MonteCarlo(f, a, b, N, initialSampleSize, sampleStepSize):
    h = (b-a)/N
    M = initialSampleSize
    M_list, S_list = [], []
    for i in range(N):  # loop on no. of samples
        s = 0
        M_list.append(M)
        for i in range(M):  # loop on  sample size
            # to get a random number between 'a' and 'b'
            x = a + (b-a)*random.random()
            s += f(x)/M  # summing for all random values
        S_list.append(s)
        M += sampleStepSize
    # multiplying by 'h' factor to get the expectation value of area
    return (sum(S_list)*h, M_list, S_list)


############################### Numerical differential equations ###############################

def forward_euler(f, y0, a, b, h):
    n = int((b-a)/h)
    h = (b-a)/n
    xList = [(a + i*h) for i in range(n)]
    yList = [None] * n  # initializing yList
    yList[0] = y0
    for i in range(n-1):
        yList[i+1] = yList[i] + h*f(yList[i], xList[i])
    return (xList, yList)


def backward_euler(f, y0, a, b, h):
    n = int((b-a)/h)
    h = (b-a)/n
    xList = [(a + i*h) for i in range(n)]
    yList = [None] * n  # initializing yList
    yList[0] = y0

    for i in range(n-1):
        def g(y): return y-yList[i]-h*f(y, xList[i+1])
        yNR = newtonRaphson(g, yList[i], min(10 ^ (-4), h/1000))[0]
        yList[i+1] = yList[i] + h*f(yNR, xList[i+1])
    return (xList, yList)


def predictor_corrector(f, y0, a, b, h):
    n = int((b-a)/h)
    h = (b-a)/n
    xList = [(a + i*h) for i in range(n)]
    yList = [None] * n  # initializing yList
    yList[0] = y0

    for i in range(n-1):
        k1 = h*f(yList[i], xList[i])
        k2 = h*f(yList[i]+k1, xList[i+1])
        yList[i+1] = yList[i] + 0.5*(k1+k2)
    return (xList, yList)


def rk2(f, y0, a, b, h):
    n = int((b-a)/h)
    h = (b-a)/n
    xList = [(a + i*h) for i in range(n)]
    yList = [None] * n  # initializing yList
    yList[0] = y0

    for i in range(n-1):
        k1 = h*f(yList[i], xList[i])
        k2 = h*f(yList[i]+k1/2, xList[i]+h/2)

        yList[i+1] = yList[i] + k2
    return (xList, yList)


def rk4(f, y0, a, b, h):
    n = int((b-a)/h)
    h = (b-a)/n
    xList = [(a + i*h) for i in range(n)]
    yList = [None] * n  # initializing yList
    yList[0] = y0

    for i in range(n-1):
        k1 = h*f(yList[i], xList[i])
        k2 = h*f(yList[i]+k1/2, xList[i]+h/2)
        k3 = h*f(yList[i]+k2/2, xList[i]+h/2)
        k4 = h*f(yList[i]+k3, xList[i+1])

        yList[i+1] = yList[i] + (1/6)*(k1+2*k2+2*k3+k4)
    return (xList, yList)


def rk4_2nd(function, h, z_0, y_0, x_0, x_n):
    n = int((x_n-x_0)/h)
    h = (x_n-x_0)/n
    X_list = [(x_0 + i*h) for i in range(n)]
    Z_list = [None] * n  # initializing zList
    Z_list[0] = z_0
    Y_list = [None] * n  # initializing yList
    Y_list[0] = y_0

    for i in range(n-1):

        # defining k1,k2,k3,k4 for all y and z=dy/dx
        v1 = h*function(X_list[i], Y_list[i], Z_list[i])
        k1 = h*Z_list[i]

        v2 = h*function(X_list[i]+h/2, Y_list[i]+k1/2, Z_list[i]+v1/2)
        k2 = h*(Z_list[i]+v1/2)

        v3 = h*function(X_list[i]+h/2, Y_list[i]+k2/2, Z_list[i]+v2/2)
        k3 = h*(Z_list[i]+v2/2)

        v4 = h*function(X_list[i+1], Y_list[i]+k3, Z_list[i]+v3)
        k4 = h*(Z_list[i]+v3)

        #finding y with the slopes and adding them to the list
        Y_list[i+1] = Y_list[i] + (1/6)*(k1+2*k2+2*k3+k4)
        Z_list[i+1] = Z_list[i] + (1/6)*(v1+2*v2+2*v3+v4)
    return (X_list, Y_list, Z_list)#returning the values of all x,y,z as list


def shoot(function, h, y_0, y_n, x_0, x_n, z_1, max_iter=1000):
    _, Y_list1, _ = rk4_2nd(function, h, z_1, y_0, x_0, x_n)
    flag=0
    a=1
    b=1
    if Y_list1[-1] > y_n:
        while flag==0:
            z_h = z_1+2**a
            yb_h = Y_list1[-1]
            _, y_list2, _ = rk4_2nd(function, h, z_h, y_0, x_0, x_n)
            if y_list2[-1] < y_n:
                z_l=-z_h
                yb_l=y_list2[-1]
                flag=1
            else:
                a=a+1
    else:
        while flag==0:
            z_l = z_1+10**b
            yb_l = Y_list1[-1]
            _, y_list2, _ = rk4_2nd(function, h, z_l, y_0, x_0, x_n)
            if y_list2[-1] > y_n:
                z_h=z_l
                yb_h=y_list2[-1]
                flag=1
            else:
                a=a+1
    iter = 0
    flag = 0
    while flag == 0 and iter < max_iter:
        # lagrange interpolation
        z = z_l + (z_h-z_l)*(y_n-yb_l)/(yb_h - yb_l)
        X_list, Y_list, Z_list = rk4_2nd(function, h, z, y_0, x_0, x_n)
        if abs(Y_list[-1]-y_n) < 10 ^ (-1):
            flag = 1
        else:
            if Y_list[-1] > y_n:
                z_h = z
                yb_h = Y_list[-1]
            else:
                z_l = z
                yb_l = Y_list[-1]
        iter += 1
    print(iter)
    return X_list, Y_list, Z_list

    
############################### Interpolation and least square fitting ###############################
'''
def get_X_and_Y_FromFile(file_name: str, delimiter=' '):
    # opening the file containing data to read its contents
    with open(f'{file_name}.csv', newline='') as file:

        reader = csv.reader(file, delimiter=delimiter,
                            quoting=csv.QUOTE_NONNUMERIC)

        # storing all values in lists
        X, Y = [], []
        for row in reader:
            X.append(row[0])
            Y.append(row[1])
    return(X, Y)


X, Y = get_X_and_Y_FromFile("a")


def poly_interpolate(X, Y, x, degree=len(X)):
    P_n = sum(Y[i]*math.prod([(x-X[k])/(X[i]-X[k])
                              for k in range(degree) if k != i]) for i in range(degree))
    return P_n


print(poly_interpolate(X, Y, 14, degree=3))


def linear_fit(X, Y):
    n = len(X)
    xbar = (1/n)*sum(X)
    ybar = (1/n)*sum(Y)
    sxx = sum([i**2 for i in X]) - n*xbar**2
    syy = sum([i**2 for i in Y]) - n*ybar**2
    sxy = sum([X[i]*Y[i] for i in range(n)]) - n*xbar*ybar

    b = sxy/sxx
    a = ybar - b*xbar

    r = sxy/abs((sxx*syy)**0.5)
    return (a, b, r)


def poly_fit(X, Y, degree=1):
    n = len(X)
    k = degree
    A = [[sum([m**(i+j) for m in X]) for j in range(k+1)] for i in range(k+1)]
    b = [[sum([(X[m]**i)*Y[m] for m in range(n)])] for i in range(k+1)]
    A = doolittle(A, b, k+1)
    x = [substitution_doolittle(A)[i][0] for i in range(k+1)]

    # For calculation of pearson's r only:
    xbar = (1/n)*sum(X)
    ybar = (1/n)*sum(Y)
    sxx = sum([i**2 for i in X]) - n*xbar**2
    syy = sum([i**2 for i in Y]) - n*ybar**2
    sxy = sum([X[i]*Y[i] for i in range(n)]) - n*xbar*ybar

    r = sxy/abs((sxx*syy)**0.5)
    return (x, r)


#print(poly_fit(X, Y, degree=4))
'''