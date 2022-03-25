def ExpAccuracyCheck(x):
    import math
    import matplotlib.pyplot as plt
    N=1
    float(N)
    n=1
    xl=[]
    yl=[]
    while abs(math.exp(x)-N)>10**(-5):
        xl.append(n)
        yl.append(math.exp(x)-N)
        print(n,"------------",math.exp(x)-N )
        N=N+((x**n)/(math.factorial(n)))
        n=n+1
    plt.plot(xl,yl)
    plt.show()

def SinAccuracyCheck(x):
    import math
    import matplotlib.pyplot as plt
    N=x
    float(N)
    n=3
    t=1
    xl=[]
    yl=[]
    while abs(math.sin(x)-N)>10**(-5):
        xl.append(t)
        yl.append((math.sin(x)-N))
        print(t,"------------",math.sin(x)-N )
        N=N+(((-1)**t)*(x**n)/(math.factorial(n)))
        n=n+2
        t=t+1
    plt.plot(xl,yl)
    plt.show()
SinAccuracyCheck(3)
ExpAccuracyCheck(3)
