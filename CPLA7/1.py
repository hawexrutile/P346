import math
import myLibrary as Lib
import matplotlib.pyplot as plt

def f(y,x):
    return (y*(math.log(y))/x)
euexp1=Lib.forward_euler(f,math.exp(1),2,10,0.5)
euexp2=Lib.forward_euler(f,math.exp(1),2,10,0.2)
euexp3=Lib.forward_euler(f,math.exp(1),2,10,0.05)
prcor1=Lib.predictor_corrector(f,math.exp(1),2,10,0.5)
prcor2=Lib.predictor_corrector(f,math.exp(1),2,10,0.2)
prcor3=Lib.predictor_corrector(f,math.exp(1),2,10,0.05)


plt.plot(euexp1[0], euexp1[1])
plt.plot(prcor1[0], prcor1[1])
plt.title("y vs x (step size 0.5) ")
plt.xlabel("y -->")
plt.ylabel("x -->")

plt.show()
plt.plot(euexp2[0], euexp2[1])
plt.plot(prcor2[0], prcor2[1])
plt.title("y vs x (step size 0.2)")
plt.xlabel("x -->")
plt.ylabel("y -->")

plt.show()

plt.plot(euexp3[0], euexp3[1])
plt.plot(prcor3[0], prcor3[1])
plt.title("y vs x (step size 0.05)")
plt.xlabel("x -->")
plt.ylabel("y -->")

plt.show()

