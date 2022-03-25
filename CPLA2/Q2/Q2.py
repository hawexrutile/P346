import math                      #input and output in list form
class myComplex:                 #Class for holding functions
    def __init__(self,z1,z2):    #z1 and z2 are the two complex numbers;[Real Part,Imaginary Part]----format
        self.a=z1[0]
        self.b=z1[1]
        self.c=z2[0]
        self.d=z2[1]
    def ComplexAdder(self):     # Adds complex numbers
        csum=[]
        rsum=self.a+self.c      #seperately adds real and complex parts
        isum=self.b+self.d
        csum.append(rsum)
        csum.append(isum)
        return csum
    def ComplexMultiplier(self):#Multiplies cmplex number
        cproduct=[]
        rproduct=(self.a*self.c)-(self.b*self.d) #real part
        iproduct=(self.a*self.d)+(self.c*self.b) #imaginary part
        cproduct.append(rproduct)
        cproduct.append(iproduct)
        return cproduct                          #output in list form
    def ComplexConjugator(self,c):               #Conjugates complex numbers
        cconjugate=[]
        cconjugate.append(c[0])
        cconjugate.append(-c[1])
        return cconjugate
    def ComplexModulus(self,c):                  #Gives modulus of complex numbers
        return math.sqrt(c[0]**2+c[1]**2)
    def ComplexPhase(self,c):                    #Gives the argument of a complex number
        return math.atan(c[1]/c[0])
    def ComplexDivider(self):                    #Divides 2 complex numbers
        cquotient=[]
        rproduct=((self.a*self.c)+(self.b*self.d))/(self.c**2+self.d**2)
        iproduct=(-(self.a*self.d)+(self.c*self.b))/(self.c**2+self.d**2)
        cquotient.append(rproduct)
        cquotient.append(iproduct)
        return (cquotient)


c=myComplex([1,2],[3,4])
d=[2,3]
#print (c.ComplexAdder())
#print (c.ComplexMultiplier())
#print (c.ComplexConjugator(d))
#print (c.ComplexModulus(d))
#print (c.ComplexPhase(d),"rad")
print (c.ComplexDivider())


