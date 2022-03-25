class Matrix:                          #classs to hold all the functions
    def __init__(self,matrixA,matrixB):
        self.A=matrixA
        self.B=matrixB
        
    def MatrixCreator(self,A):         #to create[[]],[],[]] form for the matrix:helps in programing
        n=0                            #even though defined in __innit__(); while calling the function u have to specify the matrix again
        AR1=[]; AR2=[]; AR3=[];


        while n<9:
            if n <3:
                AR1.append(A[n])   #The original matrix becomes useless
                n+=1
            elif n<6:
                AR2.append(A[n])
                n+=1
            else:
                AR3.append(A[n])
                n+=1
        A=[AR1,AR2,AR3]
        return A

    def MatrixTranspose(self,B):       #same as above-specify the matrix
        TBR1=[]; TBR2=[]; TBR3=[];     #only transposes matrices;I didn thtink it to be useful to transform vectors
        C=self.MatrixCreator(B)
        for vctr in C:
                TBR1.append(vctr.pop(0))
                TBR2.append(vctr.pop(0))
                TBR3.append(vctr.pop(0))

        TB=[TBR1,TBR2,TBR3]
        return TB

    def MatrixMutiplier(self):        #here u dont need to specify the matrices;It takes from self  #the original matrix(list form) gets distroyed
        TB=self.MatrixTranspose(self.B)    #same here; also we take transpose here to make programing simpler(now insted of row-column, it became a ror-row multiplication)
        A=self.MatrixCreator(self.A)
        AB=[]
        N=0
        for elem in A:                #row selector  of matrix A
            N=0
            while N<3:                #row selector  of matrix TB
                sum=0
                n=0
                for e in elem:        #element selector for a give row of matrix A
                    sum=sum+e*TB[N][n]
                    n=n+1             #element selector for a give row of matrix TB
                N+=1
                    
                AB.append(sum)
        return self.MatrixCreator(AB)
    def MatrixOperator(self,morv,vorm):#for matrix vector multiplication(any order)
        if len(morv)==9:#(matrix x vector)
            V=vorm
            A=self.MatrixCreator(morv)
            AV=[]
            for elem in A:
                sum=0
                n=0
                for e in elem:
                    sum=sum+e*V[n]
                    n=n+1
                AV.append(sum)
            return (AV)
            
        elif len(morv)==3:#(vector x matrix)
            V=morv
            TB=self.MatrixTranspose(vorm)
            VB=[]
            N=0
            while N<3:
                sum=0
                n=0
                for e in morv:
                    sum=sum+e*TB[N][n]
                    n=n+1
                N+=1                    
                VB.append(sum)
            return (VB)
        #need to do some error check for wrong inputs
    def VectorMultiplier(self,V1,V2):#vector-vector multiplication
        v=[]
        n=0
        sum=0
        while n<3:
                sum=sum+(V1[n]*V2[n])
                n=n+1
        v.append(sum)
        return v



        
        
A=[
    0, 1, 0,
    1, 0, 1,
    0, 1, 0
]
B=[
    1, 2, 3,
    4, 1, 0,
    7, 8, 1
]
V=[1,2,3]
m=Matrix(A,B)
#print(m.MatrixCreator(B))   #dont call them together because of the pop in transpose
#print(m.MatrixTranspose(B))
print(m.MatrixMutiplier())
#print(m.MatrixOperator(B,V))
#print(m.MatrixOperator(V,B))
#print(m.VectorMultiplier(V,V)) 

