#Program to find sum of N terms
def summer(N,selector):#input a natural number of yor choice
    n=float(N)
    sum=0
    s=selector
    if N<1:  #negative number check
        print("Please input positive natural number",N,"is not a positive natural number")
        return
    elif n.is_integer()==True:  #integer check
        if s==0:
            sumlist=range(N+1)
            for um in sumlist:  #sum of N integers
                sum=sum+um 
            return sum
        elif s==1:
            sumlist=range(N)
            for um in sumlist:  #sum of N odd integer
                sum=sum+(2*um+1) 
            return sum
        else:
            sumlist=range(N+1)
            for um in sumlist:  #sum of N odd integer
                sum=sum+(2*um) 
            return sum

    else:  #non-integer check
        print("Please input natural numbers",N,"is not a natural number")
        return
print( "sum of first,natural numbers is", summer(3.9,2) )

#print ("sum of first odd numbers is", summer()) 