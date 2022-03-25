def fact(n):
    if float(n).is_integer()==False:
        print(n, "is not a natural number") #non-integer checker
        return
    if n<0:
        print(n, "is not a positive natural number")  #Natural number checker
        return
    product=1
    l=range(1,n+1)
    for a in l:
        product=product*a
    print(n,"--------",product)

fact(5)
