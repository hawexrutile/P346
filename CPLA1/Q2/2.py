#AP Sum
def SeriesSummer(n, first_term,common_ratio_difference,selector):
    if float(n).is_integer()==False:  #non-integer check
        print("n has to be a natural number to form an AP,",n,"is not a natural number ")
        return
    elif n<=0:  #negative-integer check
        print("n has to be a positive natural number to form an AP,",n,"is not a positive natural number ")
        return
    else:
        sum=0
        lmtr=1
        clctr=[]
        crd=common_ratio_difference
        ft=first_term
        slctr=selector
        if slctr==0:
            while lmtr<=n:  #creating ap list
                clctr.append(ft)
                ft=ft+crd
                lmtr=lmtr+1

            for elem in clctr:  #summing
                sum=sum+elem 
            return sum
        elif slctr==1:
            while lmtr<=n:  #creating gp list
                clctr.append(ft)
                ft=ft*crd
                lmtr=lmtr+1

            for elem in clctr:  #summing
                sum=sum+elem
            return sum
        else:
            while lmtr<=n:
                if ft==0:  #checking for 0 in AP to avoid error while forming HP
                    print("the sequence's AP form contains a 0, please choose a different first term for the HP")
                    return
                else:
                    clctr.append(1/ft)  #creating ap list
                    ft=ft+crd
                    lmtr=lmtr+1 

            for elem in clctr:  #summing
                sum=sum+elem
            return sum
print (SeriesSummer(10,0,5,0))



