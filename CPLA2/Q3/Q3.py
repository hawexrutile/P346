def AverageDistance(n):      #gives the average distance between n natural numbers
    if isinstance(n,int)==False:
        print("please input a natural number")
        return
    if n==0:
        print("please input a natural number")
        return
    if n<0:
        print("please input a natural number")
        return
    list=[]
    sum=0
    dclctn=range(n)          #Discrete Colection
    for d in dclctn:
        for c in dclctn:
            list.append(abs(d-c)) #creates a list of all posible permutation of distances
    for elem in list:        # adds all the distances
        sum=sum+elem
    return sum/len(list)     #average 
    

print(AverageDistance(7))