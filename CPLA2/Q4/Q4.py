import math
import random
place=[
    "australia", "canberra",
    "china", "beijing",
    "japan", "tokyo",
    "seoul",
    "thailand", "bangkok",
    "cameroon", "yaounde",
    "nigeria", "abuja",
    "bloemfontein", "pretoria",
    "canada", "ottawa",
    "jamaica", "kingston",
    "argentina",
    "brazil", "brasilia",
    "chile", "santiago",
    "wellington",
    "england", "london",
    "france", "paris",
    "germany", "berlin",
    "italy", "rome",
    "netherlands", "amsterdam",
    "norway", "oslo",
    "scotland", "edinburgh",
    "spain", "madrid",
    "sweden", "stockholm"  
]
place=["delhi metro"]
def hangman():
    list=[]
    ol=[]                #old list
    nl=[]                #new list
    ilist=[]
    word=place[random.randrange(1)] #selects the word
    for w in word:
        list.append(w)               #creates a list of all the letters
    length=len(list)
    error=math.ceil(0.4*length)      #maximum allowed wrong guesses
    e=0
    for w in word:                   #replaces the letters with *
        ilist.append("*")
    while e<=error:
        p=""
        if ilist==list:
            print ("You won")
            print("the name of the place is:",word)
            return
        else:
            n=0
            for i in ilist:          # ******** former
                p+=i
            print (p)
            alpha=input("Guess :")   #input letter
            if len(alpha)>1:
                print ("Please input a single string")
                pass
            if alpha.isalpha()==False:
                print ("please input alphabets")
                pass

            while n<len(list):        
                if list[n]==alpha:   #checks for the presence of the guessed letter in the word
                    ilist[n]=alpha   #replaces * with the correctly guessed letter
                    ol=nl.copy()     #old collection of correctly guessed letters
                    nl.append(list[n])#new collection of correctly guessed letters
                    n+=1
                    
                else:                #continues checking to the next letter
                    n=n+1
            if nl==ol:               #after the itertation if new list=old list imples the guess was wrong
                print("wrong guess")
                e=e+1
            else:                    #to make old and new list same for next iteration
                ol=nl.copy()

    print ("u lost")
    print("the name of the place is:",word)
hangman()



