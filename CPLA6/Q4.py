import myLibrary as lib


def f(x): return x**3
def g(x): return x**2



Res = lib.TrapezoidalI(f, 2, 0, 100)/lib.TrapezoidalI(g, 2, 0, 100)
print("\nThe centre of mass of the 2 meter long beam having linear mass density λ(x) = x², where x is measured from one of the ends is at", Res, "meter")

############################## OUTPUT ##############################
"""

The centre of mass of the 2 meter long beam having linear mass density λ(x) = x², where x is measured from one of the ends is at 1.5000749962501885 
meter


"""