import myLibrary as Lib

accuracy = 10**(-5)

coef = [1, 0, -5, 0, 4]
x = 2.1


# function to convert to superscript:
def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)


print("\n")
print("The input polynomial is :")
terms = []
for i in range(len(coef)):
    if coef[i] != 0:
        terms.append(f"{coef[i]}x{get_super(str(len(coef)-i-1))}")
result = ' + '.join(terms)
result = result.replace('x⁰', '')
result = result.replace('¹ ', ' ')
result = result.replace('+ -', '- ')
result = result.replace('1x', 'x')
print(result)


print("\b \nThe list of roots using the Laguerre and synthetic division method is:")
print(Lib.rootsLaguerre(coef, x, accuracy))


#################################### OUTPUT ####################################
""" 

The input polynomial is :
x⁴ - 5x² + 4

The list of roots using the Laguerre and synthetic division method is:
[-2.000000000025203, -0.9999999999522425, 0.9999999999544925, 2.000000000022953]

"""
