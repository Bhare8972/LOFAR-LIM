#!/usr/bin/env python3

# Coded by Massimiliano Tomassoli, 2012.
#
# - Thanks to b49P23TIvg for suggesting that I should use a set operation
#     instead of repeated membership tests.
# - Thanks to Ian Kelly for pointing out that
#     - "minArgs = None" is better than "minArgs = -1",
#     - "if args" is better than "if len(args)", and
#     - I should use "isdisjoint".
#
def genCur(func, unique = True, minArgs = None):
    """ Generates a 'curried' version of a function. """
    def g(*myArgs, **myKwArgs):
        def f(*args, **kwArgs):
            if args or kwArgs:                  # some more args!
                # Allocates data to assign to the next 'f'.
                newArgs = myArgs + args
                newKwArgs = dict.copy(myKwArgs)
 
                # If unique is True, we don't want repeated keyword arguments.
                if unique and not kwArgs.keys().isdisjoint(newKwArgs):
                    raise ValueError("Repeated kw arg while unique = True")
 
                # Adds/updates keyword arguments.
                newKwArgs.update(kwArgs)
 
                # Checks whether it's time to evaluate func.
                if minArgs is not None and minArgs <= len(newArgs) + len(newKwArgs):
                    return func(*newArgs, **newKwArgs)  # time to evaluate func
                else:
                    return g(*newArgs, **newKwArgs)     # returns a new 'f'
            else:                               # the evaluation was forced
                return func(*myArgs, **myKwArgs)
        return f
    return g
 
def cur(f, minArgs = None):
    return genCur(f, True, minArgs)
 
def curr(f, minArgs = None):
    return genCur(f, False, minArgs)
 
if __name__ == "__main__":
    # Simple Function.
    def func(a, b, c, d, e, f, g = 100):
        print(a, b, c, d, e, f, g)
     
    # NOTE: '<====' means "this line prints to the screen".
     
    # Example 1.
    f = cur(func)                   # f is a "curried" version of func
    c1 = f(1)
    c2 = c1(2, d = 4)               # Note that c is still unbound
    c3 = c2(3)(f = 6)(e = 5)        # now c = 3
    c3()                            # () forces the evaluation              <====
                                    #   it prints "1 2 3 4 5 6 100"
    c4 = c2(30)(f = 60)(e = 50)     # now c = 30
    c4()                            # () forces the evaluation              <====
                                    #   it prints "1 2 30 4 50 60 100"
     
    print("\n------\n")
     
    # Example 2.
    f = curr(func)                  # f is a "curried" version of func
                                    # curr = cur with possibly repeated
                                    #   keyword args
    c1 = f(1, 2)(3, 4)
    c2 = c1(e = 5)(f = 6)(e = 10)() # ops... we repeated 'e' because we     <====
                                    #   changed our mind about it!
                                    #   again, () forces the evaluation
                                    #   it prints "1 2 3 4 10 6 100"
     
    print("\n------\n")
     
    # Example 3.
    f = cur(func, 6)        # forces the evaluation after 6 arguments
    c1 = f(1, 2, 3)         # num args = 3
    c2 = c1(4, f = 6)       # num args = 5
    c3 = c2(5)              # num args = 6 ==> evalution                    <====
                            #   it prints "1 2 3 4 5 6 100"
    c4 = c2(5, g = -1)      # num args = 7 ==> evaluation                   <====
                            #   we can specify more than 6 arguments, but
                            #   6 are enough to force the evaluation
                            #   it prints "1 2 3 4 5 6 -1"
     
    print("\n------\n")
     
    # Example 4.
    def printTree(func, level = None):
        if level is None:
            printTree(cur(func), 0)
        elif level == 6:
            func(g = '')()      # or just func('')()
        else:
            printTree(func(0), level + 1)
            printTree(func(1), level + 1)
     
    printTree(func)
     
    print("\n------\n")
     
    def f2(*args):
        print(", ".join(["%3d"%(x) for x in args]))
     
    def stress(f, n):
        if n: stress(f(n), n - 1)
        else: f()               # enough is enough
     
    stress(cur(f2), 100)