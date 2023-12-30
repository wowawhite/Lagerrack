import sys

def mycoolfunction(parameter=None):
    tellmesomething = sys.__name__
    return tellmesomething

booya = mycoolfunction()
print(booya)