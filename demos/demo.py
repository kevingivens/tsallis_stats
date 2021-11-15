import sys

from sympy import Symbol

sys.path.append("..")

from tsallis.tsallis import q_gaussian

if __name__ == "__main__":
    x = Symbol('x')
    print(x)