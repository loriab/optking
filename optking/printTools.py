#from __future__ import print_function
### Printing functions.
from sys import stdout
import history


def print_opt(arg):
    print(arg, file=stdout, end='')
    return

#print_opt = cleanPrint

#def printInit(printFunction=None, file=stdout):
    #if not printFunction:
#    print_opt = cleanPrint
    #else:
    #    print_opt = printFunction


def printMat(M, Ncol=7, title=None):
    if title:
        print_opt(title + '\n')
    for row in range(M.shape[0]):
        tab = 0
        for col in range(M.shape[1]):
            tab += 1
            print_opt(" %10.6f" % M[row, col])
            if tab == Ncol and col != (M.shape[1] - 1):
                print_opt("\n")
                tab = 0
        print_opt("\n")
    return


def printMatString(M, Ncol=7, title=None):
    if title:
        print_opt(title + '\n')
    s = ''
    for row in range(M.shape[0]):
        tab = 0
        for col in range(M.shape[1]):
            tab += 1
            s += " %10.6f" % M[row, col]
            if tab == Ncol and col != (M.shape[1] - 1):
                s += '\n'
                tab = 0
        s += '\n'
    return s


def printArray(M, Ncol=7, title=None):
    if title:
        print_opt(title + '\n')
    tab = 0
    for col, entry in enumerate(M):
        tab += 1
        print_opt(" %10.6f" % M[col])
        if tab == Ncol and col != (len(M) - 1):
            print_opt("\n")
            tab = 0
    print_opt("\n")
    return


def printArrayString(M, Ncol=7, title=None):
    if title:
        print_opt(title + '\n')
    tab = 0
    s = ''
    for i, entry in enumerate(M):
        tab += 1
        s += " %10.6f" % entry
        if tab == Ncol and i != (len(M) - 1):
            s += '\n'
            tab = 0
    s += '\n'
    return s


def printGeomGrad(geom, grad):
    print_opt("\tGeometry and Gradient\n")
    Natom = geom.shape[0]

    for i in range(Natom):
        print_opt("\t%20.10f%20.10f%20.10f\n" % (geom[i, 0], geom[i, 1], geom[i, 2]))
    print_opt("\n")
    for i in range(Natom):
        print_opt("\t%20.10f%20.10f%20.10f\n" % (grad[3 * i + 0], grad[3 * i + 1],
                                                 grad[3 * i + 2]))


def welcome():
    print_opt("\n\t\t\t-----------------------------------------\n")
    print_opt("\t\t\t OPTKING 3.0: for geometry optimizations \n")
    print_opt("\t\t\t     By R.A. King, Bethel University     \n")
    print_opt("\t\t\t        with contributions from          \n")
    print_opt("\t\t\t    A.V. Copan, J. Cayton, A. Heide      \n")
    print_opt("\t\t\t-----------------------------------------\n")

def generate_file_output():
    print_opt("\n  ==> Optimization Summary <==\n\n")
    print_opt("  Measures of convergence in internal coordinates in au. (Any backward steps not shown.)\n")
    print_opt(
        "  --------------------------------------------------------------------------------------------------------------- ~\n"
    )
    print_opt(
        "   Step         Total Energy             Delta E       MAX Force       RMS Force        MAX Disp        RMS Disp  ~\n"
    )
    print_opt(
        "  --------------------------------------------------------------------------------------------------------------- ~\n"
    )
    
    history.oHistory.summary(printoption=True)                                                                     
