
import psi4
#from psi4 import *
#from psi4.core import *
import numpy as np

import optking
optking.printInit(psi4.core.print_out)

#
## Collect the user-specified OPTKING keywords in a dict.
#all_options = p4util.prepare_options_for_modules()
#optking_options = all_options['OPTKING']
#optking_user_options = {}
#for opt,optval in optking_options.items():
#    if optval['has_changed'] == True:
#        optking_user_options[opt] = optval['value']
#    
## Create initial molecular structure object in optking (atomic numbers,
## cartesian coordinates, masses, ...) using class method for PSI4 molecule.
## Initialize printing.
#import optking
#optking.printInit(print_out)
##optking.printInit()  # for default
#
#mol = core.get_active_molecule()
#import optking.molsys
#Molsys = optking.molsys.MOLSYS.fromPsi4Molecule(mol)
#
# Define a function for optking that takes a Cartesian, numpy geometry, and
#  puts it in place for subsequent gradient computations.  This function
#  may move COM or reorient; will return possible changed cartesian geometry.
def setGeometry_func(newGeom):
    mol = psi4.core.get_active_molecule()
    psi_geom =  psi4.core.Matrix.from_array( newGeom )
    mol.set_geometry( psi_geom )
    mol.update_geometry()
    return np.array( mol.geometry() )
#    newGeom[:] = np.array( mol.geometry() )

calcName = 0
# Define a function for optking that returns an energy and cartesian
#  gradient in numpy array format.
def gradient_func(xyz, printResults=True):
    mol = psi4.core.get_active_molecule()
    xyz[:] = setGeometry_func(xyz)
    psi4gradientMatrix, wfn = psi4.driver.gradient(calcName, molecule=mol, return_wfn=True)
    gradientMatrix = np.array( psi4gradientMatrix )
    E = wfn.energy()
    if printResults:
        print_out( '\tEnergy: %15.10f\n' % E)
        print_out( '\tGradient\n')
        print_out( str(gradientMatrix) )
        print_out( "\n")
    return E, np.reshape(gradientMatrix, (gradientMatrix.size))

def hessian_func(xyz, printResults=False): 
    mol = psi4.core.get_active_molecule()
    xyz[:] = setGeometry_func(xyz)
    H = psi4.driver.hessian(calcName, molecule=mol)
    if printResults:
        psi4.core.print_out( 'Hessian\n')
        H.print_out()
        psi4.core.print_out( "\n")
    Hnp = np.array( H )
    return Hnp

#  This function only matters for special purposes like 1D line searching.
# The energy is usually obtained from the gradient function.
def energy_func(xyz, printResults=True):
    mol = psi4.core.get_active_molecule()
    xyz[:] = setGeometry_func(xyz)
    E, wfn = psi4.driver.energy(calcName, molecule=mol, return_wfn=True)
    if printResults:
        print_out('\tEnergy: %15.10f\n' % E)
    return E

# Also send python printing function. Otherwise print to stdout will be done

# Returns energy; or (energy, trajectory) if trajectory==True
def Psi4Opt():
    """ Psi4opt now collects all options and gets the molecular system from Psi4 instead of the options and molecular system being global variables"""

    #from psi4 import *
    #from psi4.core import *
    #import numpy as np
    
    # Collect the user-specified OPTKING keywords in a dict.
    all_options = psi4.driver.p4util.prepare_options_for_modules()
    optking_options = all_options['OPTKING']
    optking_user_options = {}
    for opt,optval in optking_options.items():
        if optval['has_changed'] == True:
            optking_user_options[opt] = optval['value']
        
    # Create initial molecular structure object in optking (atomic numbers,
    # cartesian coordinates, masses, ...) using class method for PSI4 molecule.
    # Initialize printing.
    import optking
    optking.printInit(psi4.core.print_out)
    #optking.printInit()  # for default
    
    mol = psi4.core.get_active_molecule()
    import optking.molsys
    Molsys = optking.molsys.MOLSYS.fromPsi4Molecule(mol)

    returnVal = optking.optimize(Molsys, optking_user_options, setGeometry_func, gradient_func, \
        hessian_func, energy_func)
    return returnVal


