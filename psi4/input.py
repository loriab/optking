import numpy as np

molecule hooh {
    H            1.699924772228     1.549001852664     0.729368159665
    O           -0.027495833355     1.120334367050     0.682522182417
    O           -0.047683750414    -1.071778830756    -0.755485307218
    H           -0.506770221333    -2.319613449532     0.428609578964
# unit au
}

set {
  basis dz
}

set optking {
  Geom_maxiter  1
  step_type     rfo
  hess_update   bfgs
  intrafrag_hess SIMPLE
}

# Create initial molecular structure object in optking
#  (atomic numbers, cartesians, masses, ...) using class method
#  in optking.frag.py for a PSI4 molecule.
mol = core.get_active_molecule()
import optking
Molsys = optking.molsys.MOLSYS.fromPsi4Molecule(mol)

# Collect the user-specified OPTKING keywords in a dict.
all_options = p4util.prepare_options_for_modules()
optking_options = all_options['OPTKING']
optking_user_options = {}
for opt,optval in optking_options.items():
    if optval['has_changed'] == True:
        optking_user_options[opt] = optval['value']

# Define a function for optking that takes a Cartesian, numpy geometry, and
#  puts it in place for subsequent gradient computations.  This function
#  may move COM or reorient; changes argument to match such geometry.
def setGeometry_func( newGeom ):
    psi_geom = core.Matrix.from_array( newGeom )
    mol.set_geometry( psi_geom )
    mol.update_geometry()
    newGeom[:] = np.array( mol.geometry() )

# Define a function for optking that returns an energy and cartesian
#  gradient in numpy array format.
def gradient_func(printResults=True):
    psi4gradientMatrix, wfn = driver.gradient('hf', molecule=mol, return_wfn=True)
    gradientMatrix = np.array( psi4gradientMatrix )
    E = wfn.energy()
    if printResults:
        print '\tEnergy: %15.10f' % E
        print '\tGradient'
        print gradientMatrix
    return E, np.reshape(gradientMatrix, (gradientMatrix.size))

def hessian_func(printResults=True):
    H = driver.hessian('hf', molecule=mol)
    if printResults:
        print '\t Hessian'
        print H
    return H

#  This function only matters for special purposes like 1D line searching.
# The energy is usually obtained from the gradient function.
def energy_func(printResults=True):
    E, wfn = driver.energy('hf', molecule=mol, return_wfn=True)
    if printResults:
        print '\tEnergy: %15.10f' % E
    return E

# Provide optimizer function the following arguments:
# 1. optking.molsys.MOLSYS system (defined by class method above.
# 2. a dictionary of optking options
# 3. a method which prepares a given Cartesian, numpy geometry
#     for subsequent gradient computation or analysis.
# 4. a method to compute the gradient, which returns the energy and gradient.
# 5. optional hessian function
# 6. optional energy function for line-searching algorithms

optking.optimize( Molsys, optking_user_options, setGeometry_func, gradient_func, \
                hessian_func, energy_func )

