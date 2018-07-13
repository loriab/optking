#Where I keep methods for interacting with psi4. i.e. getting gradients, hessians etc.


#ToDo delete this method once psi4 writes a method that does this (just better)
def collect_psi4_options(options):
    """Is meant to look through the dictionary of psi4 options being passed in and pick out the basis
     set and QM method used (Calcname) which are appened to the list of psi4 options, 
    """
    keywords = {}
    for opt in options['PSI4']:
        keywords[opt] = options['PSI4'][opt]

    basis = keywords['basis']
    del keywords['basis']
    QM_method = keywords['calcName']
    del keywords['calcName']

    return QM_method, basis, keywords


def get_optking_options_psi4(all_options):
    optking_user_options = {}
    optking_options = all_options['OPTKING']
    optking_user_options['OPTKING'] = {}
    for opt,optval in optking_options.items():
        if optval['has_changed'] == True:
            optking_user_options[opt] = optval['value']


    return optking_user_options

#################
## Thses are all methods that are not longer used. Keeping them around for reference if needed. Will remove completely later
#################
#def set_geometry(newGeom):
#    """Define a function for optking that takes a Cartesian, numpy geometry, and
#    puts it in place for subsequent gradient computations.  This function
#    may move COM or reorient; will return possible changed cartesian geometry.
#    """
#
#    mol = psi4.core.get_active_molecule()
#    psi_geom =  psi4.core.Matrix.from_array(newGeom)
#    mol.set_geometry(psi_geom)
#    mol.update_geometry()
#    return np.array( mol.geometry() )
#    newGeom[:] = np.array( mol.geometry() )

#def energy_func(xyz, printResults=True):
#    """This function only matters for special purposes like 1D line searching.
#    The energy is usually obtained from the gradient function.
#    """
#
#    mol = psi4.core.get_active_molecule()
#    xyz[:] = setGeometry_func(xyz) #id like for this to be in my get_energy methods
#    E, wfn = psi4.driver.energy(calcName, molecule=mol, return_wfn=True)
#    if printResults:
#        psi4.core.print_out('\tEnergy: %15.10f\n' % E)
#    return E
#
#def gradient_func(xyz, printResults=True):
#    """Define a function for optking that returns an energy and cartesian
#    gradient in numpy array format."""
#
#    mol = psi4.core.get_active_molecule()
#    #xyz[:] = setGeometry_func(xyz)
#    psi4gradientMatrix, wfn = psi4.driver.gradient(calcName, molecule=mol, return_wfn=True)
#    gradientMatrix = np.array( psi4gradientMatrix )
#    E = wfn.energy()
#    if printResults:
#       psi4.core.print_out( '\tEnergy: %15.10f\n' % E)
#       psi4.core.print_out( '\tGradient\n')
#       psi4.core.print_out( str(gradientMatrix) )
#       psi4.core.print_out( "\n")
#    return E, np.reshape(gradientMatrix, (gradientMatrix.size))
#
#
#def hessian_func(xyz, printResults=False):
#    """Gets a hessian from psi4 (cartesian)"""
#    H = psi4.driver.hessian(calcName, molecule=mol)
#    if printResults:
#        psi4.core.print_out( 'Hessian\n')
#        H.print_out()
#        psi4.core.print_out( "\n")
#    return np.array( H )
#
