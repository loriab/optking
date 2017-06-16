# optParams is a class to store all of the optimization parameters.
# The init function will receive a User Option Dictionary (uod) which can
# override default values.
# P = parameters ('self')
# Option keys in the input dictionary are interpreted case-insensitively.
# The enumerated string types are translated to all upper-case within the parameter object.
Params = 0

# Class for enumerated string options.
def stringOption(storage_name):
    def stringOption_getter(instance):
        return instance.__dict__[storage_name]

    def stringOption_setter(instance, value):
        if value.upper() in allowedStringOptions[storage_name]:
            instance.__dict__[storage_name] = value.upper()
        else:
            raise ValueError('Invalid value for ' + storage_name)
    return property(stringOption_getter, stringOption_setter)


# The keys on the left here should be lower-case, as should the storage name of the property.
allowedStringOptions = {
 'opt_type'        : ('MIN','TS','IRC'),
 'step_type'       : ('RFO', 'P_RFO', 'NR', 'SD', 'LINESEARCH_STATIC'),
 'opt_coordinates' : ('REDUNDANT', 'INTERNAL', 'DELOCALIZED', 'NATURAL', 'CARTESIAN', 'BOTH'),
 'irc_direction'   : ('FORWARD', 'BACKWARD'),
 'irc_stop'        : ('ASK', 'STOP', 'GO'),
 'g_convergence'   : ('QCHEM','MOLPRO','GAU','GAU_LOOSE','GAU_TIGHT','GAU_VERYTIGHT', \
                       'TURBOMOLE','CFOUR','NWCHEM_LOOSE'),
 'hess_update'     : ('NONE', 'BFGS', 'MS', 'POWELL', 'BOFILL'),
 'intrafrag_hess'  : ('SCHLEGEL', 'FISCHER', 'SCHLEGEL', 'SIMPLE', 'LINDH', 'LINDH_SIMPLE'),
 'frag_mode'       : ('SINGLE', 'MULTI'),
 'interfrag_mode'  : ('FIXED', 'PRINCIPAL_AXES'),
 'interfrag_hess'  : ('DEFAULT', 'FISCHER_LIKE'),
}

def enum_key( enum_type, value):
    print [key for key, val in enum_type.__dir__.items() if val == value][0]

class OPT_PARAMS(object):
    # define properties
    opt_type        = stringOption( 'opt_type' )
    step_type       = stringOption( 'step_type' )
    opt_coordinates = stringOption( 'opt_coordinates' )
    #irc_direction   = stringOption( 'irc_direction' )
    #irc_stop        = stringOption( 'irc_step' )
    g_convergence   = stringOption( 'g_convergence' )
    hess_update     = stringOption( 'hess_update' )
    intrafrag_hess  = stringOption( 'intrafrag_hess' )
    frag_mode       = stringOption( 'frag_mode' )
    #interfrag_mode  = stringOption( 'interfrag_mode' )
    #interfrag_hess  = stringOption( 'interfrag_hess' )

    def __str__(P):
        s = ''
        print "\t\t -- Optimization Parameters --"
        for attr in dir(P):
            if not hasattr( getattr(P,attr), '__self__' ): # omit bound methods
                if '__' not in attr:                       # omit these methods
                    s += "\t%-30s = %15s\n" %(attr, getattr(P,attr))
        return s

    def __init__(P, uod):
        print "\tProcessing user input options..."

        # SUBSECTION Optimization Algorithm

        # Maximum number of geometry optimization steps
        P.geom_maxiter = uod.get('geom_maxiter', 50)
        # Print level.  1 = normal
        ##P.print_lvl = uod.get('print_lvl', 1)
        P.print_lvl = uod.get('print', 1)
        ## Print all optimization parameters.
        #P.print_opt_params = uod.get('PRINT_OPT_PARAMS', False)
        # Specifies minimum search, transition-state search, or IRC following
        P.opt_type = uod.get('OPT_TYPE', 'MIN')
        # Geometry optimization step type, e.g., Newton-Raphson or Rational Function Optimization
        P.step_type = uod.get('STEP_TYPE', 'RFO')
        ## Geometry optimization coordinates to use.
        ## REDUNDANT and INTERNAL are synonyms and the default.
        ## DELOCALIZED are the coordinates of Baker.
        ## NATURAL are the coordinates of Pulay.
        ## CARTESIAN uses only cartesian coordinates.
        # BOTH uses both redundant and cartesian coordinates.
        P.opt_coordinates = uod.get('OPT_COORDINATES', 'REDUNDANT')
        # Do follow the initial RFO vector after the first step?
        P.rfo_follow_root = uod.get('RFO_FOLLOW_ROOT', False)
        # Root for RFO to follow, 0 being lowest (typical for a minimum)
        P.rfo_root = uod.get('RFO_ROOT', 0)
        # Whether to accept geometry steps that lower the molecular point group.
        P.accept_symmetry_breaking = uod.get('ACCEPT_SYMMETRY_BREAKING', False)
        # Starting level for dynamic optimization (0=nondynamic, higher=>more conservative)
        P.dynamic_level = uod.get('DYNAMIC_LEVEL', 0)
        ## IRC step size in bohr(amu)\ $^{1/2}$.
        #P.irc_step_size = uod.get('IRC_STEP_SIZE', 0.2)
        ## IRC mapping direction
        #P.irc_direction = uod.get('IRC_DIRECTION', 'FORWARD')
        ## Decide when to stop IRC calculations
        #P.irc_stop = uod.get('IRC_STOP', 'STOP')
#
        # Initial maximum step size in bohr or radian along an internal coordinate
        P.intrafrag_trust = uod.get('INTRAFRAG_TRUST', 0.5)
        # Lower bound for dynamic trust radius [au]
        P.intrafrag_trust_min = uod.get('INTRAFRAG_TRUST_MIN', 0.001)
        # Upper bound for dynamic trust radius [au]
        P.intrafrag_trust_max = uod.get('INTRAFRAG_TRUST_MAX', 1.0)
        # Maximum step size in bohr or radian along an interfragment coordinate
        #P.interfrag_trust = uod.get('INTERFRAG_TRUST', 0.5)
        ## Reduce step size as necessary to ensure convergence of back-transformation of
        ## internal coordinate step to cartesian coordinates.
        #P.ensure_bt_convergence = uod.get('ENSURE_BT_CONVERGENCE', False)
        # Do simple, linear scaling of internal coordinates to step limit (not RS-RFO)
        P.simple_step_scaling = uod.get('SIMPLE_STEP_SCALING', False)
        # Set number of consecutive backward steps allowed in optimization
        P.consecutiveBackstepsAllowed = uod.get('CONSECUTIVE_BACKSTEPS_ALLOWED', 0)
        P.working_consecutive_backsteps = 0
        # Eigenvectors of RFO matrix whose final column is smaller than this are ignored.
        P.rfo_normalization_max = uod.get('RFO_NORMALIZATION_MAX', 100)
        # Absolute maximum value of RS-RFO.
        P.rsrfo_alpha_max = uod.get('RSRFO_ALPHA_MAX', 1e8)
#
        ## Specify distances between atoms to be frozen (unchanged)
        #P.frozen_distance = uod.get('FROZEN_DISTANCE','')
        ## Specify angles between atoms to be frozen (unchanged)
        #P.frozen_bend     = uod.get('FROZEN_BEND','')
        ## Specify dihedral angles between atoms to be frozen (unchanged)
        #P.frozen_dihedral = uod.get('FROZEN_DIHEDRAL','')
        ## Specify atom and X, XY, XYZ, ... to be frozen (unchanged)
        #P.frozen_cartesian = uod.get('FROZEN_CARTESIAN','')
#
        ## Specify distances between atoms to be fixed (eq. value specified)
        #P.fixed_distant = uod.get("FIXED_DISTANCE", "")
        ## Specify angles between atoms to be fixed (eq. value specified)
        #P.fixed_bend    = uod.get("FIXED_BEND", "")
        ## Specify dihedral angles between atoms to be fixed (eq. value specified)
        #P.fixed_dihedral = uod.get("FIXED_DIHEDRAL","")
#
#
        ## Should an xyz trajectory file be kept (useful for visualization)?
        #P.print_trajectory_xyz = uod.get('PRINT_TRAJECTORY_XYZ', False)
        ## Symmetry tolerance for testing whether a mode is symmetric.
        #P.symm_tol("SYMM_TOL", 0.05)
#
        # SUBSECTION Convergence Control.
        # Set of optimization criteria. Specification of any MAX_*_G_CONVERGENCE
        # RMS_*_G_CONVERGENCE options will append to overwrite the criteria set here
        # |optking__flexible_g_convergence| is also on.      
        # See Table :ref:`Geometry Convergence <table:optkingconv>` for details.
        P.g_convergence = uod.get('G_CONVERGENCE', 'QCHEM')
        # Convergence criterion for geometry optmization: maximum force (internal coordinates, au)
        P.max_force_g_convergence =  uod.get('MAX_FORCE_G_CONVERGENCE', 3.0e-4)
        # Convergence criterion for geometry optmization: rms force  (internal coordinates, au)
        P.rms_force_g_convergence = uod.get('RMS_FORCE_G_CONVERGENCE', 3.0e-4)
        # Convergence criterion for geometry optmization: maximum energy change
        P.max_energy_g_convergence = uod.get('MAX_ENERGY_G_CONVERGENCE', 1.0e-6)
        # Convergence criterion for geometry optmization: maximum displacement (internal coordinates, au)
        P.max_disp_g_convergence = uod.get('MAX_DISP_G_CONVERGENCE', 1.2e-3)
        # Convergence criterion for geometry optmization: rms displacement (internal coordinates, au)
        P.rms_disp_g_convergence = uod.get('RMS_DISP_G_CONVERGENCE', 1.2e-3)
        # Even if a user-defined threshold is set, allow for normal, flexible convergence criteria
        P.flexible_g_convergence = uod.get('FLEXIBLE_G_CONVERGENCE', False)
#
        ## SUBSECTION Hessian Update
        # Hessian update scheme
        P.hess_update = uod.get('HESS_UPDATE', 'BFGS')
        # Number of previous steps to use in Hessian update, 0 uses all
        P.hess_update_use_last = uod.get('HESS_UPDATE_USE_LAST', 2)
        # Do limit the magnitude of changes caused by the Hessian update?
        P.hess_update_limit = uod.get('HESS_UPDATE_LIMIT', True)
        # If |hess_update_limit| is True, changes to the Hessian from the update are limited
        # to the larger of |hess_update_limit_scale| * (current value) and
        # |hess_update_limit_max| [au].  By default, a Hessian value cannot be changed by more
        # than 50% and 1 au.
        P.hess_update_limit_max = uod.get('HESS_UPDATE_LIMIT_MAX', 1.00)
        P.hess_update_limit_scale = uod.get('HESS_UPDATE_LIMIT_SCALE', 0.50)
        # Denominator check for hessian update.
        P.hess_update_den_tol = uod.get('HESS_UPDATE_DEN_TOL', 1e-7)
        # Hessian update is avoided if any internal coordinate has changed by 
        # more than this in radians/au
        P.hess_update_dq_tol = 0.5

        # SUBSECTION Using external Hessians
        # Do read Cartesian Hessian?  Only for experts - use
        # |optking__full_hess_every| instead.
        P.cart_hess_read = uod.get('CART_HESS_READ', False)
        ## Frequency with which to compute the full Hessian in the course
        ## of a geometry optimization. 0 means to compute the initial Hessian only,
        ## 1 means recompute every step, and N means recompute every N steps. The
        ## default (-1) is to never compute the full Hessian.
        #P.full_hess_every = uod.get('FULL_HESS_EVERY', -1)
        # Model Hessian to guess intrafragment force constants
        P.intrafrag_hess = uod.get('INTRAFRAG_HESS', 'SCHLEGEL')
        ## Re-estimate the Hessian at every step, i.e., ignore the currently stored Hessian.
        #P.h_guess_every = uod.get('H_GUESS_EVERY', False)

        P.working_steps_since_last_H = 0
#
        ## SUBSECTION Backtransformation to Cartesian Coordinates Control
        P.bt_max_iter = uod.get('bt_max_iter', 25)
        P.bt_dx_conv = uod.get('bt_dx_conv', 1.0e-6)
        P.bt_dx_rms_change_conv = uod.get('bt_dx_rms_change_conv',1.0e-12)
#
        ## For multi-fragment molecules, treat as single bonded molecule or via interfragment
        ## coordinates. A primary difference is that in ``MULTI`` mode, the interfragment 
        ## coordinates are not redundant.
        P.frag_mode = uod.get('FRAG_MODE', 'SINGLE')
        ## Which atoms define the reference points for interfragment coordinates?
        #P.frag_ref_atoms = uod.get('FRAG_REF_ATOMS', '')
        ## Do freeze all fragments rigid?
        #P.freeze_intrafrag = uod.get('FREEZE_INTRAFRAG', False)
        ## Do freeze all interfragment modes?
        #P.inter_frag = uod.get('FREEZE_INTERFRAG', False)
        ## When interfragment coordinates are present, use as reference points either
        ## principal axes or fixed linear combinations of atoms.
        #P.interfrag_mode = uod.get('INTERFRAG_MODE', 'FIXED')
        ## Do add bond coordinates at nearby atoms for non-bonded systems?
        #P.add_auxiliary_bonds = uod.get('ADD_AUXILIARY_BONDS', True)
        ## This factor times standard covalent distance is used to add extra stretch coordinates.
        #P.auxiliary_bond_factor = uod.get('AUXILIARYBOND_FACTOR', 2.5)
        ## Do use 1/R for the interfragment stretching coordinate instead of R?
        #P.interfrag_dist_inv = uod.get('INTERFRAG_DIST_INV', False)
        ## Model Hessian to guess interfragment force constants
        #P.interfrag_hess = uod.get('INTERFRAG_HESS', 'DEFAULT')
        # When determining connectivity, a bond is assigned if interatomic distance
        # is less than (this number) * sum of covalent radii.
        P.covalent_connect = uod.get('COVALENT_CONNECT', 1.3)
        ## When connecting disparate fragments when frag_mode = SIMPLE, a "bond"
        ## is assigned if interatomic distance is less than (this number) * sum of covalent radii. The
        ## value is then increased until all the fragments are connected (directly or indirectly).
        P.interfragment_connect = uod.get('INTERFRAGMENT_CONNECT', 1.8)
        ## General, maximum distance for the definition of H-bonds.
        #P.h_bond_connect = uod.get('h_bond_connect', 4.3)
        ## Only generate the internal coordinates and then stop (boolean)
        #P.intcos_generate_exit = uod.get('INTCOS_GENERATE_EXIT', False)
  #
  #
        ## SUBSECTION Misc.
        ## Do save and print the geometry from the last projected step at the end
        ## of a geometry optimization? Otherwise (and by default), save and print
        ## the previous geometry at which was computed the gradient that satisfied
        ## the convergence criteria.
        #P.final_geom_write = uod.get('FINAL_GEOM_WRITE', False)
        # Do test B matrix?
        P.test_B = uod.get('TEST_B', False)
        ## Do test derivative B matrix?
        P.test_derivative_B = uod.get('TEST_DERIVATIVE_B', False)
        ## Keep internal coordinate definition file.
        #P.keep_intcos = uod.get('KEEP_INTCOS', False)
        ## In constrained optimizations, for coordinates with user-specified
        ## equilibrium values, this is the initial force constant (in au) used to apply an
        ## additional force to each coordinate.
        #P.fixed_coord_force_constant = uod.get('FIXED_COORD_FORCE_CONSTANT', 0.5)
        ## If doing a static line search, scan this many points.
        #P.linesearch_static_N = uod.get('LINESEARCH_STATIC_N', 8)
        ## If doing a static line search, this fixes the shortest step, whose largest
        ## change in an internal coordinate is set to this value (in au)
        #P.linesearch_static_min = uod.get('LINESEARCH_STATIC_MIN', 0.001)
        ## If doing a static line search, this fixes the largest step, whose largest
        ## change in an internal coordinate is set to this value (in au) 
        #P.linesearch_static_max = uod.get('LINESEARCH_STATIC_MAX', 0.100)
        # Guess at Hessian in steepest-descent direction.
        P.sd_hessian = uod.get('SD_HESSIAN', 1.0)
#
        ## --- Complicated defaults ---
#
        ## Assume RFO means P-RFO for transition states.
        #if P.opt_type == 'TS':
            #if P.step_type == 'RFO' or 'STEP_TYPE' not in uod:
                #P.step_type = 'P_RFO'
#
        ## INTERNAL is a synonym
        #if P.coordinates == 'INTERNAL': P.coordinates = 'REDUNDANT'
#
        ## Initial Hessian guess for cartesians with coordinates BOTH is stupid, so don't scale
        ##   step size down too much.  Steepest descent has no good hessian either.
        #if 'INTRAFRAG_TRUST_MIN' not in uod:
            #if P.coordinates == 'BOTH':
                #P.intrafragment_trust_min = Opt_params.intrafragment_trust / 2.0;
            #if P.step_type == 'SD':
                #P.intrafragment_trust_min = Opt_params.intrafragment_trust;
#
        ## Original Lindh specification was to redo at every step.
        #if 'H_GUESS_EVERY' not in uod and P.intrafrag_hess == 'LINDH':
            #P.h_guess_every = True
#
        ## Default for cartesians: use Lindh force field for initial guess, then BFGS.
        #if P.coordinates == 'CARTESIAN':
            #if 'INTRAFRAG_HESS_GUESS' not in uod:
                #P.intrafrag_hess = 'LINDH'
                #if 'H_GUESS_EVERY' not in uod:
                    #Opt_params.H_guess_every = False;
#
        ## Set Bofill as default for TS optimizations.
        #if P.opt_type == 'TS' or P.opt_type == 'IRC':
            #if 'HESS_UPDATE' not in uod:
                #P.hess_update = 'BOFILL'
#
        ## Make trajectory file printing the default for IRC.
        #if P.opt_type == 'IRC' and 'PRINT_TRAJECTORY_XYZ_FILE' not in uod:
            #P.print_trajectory_xyz_file = True
#
        ## Read cartesian Hessian by default for IRC.
        #if P.opt_type == 'IRC' and 'CART_HESS_READ' not in uod:
            #P.read_cartesian_H = True
#
        #if P.generate_intcos_exit:
            #P.keep_intcos = True
#
        ## if steepest-descent, then make much larger default
        #if P.step_type = 'SD' and 'CONSECUTIVE_BACKSTEPS' not in uod:
            #P.consecutive_backsteps_allowed = 10;
#
        ## For RFO step, eigenvectors of augmented Hessian are divided by the last
        ## element unless it is smaller than this value {double}.  Can be used to
        ## eliminate asymmetric steps not otherwise detected (e.g. in degenerate
        ## point groups). For multi-fragment modes, we presume that smaller
        ##  Delta-E's are possible, and this threshold should be made larger.
        #if P.fragment_mode == 'MULTI' and 'RFO_NORMALIZATION_MAX' not in uod:
            #P.rfo_normalization_max = 1.0e5
#
        #if P.fixed_distance or P.fixed_bend or P.fixed_dihedral:
            #if 'INTRAFRAGMENT_TRUST' not in uod:
                #P.intrafragment_trust     = 0.1
            #if 'INTRAFRAGMENT_TRUST_MIN' not in uod:
                #P.intrafragment_trust_min = 0.1
            #if 'INTRAFRAGMENT_TRUST_MAX' not in uod:
                #P.intrafragment_trust_max = 0.1
#
        ## -- Items are below unlikely to need modified
#
        # Boundary to guess if a torsion or out-of-plane angle has passed through 180
        # during a step.
        P.fix_val_near_pi = 1.57 
#
        ## Only used for determining which atoms in a fragment are acceptable for use
        ## as reference atoms.  We avoid collinear sets.
        ## angle is 0/pi if the bond angle is within this fraction of pi from 0/pi
        #P.interfrag_collinear_tol = 0.01
#
        ## Torsional angles will not be computed if the contained bond angles are within
        # this many radians of zero or 180. (< ~1 and > ~179 degrees)
        # only used in v3d.py
        P.v3d_tors_angle_lim = 0.017
#
        # cos(torsional angle) must be this close to -1/+1 for angle to count as 0/pi
        # only used in v3d.py
        P.v3d_tors_cos_tol = 1e-10
#
        # if bend exceeds this value, then also create linear bend complement
        P.linear_bend_threshold = 3.05 # about 175 degrees
        ## If bend is smaller than this value, then never fix its associated vectors
        ## this allows iterative steps through and near zero degrees. 
        #P.small_bend_fix_threshold = 0.35
#
        ## Threshold for which entries in diagonalized redundant matrix are kept and
        # inverted while computing a generalized inverse of a matrix
        P.redundant_eval_tol = 1.0e-10
#
        ## --- SET INTERNAL OPTIMIZATION PARAMETERS ---
        P.i_max_force = False
        P.i_rms_force = False
        P.i_max_DE    = False
        P.i_max_disp  = False
        P.i_rms_disp  = False
        P.i_untampered = False
#
        if P.g_convergence == 'QCHEM':
            P.i_untampered   = True
            P.conv_max_force = 3.0e-4 ; P.i_max_force = True
            P.conv_max_DE    = 1.0e-6 ; P.i_max_DE    = True
            P.conv_max_disp  = 1.2e-3 ; P.i_max_disp  = True
        elif P.g_convergence == 'MOLPRO':
            P.i_untampered   = True
            P.conv_max_force = 3.0e-4 ; P.i_max_force = True
            P.conv_max_DE    = 1.0e-6 ; P.i_max_DE    = True
            P.conv_max_disp  = 3.0e-4 ; P.i_max_disp  = True
        elif P.g_convergence == 'GAU':
            P.i_untampered   = True
            P.conv_max_force = 4.5e-4 ; P.i_max_force = True
            P.conv_rms_force = 3.0e-4 ; P.i_rms_force = True
            P.conv_max_disp  = 1.8e-3 ; P.i_max_disp  = True
            P.conv_rms_disp  = 1.2e-3 ; P.i_rms_disp  = True
        elif P.g_convergence == 'GAU_TIGHT':
            P.i_untampered   = True
            P.conv_max_force = 1.5e-5 ; P.i_max_force = True
            P.conv_rms_force = 1.0e-5 ; P.i_rms_force = True
            P.conv_max_disp  = 6.0e-5 ; P.i_max_disp  = True
            P.conv_rms_disp  = 4.0e-5 ; P.i_rms_disp  = True
        elif P.g_convergence == 'GAU_VERYTIGHT':
            P.i_untampered   = True;
            P.conv_max_force = 2.0e-6 ; P.i_max_force = True
            P.conv_rms_force = 1.0e-6 ; P.i_rms_force = True
            P.conv_max_disp  = 6.0e-6 ; P.i_max_disp  = True
            P.conv_rms_disp  = 4.0e-6 ; P.i_rms_disp  = True
        elif P.g_convergence == 'GAU_LOOSE':
            P.i_untampered   = True
            P.conv_max_force = 2.5e-3 ; P.i_max_force = True
            P.conv_rms_force = 1.7e-3 ; P.i_rms_force = True
            P.conv_max_disp  = 1.0e-2 ; P.i_max_disp  = True
            P.conv_rms_disp  = 6.7e-3 ; P.i_rms_disp  = True
        elif P.g_convergence == 'TURBOMOLE':
            P.i_untampered   = True
            P.conv_max_force = 1.0e-3 ; P.i_max_force = True
            P.conv_rms_force = 5.0e-4 ; P.i_rms_force = True
            P.conv_max_DE    = 1.0e-6 ; P.i_max_DE    = True
            P.conv_max_disp  = 1.0e-3 ; P.i_max_disp  = True
            P.conv_rms_disp  = 5.0e-4 ; P.i_rms_disp  = True
        elif P.g_convergence == 'CFOUR':
            P.i_untampered   = True
            P.conv_rms_force = 1.0e-4 ; P.i_rms_force = True
        elif P.g_convergence == 'NWCHEM_LOOSE':
            P.i_untampered   = True
            P.conv_max_force = 4.5e-3 ; P.i_max_force = True
            P.conv_rms_force = 3.0e-3 ; P.i_rms_force = True
            P.conv_max_disp  = 5.4e-3 ; P.i_max_disp  = True
            P.conv_rms_disp  = 3.6e-3 ; P.i_rms_disp  = True
#
        ## ---  Specific optimization criteria
        if 'MAX_FORCE_G_CONVERGENCE' in uod:
            P.i_untampered   = False
            P.i_max_force    = True
            P.conv_max_force = P.max_force_g_convergence
        if 'RMS_FORCE_G_CONVERGENCE' in uod:
            P.i_untampered   = False
            P.i_rms_force    = True
            P.conv_rms_force = P.rms_force_g_convergence
        if 'MAX_ENERGY_G_CONVERGENCE' in uod:
            P.i_untampered   = False
            P.i_max_DE       = True
            P.conv_max_DE    = P.max_energy_g_convergence
        if 'MAX_DISP_G_CONVERGENCE' in uod:
            P.i_untampered   = False
            P.i_max_disp     = True
            P.conv_max_disp  = P.max_disp_g_convergence
        if 'RMS_DISP_G_CONVERGENCE' in uod:
            P.i_untampered   = False
            P.i_rms_disp     = True
            P.conv_rms_disp  = P.rms_disp_g_convergence
#
        # Even if a specific threshold were given, allow for Molpro/Qchem/G03 flex criteria
        if P.flexible_g_convergence:
            P.i_untampered   = True

    def increaseTrustRadius(P):
        maximum = P.intrafrag_trust_max
        if P.intrafrag_trust != maximum:
            new_val = P.intrafrag_trust * 3

            if new_val > maximum:
                P.intrafrag_trust = maximum
            else:
                P.intrafrag_trust = new_val

            print "\tEnergy ratio indicates good step: Trust radius increased to %6.3e." % \
                   P.intrafrag_trust
        return

    def decreaseTrustRadius(P):
        minimum = P.intrafrag_trust_min
        if P.intrafrag_trust != minimum:
            new_val = P.intrafrag_trust / 4

            if new_val < minimum:
                P.intrafrag_trust = minimum
            else:
                P.intrafrag_trust = new_val

            print "\tEnergy ratio indicates iffy step: Trust radius decreased to %6.3e." % \
           P.intrafrag_trust
        return



    '''
Need to check some dynamic initializations in set_params.cc

*  if dynamic mode is on, then other settings are overridden.
* step_type = step
* intrafragment_trust = trust
* consecutive_backsteps = backsteps
* RI = redundant internals; D = default anyway

*dynamic  step   coord   trust      backsteps              criteria
* level                                               for downmove    for upmove
*  0      RFO    RI      dynamic         no           none            none
*
*  1      RFO    RI      dynamic(D)      no           1 bad step
*
*  2      RFO    RI      small initial   yes (1)      1 bad step
*                        dynamic(D)    
*
*  3      SD     RI      large(D)        yes (1)      1 bad step
*
*  4      SD     RI+XYZ  large(D)        yes (1)      1 bad step
*
*  5      SD     XYZ     large(D)        yes (1)      1 bad step
*
*  6      SD     XYZ     small           yes (1)      1 bad step
*
*  7  abort
*
*  BackStep:
*   DE > 0 in minimization
*
*  BadStep:
*   DE > 0 and backsteps exceeded and iterations > 5  ** OR **
*   badly defined internal coordinate or derivative
*
* 
  if (options.get_int("DYNAMIC_LEVEL") == 0) // if 0, then not dynamic
    Opt_params.dynamic = 0;
  else if (INTCO_EXCEPT::dynamic_level != 0) // already set
    Opt_params.dynamic = INTCO_EXCEPT::dynamic_level;
  else
    Opt_params.dynamic = options.get_int("DYNAMIC_LEVEL");


  switch(Opt_params.dynamic) {
    case 0: // not dynamic
      break;
    case 1:
      Opt_params.coordinates = OPT_PARAMS::REDUNDANT;
      Opt_params.consecutive_backsteps_allowed = 0;
      Opt_params.step_type = OPT_PARAMS::RFO;
             printf("At level 1: Red. Int., RFO, no backsteps, dynamic trust\n");
      oprintf_out("\tAt level 1: Red. Int., RFO, no backsteps, dynamic trust\n");
      break;
    case 2:
      Opt_params.coordinates = OPT_PARAMS::REDUNDANT;
      Opt_params.consecutive_backsteps_allowed = 1;
      Opt_params.step_type = OPT_PARAMS::RFO;
      Opt_params.intrafragment_trust = 0.2;
      Opt_params.intrafragment_trust_min = 0.2; //this code overwrites changes anyway
      Opt_params.intrafragment_trust_max = 0.2;
             printf("At level 2: Red. Int., RFO, backsteps, smaller trust.\n");
      oprintf_out("\tAt level 2: Red. Int., RFO, backsteps, smaller trust.\n");
      break;
    case 3:
      Opt_params.coordinates = OPT_PARAMS::BOTH;
      Opt_params.consecutive_backsteps_allowed = 1;
      Opt_params.step_type = OPT_PARAMS::RFO;
      Opt_params.intrafragment_trust = 0.1;
      Opt_params.intrafragment_trust_min = 0.1; //this code overwrites changes anyway
      Opt_params.intrafragment_trust_max = 0.1;
             printf("At level 3: Red. Int. + XYZ, RFO, backsteps, smaller trust.\n");
      oprintf_out("\tAt level 3: Red. Int. + XYZ, RFO, backsteps, smaller trust.\n");
      break;
    case 4:
      Opt_params.coordinates = OPT_PARAMS::CARTESIAN;
      Opt_params.consecutive_backsteps_allowed = 1;
      Opt_params.step_type = OPT_PARAMS::RFO;
      Opt_params.intrafragment_H = OPT_PARAMS::LINDH;
      Opt_params.intrafragment_trust = 0.3;
      Opt_params.intrafragment_trust_min = 0.3; //this code overwrites changes anyway
      Opt_params.intrafragment_trust_max = 0.3;
             printf("At level 4: XYZ, RFO, backsteps, larger trust.\n");
      oprintf_out("\tAt level 4: XYZ, RFO, backsteps, larger trust.\n");
      break;
    case 5:   // Try repeating level 4 which is working well
      Opt_params.coordinates = OPT_PARAMS::CARTESIAN;
      Opt_params.consecutive_backsteps_allowed = 1;
      Opt_params.step_type = OPT_PARAMS::RFO;
      Opt_params.intrafragment_H = OPT_PARAMS::LINDH;
      Opt_params.intrafragment_trust = 0.2;
      Opt_params.intrafragment_trust_min = 0.2; //this code overwrites changes anyway
      Opt_params.intrafragment_trust_max = 0.2;
             printf("At level 5: XYZ, RFO, backsteps, medium trust.\n");
      oprintf_out("\tAt level 5: XYZ, RFO, backsteps, medium trust.\n");
      break;
    case 6:
      Opt_params.coordinates = OPT_PARAMS::CARTESIAN;
      Opt_params.consecutive_backsteps_allowed = 1;
      Opt_params.step_type = OPT_PARAMS::SD;
      Opt_params.sd_hessian = 0.3;
      Opt_params.intrafragment_trust = 0.3;
      Opt_params.intrafragment_trust_min = 0.3; //this code overwrites changes anyway
      Opt_params.intrafragment_trust_max = 0.3;
             printf("At level 5: XYZ, SD, backsteps, larger trust.\n");
      oprintf_out("\tAt level 5: XYZ, SD, backsteps, larger trust.\n");
      break;
    case 7:
      Opt_params.coordinates = OPT_PARAMS::CARTESIAN;
      Opt_params.consecutive_backsteps_allowed = 1;
      Opt_params.step_type = OPT_PARAMS::SD;
      Opt_params.sd_hessian = 0.6;
      Opt_params.intrafragment_trust = 0.1;
      Opt_params.intrafragment_trust_min = 0.1; //this code overwrites changes anyway
      Opt_params.intrafragment_trust_max = 0.1;
             printf("Moving to level 6: XYZ, SD, backsteps, small trust, smaller steps.\n");
      oprintf_out("\tMoving to level 6: XYZ, SD, backsteps, small trust, smaller steps.\n");
      break;
  default:
      oprintf_out("Unknown value of Opt_params.dynamic variable.\n");
  }
}
    '''
