import numpy as np

from . import covRadii
from . import frag
from . import optExceptions
from . import physconst as pc
from . import v3d
from .addIntcos import connectivityFromDistances, addCartesianIntcos
from .printTools import print_opt


class MOLSYS(object):  # new-style classes required for getter/setters
    def __init__(self, fragments, fb_fragments=None, intcos=None):
        # ordinary fragments with internal structure
        self._fragments = []
        if fragments:
            self._fragments = fragments
        # fixed body fragments defined by Euler/rotation angles
        self._fb_fragments = []
        if fb_fragments:
            self._fb_fragments = fb_fragments

    def __str__(self):
        s = ''
        for iF, F in enumerate(self._fragments):
            s += "Fragment %d\n" % (iF + 1)
            s += F.__str__()
        for iB, B in enumerate(self._fb_fragments):
            s += "Fixed boxy Fragment %d\n" % (iB + 1)
            s += B.__str__()
        return s

    @classmethod
    def fromPsi4Molecule(cls, mol):
        print_opt("\n\tGenerating molecular system for optimization from PSI4.\n")

        NF = mol.nfragments()
        print_opt("\t%d Fragments in PSI4 molecule object.\n" % NF)
        frags = []

        for iF in range(NF):
            fragMol = mol.extract_subsets(iF + 1)

            fragNatom = fragMol.natom()
            print_opt("\tCreating fragment %d with %d atoms\n" % (iF + 1, fragNatom))

            fragGeom = np.zeros((fragNatom, 3), float)
            fragGeom[:] = fragMol.geometry()

            #fragZ = np.zeros( fragNatom, int)
            fragZ = []
            for i in range(fragNatom):
                fragZ.append(int(fragMol.Z(i)))
                #fragZ[i] = fragMol.Z(i)

            fragMasses = np.zeros(fragNatom, float)
            for i in range(fragNatom):
                fragMasses[i] = fragMol.mass(i)

            frags.append(frag.FRAG(fragZ, fragGeom, fragMasses))
        return cls(frags)

    @property
    def Natom(self):
        s = 0
        for F in self._fragments:
            s += F.Natom
        return s

    @property
    def Nfragments(self):
        return len(self._fragments) + len(self._fb_fragments)

    # Return overall index of first atom in fragment, beginning 0,1,...
    def frag_1st_atom(self, iF):
        if iF >= len(self._fragments):
            return ValueError()
        start = 0
        for i in range(0, iF):
            start += self._fragments[i].Natom
        return start

    def frag_atom_range(self, iF):
        start = self.frag_1st_atom(iF)
        return range(start, start + self._fragments[iF].Natom)

    # accepts absolute atom index, returns fragment index
    def atom2frag_index(self, atom_index):
        for iF, F in enumerate(self._fragments):
            if atom_index in self.frag_atom_range(iF):
                return iF
        raise optExceptions.OPT_FAIL("atom2frag_index: atom_index impossibly large")

    # Given a list of atoms, return all the fragments to which they belong
    def atomList2uniqueFragList(self, atomList):
        fragList = []
        for a in atomList:
            f = self.atom2frag_index(a)
            if f not in fragList:
                fragList.append(f)
        return fragList

    @property
    def geom(self):
        geom = np.zeros((self.Natom, 3), float)
        for iF, F in enumerate(self._fragments):
            row = self.frag_1st_atom(iF)
            geom[row:(row + F.Natom), :] = F.geom
        return geom

    @geom.setter
    def geom(self, newgeom):
        for iF, F in enumerate(self._fragments):
            row = self.frag_1st_atom(iF)
            F.geom[:] = newgeom[row:(row + F.Natom), :]

    @property
    def masses(self):
        m = np.zeros(self.Natom, float)
        for iF, F in enumerate(self._fragments):
            start = self.frag_1st_atom(iF)
            m[start:(start + F.Natom)] = F.masses
        return m

    @property
    def Z(self):
        z = [0 for i in range(self.Natom)]
        for iF, F in enumerate(self._fragments):
            first = self.frag_1st_atom(iF)
            z[first:(first + F.Natom)] = F.Z
        return z

    @property
    def intcos(self):
        _intcos = []
        for F in self._fragments:
            _intcos += F.intcos
        return _intcos

    def frag_1st_intco(self, iF):
        if iF >= len(self._fragments):
            return ValueError()
        start = 0
        for i in range(0, iF):
            start += len(self._fragments[i]._intcos)
        return start

    def printIntcos(self):
        for iF, F in enumerate(self._fragments):
            print_opt("Fragment %d\n" % (iF + 1))
            F.printIntcos()
        return

    def addIntcosFromConnectivity(self, C=None):
        for F in self._fragments:
            if C is None:
                C = F.connectivityFromDistances()
            F.addIntcosFromConnectivity(C)

    def addCartesianIntcos(self):
        for F in self._fragments:
            addCartesianIntcos(F._intcos, F._geom)

    def printGeom(self):
        for iF, F in enumerate(self._fragments):
            print_opt("Fragment %d\n" % (iF + 1))
            F.printGeom()

    def showGeom(self):
        for iF, F in enumerate(self._fragments):
            print_opt("Fragment %d\n" % (iF + 1))
            F.showGeom()

    def consolidateFragments(self):
        if self.Nfragments == 1:
            return
        print_opt("Consolidating multiple fragments into one for optimization.\n")
        consolidatedFrag = frag.FRAG(self.Z, self.geom, self.masses)
        del self._fragments[:]
        self._fragments.append(consolidatedFrag)

    # Split any fragment not connected by bond connectivity.
    def splitFragmentsByConnectivity(self):
        tempZ = np.copy(self.Z)
        tempGeom = np.copy(self.geom)
        tempMasses = np.copy(self.masses)

        newFragments = []
        for F in self._fragments:
            C = connectivityFromDistances(F.geom, F.Z)
            atomsToAllocate = list(reversed(range(F.Natom)))

            while atomsToAllocate:
                frag_atoms = [atomsToAllocate.pop()]

                more_found = True
                while more_found:
                    more_found = False
                    addAtoms = []
                    for A in frag_atoms:
                        for B in atomsToAllocate:
                            if C[A, B]:
                                addAtoms.append(B)
                                more_found = True

                    for a in addAtoms:
                        frag_atoms.append(a)
                        atomsToAllocate.remove(a)

                frag_atoms.sort()
                subNatom = len(frag_atoms)
                subZ = np.zeros(subNatom, float)
                subGeom = np.zeros((subNatom, 3), float)
                subMasses = np.zeros(subNatom, float)
                for i, I in enumerate(frag_atoms):
                    subZ[i] = tempZ[I]
                    subGeom[i, 0:3] = tempGeom[I, 0:3]
                    subMasses[i] = tempMasses[I]
                newFragments.append(frag.FRAG(subZ, subGeom, subMasses))

        del self._fragments[:]
        self._fragments = newFragments

    # Supplements a connectivity matrix to connect all fragments.  Assumes the
    # definition of the fragments has ALREADY been determined before function called.
    def augmentConnectivityToSingleFragment(self, C):
        print_opt('\tAugmenting connectivity matrix to join fragments.\n')
        fragAtoms = []
        geom = self.geom
        for iF, F in enumerate(self._fragments):
            fragAtoms.append(
                range(self.frag_1st_atom(iF),
                      self.frag_1st_atom(iF) + F.Natom))

        # Which fragments are connected?
        nF = self.Nfragments
        frag_connectivity = np.zeros((nF, nF))
        for iF in range(nF):
            frag_connectivity[iF, iF] = 1

        Z = self.Z

        scale_dist = 1.3
        all_connected = False
        while not all_connected:
            for f2 in range(nF):
                for f1 in range(f2):
                    if frag_connectivity[f1][f2]:
                        continue  # already connected
                    minVal = 1.0e12

                    # Find closest 2 atoms between fragments.
                    for f1_atom in fragAtoms[f1]:
                        for f2_atom in fragAtoms[f2]:
                            tval = v3d.dist(geom[f1_atom], geom[f2_atom])
                            if tval < minVal:
                                minVal = tval
                                i = f1_atom
                                j = f2_atom

                    Rij = v3d.dist(geom[i], geom[j])
                    R_i = covRadii.R[int(Z[i])] / pc.bohr2angstroms
                    R_j = covRadii.R[int(Z[j])] / pc.bohr2angstroms
                    if Rij > scale_dist * (R_i + R_j):
                        continue  # ignore this as too far - for starters.  may have A-B-C situation.

                    print_opt("\tConnecting fragments with atoms %d and %d\n" % (i + 1,
                                                                                 j + 1))
                    C[i][j] = C[j][i] = True
                    frag_connectivity[f1][f2] = frag_connectivity[f2][f1] = True

                    # Now check for possibly symmetry-related atoms which are just as close
                    # We need them all to avoid symmetry breaking.
                    for f1_atom in fragAtoms[f1]:
                        for f2_atom in fragAtoms[f2]:
                            if f1_atom == i and f2_atom == j:  # already have this one
                                continue
                            tval = v3d.dist(geom[f1_atom], geom[f2_atom])
                            if np.fabs(tval - minVal) < 1.0e-10:
                                i = f1_atom
                                j = f2_atom
                                print_opt("\tAlso, with atoms %d and %d\n" % (i + 1,
                                                                              j + 1))
                                C[i][j] = C[j][i] = True

            # Test whether all frags are connected using current distance threshold
            if np.sum(frag_connectivity[0]) == nF:
                print_opt("\tAll fragments are connected in connectivity matrix.\n")
                all_connected = True
            else:
                scale_dist += 0.2
                print_opt(
                    "\tIncreasing scaling to %6.3f to connect fragments.\n" % scale_dist)
        return
