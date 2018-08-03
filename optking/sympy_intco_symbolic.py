"""Borrowed from python translation of INTDER by @jturney."""

import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdastr


Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz = \
    sp.var('A_x,A_y,A_z,B_x,B_y,B_z,C_x,C_y,C_z,D_x,D_y,D_z', real=True)

# These are internal temporary symbols used in the compute routines
_T1, _T2, _T3, _T4, _T5, _T6, _T7, _T8, _T9, _T10, _T11, _T12 = \
    sp.var('T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12', real=True)


class CoordinateError(RuntimeError):
    pass


def unit_vector(ax, ay, az, bx, by, bz):
    # return [sp.diff(sp.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2), coord) for coord in [ax, ay, az]]
    Rab = sp.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)
    return [
        (bx - ax) / Rab,
        (by - ay) / Rab,
        (bz - az) / Rab
    ]


class InternalCoordinate:
    """Encapsulates an internal coordinate.

    This is capable of computing values and B vectors.
    """

    @classmethod
    def check_completeness(cls, coords, molecule):
        """
        Class method for checking the number of non-redundant internal coordinates, i.e.
        the completeness of the chosen set. This method will print a status summary
        to stdout.
        :param coords: The set of internal coordinates to check completeness of
        :param molecule: The molecule the coordinates are based on
        :return:
        """
        if len(coords) == 0:
            raise CoordinateError('There are no coordinates')

        # Form the B matrix for the molecule and these internal coordinates
        for coord in coords:
            coord.dq_dx(molecule)

        raise NotImplementedError

    def is_valid_value(self, value):
        return True

    @staticmethod
    def form_B_matrix(molecule, coords, scale=None):
        if len(coords) == 0:
            raise CoordinateError('there are no coordinates')

        if scale is None:
            scale = np.ones(len(coords))

        b_matrix = np.zeros((len(coords), 3 * len(molecule)))
        for n, coord in enumerate(coords):
            b_elements = coord.dq_dx(molecule) * scale[n]

            for m, atom in enumerate(coord.atoms):
                loc = 3 * atom
                b_matrix[n, loc:loc + 3] = b_elements[m, :]

        return b_matrix


class SimpleInternalCoordinate(InternalCoordinate):

    def __init__(self, atoms, name):
        """
        :param atoms: A list of zero-based atom indices
        """
        self._name = name

        try:
            for v in atoms:
                if int(v) < 0:
                    raise CoordinateError('Atom identifier cannot be negative.')
        except:
            raise CoordinateError('Atoms must be an iterable list of whole numbers.')
        self._atoms = atoms

    @property
    def type(self):
        return self._type

    @property
    def atoms(self):
        return self._atoms

    @property
    def a(self):
        try:
            return self.atoms[0]
        except:
            raise CoordinateError('a() called but atoms[0] does not exist.')

    @property
    def b(self):
        try:
            return self.atoms[1]
        except:
            raise CoordinateError('b() called but atoms[1] does not exist.')

    @property
    def c(self):
        try:
            return self.atoms[2]
        except:
            raise CoordinateError('c() called but atoms[2] does not exist.')

    @property
    def d(self):
        try:
            return self.atoms[3]
        except:
            raise CoordinateError('d() called but atoms[3] does not exist.')

    def q(self, geom):
        """
        Given geometry, returns value of the internal coordinate.
        :param geom: matrix of coordinates [natoms, 3]
        :return: the value of the coordinate
        """
        raise NotImplementedError()

    def dq_dx(self, geom):
        """
        Computes the first derivative of the coordinate (aka B matrix elements)
        :param geom: matrix of coordinates [natoms, 3]
        :return:
        """
        raise NotImplementedError()

    def d2q_dx2(self, geom):
        """
        Computes the second derivative of the coordinate
        :param geom: matrix of coordinates [natoms, 3]
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def _derivatives(cls, zeroth_order, variables):
        """
        Computes the first and second derivative of zeroth_order.
        :param zeroth_order: the base equation
        :param variables: the variables to take the derivative over
        :return: tuple (zeroth_order, first_order, second_order)
        """
        first = [sp.diff(zeroth_order, coord) for coord in variables]
        second = []
        for gradient in first:
            second.extend(sp.diff(gradient, coord) for coord in variables)
        return (
            first,
            second
        )


class Stretch(SimpleInternalCoordinate):
    # The sympy equation for a stretch
    _0th_order = sp.sqrt((Ax - Bx) ** 2 + (Ay - By) ** 2 + (Az - Bz) ** 2)

    # First derivative equation for a stretch
    _1st_order = None

    # Second derivative equation for a stretch
    _2nd_order = None

    # These are the variables used in the equation
    # Allows for zipping the variables with their data
    _variables = [Ax, Ay, Az, Bx, By, Bz]

    def __init__(self, atoms):
        super(Stretch, self).__init__(atoms, "STRE")

        if not Stretch._1st_order or not Stretch._2nd_order:
            (Stretch._1st_order, Stretch._2nd_order) = \
                SimpleInternalCoordinate._derivatives(
                    Stretch._0th_order,
                    Stretch._variables)

    def q(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])

        values = list(zip(Stretch._variables, a + b))

        return Stretch._0th_order.subs(values)

    def dq_dx(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])

        values = list(zip(Stretch._variables, a + b))

        return np.array([term.subs(values) for term in Stretch._1st_order], dtype=np.float64).reshape(-1, 3)

    def d2q_dx2(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])

        values = list(zip(Stretch._variables, a + b))

        return np.array([term.subs(values) for term in Stretch._2nd_order], dtype=np.float64).reshape(-1, 6)

    def q_code(self):
        return lambdastr(Stretch._variables, Stretch._0th_order)

    def dq_dx_code(self):
        return [lambdastr(Stretch._variables, term) for term in Stretch._1st_order]

    def d2q_dx2_code(self):
        return [lambdastr(Stretch._variables, term) for term in Stretch._2nd_order]

    def __str__(self):
        return "Bond {}-{}".format(self.a, self.b)

    @staticmethod
    def compute(ax, ay, az, bx, by, bz):
        temps = [_T1, _T2, _T3, _T4, _T5, _T6]
        values = list(zip(Stretch._variables, temps))
        subs = Stretch._0th_order.subs(values)
        values = list(zip(temps, [ax, ay, az, bx, by, bz]))
        # Do not convert to t a float in case the caller passes variables
        return subs.subs(values)


class Angle(SimpleInternalCoordinate):
    # The sympy symbolic equation for a bond angle
    _0th_order = sp.acos(np.dot(unit_vector(Bx, By, Bz, Cx, Cy, Cz), unit_vector(Bx, By, Bz, Ax, Ay, Az)))

    # First derivative
    _1st_order = None

    # Second derivative
    _2nd_order = None

    # These are the variables used in the equation.
    # This allows for zipping the variables with their data.
    _variables = [Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz]

    def __init__(self, atoms):
        super(Angle, self).__init__(atoms, "ANGL")

        if not Angle._1st_order or not Angle._2nd_order:
            (Angle._1st_order, Angle._2nd_order) = \
                SimpleInternalCoordinate._derivatives(
                    Angle._0th_order,
                    Angle._variables
                )

    def q(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])
        c = list(geom[self.c])

        values = list(zip(Angle._variables, a + b + c))

        return float(Angle._0th_order.subs(values))

    def dq_dx(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])
        c = list(geom[self.c])

        values = list(zip(Angle._variables, a + b + c))

        return np.array([term.subs(values) for term in Angle._1st_order], dtype=np.float64).reshape(-1, 3)

    def d2q_dx2(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])
        c = list(geom[self.c])

        values = list(zip(Angle._variables, a + b + c))

        return np.array([term.subs(values) for term in Angle._2nd_order], dtype=np.float64).reshape(-1, 9)

    def q_code(self):
        return lambdastr(Angle._variables, Angle._0th_order)

    def dq_dx_code(self):
        return [lambdastr(Angle._variables, term) for term in Angle._1st_order]

    def d2q_dx2_code(self):
        return [lambdastr(Angle._variables, term) for term in Angle._2nd_order]

    def __str__(self):
        return "Angle {}-{}-{}".format(self.a, self.b, self.c)

    def is_valid_value(self, value):
        return 0.0 <= value <= np.pi

    def canonicalize_value(self, value):
        return value

    @staticmethod
    def compute(ax, ay, az, bx, by, bz, cx, cy, cz):
        """The angle between vector BA and BC."""
        temps = [_T1, _T2, _T3, _T4, _T5, _T6, _T7, _T8, _T9]
        values = list(zip(Angle._variables, temps))
        subs = Angle._0th_order.subs(values)
        values = list(zip(temps, [ax, ay, az, bx, by, bz, cx, cy, cz]))
        return subs.subs(values)


class OutOfPlane(SimpleInternalCoordinate):
    # The sympy symbolic equation for an out of plane angle
    _0th_order = sp.asin(
        np.dot(
            unit_vector(Bx, By, Bz, Ax, Ay, Az),
            np.cross(unit_vector(Bx, By, Bz, Cx, Cy, Cz), unit_vector(Bx, By, Bz, Dx, Dy, Dz))
        )
        / sp.sin(Angle.compute(Cx, Cy, Cz, Bx, By, Bz, Dx, Dy, Dz)))

    # First derivative
    _1st_order = None

    # Second derivative
    _2nd_order = None

    # These are the variables used in the equation
    # This allows for zipping the variables with their data.
    _variables = [Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz]

    def __init__(self, atoms):
        super(OutOfPlane, self).__init__(atoms, "OUT ")

        if not OutOfPlane._1st_order or not OutOfPlane._2nd_order:
            (OutOfPlane._1st_order, OutOfPlane._2nd_order) = \
                SimpleInternalCoordinate._derivatives(
                    OutOfPlane._0th_order,
                    OutOfPlane._variables
                )

    def q(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])
        c = list(geom[self.c])
        d = list(geom[self.d])

        values = list(zip(OutOfPlane._variables, a + b + c + d))

        return float(OutOfPlane._0th_order.subs(values))

    def dq_dx(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])
        c = list(geom[self.c])
        d = list(geom[self.d])

        values = list(zip(OutOfPlane._variables, a + b + c + d))

        return np.array([term.subs(values) for term in OutOfPlane._1st_order], dtype=np.float64).reshape(-1, 3)

    def d2q_dx2(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])
        c = list(geom[self.c])
        d = list(geom[self.d])

        values = list(zip(OutOfPlane._variables, a + b + c + d))

        return np.array([term.subs(values) for term in OutOfPlane._2nd_order], dtype=np.float64).reshape(-1, 12)

    def q_code(self):
        return lambdastr(OutOfPlane._variables, OutOfPlane._0th_order)

    def dq_dx_code(self):
        return [lambdastr(OutOfPlane._variables, term) for term in OutOfPlane._1st_order]

    def d2q_dx2_code(self):
        return [lambdastr(OutOfPlane._variables, term) for term in OutOfPlane._2nd_order]

    def __str__(self):
        return "OutOfPlane {}-{}-{}-{}".format(self.a, self.b, self.c, self.d)

    def is_valid_value(self, value):
        return -np.pi / 2.0 <= value <= np.pi / 2.0


class PeriodicCoordinate(SimpleInternalCoordinate):

    def __init__(self, atoms, discontinuity, period, name):
        super(PeriodicCoordinate, self).__init__(atoms, name)

        self._discontinuity = discontinuity
        self._period = period

    def canonicalize_value(self, value):
        newval = value - self._discontinuity
        nperiods = np.floor(newval / self._period)
        remainder = newval - nperiods * self._period
        return remainder + self._discontinuity

    def q(self, geom):
        raise NotImplementedError

    def dq_dx(self, geom):
        raise NotImplementedError

    def d2q_dx2(self, geom):
        raise NotImplementedError


class Torsion(PeriodicCoordinate):
    # The sympy symbolic equation for an out of plane angle
    # _0th_order = sp.acos(np.dot(
    #     np.cross(unit_vector(Bx, By, Bz, Ax, Ay, Az), unit_vector(Bx, By, Bz, Cx, Cy, Cz)),
    #     np.cross(unit_vector(Cx, Cy, Cz, Bx, By, Bz), unit_vector(Cx, Cy, Cz, Dx, Dy, Dz))
    # ) / (sp.sin(Angle.compute(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz)) * sp.sin(Angle.compute(Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz))))
    # _0th_order = sp.asin(np.dot(
    #     unit_vector(Bx, By, Bz, Ax, Ay, Az),
    #     np.cross(
    #         unit_vector(Cx, Cy, Cz, Bx, By, Bz), unit_vector(Cx, Cy, Cz, Dx, Dy, Dz)
    #     )
    # ) / sp.sin(Angle.compute(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz)) * sp.sin(Angle.compute(Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz)))
    # ) / sp.sqrt(1.0 - (np.dot(unit_vector(Bx, By, Bz, Cx, Cy, Cz), unit_vector(Bx, By, Bz, Ax, Ay, Az)))**2) *
    #                      sp.sqrt(1.0 - (np.dot(unit_vector(Cx, Cy, Cz, Dx, Dy, Dz), unit_vector(Cx, Cy, Cz, Bx, By, Bz)))**2))

    _0th_order = sp.atan(
        (Stretch.compute(Bx, By, Bz, Cx, Cy, Cz) *
         np.dot([Ax - Bx, Ay - By, Az - Bz], np.cross([Bx - Cx, By - Cy, Bz - Cz], [Dx - Cx, Dy - Cy, Dz - Cz])))
        /
        np.dot(
            np.cross([Ax - Bx, Ay - By, Az - Bz], [Cx - Bx, Cy - By, Cz - Bz]),
            np.cross([Bx - Cx, By - Cy, Bz - Cz], [Dx - Cx, Dy - Cy, Dz - Cz])
        )
    )

    # First derivative
    _1st_order = None

    # Second derivative
    _2nd_order = None

    _variables = [Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz]

    def __init__(self, atoms, discontinuity=-np.pi / 4.0):
        super(Torsion, self).__init__(atoms, discontinuity, 2.0 * np.pi, "TORS")

        if not Torsion._1st_order or not Torsion._2nd_order:
            (Torsion._1st_order, Torsion._2nd_order) = \
                SimpleInternalCoordinate._derivatives(
                    Torsion._0th_order,
                    Torsion._variables
                )

    def q(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])
        c = list(geom[self.c])
        d = list(geom[self.d])

        values = list(zip(Torsion._variables, a + b + c + d))
        result = Torsion._0th_order.subs(values)

        return result

    def dq_dx(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])
        c = list(geom[self.c])
        d = list(geom[self.d])

        values = list(zip(Torsion._variables, a + b + c + d))

        return np.array([term.subs(values) for term in Torsion._1st_order], dtype=np.float64).reshape(-1, 3)

    def d2q_dx2(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])
        c = list(geom[self.c])
        d = list(geom[self.d])

        values = list(zip(Torsion._variables, a + b + c + d))

        return np.array([term.subs(values) for term in Torsion._2nd_order], dtype=np.float64).reshape(-1, 12)

    def q_code(self):
        return lambdastr(Torsion._variables, Torsion._0th_order)

    def dq_dx_code(self):
        return [lambdastr(Torsion._variables, term) for term in Torsion._1st_order]

    def d2q_dx2_code(self):
        return [lambdastr(Torsion._variables, term) for term in Torsion._2nd_order]

    @staticmethod
    def compute(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz):
        values = list(zip(Torsion._variables, [ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz]))
        return Torsion._0th_order.subs(values)


class LinX(SimpleInternalCoordinate):
    _0th_order = sp.cos(Torsion.compute(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz)) * \
                 sp.sin(Angle.compute(Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz))

    _1st_order = None

    _2nd_order = None

    _variables = [Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz]

    def __init__(self, atoms):
        super(LinX, self).__init__(atoms, "LINX")

        if not LinX._1st_order or not LinX._2nd_order:
            (LinX._1st_order, LinX._2nd_order) = \
                SimpleInternalCoordinate._derivatives(
                    LinX._0th_order,
                    LinX._variables
                )

    def q(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])
        c = list(geom[self.c])
        d = list(geom[self.d])

        values = list(zip(LinX._variables, a + b + c + d))

        return LinX._0th_order.subs(values)

    def dq_dx(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])
        c = list(geom[self.c])
        d = list(geom[self.d])

        values = list(zip(LinX._variables, a + b + c + d))

        return np.array([term.subs(values) for term in LinX._1st_order], dtype=np.float64).reshape(-1, 3)

    def d2q_dx2(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])
        c = list(geom[self.c])
        d = list(geom[self.d])

        values = list(zip(LinX._variables, a + b + c + d))

        return np.array([term.subs(values) for term in LinX._2nd_order], dtype=np.float64)


class LinY(SimpleInternalCoordinate):
    _0th_order = sp.sin(Torsion.compute(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz)) * \
                 sp.sin(Angle.compute(Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz))

    _1st_order = None

    _2nd_order = None

    _variables = [Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz]

    def __init__(self, atoms):
        super(LinY, self).__init__(atoms, "LINX")

        if not LinY._1st_order or not LinY._2nd_order:
            (LinY._1st_order, LinY._2nd_order) = \
                SimpleInternalCoordinate._derivatives(
                    LinY._0th_order,
                    LinY._variables
                )

    def q(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])
        c = list(geom[self.c])
        d = list(geom[self.d])

        values = list(zip(LinY._variables, a + b + c + d))

        return LinY._0th_order.subs(values)

    def dq_dx(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])
        c = list(geom[self.c])
        d = list(geom[self.d])

        values = list(zip(LinY._variables, a + b + c + d))

        return np.array([term.subs(values) for term in LinY._1st_order], dtype=np.float64).reshape(-1, 3)

    def d2q_dx2(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])
        c = list(geom[self.c])
        d = list(geom[self.d])

        values = list(zip(LinY._variables, a + b + c + d))

        return np.array([term.subs(values) for term in LinY._2nd_order], dtype=np.float64)


class SPF(SimpleInternalCoordinate):
    _variables = [Ax, Ay, Az, Bx, By, Bz]

    def __init__(self, atoms, reference_bond_length):
        super(SPF, self).__init__(atoms, "SPF")

        self._0th_order = 1.0 - (reference_bond_length / sp.sqrt((Ax - Bx) ** 2 + (Ay - By) ** 2 + (Az - Bz) ** 2))
        (self._1st_order, self._2nd_order) = SimpleInternalCoordinate._derivatives(self._0th_order, SPF._variables)

    def q(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])

        values = list(zip(SPF._variables, a + b))

        return self._0th_order.subs(values)

    def dq_dx(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])

        values = list(zip(SPF._variables, a + b))

        return np.array([term.subs(values) for term in self._1st_order], dtype=np.float64).reshape(-1, 3)

    def d2q_dx2(self, geom):
        a = list(geom[self.a])
        b = list(geom[self.b])

        values = list(zip(SPF._variables, a + b))

        return np.array([term.subs(values) for term in self._2nd_order], dtype=np.float64)

    def __str__(self):
        return "SPF {}-{}".format(self.a, self.b)


def generate_static():
    """Write a file sympy_intco_computers.py of functions evaluating 0th, 1st, & 2nd
    derivatives of Stretch, Angle, OutOfPlane, and Torsion internal coordinates.
    
    """
    text = []
    text.append('import math')
    text.append('from math import sqrt')
    text.append('import numpy as np')
    
    intco = Stretch([0, 1])
    dim = len(Stretch._variables)

    scode = intco.q_code()
    text.append('\n\ndef stretch_q(gA, gB):')
    text.append('\t"""Zeroth derivative of stretch coordinate as scalar."""\n')
    text.append('\tAx, Ay, Az = gA')
    text.append('\tBx, By, Bz = gB')
    text.append('\n\tlam = {}'.format(scode))
    text.append('\treturn lam(Ax, Ay, Az, Bx, By, Bz)')

    scode = intco.dq_dx_code()
    text.append('\n\ndef stretch_dq_dx(gA, gB):')
    text.append('\t"""First derivative of stretch coordinate as ({}, 3) ndarray."""\n'.format(dim // 3))
    text.append('\tAx, Ay, Az = gA')
    text.append('\tBx, By, Bz = gB')
    text.append('\n\tlams = [{}]'.format(',\n\t\t\t'.join(scode)))
    text.append('\treturn np.array([lam(Ax, Ay, Az, Bx, By, Bz) for lam in lams], dtype=np.float64).reshape(-1, 3)')

    scode = intco.d2q_dx2_code()
    text.append('\n\ndef stretch_d2q_dx2(gA, gB):')
    text.append('\t"""Second derivative of stretch coordinate as ({}, {}) ndarray."""\n'.format(dim, dim))
    text.append('\tAx, Ay, Az = gA')
    text.append('\tBx, By, Bz = gB')
    text.append('\n\tlams = [{}]'.format(',\n\t\t\t'.join(scode)))
    text.append('\treturn np.array([lam(Ax, Ay, Az, Bx, By, Bz) for lam in lams], dtype=np.float64).reshape(-1, {})'.format(dim))

    intco = Angle([0, 1, 2])
    dim = len(Angle._variables)

    scode = intco.q_code()
    text.append('\n\ndef angle_q(gA, gB, gC):')
    text.append('\t"""Zeroth derivative of Angle coordinate as scalar."""\n')
    text.append('\tAx, Ay, Az = gA')
    text.append('\tBx, By, Bz = gB')
    text.append('\tCx, Cy, Cz = gC')
    text.append('\n\tlam = {}'.format(scode))
    text.append('\treturn lam(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz)')

    scode = intco.dq_dx_code()
    text.append('\n\ndef angle_dq_dx(gA, gB, gC):')
    text.append('\t"""First derivative of Angle coordinate as ({}, 3) ndarray."""\n'.format(dim // 3))
    text.append('\tAx, Ay, Az = gA')
    text.append('\tBx, By, Bz = gB')
    text.append('\tCx, Cy, Cz = gC')
    text.append('\n\tlams = [{}]'.format(',\n\t\t\t'.join(scode)))
    text.append('\treturn np.array([lam(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz) for lam in lams], dtype=np.float64).reshape(-1, 3)')

    scode = intco.d2q_dx2_code()
    text.append('\n\ndef angle_d2q_dx2(gA, gB, gC):')
    text.append('\t"""Second derivative of Angle coordinate as ({}, {}) ndarray."""\n'.format(dim, dim))
    text.append('\tAx, Ay, Az = gA')
    text.append('\tBx, By, Bz = gB')
    text.append('\tCx, Cy, Cz = gC')
    text.append('\n\tlams = [{}]'.format(',\n\t\t\t'.join(scode)))
    text.append('\treturn np.array([lam(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz) for lam in lams], dtype=np.float64).reshape(-1, {})'.format(dim))

    intco = OutOfPlane([0, 1, 2, 3])
    dim = len(OutOfPlane._variables)

    scode = intco.q_code()
    text.append('\n\ndef outofplane_q(gA, gB, gC, gD):')
    text.append('\t"""Zeroth derivative of OutOfPlane coordinate as scalar."""\n')
    text.append('\tAx, Ay, Az = gA')
    text.append('\tBx, By, Bz = gB')
    text.append('\tCx, Cy, Cz = gC')
    text.append('\tDx, Dy, Dz = gD')
    text.append('\n\tlam = {}'.format(scode))
    text.append('\treturn lam(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz)')

    scode = intco.dq_dx_code()
    text.append('\n\ndef outofplane_dq_dx(gA, gB, gC, gD):')
    text.append('\t"""First derivative of OutOfPlane coordinate as ({}, 3) ndarray."""\n'.format(dim // 3))
    text.append('\tAx, Ay, Az = gA')
    text.append('\tBx, By, Bz = gB')
    text.append('\tCx, Cy, Cz = gC')
    text.append('\tDx, Dy, Dz = gD')
    text.append('\n\tlams = [{}]'.format(',\n\t\t\t'.join(scode)))
    text.append('\treturn np.array([lam(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz) for lam in lams], dtype=np.float64).reshape(-1, 3)')

    scode = intco.d2q_dx2_code()
    text.append('\n\ndef outofplane_d2q_dx2(gA, gB, gC, gD):')
    text.append('\t"""Second derivative of OutOfPlane coordinate as ({}, {}) ndarray."""\n'.format(dim, dim))
    text.append('\tAx, Ay, Az = gA')
    text.append('\tBx, By, Bz = gB')
    text.append('\tCx, Cy, Cz = gC')
    text.append('\tDx, Dy, Dz = gD')
    text.append('\n\tlams = [{}]'.format(',\n\t\t\t'.join(scode)))
    text.append('\treturn np.array([lam(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz) for lam in lams], dtype=np.float64).reshape(-1, {})'.format(dim))

    intco = Torsion([0, 1, 2, 3])
    dim = len(Torsion._variables)

    scode = intco.q_code()
    text.append('\n\ndef torsion_q(gA, gB, gC, gD):')
    text.append('\t"""Zeroth derivative of Torsion coordinate as scalar."""\n')
    text.append('\tAx, Ay, Az = gA')
    text.append('\tBx, By, Bz = gB')
    text.append('\tCx, Cy, Cz = gC')
    text.append('\tDx, Dy, Dz = gD')
    text.append('\n\tlam = {}'.format(scode))
    text.append('\treturn lam(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz)')

    scode = intco.dq_dx_code()
    text.append('\n\ndef torsion_dq_dx(gA, gB, gC, gD):')
    text.append('\t"""First derivative of Torsion coordinate as ({}, 3) ndarray."""\n'.format(dim // 3))
    text.append('\tAx, Ay, Az = gA')
    text.append('\tBx, By, Bz = gB')
    text.append('\tCx, Cy, Cz = gC')
    text.append('\tDx, Dy, Dz = gD')
    text.append('\n\tlams = [{}]'.format(',\n\t\t\t'.join(scode)))
    text.append('\treturn np.array([lam(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz) for lam in lams], dtype=np.float64).reshape(-1, 3)')

    scode = intco.d2q_dx2_code()
    text.append('\n\ndef torsion_d2q_dx2(gA, gB, gC, gD):')
    text.append('\t"""Second derivative of Torsion coordinate as ({}, {}) ndarray."""\n'.format(dim, dim))
    text.append('\tAx, Ay, Az = gA')
    text.append('\tBx, By, Bz = gB')
    text.append('\tCx, Cy, Cz = gC')
    text.append('\tDx, Dy, Dz = gD')
    text.append('\n\tlams = [{}]'.format(',\n\t\t\t'.join(scode)))
    text.append('\treturn np.array([lam(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz) for lam in lams], dtype=np.float64).reshape(-1, {})'.format(dim))


    text.append('\n\nif __name__ == "__main__":')
    text.append('\tgA = [1., 1., 1.]')
    text.append('\tgB = [0., 0., 0.]')
    text.append('\tgC = [-1., 0., 0.]')
    text.append('\tgD = [-2., 1., 0.]')

    text.append('\n\tans = stretch_q(gA, gB)')
    text.append('\tassert np.isclose(1.73205080756888, ans)')

    text.append('\n\tans = stretch_dq_dx(gA, gB)')
    text.append("""\tassert np.allclose(
np.array([[ 0.57735027,  0.57735027,  0.57735027],
       [-0.57735027, -0.57735027, -0.57735027]]), ans)""")

    text.append('\n\tans = stretch_d2q_dx2(gA, gB)')
    text.append("""\tassert np.allclose(
np.array([[ 0.38490018, -0.19245009, -0.19245009, -0.38490018,  0.19245009, 0.19245009],
       [-0.19245009,  0.38490018, -0.19245009,  0.19245009, -0.38490018, 0.19245009],
       [-0.19245009, -0.19245009,  0.38490018,  0.19245009,  0.19245009, -0.38490018],
       [-0.38490018,  0.19245009,  0.19245009,  0.38490018, -0.19245009, -0.19245009],
       [ 0.19245009, -0.38490018,  0.19245009, -0.19245009,  0.38490018, -0.19245009],
       [ 0.19245009,  0.19245009, -0.38490018, -0.19245009, -0.19245009, 0.38490018]]), ans)""")
    text.append('\tprint("Stretch Passed!")')

    text.append('\n\tans = angle_q(gA, gB, gC)')
    text.append('\tassert np.isclose(2.186276035465284, ans)')

    text.append('\n\tans = angle_dq_dx(gA, gB, gC)')
    text.append("""\tassert np.allclose(
np.array([[ 0.47140452, -0.23570226, -0.23570226],
       [-0.47140452,  0.94280904,  0.94280904],
       [ 0.        , -0.70710678, -0.70710678]]), ans)""")

    text.append('\n\tans = angle_d2q_dx2(gA, gB, gC)')
    text.append("""\tassert np.allclose(
np.array([[-3.14269681e-01, -7.85674201e-02, -7.85674201e-02,
         3.14269681e-01,  7.85674201e-02,  7.85674201e-02,
         0.00000000e+00,  8.32667268e-17,  1.11022302e-16],
       [-7.85674201e-02,  3.92837101e-02,  2.74985970e-01,
         7.85674201e-02,  3.14269681e-01, -6.28539361e-01,
         0.00000000e+00, -3.53553391e-01,  3.53553391e-01],
       [-7.85674201e-02,  2.74985970e-01,  3.92837101e-02,
         7.85674201e-02, -6.28539361e-01,  3.14269681e-01,
         0.00000000e+00,  3.53553391e-01, -3.53553391e-01],
       [ 3.14269681e-01,  7.85674201e-02,  7.85674201e-02,
        -3.14269681e-01, -7.85674201e-01, -7.85674201e-01,
         6.79869978e-17,  7.07106781e-01,  7.07106781e-01],
       [ 7.85674201e-02,  3.14269681e-01, -6.28539361e-01,
        -7.85674201e-01, -1.02137646e+00,  1.33564614e+00,
         7.07106781e-01,  7.07106781e-01, -7.07106781e-01],
       [ 7.85674201e-02, -6.28539361e-01,  3.14269681e-01,
        -7.85674201e-01,  1.33564614e+00, -1.02137646e+00,
         7.07106781e-01, -7.07106781e-01,  7.07106781e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         6.79869978e-17,  7.07106781e-01,  7.07106781e-01,
         0.00000000e+00, -7.07106781e-01, -7.07106781e-01],
       [ 8.32667268e-17, -3.53553391e-01,  3.53553391e-01,
         7.07106781e-01,  7.07106781e-01, -7.07106781e-01,
        -7.07106781e-01, -3.53553391e-01,  3.53553391e-01],
       [ 1.11022302e-16,  3.53553391e-01, -3.53553391e-01,
         7.07106781e-01, -7.07106781e-01,  7.07106781e-01,
        -7.07106781e-01,  3.53553391e-01, -3.53553391e-01]]), ans)""")
    text.append('\tprint("Angle Passed!")')

    text.append('\n\tans = outofplane_q(gA, gB, gC, gD)')
    text.append('\tassert np.isclose(-0.6154797086703873, ans)')

    text.append('\n\tans = outofplane_dq_dx(gA, gB, gC, gD)')
    text.append("""\tassert np.allclose(
np.array([[ 2.35702260e-01,  2.35702260e-01, -4.71404521e-01],
       [-2.35702260e-01, -2.35702260e-01,  1.88561808e+00],
       [ 0.00000000e+00, -5.43895982e-16, -2.12132034e+00],
       [ 0.00000000e+00,  1.35973996e-16,  7.07106781e-01]]), ans)""")

    text.append('\n\tans = outofplane_d2q_dx2(gA, gB, gC, gD)')
    text.append("""\tassert np.allclose(
np.array([[-3.92837101e-02, -2.74985970e-01,  7.85674201e-02,
         3.92837101e-02,  2.74985970e-01, -7.85674201e-02,
         0.00000000e+00,  2.22044605e-16,  3.53553391e-01,
         0.00000000e+00, -8.32667268e-17, -3.53553391e-01],
       [-2.74985970e-01, -3.92837101e-02,  7.85674201e-02,
         2.74985970e-01,  3.92837101e-02, -7.85674201e-02,
         0.00000000e+00,  2.22044605e-16, -3.53553391e-01,
         0.00000000e+00, -8.32667268e-17,  3.53553391e-01],
       [ 7.85674201e-02,  7.85674201e-02,  3.14269681e-01,
        -7.85674201e-02, -7.85674201e-02, -3.14269681e-01,
         0.00000000e+00, -4.44089210e-16,  3.33066907e-16,
        -2.77555756e-17,  1.66533454e-16, -5.55111512e-17],
       [ 3.92837101e-02,  2.74985970e-01, -7.85674201e-02,
        -3.92837101e-02, -2.74985970e-01, -1.33564614e+00,
         0.00000000e+00, -3.92523115e-17,  1.06066017e+00,
        -7.78000756e-17,  1.55600151e-16,  3.53553391e-01],
       [ 2.74985970e-01,  3.92837101e-02, -7.85674201e-02,
        -2.74985970e-01, -3.92837101e-02, -1.33564614e+00,
         5.43895982e-16,  1.04853965e-15,  3.18198052e+00,
        -1.45787073e-16, -2.52321835e-16, -1.76776695e+00],
       [-7.85674201e-02, -7.85674201e-02, -3.14269681e-01,
        -1.33564614e+00, -1.33564614e+00,  3.14269681e-01,
         2.12132034e+00,  2.12132034e+00, -8.88178420e-16,
        -7.07106781e-01, -7.07106781e-01,  2.22044605e-16],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  5.43895982e-16,  2.12132034e+00,
         0.00000000e+00, -5.43895982e-16, -2.12132034e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [ 2.94610324e-16,  2.94610324e-16, -5.89220647e-16,
        -9.06493304e-17,  9.97142634e-16,  2.12132034e+00,
        -5.43895982e-16, -8.15843973e-16, -4.24264069e+00,
        -2.26442191e-32,  1.22376596e-15,  2.12132034e+00],
       [ 3.53553391e-01, -3.53553391e-01,  1.11022302e-16,
         1.06066017e+00,  3.18198052e+00, -4.44089210e-16,
        -2.12132034e+00, -4.24264069e+00,  3.53553391e-01,
         7.07106781e-01,  1.41421356e+00, -3.53553391e-01],
       [-1.69967494e-17, -1.69967494e-17,  3.39934989e-17,
        -6.79869978e-17, -1.35973996e-16, -7.07106781e-01,
         0.00000000e+00,  0.00000000e+00,  7.07106781e-01,
         0.00000000e+00, -1.35973996e-16,  0.00000000e+00],
       [-9.06493304e-17, -9.06493304e-17,  1.81298661e-16,
         1.58636328e-16, -2.49285659e-16, -7.07106781e-01,
         0.00000000e+00,  1.22376596e-15,  1.41421356e+00,
        -1.35973996e-16, -6.79869978e-16, -7.07106781e-01],
       [-3.53553391e-01,  3.53553391e-01, -8.32667268e-17,
         3.53553391e-01, -1.76776695e+00,  4.44089210e-16,
         0.00000000e+00,  2.12132034e+00, -3.53553391e-01,
         2.94392336e-17, -7.07106781e-01,  3.53553391e-01]]), ans)""")
    text.append('\tprint("OutOfPlane Passed!")')

    text.append('\n\tans = torsion_q(gA, gB, gC, gD)')
    text.append('\tassert np.isclose(0.785398163397448, ans)')

    text.append('\n\tans = torsion_dq_dx(gA, gB, gC, gD)')
    text.append("""\tassert np.allclose(
np.array([[ 0. , -0.5,  0.5],
       [ 0. ,  1. , -2. ],
       [ 0. , -0.5,  2.5],
       [ 0. ,  0. , -1. ]]), ans)""")

    text.append('\n\tans = torsion_d2q_dx2(gA, gB, gC, gD)')
    text.append("""\tassert np.allclose(
np.array([[ 0. ,  0. ,  0. ,  0. ,  0.5, -0.5,  0. , -0.5,  0.5,  0. ,  0. , 0. ],
       [ 0. ,  0.5,  0. ,  0. , -1. ,  0. ,  0. ,  0.5,  0. ,  0. ,  0. , 0. ],
       [ 0. ,  0. , -0.5,  0. ,  0. ,  1. ,  0. ,  0. , -0.5,  0. ,  0. , 0. ],
       [ 0. ,  0. ,  0. ,  0. , -1. ,  2. ,  0. ,  1. , -2. ,  0. ,  0. , 0. ],
       [ 0.5, -1. ,  0. , -1. ,  2.5,  1.5,  0.5, -1.5, -2.5,  0. ,  0. , 1. ],
       [-0.5,  0. ,  1. ,  2. ,  1.5, -2.5, -2.5, -2.5,  1.5,  1. ,  1. , 0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0.5, -2.5,  0. , -0.5,  2.5,  0. ,  0. , 0. ],
       [-0.5,  0.5,  0. ,  1. , -1.5, -2.5, -0.5,  1. ,  4.5,  0. ,  0. , -2. ],
       [ 0.5,  0. , -0.5, -2. , -2.5,  1.5,  2.5,  4.5, -1. , -1. , -2. , 0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. , -1. ,  0. ,  0. , 0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. , -2. ,  0. ,  0. , 1. ],
       [ 0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. , -2. ,  0. ,  0. ,  1. , 0. ]]), ans)""")
    text.append('\tprint("Torsion Passed!")')

    text.append('\tprint("All Passed!")')
    text.append('')

    with open('sympy_intco_computers.py', 'w') as fp:
        fp.write('\n'.join(text).replace('\t', '    '))


if __name__ == "__main__":
    generate_static()

