# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

"""
Defines SymmetryGroup parent class and PointGroup and SpaceGroup classes.
Shyue Ping Ong thanks Marc De Graef for his generous sharing of his
SpaceGroup data as published in his textbook "Structure of Materials".
"""

import os
import re
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from fractions import Fraction
from itertools import product
from monty.serialization import loadfn
from monty.design_patterns import cached_class

import numpy as np

from . import symm_op
SymmOp = symm_op.SymmOp

SYMM_DATA = None

class SymmGroup:
    def __init__(self, name):
        self.pgGroup = name[0]
        self.pgOrder = name[1]
        self.true_symNo = 0
        self.symm_opsL = []
        self.symm_opsR = []
        self.text_to_symm_ops(self.fill_symmetry_class())
        self.add_identity_op()

    def add_identity_op(self):
        self.symm_opsR.append(SymmOp.identity())

    def fill_symmetry_class(self):
        if self.pgGroup == 'C':
            fileContent = "rot_axis " + self.pgOrder + " 0 0 1"
        return fileContent

    def text_to_symm_ops(self, text):
        text = text.split()
        i = 0
        while i < len(text):
            if text[i] == 'rot_axis':
                i += 1
                fold = int(text[i])
                self.true_symNo += fold - 1
                axis = np.zeros(3)
                for j in range(3):
                    i += 1
                    axis[j] = float(text[i])
                i += 1

                rot_ang = ang_incr = 360. / fold
                for j in range(1, fold):
                    identity = SymmOp.from_rotation_and_translation()
                    elem = SymmOp.from_axis_angle_and_translation(axis, rot_ang)
                    trans_elem = SymmOp.from_rotation_and_translation(elem.rotation_matrix.transpose(), elem.translation_vector)
                    rot_ang += ang_incr
                    self.symm_opsL.append(identity)
                    self.symm_opsR.append(trans_elem)

        self.chain_length = [1 for i in range(self.true_symNo)]
        self.compute_subgroup()

    def compute_subgroup(self):
        tried = np.zeros((self.true_symNo, self.true_symNo))
        ij = [0, 0]

        identity = SymmOp.from_rotation_and_translation()

        while self.found_not_tried(tried, ij, self.true_symNo):
            i, j = ij
            tried[i][j] = 1
            L1 = self.symm_opsL[i]
            R1 = self.symm_opsR[i]

            L2 = self.symm_opsL[j]
            R2 = self.symm_opsR[j]
            newL = L1 * L2
            newR = R1 * R2
            new_chain_length = self.chain_length[i] + self.chain_length[j]
            newR3 = SymmOp.from_rotation_and_translation(newR.rotation_matrix)
            if newL == identity and newR3 == identity:
                continue
            found = False
            for l in range(self.SymsNo):
                L1 = self.symm_opsL[l]
                R1 = self.symm_opsR[l]
                if newL == L1 and newR == R1:
                    found = True
                    break
            if not found:
                self.add_matrices(newL, newR, new_chain_length)
                old_tried = tried
                tried = np.zeros((old_tried.shape[0]+1, old_tried.shape[1]+1))
                tried[:old_tried.shape[0], :old_tried.shape[1]] = old_tried

    def found_not_tried(self, tried, ij, true_symNo):
        ij[0] = 0
        ij[1] = 0
        n = 0
        while n != tried.shape[1]:
            i, j = ij
            #print(ij, n, true_symNo)
            if not (i >= true_symNo and j >= true_symNo) and tried[i][j] == 0:
                return True
            if i != n:
                ij[0] += 1
            else:
                ij[1] -= 1
                if ij[1] == -1:
                    n += 1
                    ij[1] = n
                    ij[0] = 0
        return False

    def add_matrices(self, L, R, chain_length):
        self.chain_length.append(chain_length)
        if self.true_symNo == len(self.symm_opsL):
            self.symm_opsL.append(L)
            self.symm_opsR.append(R)
        else:
            self.symm_opsL[true_symNo] = L
            self.symm_opsR[true_symNo] = R
        self.true_symNo += 1

    @property
    def SymsNo(self):
        return len(self.symm_opsR)


def _get_symm_data(name):
    global SYMM_DATA
    if SYMM_DATA is None:
        SYMM_DATA = loadfn(os.path.join(os.path.dirname(__file__), "symm_data.json"))
        #print(SYMM_DATA["point_group_encoding"])
    return SYMM_DATA[name]


class SymmetryGroup(Sequence, metaclass=ABCMeta):
    """
    Abstract class representation a symmetry group.
    """

    @property
    @abstractmethod
    def symmetry_ops(self):
        """
        :return: List of symmetry operations
        """
        pass

    def __contains__(self, item):
        for i in self.symmetry_ops:
            if np.allclose(i.affine_matrix, item.affine_matrix):
                return True
        return False

    def __hash__(self):
        return self.__len__()

    def __getitem__(self, item):
        return self.symmetry_ops[item]

    def __len__(self):
        return len(self.symmetry_ops)

    def is_subgroup(self, supergroup):
        """
        True if this group is a subgroup of the supplied group.

        Args:
            supergroup (SymmetryGroup): Supergroup to test.

        Returns:
            True if this group is a subgroup of the supplied group.
        """
        warnings.warn("This is not fully functional. Only trivial subsets are tested right now. ")
        return set(self.symmetry_ops).issubset(supergroup.symmetry_ops)

    def is_supergroup(self, subgroup):
        """
        True if this group is a supergroup of the supplied group.

        Args:
            subgroup (SymmetryGroup): Subgroup to test.

        Returns:
            True if this group is a supergroup of the supplied group.
        """
        warnings.warn("This is not fully functional. Only trivial subsets are tested right now. ")
        return set(subgroup.symmetry_ops).issubset(self.symmetry_ops)

    def to_latex_string(self) -> str:
        r"""
        Returns:
            A latex formatted group symbol with proper subscripts and overlines.
        """
        sym = re.sub(r"_(\d+)", r"$_{\1}$", self.to_pretty_string())
        return re.sub(r"-(\d)", r"$\\overline{\1}$", sym)


class PointGroup(SymmetryGroup):
    """
    Class representing a Point Group, with generators and symmetry operations.

    .. attribute:: symbol

        Full International or Hermann-Mauguin Symbol.

    .. attribute:: generators

        List of generator matrices. Note that 3x3 matrices are used for Point
        Groups.

    .. attribute:: symmetry_ops

        Full set of symmetry operations as matrices.
    """

    def __init__(self, int_symbol):
        """
        Initializes a Point Group from its international symbol.

        Args:
            int_symbol (str): International or Hermann-Mauguin Symbol.
        """
        self.symbol = int_symbol
        self.generators = [
            _get_symm_data("generator_matrices")[c] for c in _get_symm_data("point_group_encoding")[int_symbol]
        ]
        self._symmetry_ops = {SymmOp.from_rotation_and_translation(m) for m in self._generate_full_symmetry_ops()}
        self.order = len(self._symmetry_ops)

    @property
    def symmetry_ops(self):
        """
        :return: List of symmetry operations for SpaceGroup
        """
        return self._symmetry_ops

    def _generate_full_symmetry_ops(self):
        symm_ops = list(self.generators)
        new_ops = self.generators
        while len(new_ops) > 0:
            gen_ops = []
            for g1, g2 in product(new_ops, symm_ops):
                op = np.dot(g1, g2)
                if not in_array_list(symm_ops, op):
                    gen_ops.append(op)
                    symm_ops.append(op)
            new_ops = gen_ops
        return symm_ops

    def get_orbit(self, p, tol=1e-5):
        """
        Returns the orbit for a point.

        Args:
            p: Point as a 3x1 array.
            tol: Tolerance for determining if sites are the same. 1e-5 should
                be sufficient for most purposes. Set to 0 for exact matching
                (and also needed for symbolic orbits).

        Returns:
            ([array]) Orbit for point.
        """
        orbit = []
        for o in self.symmetry_ops:
            pp = o.operate(p)
            if not in_array_list(orbit, pp, tol=tol):
                orbit.append(pp)
        return orbit


class SpaceGroup(SymmetryGroup):
    """
    Class representing a SpaceGroup.

    .. attribute:: symbol

        Full International or Hermann-Mauguin Symbol.

    .. attribute:: int_number

        International number

    .. attribute:: generators

        List of generator matrices. Note that 4x4 matrices are used for Space
        Groups.

    .. attribute:: order

        Order of Space Group
    """

    SYMM_OPS = loadfn(os.path.join(os.path.dirname(__file__), "symm_ops.json"))
    SG_SYMBOLS = set(_get_symm_data("space_group_encoding").keys())
    for op in SYMM_OPS:
        op["hermann_mauguin"] = re.sub(r" ", "", op["hermann_mauguin"])
        op["universal_h_m"] = re.sub(r" ", "", op["universal_h_m"])
        SG_SYMBOLS.add(op["hermann_mauguin"])
        SG_SYMBOLS.add(op["universal_h_m"])

    gen_matrices = _get_symm_data("generator_matrices")
    # POINT_GROUP_ENC = SYMM_DATA["point_group_encoding"]
    sgencoding = _get_symm_data("space_group_encoding")
    abbrev_sg_mapping = _get_symm_data("abbreviated_spacegroup_symbols")
    translations = {k: Fraction(v) for k, v in _get_symm_data("translations").items()}
    full_sg_mapping = {v["full_symbol"]: k for k, v in _get_symm_data("space_group_encoding").items()}

    def __init__(self, int_symbol):
        """
        Initializes a Space Group from its full or abbreviated international
        symbol. Only standard settings are supported.

        Args:
            int_symbol (str): Full International (e.g., "P2/m2/m2/m") or
                Hermann-Mauguin Symbol ("Pmmm") or abbreviated symbol. The
                notation is a LaTeX-like string, with screw axes being
                represented by an underscore. For example, "P6_3/mmc".
                Alternative settings can be access by adding a ":identifier".
                For example, the hexagonal setting  for rhombohedral cells can be
                accessed by adding a ":H", e.g., "R-3m:H". To find out all
                possible settings for a spacegroup, use the get_settings
                classmethod. Alternative origin choices can be indicated by a
                translation vector, e.g., 'Fm-3m(a-1/4,b-1/4,c-1/4)'.
        """

        int_symbol = re.sub(r" ", "", int_symbol)
        if int_symbol in SpaceGroup.abbrev_sg_mapping:
            int_symbol = SpaceGroup.abbrev_sg_mapping[int_symbol]
        elif int_symbol in SpaceGroup.full_sg_mapping:
            int_symbol = SpaceGroup.full_sg_mapping[int_symbol]

        for spg in SpaceGroup.SYMM_OPS:
            if int_symbol in [spg["hermann_mauguin"], spg["universal_h_m"]]:
                ops = [SymmOp.from_xyz_string(s) for s in spg["symops"]]
                self.symbol = re.sub(r":", "", re.sub(r" ", "", spg["universal_h_m"]))
                if int_symbol in SpaceGroup.sgencoding:
                    self.full_symbol = SpaceGroup.sgencoding[int_symbol]["full_symbol"]
                    self.point_group = SpaceGroup.sgencoding[int_symbol]["point_group"]
                else:
                    self.full_symbol = re.sub(r" ", "", spg["universal_h_m"])
                    self.point_group = spg["schoenflies"]
                self.int_number = spg["number"]
                self.order = len(ops)
                self._symmetry_ops = ops
                break
        else:
            if int_symbol not in SpaceGroup.sgencoding:
                raise ValueError("Bad international symbol %s" % int_symbol)

            data = SpaceGroup.sgencoding[int_symbol]

            self.symbol = int_symbol
            # TODO: Support different origin choices.
            enc = list(data["enc"])
            inversion = int(enc.pop(0))
            ngen = int(enc.pop(0))
            symm_ops = [np.eye(4)]
            if inversion:
                symm_ops.append(np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
            for i in range(ngen):
                m = np.eye(4)
                m[:3, :3] = SpaceGroup.gen_matrices[enc.pop(0)]
                m[0, 3] = SpaceGroup.translations[enc.pop(0)]
                m[1, 3] = SpaceGroup.translations[enc.pop(0)]
                m[2, 3] = SpaceGroup.translations[enc.pop(0)]
                symm_ops.append(m)
            self.generators = symm_ops
            self.full_symbol = data["full_symbol"]
            self.point_group = data["point_group"]
            self.int_number = data["int_number"]
            self.order = data["order"]

            self._symmetry_ops = None

    def _generate_full_symmetry_ops(self):
        symm_ops = np.array(self.generators)
        for op in symm_ops:
            op[0:3, 3] = np.mod(op[0:3, 3], 1)
        new_ops = symm_ops
        while len(new_ops) > 0 and len(symm_ops) < self.order:
            gen_ops = []
            for g in new_ops:
                temp_ops = np.einsum("ijk,kl", symm_ops, g)
                for op in temp_ops:
                    op[0:3, 3] = np.mod(op[0:3, 3], 1)
                    ind = np.where(np.abs(1 - op[0:3, 3]) < 1e-5)
                    op[ind, 3] = 0
                    if not in_array_list(symm_ops, op):
                        gen_ops.append(op)
                        symm_ops = np.append(symm_ops, [op], axis=0)
            new_ops = gen_ops
        assert len(symm_ops) == self.order
        return symm_ops

    @classmethod
    def get_settings(cls, int_symbol):
        """
        Returns all the settings for a particular international symbol.

        Args:
            int_symbol (str): Full International (e.g., "P2/m2/m2/m") or
                Hermann-Mauguin Symbol ("Pmmm") or abbreviated symbol. The
                notation is a LaTeX-like string, with screw axes being
                represented by an underscore. For example, "P6_3/mmc".

        """
        symbols = []
        if int_symbol in SpaceGroup.abbrev_sg_mapping:
            symbols.append(SpaceGroup.abbrev_sg_mapping[int_symbol])
            int_number = SpaceGroup.sgencoding[int_symbol]["int_number"]
        elif int_symbol in SpaceGroup.full_sg_mapping:
            symbols.append(SpaceGroup.full_sg_mapping[int_symbol])
            int_number = SpaceGroup.sgencoding[int_symbol]["int_number"]
        else:
            for spg in SpaceGroup.SYMM_OPS:
                if int_symbol in [
                    re.split(r"\(|:", spg["hermann_mauguin"])[0],
                    re.split(r"\(|:", spg["universal_h_m"])[0],
                ]:
                    int_number = spg["number"]
                    break

        for spg in SpaceGroup.SYMM_OPS:
            if int_number == spg["number"]:
                symbols.append(spg["hermann_mauguin"])
                symbols.append(spg["universal_h_m"])
        return set(symbols)

    @property
    def symmetry_ops(self):
        """
        Full set of symmetry operations as matrices. Lazily initialized as
        generation sometimes takes a bit of time.
        """
        if self._symmetry_ops is None:
            self._symmetry_ops = [SymmOp(m) for m in self._generate_full_symmetry_ops()]
        return self._symmetry_ops

    def get_orbit(self, p, tol=1e-5):
        """
        Returns the orbit for a point.

        Args:
            p: Point as a 3x1 array.
            tol: Tolerance for determining if sites are the same. 1e-5 should
                be sufficient for most purposes. Set to 0 for exact matching
                (and also needed for symbolic orbits).

        Returns:
            ([array]) Orbit for point.
        """
        orbit = []
        for o in self.symmetry_ops:
            pp = o.operate(p)
            pp = np.mod(np.round(pp, decimals=10), 1)
            if not in_array_list(orbit, pp, tol=tol):
                orbit.append(pp)
        return orbit

    def is_compatible(self, lattice, tol=1e-5, angle_tol=5):
        """
        Checks whether a particular lattice is compatible with the
        *conventional* unit cell.

        Args:
            lattice (Lattice): A Lattice.
            tol (float): The tolerance to check for equality of lengths.
            angle_tol (float): The tolerance to check for equality of angles
                in degrees.
        """
        abc = lattice.lengths
        angles = lattice.angles
        crys_system = self.crystal_system

        def check(param, ref, tolerance):
            return all(abs(i - j) < tolerance for i, j in zip(param, ref) if j is not None)

        if crys_system == "cubic":
            a = abc[0]
            return check(abc, [a, a, a], tol) and check(angles, [90, 90, 90], angle_tol)
        if crys_system == "hexagonal" or (
            crys_system == "trigonal"
            and (
                self.symbol.endswith("H")
                or self.int_number
                in [
                    143,
                    144,
                    145,
                    147,
                    149,
                    150,
                    151,
                    152,
                    153,
                    154,
                    156,
                    157,
                    158,
                    159,
                    162,
                    163,
                    164,
                    165,
                ]
            )
        ):
            a = abc[0]
            return check(abc, [a, a, None], tol) and check(angles, [90, 90, 120], angle_tol)
        if crys_system == "trigonal":
            a = abc[0]
            alpha = angles[0]
            return check(abc, [a, a, a], tol) and check(angles, [alpha, alpha, alpha], angle_tol)
        if crys_system == "tetragonal":
            a = abc[0]
            return check(abc, [a, a, None], tol) and check(angles, [90, 90, 90], angle_tol)
        if crys_system == "orthorhombic":
            return check(angles, [90, 90, 90], angle_tol)
        if crys_system == "monoclinic":
            return check(angles, [90, None, 90], angle_tol)
        return True

    @property
    def crystal_system(self):
        """
        :return: Crystal system for space group.
        """
        i = self.int_number
        if i <= 2:
            return "triclinic"
        if i <= 15:
            return "monoclinic"
        if i <= 74:
            return "orthorhombic"
        if i <= 142:
            return "tetragonal"
        if i <= 167:
            return "trigonal"
        if i <= 194:
            return "hexagonal"
        return "cubic"

    def is_subgroup(self, supergroup):
        """
        True if this space group is a subgroup of the supplied group.

        Args:
            group (Spacegroup): Supergroup to test.

        Returns:
            True if this space group is a subgroup of the supplied group.
        """
        if len(supergroup.symmetry_ops) < len(self.symmetry_ops):
            return False

        groups = [[supergroup.int_number]]
        all_groups = [supergroup.int_number]
        max_subgroups = {int(k): v for k, v in _get_symm_data("maximal_subgroups").items()}
        while True:
            new_sub_groups = set()
            for i in groups[-1]:
                new_sub_groups.update([j for j in max_subgroups[i] if j not in all_groups])
            if self.int_number in new_sub_groups:
                return True

            if len(new_sub_groups) == 0:
                break

            groups.append(new_sub_groups)
            all_groups.extend(new_sub_groups)
        return False

    def is_supergroup(self, subgroup):
        """
        True if this space group is a supergroup of the supplied group.

        Args:
            subgroup (Spacegroup): Subgroup to test.

        Returns:
            True if this space group is a supergroup of the supplied group.
        """
        return subgroup.is_subgroup(self)

    @classmethod
    def from_int_number(cls, int_number, hexagonal=True):
        """
        Obtains a SpaceGroup from its international number.

        Args:
            int_number (int): International number.
            hexagonal (bool): For rhombohedral groups, whether to return the
                hexagonal setting (default) or rhombohedral setting.

        Returns:
            (SpaceGroup)
        """
        sym = sg_symbol_from_int_number(int_number, hexagonal=hexagonal)
        if not hexagonal and int_number in [146, 148, 155, 160, 161, 166, 167]:
            sym += ":R"
        return SpaceGroup(sym)

    def __str__(self):
        return "Spacegroup %s with international number %d and order %d" % (
            self.symbol,
            self.int_number,
            len(self.symmetry_ops),
        )

    def to_pretty_string(self):
        """
        :return: Spacegroup string.
        """
        return self.symbol


def sg_symbol_from_int_number(int_number, hexagonal=True):
    """
    Obtains a SpaceGroup name from its international number.

    Args:
        int_number (int): International number.
        hexagonal (bool): For rhombohedral groups, whether to return the
            hexagonal setting (default) or rhombohedral setting.

    Returns:
        (str) Spacegroup symbol
    """
    syms = []
    for n, v in _get_symm_data("space_group_encoding").items():
        if v["int_number"] == int_number:
            syms.append(n)
    if len(syms) == 0:
        raise ValueError("Invalid international number!")
    if len(syms) == 2:
        for sym in syms:
            if "e" in sym:
                return sym
        if hexagonal:
            syms = list(filter(lambda s: s.endswith("H"), syms))
        else:
            syms = list(filter(lambda s: not s.endswith("H"), syms))
    return syms.pop()


def in_array_list(array_list, a, tol=1e-5):
    """
    Extremely efficient nd-array comparison using numpy's broadcasting. This
    function checks if a particular array a, is present in a list of arrays.
    It works for arrays of any size, e.g., even matrix searches.

    Args:
        array_list ([array]): A list of arrays to compare to.
        a (array): The test array for comparison.
        tol (float): The tolerance. Defaults to 1e-5. If 0, an exact match is
            done.

    Returns:
        (bool)
    """
    if len(array_list) == 0:
        return False
    axes = tuple(range(1, a.ndim + 1))
    if not tol:
        return np.any(np.all(np.equal(array_list, a[None, :]), axes))
    return np.any(np.sum(np.abs(array_list - a[None, :]), axes) < tol)
