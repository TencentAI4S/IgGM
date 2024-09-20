# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
from collections import OrderedDict

import numpy as np

# mapping between 1-char & 3-char residue names
RESD_MAP_1TO3 = OrderedDict([
    ('A', 'ALA'),
    ('R', 'ARG'),
    ('N', 'ASN'),
    ('D', 'ASP'),
    ('C', 'CYS'),
    ('Q', 'GLN'),
    ('E', 'GLU'),
    ('G', 'GLY'),
    ('H', 'HIS'),
    ('I', 'ILE'),
    ('L', 'LEU'),
    ('K', 'LYS'),
    ('M', 'MET'),
    ('F', 'PHE'),
    ('P', 'PRO'),
    ('S', 'SER'),
    ('T', 'THR'),
    ('W', 'TRP'),
    ('Y', 'TYR'),
    ('V', 'VAL')
])

RESD_MAP_3TO1 = {v: k for k, v in RESD_MAP_1TO3.items()}

# note that order not compact with alphafold
RESD_NAMES_1C = sorted(list(RESD_MAP_1TO3.keys()))
RESD_NAMES_3C = sorted(list(RESD_MAP_1TO3.values()))
RESD_NUM = len(RESD_NAMES_1C)  # := 20.
RESD_WITH_X = RESD_NAMES_1C + ['X']

RESD_ORDER_WITH_X = {restype: i for i, restype in enumerate(RESD_WITH_X)}
N_ATOMS_PER_RESD = 14  # TRP
N_ANGLS_PER_RESD = 7  # TRP (omega, phi, psi, chi1, chi2, chi3, and chi4)

# atom names for each residue type (excluding hydrogen atoms)
# residue_atoms in openfold
ATOM_NAMES_PER_RESD = {
    'ALA': ['C', 'CA', 'CB', 'N', 'O'],
    'ARG': ['C', 'CA', 'CB', 'CG', 'CD', 'CZ', 'N', 'NE', 'O', 'NH1', 'NH2'],
    'ASP': ['C', 'CA', 'CB', 'CG', 'N', 'O', 'OD1', 'OD2'],
    'ASN': ['C', 'CA', 'CB', 'CG', 'N', 'ND2', 'O', 'OD1'],
    'CYS': ['C', 'CA', 'CB', 'N', 'O', 'SG'],
    'GLU': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O', 'OE1', 'OE2'],
    'GLN': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'NE2', 'O', 'OE1'],
    'GLY': ['C', 'CA', 'N', 'O'],
    'HIS': ['C', 'CA', 'CB', 'CG', 'CD2', 'CE1', 'N', 'ND1', 'NE2', 'O'],
    'ILE': ['C', 'CA', 'CB', 'CG1', 'CG2', 'CD1', 'N', 'O'],
    'LEU': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'N', 'O'],
    'LYS': ['C', 'CA', 'CB', 'CG', 'CD', 'CE', 'N', 'NZ', 'O'],
    'MET': ['C', 'CA', 'CB', 'CG', 'CE', 'N', 'O', 'SD'],
    'PHE': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O'],
    'PRO': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O'],
    'SER': ['C', 'CA', 'CB', 'N', 'O', 'OG'],
    'THR': ['C', 'CA', 'CB', 'CG2', 'N', 'O', 'OG1'],
    'TRP': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'N', 'NE1', 'O'],
    'TYR': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O', 'OH'],
    'VAL': ['C', 'CA', 'CB', 'CG1', 'CG2', 'N', 'O']
}

# compact atom encoding with 14 columns
restype_name_to_atom14_names = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', ''],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', '', '', ''],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', '', '', '', '', '', ''],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', '', '', '', '', '', ''],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG', '', '', '', '', '', '', '', ''],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', '', '', '', '', ''],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', ''],
    'GLY': ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', ''],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', '', '', '', ''],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '', '', '', '', '', ''],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', ''],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', ''],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', '', '', '', '', '', ''],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', ''],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', ''],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG', '', '', '', '', '', '', '', ''],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '', '', '', '', '', '', ''],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', '', ''],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', ''],
    'UNK': ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
}

# atom type
ATOM_TYPES = [
    'N',
    'CA',
    'C',
    'CB',
    'O',
    'CG',
    'CG1',
    'CG2',
    'OG',
    'OG1',
    'SG',
    'CD',
    'CD1',
    'CD2',
    'ND1',
    'ND2',
    'OD1',
    'OD2',
    'SD',
    'CE',
    'CE1',
    'CE2',
    'CE3',
    'NE',
    'NE1',
    'NE2',
    'OE1',
    'OE2',
    'CH2',
    'NH1',
    'NH2',
    'OH',
    'CZ',
    'CZ2',
    'CZ3',
    'NZ',
    'OXT',
]
ATOM_ORDER = {atom_type: i for i, atom_type in enumerate(ATOM_TYPES)}
ATOM_TYPE_NUM = len(ATOM_TYPES)  # := 37.

# various formats of per-residue atom name list
ATOM_NAMES_PER_RESD_N3 = {x: ['N', 'CA', 'C'] for x in RESD_NAMES_3C}
ATOM_NAMES_PER_RESD_N4 = {x: ['N', 'CA', 'C', 'O'] for x in RESD_NAMES_3C}
ATOM_NAMES_PER_RESD_N14_TF = ATOM_NAMES_PER_RESD
ATOM_NAMES_PER_RESD_N14_AF = restype_name_to_atom14_names
ATOM_NAMES_PER_RESD_N37 = {x: ATOM_TYPES for x in RESD_NAMES_3C}

# atom coordinate information for each residue type (name, rigid-group, and idealized coordinates)
# rigid-groups are numbered as: 0 for backbone, 3 for psi-group, and 4-7 for chi1-chi4 groups
ATOM_INFOS_PER_RESD = {
    'ALA': [
        ['N', 0, (-0.525, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.529, -0.774, -1.205)],
        ['O', 3, (0.627, 1.062, 0.000)],
    ],
    'ARG': [
        ['N', 0, (-0.524, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.524, -0.778, -1.209)],
        ['O', 3, (0.626, 1.062, 0.000)],
        ['CG', 4, (0.616, 1.390, -0.000)],
        ['CD', 5, (0.564, 1.414, 0.000)],
        ['NE', 6, (0.539, 1.357, -0.000)],
        ['NH1', 7, (0.206, 2.301, 0.000)],
        ['NH2', 7, (2.078, 0.978, -0.000)],
        ['CZ', 7, (0.758, 1.093, -0.000)],
    ],
    'ASN': [
        ['N', 0, (-0.536, 1.357, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.531, -0.787, -1.200)],
        ['O', 3, (0.625, 1.062, 0.000)],
        ['CG', 4, (0.584, 1.399, 0.000)],
        ['ND2', 5, (0.593, -1.188, 0.001)],
        ['OD1', 5, (0.633, 1.059, 0.000)],
    ],
    'ASP': [
        ['N', 0, (-0.525, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, 0.000, -0.000)],
        ['CB', 0, (-0.526, -0.778, -1.208)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.593, 1.398, -0.000)],
        ['OD1', 5, (0.610, 1.091, 0.000)],
        ['OD2', 5, (0.592, -1.101, -0.003)],
    ],
    'CYS': [
        ['N', 0, (-0.522, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, 0.000, 0.000)],
        ['CB', 0, (-0.519, -0.773, -1.212)],
        ['O', 3, (0.625, 1.062, -0.000)],
        ['SG', 4, (0.728, 1.653, 0.000)],
    ],
    'GLN': [
        ['N', 0, (-0.526, 1.361, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, 0.000)],
        ['CB', 0, (-0.525, -0.779, -1.207)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.615, 1.393, 0.000)],
        ['CD', 5, (0.587, 1.399, -0.000)],
        ['NE2', 6, (0.593, -1.189, -0.001)],
        ['OE1', 6, (0.634, 1.060, 0.000)],
    ],
    'GLU': [
        ['N', 0, (-0.528, 1.361, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.526, -0.781, -1.207)],
        ['O', 3, (0.626, 1.062, 0.000)],
        ['CG', 4, (0.615, 1.392, 0.000)],
        ['CD', 5, (0.600, 1.397, 0.000)],
        ['OE1', 6, (0.607, 1.095, -0.000)],
        ['OE2', 6, (0.589, -1.104, -0.001)],
    ],
    'GLY': [
        ['N', 0, (-0.572, 1.337, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.517, -0.000, -0.000)],
        ['O', 3, (0.626, 1.062, -0.000)],
    ],
    'HIS': [
        ['N', 0, (-0.527, 1.360, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, 0.000, 0.000)],
        ['CB', 0, (-0.525, -0.778, -1.208)],
        ['O', 3, (0.625, 1.063, 0.000)],
        ['CG', 4, (0.600, 1.370, -0.000)],
        ['CD2', 5, (0.889, -1.021, 0.003)],
        ['ND1', 5, (0.744, 1.160, -0.000)],
        ['CE1', 5, (2.030, 0.851, 0.002)],
        ['NE2', 5, (2.145, -0.466, 0.004)],
    ],
    'ILE': [
        ['N', 0, (-0.493, 1.373, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, -0.000)],
        ['CB', 0, (-0.536, -0.793, -1.213)],
        ['O', 3, (0.627, 1.062, -0.000)],
        ['CG1', 4, (0.534, 1.437, -0.000)],
        ['CG2', 4, (0.540, -0.785, -1.199)],
        ['CD1', 5, (0.619, 1.391, 0.000)],
    ],
    'LEU': [
        ['N', 0, (-0.520, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.522, -0.773, -1.214)],
        ['O', 3, (0.625, 1.063, -0.000)],
        ['CG', 4, (0.678, 1.371, 0.000)],
        ['CD1', 5, (0.530, 1.430, -0.000)],
        ['CD2', 5, (0.535, -0.774, 1.200)],
    ],
    'LYS': [
        ['N', 0, (-0.526, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, 0.000)],
        ['CB', 0, (-0.524, -0.778, -1.208)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.619, 1.390, 0.000)],
        ['CD', 5, (0.559, 1.417, 0.000)],
        ['CE', 6, (0.560, 1.416, 0.000)],
        ['NZ', 7, (0.554, 1.387, 0.000)],
    ],
    'MET': [
        ['N', 0, (-0.521, 1.364, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, 0.000, 0.000)],
        ['CB', 0, (-0.523, -0.776, -1.210)],
        ['O', 3, (0.625, 1.062, -0.000)],
        ['CG', 4, (0.613, 1.391, -0.000)],
        ['SD', 5, (0.703, 1.695, 0.000)],
        ['CE', 6, (0.320, 1.786, -0.000)],
    ],
    'PHE': [
        ['N', 0, (-0.518, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, 0.000, -0.000)],
        ['CB', 0, (-0.525, -0.776, -1.212)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.607, 1.377, 0.000)],
        ['CD1', 5, (0.709, 1.195, -0.000)],
        ['CD2', 5, (0.706, -1.196, 0.000)],
        ['CE1', 5, (2.102, 1.198, -0.000)],
        ['CE2', 5, (2.098, -1.201, -0.000)],
        ['CZ', 5, (2.794, -0.003, -0.001)],
    ],
    'PRO': [
        ['N', 0, (-0.566, 1.351, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, 0.000)],
        ['CB', 0, (-0.546, -0.611, -1.293)],
        ['O', 3, (0.621, 1.066, 0.000)],
        ['CG', 4, (0.382, 1.445, 0.0)],
        ['CD', 5, (0.477, 1.424, 0.0)],
    ],
    'SER': [
        ['N', 0, (-0.529, 1.360, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.518, -0.777, -1.211)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['OG', 4, (0.503, 1.325, 0.000)],
    ],
    'THR': [
        ['N', 0, (-0.517, 1.364, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, -0.000)],
        ['CB', 0, (-0.516, -0.793, -1.215)],
        ['O', 3, (0.626, 1.062, 0.000)],
        ['CG2', 4, (0.550, -0.718, -1.228)],
        ['OG1', 4, (0.472, 1.353, 0.000)],
    ],
    'TRP': [
        ['N', 0, (-0.521, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, 0.000)],
        ['CB', 0, (-0.523, -0.776, -1.212)],
        ['O', 3, (0.627, 1.062, 0.000)],
        ['CG', 4, (0.609, 1.370, -0.000)],
        ['CD1', 5, (0.824, 1.091, 0.000)],
        ['CD2', 5, (0.854, -1.148, -0.005)],
        ['CE2', 5, (2.186, -0.678, -0.007)],
        ['CE3', 5, (0.622, -2.530, -0.007)],
        ['NE1', 5, (2.140, 0.690, -0.004)],
        ['CH2', 5, (3.028, -2.890, -0.013)],
        ['CZ2', 5, (3.283, -1.543, -0.011)],
        ['CZ3', 5, (1.715, -3.389, -0.011)],
    ],
    'TYR': [
        ['N', 0, (-0.522, 1.362, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, -0.000, -0.000)],
        ['CB', 0, (-0.522, -0.776, -1.213)],
        ['O', 3, (0.627, 1.062, -0.000)],
        ['CG', 4, (0.607, 1.382, -0.000)],
        ['CD1', 5, (0.716, 1.195, -0.000)],
        ['CD2', 5, (0.713, -1.194, -0.001)],
        ['CE1', 5, (2.107, 1.200, -0.002)],
        ['CE2', 5, (2.104, -1.201, -0.003)],
        ['OH', 5, (4.168, -0.002, -0.005)],
        ['CZ', 5, (2.791, -0.001, -0.003)],
    ],
    'VAL': [
        ['N', 0, (-0.494, 1.373, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, -0.000)],
        ['CB', 0, (-0.533, -0.795, -1.213)],
        ['O', 3, (0.627, 1.062, -0.000)],
        ['CG1', 4, (0.540, 1.429, -0.000)],
        ['CG2', 4, (0.533, -0.776, 1.203)],
    ],
}

# torsion angle information for each residue type (name, symmetric or not, and atom names)
ANGL_INFOS_PER_RESD = {
    'ALA': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
    ],
    'ARG': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'CD']],
        ['chi3', False, ['CB', 'CG', 'CD', 'NE']],
        ['chi4', False, ['CG', 'CD', 'NE', 'CZ']],
    ],
    'ASN': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'OD1']],
    ],
    'ASP': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', True, ['CA', 'CB', 'CG', 'OD1']],
    ],
    'CYS': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'SG']],
    ],
    'GLN': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'CD']],
        ['chi3', False, ['CB', 'CG', 'CD', 'OE1']],
    ],
    'GLU': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'CD']],
        ['chi3', True, ['CB', 'CG', 'CD', 'OE1']],
    ],
    'GLY': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
    ],
    'HIS': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'ND1']],
    ],
    'ILE': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG1']],
        ['chi2', False, ['CA', 'CB', 'CG1', 'CD1']],
    ],
    'LEU': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'CD1']],
    ],
    'LYS': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'CD']],
        ['chi3', False, ['CB', 'CG', 'CD', 'CE']],
        ['chi4', False, ['CG', 'CD', 'CE', 'NZ']],
    ],
    'MET': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'SD']],
        ['chi3', False, ['CB', 'CG', 'SD', 'CE']],
    ],
    'PHE': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', True, ['CA', 'CB', 'CG', 'CD1']],
    ],
    'PRO': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'CD']],
    ],
    'SER': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'OG']],
    ],
    'THR': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'OG1']],
    ],
    'TRP': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'CD1']],
    ],
    'TYR': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', True, ['CA', 'CB', 'CG', 'CD1']],
    ],
    'VAL': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG1']],
    ],
}

# make atom14 mask
restype_atom14_to_atom37 = []  # mapping (restype, atom14) --> atom37
restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
restype_atom14_mask = []

for rt in RESD_NAMES_1C:
    atom_names = restype_name_to_atom14_names[RESD_MAP_1TO3[rt]]
    restype_atom14_to_atom37.append([(ATOM_ORDER[name] if name else 0) for name in atom_names])
    atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
    restype_atom37_to_atom14.append(
        [(atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0) for name in ATOM_TYPES]
    )
    restype_atom14_mask.append([(1.0 if name else 0.0) for name in atom_names])

# Add dummy mapping for restype 'UNK'
restype_atom14_to_atom37.append([0] * 14)
restype_atom37_to_atom14.append([0] * 37)
restype_atom14_mask.append([0.0] * 14)

restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int32)
restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, dtype=np.int32)
restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)

# create the corresponding mask
restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
for restype, restype_letter in enumerate(RESD_NAMES_1C):
    restype_name = RESD_MAP_1TO3[restype_letter]
    atom_names = ATOM_NAMES_PER_RESD[restype_name]
    for atom_name in atom_names:
        atom_type = ATOM_ORDER[atom_name]
        restype_atom37_mask[restype, atom_type] = 1
