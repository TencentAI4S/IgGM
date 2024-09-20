"""PDB file fixer."""

import os
import shutil
import logging
import subprocess

from openmm.app import PDBFile
from pdbfixer import PDBFixer

from IgGM.utils import get_rand_str


class PdbFixer():
    """PDB file fixer."""

    def __init__(self):
        """Constructor function."""

        # setup configurations
        self.n_repts = 2  # maximal number of repeated runs
        self.out_dpath = os.getenv('DEBUG_OUT_DIR')  # for exporting problematic PDB files


    def add_atoms_cli(self, pdb_fpath_src, pdb_fpath_dst):
        """Add missing atoms via the CLI interface.

        Args:
        * pdb_fpath_src: path to the input PDB file
        * pdb_fpath_dst: path to the output PDB file
        * add_sc_atoms: (optional) whether to add side-chain atoms

        Returns: n/a
        """

        # add missing atoms
        os.makedirs(os.path.dirname(os.path.realpath(pdb_fpath_dst)), exist_ok=True)
        cmd_str = f'pdbfixer {pdb_fpath_src} --output {pdb_fpath_dst} --add-atoms heavy'
        for _ in range(self.n_repts):
            subprocess.call(cmd_str, shell=True)
            if os.path.exists(pdb_fpath_dst):
                break

        # save the problematic PDB file for future investigation - temporary solution
        if not os.path.exists(pdb_fpath_dst):
            rand_str = get_rand_str()
            pdb_fpath_dbg = os.path.join(self.out_dpath, f'{rand_str}.pdb')
            os.makedirs(self.out_dpath, exist_ok=True)
            shutil.copyfile(pdb_fpath_src, pdb_fpath_dbg)
            logging.warning('failed to add side-chain atoms for %s', pdb_fpath_src)
            logging.warning('problematic PDB file saved to %s', pdb_fpath_dbg)
            os.makedirs(os.path.dirname(os.path.realpath(pdb_fpath_dst)), exist_ok=True)
            shutil.copyfile(pdb_fpath_src, pdb_fpath_dst)


    def add_atoms_api(self, pdb_fpath_src, pdb_fpath_dst, bb_only=False):
        """Add missing atoms via the Python API interface.

        Args:
        * pdb_fpath_src: path to the input PDB file
        * pdb_fpath_dst: path to the output PDB file
        * bb_only: (optional) whether to only add missing backbone atoms

        Returns: n/a
        """

        # load the structure
        fixer = PDBFixer(filename=pdb_fpath_src)

        # determine which atoms should be added
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        if bb_only:
            for key, atoms in fixer.missingAtoms.items():
                fixer.missingAtoms[key] = [x for x in atoms if x.name == 'O']
            fixer.missingTerminals = {}

        # add missing atoms
        fixer.addMissingAtoms()

        # save the resulting structure into a PDB file
        os.makedirs(os.path.dirname(os.path.realpath(pdb_fpath_dst)), exist_ok=True)
        with open(pdb_fpath_dst, 'w', encoding='UTF-8') as o_file:
            PDBFile.writeFile(fixer.topology, fixer.positions, o_file)
