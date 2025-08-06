# Carry out Hirshfeld partitioning of the charge density
# https://link.springer.com/content/pdf/10.1007/BF00549096.pdf

import sys
import logging
import numpy as np
from pyscf import scf, dft, mcscf
from .sph_dft_atom_ks import get_atm_nrks, free_atom_info

class HirshfeldAnalysis:
    """
    This class computes Hirshfeld-partitioned integrals
    over the molecular electronic density.
    """

    def __init__(self, mf, xc):
        # assert(isinstance(mf, scf.hf.SCF) or
        #        isinstance(mf, mcscf.casci.CASCI))
        self.result = {}
        self.mf = mf
        self.mol = mf.mol
        self.mol.verbose = 1
        self.xc = xc
        self.make_grids()

    def make_grids(self):
        grids = getattr(self.mf, "grids", None)
        if grids is None:
            grids = dft.Grids(self.mol)
            grids.atom_grid = (35, 74)
            grids.build()
        self.grids = grids
        # (N_atom x N_gridpoints) array of
        # r - r_atom, where r is a grid point and r_atom is an
        # atom center.
        self.coords_atoms = self.grids.coords[None, :, :] - self.mol.atom_coords()[:, None, :] # (natm, grid_sz, 3)
        # distance from each atom to each grid point
        self.rad_atoms = np.linalg.norm(self.coords_atoms, axis=-1) # (natm, grid_sz)

    def perform_free_atom(self):

        result = self.result
        mf = self.mf

        result["mf_elem"] = {}
        result["V_free_elem"] = {}
        result["spl_free_elem"] = {}
        mf_elems = get_atm_nrks(mf, xc=self.xc, basis=self.mol.basis)
        for elem in mf_elems:
            mf_elem = mf_elems[elem]
            result["mf_elem"][elem] = mf_elem
            result["V_free_elem"][elem], result["spl_free_elem"][elem] = free_atom_info(mf_elem)

        result["V_free"] = np.zeros(self.mol.natm)
        for atom in range(self.mol.natm):
            elem = self.mol.atom_symbol(atom)
            result["V_free"][atom] = result["V_free_elem"][elem]
        
        # rho_free = atoms x integration points array of 
        #            "proatom" densities
        rho_free = np.empty((self.mol.natm, len(self.grids.coords)))
        for atom in range(self.mol.natm):
            elem = self.mol.atom_symbol(atom)
            rho_free[atom] = result["spl_free_elem"][elem](self.rad_atoms[atom])
        # weights_free = atoms x integration points array of 
        #   percentages -- for partitioning the total el- density.
        #
        # weights_free[i,p] = rho_free[i,p] / sum_j rho_free[j, p]
        tot_free = rho_free.sum(axis=0) # (grid_sz,)
        weights_free = rho_free / (tot_free + (tot_free < 1e-15)) # (natm, grid_sz)
        result['rho_free'] = rho_free
        result['tot_free'] = tot_free
        result['weights_free'] = weights_free
        return self
    

    def perform_hirshfeld(self, fn=None, orb_idx=None):
        """ Compute self.result object.

            If fn is not None, result['custom']
            will also be filled with an array of integrals, 1 for each atom.

            fn should take r : array (natm, pts, R^3) -> array (natm, pts, S)
            Here, r are coordinates in units of Bohr.

            The return shape from `integrate` will be (atoms,) + S

            The integral done for every atom, a (at r_a), is,

               int fn(r - r_a) chg_a dr^3

            where chg_a = -1*[rho - rho_ref]*(weight function for atom a)
        """
        result = self.result
        ni  = dft.numint.NumInt()
        # set logging to stdout

        if orb_idx is None:
            dm = self.mf.make_rdm1()
            if isinstance(dm, tuple) or len(dm.shape) == 3:
                dm = dm[0]+dm[1]
            rho = ni.get_rho(mol, dm, self.grids) # (grid_sz,)
            Ntot = np.vdot(rho, self.grids.weights)
            rho *= mol.nelectron / Ntot
        else:
            orb = self.mf.mo_coeff[:, orb_idx]
            dm = np.einsum('i,j->ij', orb, orb)
            rho = ni.get_rho(self.mol, dm, self.grids) # (grid_sz,)
            Ntot = np.vdot(rho, self.grids.weights)
            rho *= 1 / Ntot
        
        # print(f"integrated electrons: {Ntot}")
        # print(f"nelectron: {mol.nelectron}")
        # print(f"charge: {mol.charge}")
        # print(f"nuc charge: {mol.atom_charges().sum()}")

        # electron density partitioned onto every atom
        # rho_eff.sum(axis=0) == rho
        # multiply by grids.weights to integrate numerically
        rho_eff = rho * result['weights_free'] # (natm, grid_sz)

        # integral of r^3 - proxy for atomic volume
        # V_eff = (rho_eff * self.rad_atoms ** 3 * self.grids.weights).sum(axis=-1) # (natm,)
        # number of electrons on each atom
        elec_eff = (rho_eff * self.grids.weights).sum(axis=-1)
        elec_eff = np.round(elec_eff, 7)
        # net charge on each atom
        chrg_eff = - elec_eff + self.mol.atom_charges()
        # dipole_eff = - (self.coords_atoms * rho_eff[:, :, None] * self.grids.weights[:, None]).sum(axis=-2)

        # result["V_eff"] = V_eff
        result["elec_eff"] = elec_eff
        result["charge_eff"] = chrg_eff
        # result["dipole_eff"] = dipole_eff
        if fn is None:
            result["custom"] = None
        else:
            # normalize total pro-atom charge
            Ntot_free = np.vdot(self.tot_free, self.grids.weights)
            self.tot_free *= mol.nelectron / Ntot_free

            F = fn(self.coords_atoms)
            assert F.shape[:2] == self.coords_atoms.shape[:2], \
                        "Invalid return shape from fn."
            ans = np.einsum('ai...,ai,i->a...', F,
                    self.tot_free*self.weights_free - rho_eff, self.grids.weights)
            result["custom"] = ans
        return self

    def run(self, fn=None):
        self.perform_free_atom().perform_hirshfeld(fn)
        return self
    
    def run_by_orb(self, fn=None, orb_indices=None):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        self.perform_free_atom()
        result = []
        if orb_indices is None:
            orb_indices = np.arange(len(self.mol.ao_labels()))
        for orb_idx in orb_indices:
#            logging.info(f"Running Hirshfeld for orbital {orb_idx}")
            self.perform_hirshfeld(fn, orb_idx=orb_idx)
            result.append(self.result['elec_eff'].copy())
        result = np.array(result)
        return result

if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = """
    O 0 0 0; H 0 0 1; H 0 1 0;
    O 0 0 2; H 0 0 3; H 0 1 2
    """
    mol.basis = "cc-pVDZ"
    mol.build()
    mf = dft.RKS(mol, xc="PBE").run()

    anal = HirshfeldAnalysis(mf).run()
    print(anal.result["V_free"])
    print(anal.result["V_eff"])
