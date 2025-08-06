# Hirshfeld Analysis module for PySCF

Original script from [https://github.com/frobnitzem/hirshfeld]

## Install

```
git clone https://github.com/yhkim55/hirshfeld-orb.git

# Set pyscf extended module path:
echo 'export PYSCF_EXT_PATH=/home/abc/local/path:$PYSCF_EXT_PATH' >> ~/.bashrc
```
## Usage
```
from pyscf import gto, scf
mol = gto.Mole()
mol.atom = """
O 0 0 0; H 0 0 1; H 0 1 0;
O 0 0 2; H 0 0 3; H 0 1 2
"""
mol.basis = "cc-pVDZ"
mol.build()
mf = scf.RHF(mol).run()

# DFT xc functional is used to calculate free-atom density, PBE is used in this example

# Hirshfeld population analysis for orbitals 0,1,and 2
hirsh_popul = HirshfeldAnalysis(mf, "PBE").run_by_orb(orb_indices=[0,1,2])

# Hirshfeld population analysis for all orbitals
hirsh_popul = HirshfeldAnalysis(mf, "PBE").run_by_orb()
