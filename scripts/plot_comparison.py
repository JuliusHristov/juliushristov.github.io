#!/usr/bin/env python3
"""Plot manuscript primitive scalar ratios vs N=infinity lattice 0++ levels.

Source: Athenodorou & Teper, JHEP 12 (2021) 082,
DOI 10.1007/JHEP12(2021)082 (continuum N=infinity table).
Lattice values: 0++ ground 3.072(14), 0++ first excited 5.845(50),
in units of sqrt(sigma). Errors in the ratio are propagated assuming
independent measurements; covariance is not available here.
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

PAPER_C2 = np.array([0.4220071, 1.03534, 1.87403, 2.93503, 4.21715, 5.72001])
LATTICE = np.array([3.072, 5.845])
LATTICE_ERR = np.array([.014,.050])

parser=argparse.ArgumentParser()
parser.add_argument('--out', type=Path, default=Path('scalar_spectrum_comparison.png'))
args=parser.parse_args()
ratio=np.sqrt(PAPER_C2/PAPER_C2[0])
latratio=LATTICE/LATTICE[0]
laterr=np.array([0., latratio[1] * np.sqrt((LATTICE_ERR[1]/LATTICE[1])**2 +
                                                (LATTICE_ERR[0]/LATTICE[0])**2)])
fig,ax=plt.subplots(figsize=(9,5.2))
x=np.arange(len(ratio))
ax.plot(x,ratio,'o-',label='Primitive scalar tower (manuscript)')
ax.errorbar([0,1],latratio,yerr=laterr,fmt='s',capsize=5,
            label='Large-N lattice 0++ (Athenodorou–Teper 2021)')
ax.set_xticks(x,[f'n={n}' for n in x]);ax.set_ylabel('Mass / ground-state mass')
ax.set_xlabel('Radial excitation index (proposed correspondence)')
ax.set_title('Primitive scalar-singlet spectrum and lattice comparison')
ax.grid(alpha=.18);ax.legend();fig.tight_layout()
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out,dpi=180)
print('Saved',args.out)
print(f'Model first excitation: {ratio[1]:.4f}')
print(f'Lattice first excitation: {latratio[1]:.4f} ± {laterr[1]:.4f}')
