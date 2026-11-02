#!/usr/bin/env python3
"""Lefschetz SO(4)-singlet scalar Sturm–Liouville solver (manuscript Ch. 4).

Independent cubic B-spline discretization of Eqs. (4.2)-(4.6).
The SO(4)-singlet/Lefschetz label does not itself establish 0-+ quantum
numbers or correspondence to lattice pseudoscalar glueballs.
"""
import argparse, csv
from pathlib import Path
import numpy as np
from scipy.special import roots_legendre
from scipy.interpolate import BSpline
from scipy.linalg import eigh
from solve_primitive import radial_weight

REPORTED_C2=[0.9981,1.7792,2.7771,3.9935,5.4302]

def solve(tau_max=16.,intervals=150,quad_order=8,count=5):
    gg,ww=roots_legendre(quad_order)
    breaks=np.linspace(0,tau_max,intervals+1)
    t=np.concatenate([.5*(breaks[k+1]-breaks[k])*gg+.5*(breaks[k+1]+breaks[k])
                      for k in range(intervals)])
    weights=np.concatenate([.5*(breaks[k+1]-breaks[k])*ww for k in range(intervals)])
    tw,hw=radial_weight()
    hhat=np.interp(t,tw,hw)/hw[0]
    sh=np.sinh(t)
    p=(np.sinh(2*t)-2*t)**(2/3)
    W=sh**2*hhat
    r=8*sh**2/(3*(np.sinh(2*t)-2*t)) # p'/p
    VL=.5*r*r
    degree=3
    knots=np.r_[np.repeat(0,4),breaks[1:-1],np.repeat(tau_max,4)]
    nb=len(knots)-degree-1
    spl=BSpline(knots,np.eye(nb),degree,axis=0)
    # f'(0)=0; f(tau_max)=0. Do not impose a spurious right-end derivative.
    proj=np.zeros((nb,nb-2));proj[0,0]=proj[1,0]=1.
    for j in range(2,nb-1):proj[j,j-1]=1.
    f=spl(t)@proj
    fp=spl.derivative(1)(t)@proj
    D=fp+f/np.tanh(t)[:,None]
    energy=.5*((D.T*(weights*p))@D+(f.T*(weights*p*VL))@f)
    norm=(f.T*(weights*W))@f
    vals=eigh(energy,norm,subset_by_index=(0,count-1))[0]
    return vals

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tau-max',type=float,default=16.)
    ap.add_argument('--intervals',type=int,default=150)
    ap.add_argument('--count',type=int,default=5)
    ap.add_argument('--csv',type=Path)
    a=ap.parse_args()
    values=solve(a.tau_max,a.intervals,count=a.count)
    print('Lefschetz c² eigenvalues:')
    for i,v in enumerate(values):
        ref=REPORTED_C2[i] if i<len(REPORTED_C2) else float('nan')
        print(f' n={i}: {v:.8f}  (manuscript {ref:.5f})')
    if a.csv:
        a.csv.parent.mkdir(parents=True,exist_ok=True)
        with a.csv.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['mode','calculated_c_squared','manuscript_c_squared'])
            for i,v in enumerate(values):writer.writerow([i,f'{v:.10g}',REPORTED_C2[i] if i<len(REPORTED_C2) else ''])
        print('Wrote',a.csv)
