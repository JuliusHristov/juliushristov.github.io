#!/usr/bin/env python3
"""Numerical Galerkin check of the primitive SO(4)-singlet scalar spectrum.

Equations: Hristov manuscript, Chapter 2 (weight) and Chapter 3,
particularly Eq. (3.5). Independent cubic B-spline / Gauss quadrature
implementation. Does NOT establish the Yang-Mills/collective identification.

Install: pip install numpy scipy matplotlib
Run: python scripts/solve_primitive.py --tau-max 27 --intervals 220
Optional: --csv results/primitive_spectrum.csv

The lowest finite-interval eigenvalue approximates the fixed A-period zero
mode and must not be interpreted as a physical glueball. Its numerical value
is not exactly zero because the finite-interval trial space has clamped UV
boundary conditions. The next eigenvalue is the first physical candidate.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import BSpline
from scipy.linalg import eigh
from scipy.special import roots_legendre

REPORTED_C2 = np.array([0.4220071, 1.03534, 1.87403, 2.93503, 4.21715, 5.72001])


def radial_weight(t_max=38., n_grid=152001):
    """Solve -[p h']'=S with tip regularity and h(infinity)=0 approximately.

    Uses a finite asymptotic cutoff; increase t_max for a tail check.
    The check h(0) should be near 0.13407856 in manuscript normalization.
    """
    t = np.linspace(0.0, t_max, n_grid)
    u0 = (2.0 / 3.0) ** 1.5
    u = u0 / np.cosh(t)
    source = u * np.log(u) - u - u0 * np.log(u0) + u0
    p = np.maximum(np.sinh(2 * t) - 2 * t, 0.0) ** (2.0 / 3.0)
    cumulative_source = cumulative_trapezoid(source, t, initial=0.0)
    flux = np.divide(cumulative_source, p, out=np.zeros_like(p), where=p > 0)
    # Integrate from large radius back to the cap, maintaining positive h.
    h = -cumulative_trapezoid(flux[::-1], t[::-1], initial=0.0)[::-1]
    return t, h


def regular_clamped_splines(t, tau_max, intervals):
    """C^2 cubic splines: v'(0)=0 and v(T)=v'(T)=0."""
    degree = 3
    breakpoints = np.linspace(0., tau_max, intervals + 1)
    knots = np.r_[np.repeat(0., degree + 1), breakpoints[1:-1],
                  np.repeat(tau_max, degree + 1)]
    n_basis = len(knots) - degree - 1
    spline = BSpline(knots, np.eye(n_basis), degree, axis=0, extrapolate=False)
    # First two coefficients equal => derivative vanishes at the tip.
    # Last two coefficients zero => v(T) = v'(T) = 0.
    project = np.zeros((n_basis, n_basis - 3))
    project[0, 0] = 1.
    project[1, 0] = 1.
    for k in range(2, n_basis - 2):
        project[k, k - 1] = 1.
    return spline(t) @ project, spline.derivative(1)(t) @ project, \
        spline.derivative(2)(t) @ project


def compute(tau_max=27., intervals=220, quad_order=8, count=6, tail_max=38.):
    if tau_max >= tail_max - 2:
        raise ValueError('Choose tail_max at least 2 greater than tau_max.')
    if intervals < 10 or quad_order < 3:
        raise ValueError('Increase intervals and quadrature order.')
    full_t, full_h = radial_weight(t_max=tail_max)
    h0 = full_h[0]
    breaks = np.linspace(0., tau_max, intervals + 1)
    gauss_x, gauss_w = roots_legendre(quad_order)
    t = np.concatenate([(breaks[i+1] - breaks[i]) * gauss_x/2 +
                        (breaks[i+1]+breaks[i])/2 for i in range(intervals)])
    dt = np.concatenate([np.full(quad_order, (breaks[i+1]-breaks[i])/2)*gauss_w
                         for i in range(intervals)])
    hhat = np.interp(t, full_t, full_h) / h0
    p = np.maximum(np.sinh(2 * t) - 2 * t, 0.) ** (2.0 / 3.0)
    sh = np.sinh(t)
    mass_weight = sh**2 * hhat
    v, vp, vpp = regular_clamped_splines(t, tau_max, intervals)
    q = np.cosh(t)[:, None]*v + 0.5*sh[:, None]*vp
    d_primitive = vpp + 4.0/np.tanh(t)[:, None]*vp + 4*v
    # Manuscript Eqs. (3.4)-(3.6): Q = 1/4 int p sinh^2 |D v|^2,
    # N = int W (|v|^2+2|q|^2).  Nothing else is fitted.
    norm_matrix = (v.T*(dt*mass_weight))@v + 2*(q.T*(dt*mass_weight))@q
    energy_matrix = 0.25*(d_primitive.T*(dt*p*sh**2))@d_primitive
    eigenvalues = eigh(energy_matrix, norm_matrix,
                      subset_by_index=(0, min(len(norm_matrix)-1, count+1)),
                      check_finite=True)[0]
    return h0, eigenvalues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tau-max', type=float, default=27.)
    parser.add_argument('--intervals', type=int, default=220)
    parser.add_argument('--quad-order', type=int, default=8)
    parser.add_argument('--tail-max', type=float, default=38.)
    parser.add_argument('--count', type=int, default=6)
    parser.add_argument('--csv', type=Path)
    args = parser.parse_args()
    h0, eigen = compute(args.tau_max, args.intervals, args.quad_order,
                        args.count, args.tail_max)
    physical = eigen[1:args.count+1]
    print(f'h(0)={h0:.9f} (manuscript approx 0.13407856)')
    print(f'finite-T near-zero A-period candidate: c²={eigen[0]:.9g} (excluded)')
    print('mode   calculated c²    calculated M/M0   manuscript c²')
    for k, c2 in enumerate(physical):
        reference = f'{REPORTED_C2[k]:.7f}' if k<len(REPORTED_C2) else '--'
        print(f'{k:>3}    {c2:>14.7f}      {np.sqrt(c2/physical[0]):>9.5f}      {reference}')
    print('Convergence check: repeat with more intervals and a larger tau-max.')
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['mode','calculated_c_squared','calculated_mass_ratio',
                             'manuscript_c_squared','tau_max','intervals'])
            for k, c2 in enumerate(physical):
                writer.writerow([k, f'{c2:.10g}', f'{np.sqrt(c2/physical[0]):.9g}',
                                 REPORTED_C2[k] if k<len(REPORTED_C2) else '',
                                 args.tau_max, args.intervals])
        print('Wrote',args.csv)

if __name__ == '__main__':
    main()
