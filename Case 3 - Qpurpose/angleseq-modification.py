import numpy as np
import pyqsp
from pyqsp import angle_sequence, response
from pyqsp.poly import polynomial_generators, PolyTaylorSeries

"""
The python package pyqsp allows you to generate new angle sequences: https://github.com/ichuang/pyqsp

The sample code under "A guide within a guide" is the most efficient way to get started.
However, for use in the QSVT algorithm, 'phiset' as output by the sample code: 

(phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
    poly,
    method='sym_qsp',
    chebyshev_basis=True)

must be slightly modified. The function QSVT_format below implements the required
modification. 
"""


def QSVT_format(phiset):
    n = len(phiset) - 1
    Phi = np.zeros(n)
    Phi[1:n] = phiset[1:n] - np.pi / 2
    Phi[0] = phiset[0] + phiset[-1] + ((n - 2) % 4) * np.pi / 2
    # If you use output from the MATLAB package qsppack you have to replace (n-2) by (n-1) in the above formula!
    return Phi


def QSVT_cosinus(degree_cutoff: int, max_scale: float, M_step: int):
    func = lambda x: max_scale * np.cos(M_step * x)

    poly = PolyTaylorSeries().taylor_series(
        func=func, degree=degree_cutoff, max_scale=max_scale, chebyshev_basis=True, cheb_samples=2 * degree_cutoff
    )

    angles, reduced_angles, parity = angle_sequence.QuantumSignalProcessingPhases(
        poly, method="sym_qsp", chebyshev_basis=True
    )

    return QSVT_format(angles), poly


def QSVT_sinus(degree_cutoff: int, max_scale: float, M_step: int):
    func = lambda x: max_scale * np.sin(M_step * x)

    poly = PolyTaylorSeries().taylor_series(
        func=func, degree=degree_cutoff, max_scale=max_scale, chebyshev_basis=True, cheb_samples=2 * degree_cutoff
    )

    angles, reduced_angles, parity = angle_sequence.QuantumSignalProcessingPhases(
        poly, method="sym_qsp", chebyshev_basis=True
    )

    return QSVT_format(angles), poly


M_step = 10
max_scale = 0.7
true_func = lambda x: max_scale * np.sin(M_step * x)

ang, poly = QSVT_sinus(20, max_scale, M_step)


response.PlotQSPResponse(ang, pcoefs=poly, target=true_func, sym_qsp=True, simul_error_plot=True)
