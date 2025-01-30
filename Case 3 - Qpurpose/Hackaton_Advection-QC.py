import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT, RYGate, RZGate, MCXGate, ModularAdderGate
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_aer import AerSimulator
from pyqsp import angle_sequence, response
from pyqsp.poly import polynomial_generators, PolyTaylorSeries
import scipy as sp
#from pennylane.templates.state_preparations.mottonen import compute_theta, gray_code

# The block encoding is built from three special gates: shift, a special 2 qubit gate prep and a modular adder

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


def Shift_gate(n):
    """
    n : number of qubits.

    Returns a gate implementing the shift operation
    |k> --> |k+1 (mod 2**n)> in the computational basis.
    """
    qc = QuantumCircuit(n, name="Shift")
    for j in range(n - 1):
        qc.mcx([k for k in range(n - 1 - j)], n - 1 - j)  # Using multi controlled NOT gates
    qc.x(0)
    return qc.to_gate()


def Advection_prep(n, dt, c):
    """
    n: number of spatial qubits. N = 2**n is the number of spatial grid points.
    dt: time step.
    nu: diffusion coefficient.

    Returns a 2 qubit gate state preparation gate implementing
    |0> --> sqrt(nu*dt/dx**2)|0> + sqrt(1-2*nu*dt/dx**2)|1> + sqrt(nu*dt/dx**2) |2>
    """
    d = 4  # The domain is fixed to be [0,4]
    dx = d / 2**n
    nu = 0.02
    a = 1-2*dt*nu/dx**2
    b = dt * c / dx
    if b > 0:
        suptheta = np.arccos(np.sqrt(b) - 1) + np.pi  # The rotation angle needed to prepare a using an RY gate
        diatheta = np.arccos(np.sqrt(a)) + np.pi   # The rotation angle needed to prepare a using an RY gate
        subtheta = np.arccos(b) + np.pi  # The rotation angle needed to prepare a using an RY gate
    else:
        print("The chosen values n,dt,c are not admissible. Arrange that 1>c*dt/dx")
        exit(1)

    dia = RYGate(0).control(2, ctrl_state="01")
    sup = RYGate(suptheta).control(2, ctrl_state="00")
    sub = RYGate(-subtheta).control(2, ctrl_state="10")

    # qc.ry(suptheta, last-1)
    # qc.cx(1, last-1)
    # qc.ry(subtheta, last-1)
    # qc.cx(2, last-1)
    # qc.ry(suptheta, last-1)
    # qc.cx(1, last-1)
    return [sup, dia, sub]


#######

# Advection block encoding following pennylane's implementation


def UA(circ: QuantumCircuit, thetas, wire_map, control_wires, ancilla):
    for theta, control_index in zip(thetas, control_wires):
        circ.ry(2 * theta, ancilla)
        circ.cx(wire_map[control_index], ancilla)


def UB(circ: QuantumCircuit, wires_i, wires_j):
    for i, j in zip(wires_i, wires_j):
        circ.swap(i, j)


def A_matrix(n: int) -> np.ndarray:
    N = 2**n
    A = np.zeros((N, N))
    A += np.diag(np.ones(N - 1), k=1)
    A -= np.diag(np.ones(N - 1), k=-1)
    A[0, -1] = -1
    A[-1, 0] = 1
    return A


def rotation_angles(A: np.ndarray):
    alphas = np.arccos(A).flatten()
    return compute_theta(alphas)


def advection_block_encoding(n, dt, c):
    A = A_matrix(n)
    thetas = rotation_angles(A)
    s = int(np.log2(A.shape[0]))

    qr1 = QuantumRegister(s, name="Q1")
    qr2 = QuantumRegister(s, name="Q2")
    ancilla = QuantumRegister(1, name="anc")

    cr1 = ClassicalRegister(s, name="c1")
    cr2 = ClassicalRegister(s, name="c2")
    c_ancilla = ClassicalRegister(1, name="anc_c")

    qc = QuantumCircuit(ancilla, qr1, qr2, c_ancilla, cr1, cr2, name="U")

    code = gray_code(2 * np.log2(len(A)))
    n_selections = len(code)
    control_wires = [int(np.log2(int(code[i], 2) ^ int(code[(i + 1) % n_selections], 2))) for i in range(n_selections)]
    qr1_map = [qr1[i] for i in range(s)]
    cr1_map = [cr1[i] for i in range(s)]
    qr2_map = [qr2[i] for i in range(s)]
    total_map = qr1_map + qr2_map

    wire_map = {control_index: wire for control_index, wire in enumerate(total_map)}

    qc.h(qr1_map)
    UA(qc, thetas, wire_map, control_wires, ancilla[0])
    UB(qc, qr1_map, qr2_map)
    qc.h(qr1_map)

    qc.measure([ancilla[0]] + qr1_map, [c_ancilla[0]] + cr1_map)

    return qc

def Block_encoding(n, dt, c):
    """
    n : number of spatial qubits
    dt : timestep
    c : advection coefficent.

    """
    # Setting up the circuit
    anc = QuantumRegister(2, name="anc")
    qr1 = QuantumRegister(n, name="Q1")
    qr2 = QuantumRegister(n, name="Q2")
    qc = QuantumCircuit(anc, qr1, qr2, name="U_diff")

    # Preparing the needed gates
    [sup, dia, sub] = Advection_prep(n, dt, c)
    S = Shift_gate(n)
    adder = ModularAdderGate(n)

    # Constructing the circuit
    # Hadamard the shit out of it
    for j in range(1, n):
        qc.h(qr1[j])

    qc.append(sup, qr2[1:3] + anc[0:1])
    qc.append(dia, qr2[1:3] + anc[0:1])
    qc.append(sub, qr2[1:3] + anc[0:1])
    qc.append(S.inverse(), qr1[:])
    qc.append(adder, qr2[:] + qr1[:])

    for j in range(n):
        qc.swap(qr1[j], qr2[j])

    qc.swap(anc[0], anc[1])

    qc.append(adder.inverse(), qr2[:] + qr1[:])
    qc.append(S, qr1[:])
    qc.append(sub.inverse(), qr2[1:3] + anc[0:1])
    qc.append(dia.inverse(), qr2[1:3] + anc[0:1])
    qc.append(sup.inverse(), qr2[1:3] + anc[0:1])

    for j in range(1, n):
        qc.h(qr1[j])

    return qc


# The following function extracts QSVT angle sequences from the file 'QSP_angles.txt'
# It contains the angle sequences for x^(5k) for 0<k<21


def extract_angle_seq(file="QSP_angles.txt"):
    """
    Returns a list of angle sequences
    angle_seq[k] is the angle sequence of x**(5(k+1))
    """

    angle_seqs = []

    with open(file, "r") as file:
        lines = file.readlines()[1:]

        for line in lines:
            seq = list(map(float, line.strip().split()))
            angle_seqs.append(np.array(seq))
    return angle_seqs


# The following function implements QSVT on the block encoding provided by Block_encoding
# The full circuit is then simulated with the aer-simulator using Gaussian initial conditions.
# A post-selection procedure picks out the successfull runs and arranges the results in a vector


def Advection_QSVT(deg: int, taylor_cutoff: int, n: int, dt: float, c: float, shots=10**4, show_gate_count=False):
    """
    deg: number of time steps
    taylor_cutoff: the degree of the Taylor series used in the QSVT algorithm for the cosinus
    n: number of spatial qubits
    dt: time step
    c: advection coefficient
    shots: number of shots used in the aer-simulator
    show_gate_count: True or False according to whether gate counts should be printed

    The function implements QSVT for the function e^{ix} on Block_encoding(n,dt,nu)
    yielding a quantum circuit. The circuit is then measured in the computational basis.

    The circuit is initalized with normalized Gaussian initial conditions over
    [0,d=4] with N = 2**n uniformly distributed grid points.

    The circuit is simulated using the aer-simulator with shots = shots.

    A post-selection procedure picks out the successfull runs and arranges the results in
    a vector z.

    Returns x,z the spatial grid values and the simulated y values in z.
    """
    # Setting up the circuit
    qra = QuantumRegister(2)  # Ancilla register on 1 qubit used in QSVT
    qra2 = QuantumRegister(2)  # Ancilla register on 1 qubit used in Block_encoding
    qr1 = QuantumRegister(n)  # qr1 and qr2 are the same as in Block_encoding
    qr2 = QuantumRegister(n)

    cra = ClassicalRegister(2)
    cra2 = ClassicalRegister(2)
    cr1 = ClassicalRegister(n)
    cr2 = ClassicalRegister(n)

    qc = QuantumCircuit(qra, qra2, qr1, qr2, cra, cra2, cr1, cr2)
    #qc = QuantumCircuit(qra, qr1, qr2, cra, cr1, cr2)

    # Preparing the initial conditions
    N = 2**n
    d = 4  # spatial domain [0,d]
    dx = 4 / N
    x = np.linspace(0, d, N, endpoint=False)
    y = np.exp(-20 * (x - d / 3) ** 2)  # Gaussian initial conditions
    y = y / np.linalg.norm(y)  # normalized to be a unit vector
    qc.prepare_state(Statevector(y), qr2)

    U = Block_encoding(n, dt, c)  # Block encoding circuit

    U.draw(output='latex', filename='block.pdf')

    # TODO Set maxscale such that the output is rescaled
    # TODO
    (Phi_cos,_) = QSVT_cosinus(degree_cutoff=taylor_cutoff, max_scale=0.2, M_step=deg)  # Extracting the angle sequence
    (Phi_sin,_) = QSVT_sinus(degree_cutoff=taylor_cutoff + 2, max_scale=0.2, M_step=deg)  # Extracting the angle sequence

    # Applying the QSVT circuit
    qc.h(qra[:])
    s = 0
    for k in range(len(Phi_cos) - 1, -1, -1):
        qc.mcx(qr1[:], qra[0], ctrl_state=n * "0")
        qc.crz(2 * Phi_cos[k], qra[1], qra[0], ctrl_state="0")
        qc.crz(2 * Phi_sin[k+1], qra[1], qra[0], ctrl_state="1")
        qc.mcx(qr1[:], qra[0], ctrl_state=n * "0")

        if s == 0:
            qc.append(U, qra2[:] + qr1[:] + qr2[:])
            #qc.append(U, qr1[:] + qr2[:])
            s = 1
        else:
            if k != 0:
                qc.append(U.inverse(), qra2[:] + qr1[:] + qr2[:])
                #qc.append(U.inverse(), qr1[:] + qr2[:])
                s = 0
            else:
                qc.append(U.control(1), qra[0:1] + qra2[:] + qr1[:] + qr2[:])
                #qc.append(U.control(1), qra[0:1] + qr1[:] + qr2[:])

    
    qc.mcx(qr1[:], qra[0], ctrl_state=n * "0")
    qc.crz(2 * Phi_sin[0], qra[0], qra[1], ctrl_state="1")
    qc.mcx(qr1[:], qra[0], ctrl_state=n * "0")

    qc.p(-np.pi / 2, qra[0])
    qc.h(qra[:])
    
    #qc.draw(output='latex', filename='circuit.pdf')

    # Measurements
    qc.measure(qra, cra)
    qc.measure(qra2, cra2)
    qc.measure(qr1, cr1)
    qc.measure(qr2, cr2)

    # Running the circuit
    sim = AerSimulator()
    qc_comp = transpile(qc, sim)
    res = sim.run(qc_comp, shots=shots).result()
    counts = res.get_counts(0)

    # Printing gate counts
    if show_gate_count:
        dict = qc_comp.count_ops()
        gate_1q = 0
        gate_2q = 0
        for key in dict:
            if key[0] == "c":
                gate_2q += dict[key]
            elif key != "measure":
                gate_1q += dict[key]

        print("1 qubit gates:", gate_1q)
        print("2 qubit gates:", gate_2q)
        print("Total:", gate_1q + gate_2q)

        print("Circuit depth after transpiling:", qc_comp.depth())

    # Postselection
    select = (n + 4) * "0"
    total = 0  # Tracks the number of successfull outcomes
    z = np.zeros(N)  # The results are encoded in z
    for key in counts:
        L = key.split()
        if L[1] + L[2] + L[3] == select:
            print(L)
            z[int(L[0], 2)] = np.sqrt(counts[key] / shots)  # By construction all amplitudes are positive real numbers
            total += counts[key]  # so this actually recovers them!
    success_rate = total / shots
    print("Success rate =", success_rate)
    return x, z


def Euler_advection(deg, n, dt, c):
    """
    deg: number of time steps
    n: N=2**n is the number of spatial grid points
    dt: time step
    nu: diffusion coefficient

    Returns x,y,w where x are the spatial grid points,
    y and z are the function values at x at time t=0 and t=deg*dt, respectively.
    """
    N = 2**n
    d = 4  # Domain [0,d]
    dx = d / N
    x = np.linspace(0, d, N, endpoint=False)

    b = (-1j * dt * c) / (2 * dx)
    B = b * np.diag(np.ones(N - 1), 1) - b * np.diag(np.ones(N - 1), -1)
    B[0][-1] = -1 * b
    B[-1][0] = 1 * b

    y = np.exp(-20 * (x - d / 3) ** 2)
    y = y / np.linalg.norm(y)

    e_iht = sp.linalg.expm(-1j * B * deg)

    w = np.matmul(e_iht, y)
    return x, y, w


def Compare_plots(deg=10, n=5, dt=0.1, c=0.02, shots=10**6):
    # Plots the initial distribution and the results of the classical and quantum simulations at t = deg*dt
    x, y, w = Euler_advection(deg, n, dt, c)
    x, z = Advection_QSVT(deg, 10, n, dt, c, shots=shots, show_gate_count=True)
    T = deg * dt
    plt.plot(x, y, x, w, x, z)
    plt.legend(["Classical T=0", "Classical T=" + str(T), "Quantum T=" + str(T)])
    #plt.legend(["Classical T=0", "Classical T=" + str(T)])
    plt.show()

def Visualize_matrix(n = 5, dt = 0.1, c = 0.02):
 
    U = Block_encoding(n,dt,c)                       # Block encoding circuit 
    #U.draw(output='latex', filename='circuit.pdf')
     
    u = np.zeros(2**n).astype(complex)
    for j in range(2**n):
        b = (n+2)*'0' + f"{j:0{n}b}"#[::-1]
        state = Statevector.from_label(b).evolve(U).reverse_qargs()
        for i in range(0,2**n):
            index = i*2**(n+2)
            u[i] = np.round(state[index], decimals=3)
            #print(index)
        print(u)
        stateM = np.round(np.real(state), decimals=3)
        #print(stateM)

#Compare_plots(deg = 10, n = 6, dt = 0.05, c = 0.6, shots = 10**6)
Visualize_matrix(n = 3, dt = 0.05, c = 0.8)
