import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT, RYGate, RZGate, MCXGate, ModularAdderGate
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

# The block encoding is built from three special gates: shift, a special 2 qubit gate prep and a modular adder

def Shift_gate(n):
    """
    n : number of qubits.
    
    Returns a gate implementing the shift operation  
    |k> --> |k+1 (mod 2**n)> in the computational basis.
    """
    qc = QuantumCircuit(n,name = 'Shift')
    for j in range(n-1):
        qc.mcx([k for k in range(n-1-j)],n-1-j)         # Using multi controlled NOT gates 
    qc.x(0)
    return qc.to_gate()
    
def Advection_prep(n,dt,c):
    """
    n: number of spatial qubits. N = 2**n is the number of spatial grid points. 
    dt: time step.      
    nu: diffusion coefficient. 
    
    Returns a 2 qubit gate state preparation gate implementing 
    |0> --> sqrt(nu*dt/dx**2)|0> + sqrt(1-2*nu*dt/dx**2)|1> + sqrt(nu*dt/dx**2) |2>
    """
    d = 4                               # The domain is fixed to be [0,4]
    dx = d/2**n                
    #a = 1-2*dt*nu/dx**2
    b = 1-dt*c/dx
    if b>0:
        suptheta = np.arccos(np.sqrt(b) - 1)   # The rotation angle needed to prepare a using an RY gate  
        diatheta = 0 #np.arccos(np.sqrt(a)) + np.pi   # The rotation angle needed to prepare a using an RY gate  
        subtheta = np.arccos(-b) + np.pi   # The rotation angle needed to prepare a using an RY gate  
    else:
        print('The chosen values n,dt,c are not admissible. Arrange that 1>c*dt/dx')
        exit(1)

    sup = RYGate(-suptheta).control(2, ctrl_state='00')
    dia = RYGate(diatheta).control(2, ctrl_state='01')
    sub = RYGate(subtheta).control(2, ctrl_state='10')
        
    #qc.ry(suptheta, last-1)
    #qc.cx(1, last-1)           
    #qc.ry(subtheta, last-1)
    #qc.cx(2, last-1)           
    #qc.ry(suptheta, last-1)
    #qc.cx(1, last-1)           
    return [sup, dia, sub]

def Block_encoding(n,dt,c):
    """
    n : number of spatial qubits
    dt : timestep
    c : advection coefficent.
    
    """ 
    # Setting up the circuit 
    anc = QuantumRegister(2, name='anc')
    qr1 = QuantumRegister(n, name='Q1')
    qr2 = QuantumRegister(n, name='Q2')
    qc = QuantumCircuit(anc,qr1,qr2, name = 'U_diff')
    
    # Preparing the needed gates 
    [sup, dia, sub] = Advection_prep(n,dt,c)
    S = Shift_gate(n)
    adder = ModularAdderGate(n)

    # Constructing the circuit 
    # Hadamard the shit out of it
    for j in range(1, n):
        qc.h(qr1[j])
        
    qc.append(sup,qr2[1:3]+anc[0:1])
    qc.append(dia,qr2[1:3]+anc[0:1])
    qc.append(sub,qr2[1:3]+anc[0:1])
    qc.append(S.inverse(),qr1[:])
    qc.append(adder,qr2[:]+qr1[:])
    
    for j in range(n):
        qc.swap(qr1[j],qr2[j])
        
    qc.swap(anc[0], anc[1])
    
    qc.append(adder.inverse(),qr2[:]+qr1[:])
    qc.append(S,qr1[:])
    qc.append(sub.inverse(),qr2[1:3]+anc[0:1])
    qc.append(dia.inverse(),qr2[1:3]+anc[0:1])
    qc.append(sup.inverse(),qr2[1:3]+anc[0:1])
    
    for j in range(1, n):
        qc.h(qr1[j])
        
    return qc


    
# The following function extracts QSVT angle sequences from the file 'QSP_angles.txt'
# It contains the angle sequences for x^(5k) for 0<k<21

def extract_angle_seq(file = 'QSP_angles.txt'):
    """
    Returns a list of angle sequences 
    angle_seq[k] is the angle sequence of x**(5(k+1))
    """
    
    angle_seqs = []
    
    with open(file, 'r') as file:
        lines = file.readlines()[1:]

        for line in lines:
            seq = list(map(float, line.strip().split()))
            angle_seqs.append(np.array(seq))   
    return angle_seqs

# The following function implements QSVT on the block encoding provided by Block_encoding
# The full circuit is then simulated with the aer-simulator using Gaussian initial conditions. 
# A post-selection procedure picks out the successfull runs and arranges the results in a vector

def Diffusion_QSVT(deg,n,dt,c,shots = 10**6,show_gate_count = False):
    """
    deg: number of time steps
    n: number of spatial qubits 
    dt: time step
    nu: diffusion coefficient
    shots: number of shots used in the aer-simulator
    show_gate_count: True or False according to whether gate counts should be printed
    
    The function implements QSVT for the function x**deg on Block_encoding(n,dt,nu)
    yielding a quantum circuit. The circuit is then measured in the computational basis. 
    
    The circuit is initalized with normalized Gaussian initial conditions over
    [0,d=4] with N = 2**n uniformly distributed grid points.  
    
    The circuit is simulated using the aer-simulator with shots = shots. 
    
    A post-selection procedure picks out the successfull runs and arranges the results in
    a vector z.

    Returns x,z the spatial grid values and the simulated y values in z. 
    """
    # Setting up the circuit 
    qra = QuantumRegister(1)      # Ancilla register on 1 qubit used in QSVT 
    qra0 = QuantumRegister(2)      # Ancilla register on 2 qubits same as in in Block_encoding 
    qr1 = QuantumRegister(n)      # qr1 and qr2 are the same as in Block_encoding 
    qr2 = QuantumRegister(n)
    
    cra = ClassicalRegister(1)
    cra0 = ClassicalRegister(2)
    cr1 = ClassicalRegister(n)
    cr2 = ClassicalRegister(n)
    
    qc = QuantumCircuit(qra,qra0,qr1,qr2,cra,cra0,cr1,cr2)
    
    # Preparing the initial conditions 
    N = 2**n 
    d = 4                                             # spatial domain [0,d]
    dx = 4/N 
    x = np.linspace(0,d,N,endpoint = False)
    y = np.exp(-20*(x-d/3)**2)                        # Gaussian initial conditions 
    y = y/np.linalg.norm(y)                           # normalized to be a unit vector
    qc.prepare_state(Statevector(y),qr2)
    
    U = Block_encoding(n,dt,c)                       # Block encoding circuit 
    
    #U.draw(output='latex', filename='circuit.pdf')
    
    Phi = extract_angle_seq()[int(deg/5)-1]           # Extracting the angle sequence  
    
    # Applying the QSVT circuit 
    qc.h(qra[0])
    s = 0
    for k in range(len(Phi)-1,-1,-1):
        if s == 0:
            qc.append(U,qra0[:]+qr1[:]+qr2[:])
            s = 1
        else:
            qc.append(U.inverse(),qra0[:]+qr1[:]+qr2[:])
            s = 0
        qc.mcx(qr1[:],qra[0],ctrl_state = n*'0')
        qc.rz(2*Phi[k],qra[0])
        qc.mcx(qr1[:],qra[0],ctrl_state = n*'0')
    qc.h(qra[0])
    
    # Measurements
    qc.measure(qra,cra)
    qc.measure(qra0,cra0)
    qc.measure(qr1,cr1)
    qc.measure(qr2,cr2)
    
    # Running the circuit     
    sim = AerSimulator()
    qc_comp = transpile(qc,sim)
    res = sim.run(qc_comp,shots = shots).result()
    counts = res.get_counts(0)
    
    # Printing gate counts 
    if show_gate_count:
        dict = qc_comp.count_ops()
        gate_1q = 0
        gate_2q = 0
        for key in dict:
            if key[0] == 'c':
                gate_2q += dict[key]
            elif key != 'measure':
                gate_1q += dict[key]
            
        print("1 qubit gates:", gate_1q)
        print("2 qubit gates:", gate_2q)
        print("Total:", gate_1q+gate_2q)

        print('Circuit depth after transpiling:', qc_comp.depth())
        
    # Postselection
    select = (n+1)*'0'
    total = 0                      # Tracks the number of successfull outcomes
    z = np.zeros(N)                # The results are encoded in z 
    for key in counts:
        L = key.split()
        if L[1]+L[2] == select:
            z[int(L[0],2)] = np.sqrt(counts[key]/shots)    # By construction all amplitudes are positive real numbers
            total += counts[key]                           # so this actually recovers them!
    success_rate = total/shots
    print('Success rate =', success_rate)
    return x,z

def Euler_cl(deg,n,dt,c):
    """
    deg: number of time steps
    n: N=2**n is the number of spatial grid points
    dt: time step 
    nu: diffusion coefficient 
    
    Returns x,y,w where x are the spatial grid points, 
    y and z are the function values at x at time t=0 and t=deg*dt, respectively.
    """    
    N = 2**n
    d = 4                     # Domain [0,d]
    dx = d/N 
    x = np.linspace(0,d,N,endpoint = False)
    
    b = dt*c/dx**2
    a = 1-dt*c/dx
    B = b*np.diag(np.ones(N-1),-1)+a*np.diag(np.ones(N),0)+b*np.diag(np.ones(N-1),1)
    B[0][-1] = b; B[-1][0] = b
    
    y = np.exp(-20*(x-d/3)**2)
    y = y/np.linalg.norm(y)
    
    C = np.linalg.matrix_power(B,deg)
    w = np.matmul(C,y)
    return x,y,w 

def Compare_plots(deg = 10,n = 5,dt = 0.1,c = 0.02,shots = 10**6):
    # Plots the initial distribution and the results of the classical and quantum simulations at t = deg*dt  
    x,y,w = Euler_cl(deg,n,dt,c)  
    x,z = Diffusion_QSVT(deg,n,dt,c,shots = shots, show_gate_count = True)
    T = deg*dt 
    plt.plot(x,y,x,w,x,z)
    plt.legend(['Classical T=0','Classical T='+str(T),'Quantum T='+str(T)])
    plt.show()

Compare_plots(deg = 10,n = 6,dt = 0.05,c = 0.02, shots = 10**6)
