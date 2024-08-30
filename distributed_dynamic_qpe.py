from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, transpile, execute
from qiskit.providers.aer import noise
from qiskit.providers.aer.noise import pauli_error
import DQCS
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt


def create_initial_circuit(q, c):
    circ = QuantumCircuit(q, c)
    unitary = QuantumCircuit(2)
    unitary.ry(np.pi, 0)
    unitary.rx(np.pi, 1)
    return circ, unitary


def QPE(unitary_circuit, quantum_circuit, num_qubits):
    eig_val, vec = linalg.eig(execute(unitary_circuit, Aer.get_backend('unitary_simulator'), shots=1).result().get_unitary(unitary_circuit))
    eig_vec = vec[:, 1]
    norm = np.linalg.norm(eig_vec)**2
    quantum_circuit.initialize(eig_vec / norm, [num_qubits, num_qubits + 1])
    quantum_circuit.h([qubit for qubit in range(num_qubits)])
    reps = 1
    for counting_qubit in range(num_qubits):
        for i in range(reps):
            quantum_circuit.compose(unitary_circuit.control(), qubits=[(num_qubits - 1 - counting_qubit), num_qubits, num_qubits + 1], inplace=True)
        reps *= 2
    return quantum_circuit


def dynamic_qft(qc, num_qubits, c, inv):
    for i in range(num_qubits):
        qc.h(i)
        qc.p(np.pi/2, i)
        qc.measure(i, c[i])
        for j in range(i + 1, num_qubits):
            if inv:
                qc.p(-np.pi / 2**(j - i), j).c_if(c[i], 1)
            else:
                qc.p(np.pi / 2**(j - i), j).c_if(c[i], 1)


def qft_dagger(qc, num_qubits):
    for j in range(num_qubits):
        for m in range(j):
            qc.cp(-np.pi / float(2**(j - m)), m, j)
        qc.h(j)


def noise_model(ep_s, ep_t):
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(noise.depolarizing_error(ep_s, 1), ['x', 'y', 'z', 'h', 'u'])
    noise_model.add_all_qubit_quantum_error(pauli_error([('X', ep_t), ('I', 1 - ep_t)]).tensor(noise.depolarizing_error(ep_s, 1)), ['cx'])
    return noise_model


num_qubits, n_shots = 3, 1000
q = QuantumRegister(num_qubits + 2)
c = ClassicalRegister(num_qubits)
circ, unitary = create_initial_circuit(q, c)
circuit = QPE(unitary, circ, num_qubits)
qft_dagger(circuit, num_qubits)
for i in range(num_qubits):
    circuit.measure(i, i)
circ, unitary = create_initial_circuit(q, c)
circ = QPE(unitary, circ, num_qubits)
gate_app, qc = DQCS.DistributedCircuits(transpile(circ, basis_gates=['u', 'h', 'cx'], optimization_level=3), {"1": [i for i in range(num_qubits)], "2": [num_qubits, num_qubits+1]}).create_circuit()
q, c = QuantumRegister(circ.num_qubits + 4, 'q'), ClassicalRegister(4, 'c')
circuit_0 = QuantumCircuit(q, c)
circuit_0.append(qc.to_instruction(), q[0:circ.num_qubits + 4], c[0:4])
dynamic_qft(circuit_0, num_qubits, c, True)
val = '100'
count, counts, x = [], [], []
for i in range(0, 10):
    count.append(execute(circuit, Aer.get_backend('qasm_simulator'), basis_gates=noise_model(i * 0.005, i * 0.005).basis_gates,
                         noise_model=noise_model(i * 0.005, i * 0.005), shots=n_shots).result().get_counts()[val]/n_shots)
    counts.append(execute(circuit_0, Aer.get_backend('qasm_simulator'), basis_gates=noise_model(i * 0.005, i * 0.005).basis_gates,
                          noise_model=noise_model(i * 0.005, i * 0.005), shots=n_shots).result().get_counts()['0' + val]/n_shots)
    x.append(i * 0.05)
plt.plot(x, counts, "b*-")
plt.plot(x, count, "r*-")
plt.xlabel(r"$\epsilon$")
plt.ylabel("Probability")
plt.legend(["DDQPE", "QPE"])
plt.show()