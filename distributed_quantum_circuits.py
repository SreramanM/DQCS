import DQCS
from qiskit import Aer, execute
from qiskit import ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
import time


def get_hist(circuit, qubits, n_shots):
    hist, hist_dict = execute(circuit, Aer.get_backend('qasm_simulator'), shots=n_shots).result().get_counts(), {}
    for key in hist:
        hist_dict[key.split(" ")[0][::-1][0:qubits]] = 0
    for key in hist:
        hist_dict[key.split(" ")[0][::-1][0:qubits]] = hist_dict[key.split(" ")[0][::-1][0:qubits]] + hist[key]
    hist = {k: v / total for total in (sum(hist_dict.values(), 0.0),) for k, v in hist_dict.items()}
    return hist


# 1) Define the qc on one register
print("Generating the distributed quantum circuit 1")
n_shots = 1000
time.sleep(2.5)
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
# nodes - define the qubits in each node
nodes = {"1": [0, 1], "2": [2]}
valid_node = 1
qubits = [qubit for sublist in list(nodes.values()) for qubit in sublist]
# Check if nodes are specified for all the qubits
for qubit in range(qc.num_qubits):
    if qubit not in qubits:
        print("please specify the node for qubit", qubit)
        valid_node = 0
if valid_node == 1:
    gate_app, circ = DQCS.DistributedCircuits(qc, nodes).create_circuit()
    circ.measure_all()
    hist_dict = get_hist(circ, qc.num_qubits, n_shots)
    print("Result:", hist_dict)
# 2) Define the qc on multiple registers
print("Generating the distributed quantum circuit 2")
q_0 = QuantumRegister(2, 'a')
q_1 = QuantumRegister(2, 'b')
qc = QuantumCircuit(q_0, q_1)
nodes = {"1": [q_0[0], q_0[1]], "2": [q_1[0]], "3": [q_1[1]]}
qc.h(q_0[0])
qc.cx(q_0[0], q_1[1])
# Define empty circuit with communication registers
comm = QuantumRegister(4, 'c')
c = ClassicalRegister(4, 'cl')
circ = QuantumCircuit(q_0, q_1, comm, c)
gate_app, circ = DQCS.CreateDistributedCircuits(qc, c, circ, comm, nodes, 1).create_circuit()
circ.measure_all()
hist_dict = get_hist(circ, qc.num_qubits, n_shots)
print("Result:", hist_dict)