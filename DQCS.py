import random

import numpy as np
from qiskit import Aer, execute
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer import noise


class NonLocalQuantumGates:
    def __init__(self, qc, c, ctrl_qubit, tgt_qubit, comm_qubits, p):
        self.quantum_circuit, self.c, self.ctrl_qubit, self.tgt_qubit, self.comm_qubits, self.p = qc, c, ctrl_qubit, \
                                                                                                  tgt_qubit, comm_qubits,\
                                                                                                  p

    def non_local_cx(self):
        if self.ctrl_qubit == self.tgt_qubit:
            return print("Please check the quantum circuit")
        e_link = 0
        n_steps = 0
        while e_link != 1:
            p_random = np.average(random.sample(range(1, 10), 3)) / 10
            n_steps += 1
            if self.p > p_random:
                self.quantum_circuit.h(self.comm_qubits[0])
                self.quantum_circuit.cx(self.comm_qubits[0], self.comm_qubits[1])
                e_link = 1
        self.quantum_circuit.cx(self.ctrl_qubit, self.comm_qubits[0])
        self.quantum_circuit.measure(self.comm_qubits[0], self.c[0])
        self.quantum_circuit.cx(self.comm_qubits[1], self.tgt_qubit)
        self.quantum_circuit.h(self.comm_qubits[1])
        self.quantum_circuit.measure(self.comm_qubits[1], self.c[1])
        self.quantum_circuit.z(self.ctrl_qubit).c_if(self.c[1], 1)
        self.quantum_circuit.x(self.tgt_qubit).c_if(self.c[0], 1)
        self.quantum_circuit.reset(self.comm_qubits[0])
        self.quantum_circuit.reset(self.comm_qubits[1])
        return self.quantum_circuit, n_steps

    def non_local_ccx(self):
        e_link = 0
        n_steps = 0
        while e_link != 1:
            p_random = np.average(random.sample(range(1, 10), 3)) / 10
            n_steps += 1
            if self.p > p_random:
                self.quantum_circuit.h(self.comm_qubits[0])
                self.quantum_circuit.cx(self.comm_qubits[0], self.comm_qubits[1])
                e_link = 1
        e_link = 0
        while e_link != 1:
            p_random = np.average(random.sample(range(1, 10), 3)) / 10
            n_steps += 1
            if self.p > p_random:
                self.quantum_circuit.h(self.comm_qubits[2])
                self.quantum_circuit.cx(self.comm_qubits[2], self.comm_qubits[3])
                e_link = 1
        self.quantum_circuit.cx(self.ctrl_qubit[0], self.comm_qubits[0])
        self.quantum_circuit.measure(self.comm_qubits[0], self.c[0])
        self.quantum_circuit.x(self.comm_qubits[1]).c_if(self.c[0], 1)
        self.quantum_circuit.cx(self.ctrl_qubit[1], self.comm_qubits[2])
        self.quantum_circuit.measure(self.comm_qubits[2], self.c[1])
        self.quantum_circuit.x(self.comm_qubits[3]).c_if(self.c[1], 1)
        self.quantum_circuit.ccx(self.comm_qubits[3], self.comm_qubits[1], self.tgt_qubit)
        self.quantum_circuit.h(self.comm_qubits[3])
        self.quantum_circuit.measure(self.comm_qubits[3], self.c[2])
        self.quantum_circuit.h(self.comm_qubits[1])
        self.quantum_circuit.measure(self.comm_qubits[1], self.c[3])
        self.quantum_circuit.z(self.ctrl_qubit[0]).c_if(self.c[2], 1)
        self.quantum_circuit.z(self.ctrl_qubit[1]).c_if(self.c[3], 1)
        for i in range(len(self.comm_qubits)):
            self.quantum_circuit.reset(self.comm_qubits[i])
        return self.quantum_circuit, n_steps


class CreateDistributedCircuits:
    def __init__(self, qc, c, circ, comm, nodes, p):
        self.qc, self.c, self.circ, self.comm, self.nodes, self.p, self.n = qc, c, circ, comm, nodes, p, qc.num_qubits

    def create_circuit(self):
        def is_same_node(qubit_pair):
            return any(all(qubit in self.nodes[key] for qubit in qubit_pair) for key in self.nodes)

        gate_info = [(gate[0].name, gate[1], gate[0].params) for gate in self.qc.data]
        local_graph, non_local_graph = [], []
        num_gates = [0, 0, 0, 0, local_graph, non_local_graph]
        for gate, qubits, params in gate_info:
            if gate == 'h':
                self.circ.h(qubits[0])
                num_gates[0] += 1
            elif gate == 'u':
                self.circ.u(*params, qubits[0])
                num_gates[0] += 1
            elif gate == 'cx':
                if is_same_node(qubits):
                    self.circ.cx(qubits[0], qubits[1])
                    local_graph.append(qubits)
                    num_gates[1] += 1
                else:
                    self.circ, n_steps = NonLocalQuantumGates(self.circ, self.c, *qubits, self.comm[:2],
                                                              self.p).non_local_cx()
                    non_local_graph.append(qubits)
                    num_gates[3] += n_steps
                    num_gates[2] += 1
            elif gate == 'ccx':
                if is_same_node(qubits[:2]) and is_same_node(qubits[1:]):
                    self.circ.ccx(*qubits)
                    num_gates[0] += 1
                else:
                    ctrl_qubits = [qubits[0], qubits[1]]
                    target_qubit = [qubits[2]]
                    comm_qubits = self.comm[:4]
                    self.circ, n_steps = NonLocalQuantumGates(self.circ, self.c, ctrl_qubits, target_qubit, comm_qubits,
                                                              self.p).non_local_ccx()
                    num_gates[2] += 1
        return True, self.circ


class DistributedCircuits:
    def __init__(self, quantum_circuit, nodes):
        self.qc = quantum_circuit
        self.nodes = nodes
        self.n = quantum_circuit.num_qubits

    def create_circuit(self):
        q = QuantumRegister(self.n + 4, 'a')
        c = ClassicalRegister(4, 'c')
        quantum_circuit = QuantumCircuit(q, c)
        n = 0
        for gate, qargs, _ in self.qc:
            gate_name = gate.name
            gate_app = 0
            if gate_name == 'h':
                quantum_circuit.h(q[qargs[0].index])
            elif gate_name == 'u':
                quantum_circuit.u(*gate.params, q[qargs[0].index])
            elif gate_name == 'cx':
                if qargs[0].index == qargs[1].index:
                    print("Please check the quantum circuit")
                    break
                for key in self.nodes:
                    cond = qargs[0].index in self.nodes[key] and qargs[1].index in self.nodes[key]
                    if cond:
                        break
                if cond:
                    quantum_circuit.cx(q[qargs[0].index], q[qargs[1].index])
                    gate_app = 1
                else:
                    quantum_circuit, n_steps = NonLocalQuantumGates(quantum_circuit, c, q[qargs[0].index],
                                                                    q[qargs[1].index],
                                                                    [self.n, self.n + 1], 1).non_local_cx()
                    n = n + n_steps
                    gate_app = 1
            elif gate_name == 'ccx':
                gate_app = 0
                for key in self.nodes:
                    cond = (qargs[0].index in self.nodes[key]) and (qargs[1].index in self.nodes[key]) and (
                            qargs[2].index in self.nodes[key])
                    if cond:
                        quantum_circuit.ccx(q[qargs[0].index], q[qargs[1].index], qargs[2].index)
                        gate_app = 1
                    elif (qargs[0].index in self.nodes[key]) and (qargs[1].index in self.nodes[key]):
                        ctrl_qubits = [q[qargs[0].index], q[qargs[1].index]]
                        comm_qubits = [self.n, self.n + 1, self.n + 2, self.n + 3]
                        target_qubit = [q[qargs[2].index]]
                        quantum_circuit, n_steps = NonLocalQuantumGates(quantum_circuit, c, ctrl_qubits, target_qubit,
                                                                        comm_qubits,
                                                                        1).non_local_ccx()
                        gate_app = 1
                if gate_app == 0:
                    return gate_app, print("The ctrl qubits are in different nodes")
            elif gate_name == 'measure':
                quantum_circuit.measure(qargs[0].index, 0)
            elif gate_name not in ['h', 'u', 'cx', 'ccx', 'barrier', 'reset']:
                gate_app = 0
                print("The quantum circuit has unknown quantum gates")
                return gate_app
        return gate_app, quantum_circuit


class NoiseModel():
    def __init__(self, quantum_circuit, num_shots, repetitions, p1, p2, p_reset, p_meas, num_qubits):
        self.qc = quantum_circuit
        self.shots = num_shots
        self.reps = repetitions
        self.p1 = p1
        self.p2 = p2
        self.p_reset = p_reset
        self.p_meas = p_meas
        self.qubits = num_qubits

    def job_execute(self):
        noise_model = noise.NoiseModel()
        error_1 = noise.depolarizing_error(self.p1, 1)
        noise_model.add_all_qubit_quantum_error(error_1, ['x', 'z', 'h', 'u'])
        error_gate1 = noise.pauli_error([('X', self.p2), ('I', 1 - self.p2)])
        error_gate2 = error_gate1.tensor(error_gate1)
        noise_model.add_all_qubit_quantum_error(error_gate2, ['cx'])
        error_reset = noise.pauli_error([('X', self.p_reset), ('Z', self.p_reset), ('I', 1 - 2 * self.p_reset)])
        error_meas = error_reset.compose(error_reset)
        noise_model.add_all_qubit_quantum_error(error_meas, "measure")
        basis_gates = noise_model.basis_gates
        for expt in range(0, self.reps + 1):
            hist = execute(self.qc, Aer.get_backend('qasm_simulator'),
                           basis_gates=basis_gates,
                           noise_model=noise_model, shots=self.shots).result().get_counts()
        return hist
