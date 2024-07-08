# DQCS
# Author: Sreraman Muralidharan
Distributed quantum computing simulator

Modules: 
a) NonLocalQuantumGates:
This module facilitates the application of non-local quantum gates between the qubits residing in different nodes. With this functionality, quantum gates can span across nodes, paving the way for more extensive and intricate distributed quantum algorithms.

b) DistributedCircuits:
The DistributedCircuits module empowers users to effortlessly create distributed quantum circuits using a node dictionary and the quantum circuit. The feature of this module lies in its capability to handle communication registers automatically, freeing the user from having to explicitly specify them.

c) CreateDistributedCircuits:
For advanced users seeking more control, the CreateDistributedCircuits module offers the flexibility to create distributed quantum circuits using a node dictionary and the quantum circuit. Here, both the quantum registers and the communication registers can be provided as inputs, granting a finer degree of customisation.

d) NoiseModel (optional):
To enhance the fidelity of simulations, the library also offers an optional NoiseModel module. This component introduces realistic noise modeling, incorporating single qubit and two-qubit errors, and enabling a more accurate representation of real-world quantum systems.

The code "distributed_quantum_circuits.py" shows how to use the 4 classes

qc = QuantumCircuit(4)

qc.h(0)

qc.cx(1, 2)

qc.cx(0,3)

nodes = {"1": [0, 1], "2": [2, 3]}

distributed.DistributedCircuits(qc, nodes).create_circuit()

