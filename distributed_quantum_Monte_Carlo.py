#Quantum amplitude esitmaton in distributed quantum computer, Cramer-Rao
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, transpile, Aer, execute
import distributed
import dynamic_quantum_circuit_creator
from scipy import optimize
import time
import numpy as np
import matplotlib.pyplot as plt
import math

b_max = math.pi / 5
nbit = 3
analyticResult = (b_max / 2.0 - math.sin(2 * b_max) / 4.0 ) / b_max
print("Analytical Result:", analyticResult)
ndiv = 2**nbit
discretizedResult = 0.0
for i in range(ndiv):
    discretizedResult += math.sin(b_max / ndiv * (i + 0.5))**2
discretizedResult = discretizedResult / ndiv
print("Discretized Result:", discretizedResult)
def P(qc, qx, nbit):
    qc.h(qx)
def R(qc, qx, qx_measure, nbit, b_max):
    qc.ry(b_max / 2**nbit * 2 * 0.5, qx_measure)
    for i in range(nbit):
        qc.cu(2**i * b_max / 2**nbit * 2, 0, 0,0, qx[i], qx_measure[0])
def Rinv(qc, qx, qx_measure, nbit, b_max):
    for i in range(nbit)[::-1]:
        qc.cu(-2**i * b_max / 2**nbit * 2, 0, 0, 0, qx[i], qx_measure[0])
    qc.ry(-b_max / 2**nbit * 2 * 0.5, qx_measure)
def multi_control_NOT(qc, qx, qx_measure, qx_ancilla, nbit, b_max):
    if nbit == 1:
        qc.cz(qx[0], qx_measure[0])
    elif nbit == 2:
        qc.h(qx_measure[0])
        qc.ccx(qx[0], qx[1], qx_measure[0])
        qc.h(qx_measure[0])
    elif nbit > 2.0:
        qc.ccx(qx[0], qx[1], qx_ancilla[0])
        for i in range(nbit - 3):
            qc.ccx(qx[i + 2], qx_ancilla[i], qx_ancilla[i + 1])
        qc.h(qx_measure[0])
        qc.ccx(qx[nbit - 1], qx_ancilla[nbit - 3], qx_measure[0])
        qc.h(qx_measure[0])
        for i in range(nbit - 3)[::-1]:
            qc.ccx(qx[i + 2], qx_ancilla[i], qx_ancilla[i + 1])
        qc.ccx(qx[0], qx[1], qx_ancilla[0])
def reflect(qc, qx, qx_measure, qx_ancilla, nbit, b_max):
    for i in range(nbit):
        qc.x(qx[i])
    qc.x(qx_measure[0])
    qc.barrier()
    multi_control_NOT(qc, qx, qx_measure, qx_ancilla, nbit, b_max)
    qc.barrier()
    qc.x(qx_measure[0])
    for i in range(nbit):
        qc.x(qx[i])
def Q_grover(qc, qx, qx_measure, qx_ancilla, nbit, b_max):
    qc.z(qx_measure[0])
    Rinv(qc, qx, qx_measure, nbit, b_max)
    qc.barrier()
    P(qc, qx, nbit)
    reflect(qc, qx, qx_measure, qx_ancilla, nbit, b_max)
    P(qc, qx, nbit)
    qc.barrier()
    R(qc, qx, qx_measure, nbit, b_max)
def create_grover_circuit(numebr_grover_list, nbit, b_max):
    qc_list = []
    for igrover in range(len(numebr_grover_list)):
        qx = QuantumRegister(nbit,'q')
        qx_measure = QuantumRegister(1,'m')
        cr = ClassicalRegister(1)
        if (nbit > 2):
            qx_ancilla = QuantumRegister(nbit - 2)
            qc = QuantumCircuit(qx, qx_ancilla, qx_measure, cr)
        else:
            qx_ancilla = 0
            qc = QuantumCircuit(qx, qx_measure, cr)
        P(qc, qx, nbit)
        R(qc, qx, qx_measure, nbit, b_max)
        for ikAA in range(numebr_grover_list[igrover]):
            Q_grover(qc, qx, qx_measure, qx_ancilla, nbit, b_max)
        qc.measure(qx_measure[0], cr[0])
        qc_list.append(qc)
    return qc_list
def run_grover(qc_list, number_grover_list, shots_list, backend):
    hit_list = []
    for k in range(len(number_grover_list)):
        job = execute(qc_list[k], backend=backend, shots=shots_list[k])
        lapse = 0
        interval = 0.00001
        time.sleep(interval)
        while job.status().name != 'DONE':
            time.sleep(interval)
            lapse += 1
        counts = job.result().get_counts(qc_list[k]).get("1", 0)
        hit_list.append(counts)
    return hit_list
def calculate_theta(hit_list, number_grover_list, shots_list):
    small = 1.e-15
    confidenceLevel = 5
    thetaCandidate_list = []
    rangeMin = 0.0 + small
    rangeMax = 1.0 - small
    for igrover in range(len(number_grover_list)):

        def loglikelihood(p):
            ret = np.zeros_like(p)
            theta = np.arcsin(np.sqrt(p))
            for n in range(igrover + 1):
                ihit = hit_list[n]
                arg = (2 * number_grover_list[n] + 1) * theta
                ret = ret + 2 * ihit * np.log(np.abs(np.sin(arg))) + 2 * (
                        shots_list[n] - ihit) * np.log(np.abs(np.cos(arg)))
            return -ret
        searchRange = (rangeMin, rangeMax)
        searchResult = optimize.brute(loglikelihood, [searchRange])
        pCandidate = searchResult[0]
        thetaCandidate_list.append(np.arcsin(np.sqrt(pCandidate)))
        perror = CalcErrorCramérRao(igrover, shots_list, pCandidate, number_grover_list)
        rangeMax = min(pCandidate+confidenceLevel*perror,1.0 - small)
        rangeMin = max(pCandidate-confidenceLevel*perror,0.0 + small)
    return thetaCandidate_list
shots_list = [100, 100, 100, 100, 100, 100, 100]
number_grover_list = [0, 1, 2, 4, 8, 16, 32]
if len(shots_list) != len(number_grover_list):
    raise Exception(
        'The length of shots_list should be equal to the length of number_grover_list.'
    )
backend = Aer.get_backend('qasm_simulator')
def CalcErrorCramérRao(M, shot_list, p0, number_grover_list):
    FisherInfo = 0
    for k in range(M + 1):
        Nk = shot_list[k]
        mk = number_grover_list[k]
        FisherInfo += Nk / (p0 * (1 - p0)) * (2 * mk + 1)**2
    return np.sqrt(1 / FisherInfo)

def CalcNumberOracleCalls(M, shot_list, number_grover_list):
    Norac = 0
    for k in range(M + 1):
        Nk = shots_list[k]
        mk = number_grover_list[k]
        Norac += Nk * (2 * mk + 1)
    return Norac
def run_grover_dist(qc_list_dist, number_grover_list, shots_list, backend):
    hit_list = []
    for k in range(len(number_grover_list)):
        job = execute(qc_list_dist[k], backend=backend, shots=shots_list[k])
        lapse = 0
        while job.status().name != 'DONE':
            lapse += 1
        counts = job.result().get_counts(qc_list_dist[k])
        counts_1 = 0
        for key,val in counts.items():
            if key[2] == '1':
                counts_1 = counts_1+val
        hit_list.append(counts_1)
    return hit_list

qc_list = create_grover_circuit(number_grover_list, nbit,
                                b_max)
qc_list = transpile(qc_list, basis_gates=['u', 'h', 'cx'], optimization_level=3)
hit_list = run_grover(qc_list, number_grover_list, shots_list,
                      backend)
thetaCandidate_list = calculate_theta(
    hit_list, number_grover_list, shots_list)
print(np.abs(np.sin(thetaCandidate_list)**2 - discretizedResult))
# circuit= []
# p = 1
# for i in range(0, len(number_grover_list)):
#     x = 'q{}'.format(i)
#     q_0 = QuantumRegister(3, 'q')
#     q_1 = QuantumRegister(1, str(x))
#     q_2 = QuantumRegister(1, 'm')
#     qc = QuantumCircuit(q_0, q_1, q_2)
#     nodes = {"1": [q_0[0], q_0[1]], "2": [q_0[2]], "3": [q_1[0], q_2]}
#     comm = QuantumRegister(2, 'c')
#     c = ClassicalRegister(3, 'c0')
#     circ = QuantumCircuit(q_0, q_1, q_2, comm, c)
#     gate_app, circ = distributed.CreateDistributedCircuits(qc_list[i], c, circ, comm, nodes, p).create_circuit()
#     circ.measure(4,0)
#     circuit.append(circ)
# counts_list = run_grover_dist(circuit, number_grover_list, shots_list, backend)
# thetaCandidate_list_1 = calculate_theta(counts_list, number_grover_list, shots_list)
# print(np.abs(np.sin(thetaCandidate_list_1)**2 - discretizedResult))
# error_list = np.abs(np.sin(thetaCandidate_list)**2 - discretizedResult)
# OracleCall_list = []
# ErrorCramérRao_list = []
# for i in range(len(number_grover_list)):
#     OracleCall_list.append(
#         CalcNumberOracleCalls(i, shots_list, number_grover_list))
#     ErrorCramérRao_list.append(
#         CalcErrorCramérRao(i, shots_list, discretizedResult, number_grover_list))
#
# error_list_1 = np.abs(np.sin(thetaCandidate_list_1)**2 - discretizedResult)
# p1 = plt.plot(OracleCall_list, error_list, 'o')
# p2 = plt.plot(OracleCall_list, ErrorCramérRao_list)
# p3 = plt.plot(OracleCall_list, error_list_1, 'r*')
# plt.xscale('log')
# plt.xlabel("Number of oracle calls")
# plt.yscale('log')
# plt.ylabel("Estimation Error")
# plt.legend((p1[0], p2[0], p3[0]), ("Estimated Value", "Cramér-Rao", "Distributed"))
# plt.show()