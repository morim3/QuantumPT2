

import math
import sys
from typing import Any
from openfermion import QubitOperator, jordan_wigner
from typing import Optional, Union, Tuple, List, Sequence, Mapping
from quri_parts.openfermion.operator import operator_from_openfermion_op
from quri_parts.circuit.transpile import RZSetTranspiler
from quri_parts.core.operator import (
    pauli_label,
    Operator,
    PauliLabel,
    pauli_product,
    PAULI_IDENTITY,
)
from quri_parts.core.measurement import individual_pauli_measurement
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit, PauliRotation
from quri_parts.core.operator.representation import (
    BinarySymplecticVector,
    pauli_label_to_bsv,
    transition_amp_representation,
    transition_amp_comp_basis,
)
from quri_parts.core.state import ComputationalBasisState, ParametricCircuitQuantumState
# from quri_parts.core.sampling.shots_allocator import (
#     create_equipartition_shots_allocator,
#     create_proportional_shots_allocator
# )
import numpy as np
import scipy
from scipy.sparse import coo_matrix
import itertools
import time

sys.path.append("../")

from utils.challenge_2024 import ChallengeSampling, ExceededError, problem_hamiltonian
challenge_sampling = ChallengeSampling()


def _rounddown_to_unit(n: float, shot_unit: int) -> int:
    return shot_unit * math.floor(n / shot_unit)


class QuantumPT2:

    def construct_circuit(self, pauli_measure: PauliLabel):
        circuit = LinearMappedUnboundParametricQuantumCircuit(
            self.hf_state.qubit_count)
        circuit += self.hf_state.circuit
        # print(cisd_coeffs)
        for coeffs, virtual, occupied in self.cisd_coeffs[::-1]:
            if len(virtual) == 1:
                pauli = [1, 2]
            else:
                pauli = [1, 1, 1, 2]

            theta = -2. * np.arctan(coeffs)
            circuit.add_PauliRotation_gate(virtual+occupied, pauli, theta)

        if pauli_measure != PAULI_IDENTITY:
            circuit.add_Pauli_gate(*pauli_measure.index_and_pauli_id_list)

        return circuit

    def __init__(
        self,
        hamiltonian: Operator,
        cisd_coeffs,
        n_qubits: int,
        n_ele_cas: int,
        sampler,
        sampling_shots: int = 10**4,
        atol: float = 1e-5,
        total_shots=10**7

    ):

        hf_bits = 2 ** n_ele_cas - 1
        self.hf_state = ComputationalBasisState(n_qubits, bits=hf_bits)
        self.cisd_coeffs = cisd_coeffs

        self.hamiltonian: Operator = hamiltonian
        self.n_qubits: int = n_qubits
        self.n_ele_cas: int = n_ele_cas
        self.sampling_shots: int = sampling_shots
        self.atol: float = atol
        self.sampler = sampler

        self.total_shots = total_shots

    def run(self):

        weights = {
            m: abs(self.hamiltonian[m].real)
            for m in self.hamiltonian
        }
        weights_sum = sum(weights.values())

        shot_unit = 10
        shot_allocs = {
            m: int(self.total_shots * weights[m] / weights_sum / shot_unit) * shot_unit
            for m in self.hamiltonian
        }
        # shot_allocs = {
        #     m: int(self.total_shots / (len(self.hamiltonian)-1))
        #     for m in self.hamiltonian if m != PAULI_IDENTITY
        # }
        shots_map = {pauli_set: n_shots for pauli_set,
                     n_shots in shot_allocs.items()}
        results_dict = {}

        print("num non-zero shots paulis: ", np.count_nonzero(np.array(list(shot_allocs.values()))))

        for n_pauli, (pauli_set, n_shot) in enumerate(shots_map.items()):

            if n_shot == 0:
                continue

            # if n_pauli % 50 == 0:
            #     print("sampler time", time.time() - challenge_sampling.init_time)

            if pauli_set == PAULI_IDENTITY:
                continue

            print(
                f"pauli {pauli_set}, iter {n_pauli} in total {len(shots_map)}, shots {n_shot}")

            circuit = self.construct_circuit(pauli_set)
            transpiled_circuit = RZSetTranspiler()(circuit)

            try:
                counts = self.sampler(transpiled_circuit, n_shot)
            except ExceededError as e:
                print(e)
                return results_dict

            normalize_term = sum(counts.values())
            counts_normalized = {}
            for key, count in counts.items():
                counts_normalized[key] = count / normalize_term

            for key, count in counts_normalized.items():
                if key in results_dict:
                    results_dict[key] += counts_normalized[key] * \
                        self.hamiltonian[pauli_set].real
                else:
                    results_dict[key] = counts_normalized[key] * \
                        self.hamiltonian[pauli_set].real

        return results_dict


class PauliRotationCircuit:
    def __init__(
        self, generators: list, coeffs: list, param_names: list, n_qubits: int
    ):
        self.generators: list = generators
        self.coeffs: list = coeffs
        self.param_names: list = param_names
        self.n_qubits: int = n_qubits
        self.fusion_mem: list = []
        self.generetors_history: list = []

    def __call__(self):
        return self.construct_circuit()

    def construct_circuit(
        self, generators=None
    ) -> LinearMappedUnboundParametricQuantumCircuit:
        circuit = LinearMappedUnboundParametricQuantumCircuit(self.n_qubits)
        if generators is None:
            generators = self.generators
        for generator, coeff, name in zip(generators, self.coeffs, self.param_names):
            # print(generator)
            param_name = circuit.add_parameter(name)
            if isinstance(generator, str):
                generator = pauli_label(generator)
            else:
                raise
            pauli_index_list, pauli_id_list = zip(*generator)
            coeff = coeff.real
            circuit.add_ParametricPauliRotation_gate(
                pauli_index_list,
                pauli_id_list,
                {param_name: -2.0 * coeff},
            )
        return circuit


def diagonalize_effective_ham(
    ham_qp: Operator, comp_bases_qp: list[ComputationalBasisState]
) -> Tuple[np.ndarray, np.ndarray]:
    effective_ham_sparse = generate_truncated_hamiltonian(
        ham_qp, comp_bases_qp)
    assert np.allclose(effective_ham_sparse.todense().imag, 0)
    effective_ham_sparse = effective_ham_sparse.real
    if effective_ham_sparse.shape[0] > 10:
        eig_qsci, vec_qsci = scipy.sparse.linalg.eigsh(
            effective_ham_sparse, k=1, which="SA"
        )
        eig_qsci = eig_qsci.item()
        vec_qsci = vec_qsci.squeeze()
    else:
        eig_qsci, vec_qsci = np.linalg.eigh(effective_ham_sparse.todense())
        eig_qsci = eig_qsci[0]
        vec_qsci = np.array(vec_qsci[:, 0])

    return vec_qsci, eig_qsci


def generate_truncated_hamiltonian(
    hamiltonian: Operator,
    states: Sequence[ComputationalBasisState],
) -> scipy.sparse.spmatrix:
    """Generate truncated Hamiltonian on the given basis states."""
    dim = len(states)
    values = []
    row_ids = []
    column_ids = []
    h_transition_amp_repr = transition_amp_representation(hamiltonian)
    for m in range(dim):
        for n in range(m, dim):
            mn_val = transition_amp_comp_basis(
                h_transition_amp_repr, states[m].bits, states[n].bits
            )
            if mn_val:
                values.append(mn_val)
                row_ids.append(m)
                column_ids.append(n)
                if m != n:
                    values.append(mn_val.conjugate())
                    row_ids.append(n)
                    column_ids.append(m)
    truncated_hamiltonian = coo_matrix(
        (values, (row_ids, column_ids)), shape=(dim, dim)
    ).tocsc(copy=False)
    truncated_hamiltonian.eliminate_zeros()

    return truncated_hamiltonian


def generate_truncated_hamiltonian_rectangle(
    hamiltonian: Operator,
    states1: Sequence[ComputationalBasisState],
    states2: Sequence[ComputationalBasisState]
) -> scipy.sparse.spmatrix:
    """Generate truncated Hamiltonian on the given basis states."""

    dim1 = len(states1)
    dim2 = len(states2)
    values = []
    row_ids = []
    column_ids = []
    h_transition_amp_repr = transition_amp_representation(hamiltonian)
    for m in range(dim1):
        for n in range(dim2):
            mn_val = transition_amp_comp_basis(
                h_transition_amp_repr, states1[m].bits, states2[n].bits
            )
            if mn_val:
                values.append(mn_val)
                row_ids.append(m)
                column_ids.append(n)

    truncated_hamiltonian = coo_matrix(
        (values, (row_ids, column_ids)), shape=(dim1, dim2)
    ).tocsc(copy=False)
    truncated_hamiltonian.eliminate_zeros()

    return truncated_hamiltonian


def num_basis_symmetry_adapted_cisd(n_qubits: int):
    return (n_qubits**4 - 4 * n_qubits**3 + 20 * n_qubits**2 + 64) // 64
    # 8478


def pick_up_bits_from_counts(
    counts: Mapping[int, Union[int, float]],
    n_qubits,
    n_ele,
    R_max=None,
    threshold=None,
):

    sorted_keys = sorted(counts.keys(), key=lambda x: abs(counts[x]), reverse=True)

    if threshold is None:
        heavy_bits = sorted_keys
    else:
        heavy_bits = [bit for bit in sorted_keys if abs(counts[bit]) >= threshold]

    heavy_bits = [i for i in heavy_bits if bin(i).count("1") == n_ele]

    heavy_bits = [i for i in heavy_bits if bin(
        i >> (n_qubits//2)).count("1") != 2 and bin(i >> (n_qubits//2)).count("1") != 0]

    if R_max is not None:
        heavy_bits = heavy_bits[:R_max]
    comp_bases_qp = [
        ComputationalBasisState(n_qubits, bits=int(key)) for key in heavy_bits
    ]
    return comp_bases_qp, heavy_bits


def round_hamiltonian(op: Operator, num_pickup: int = None, coeff_cutoff: float = None):
    ret_op = Operator()

    sorted_pauli = sorted(op.keys(), key=lambda x: abs(op[x]), reverse=True)

    if coeff_cutoff is None:
        coeff_cutoff = 0

    for pauli in sorted_pauli:

        if pauli == PAULI_IDENTITY or np.all(np.array(pauli.index_and_pauli_id_list[1]) == 3):
            continue

        coeff = op[pauli]
        if abs(coeff) < coeff_cutoff:
            pass
        else:
            ret_op += Operator({pauli: coeff})

    if num_pickup is not None:
        sorted_pauli = sorted_pauli[:num_pickup]

    return ret_op


class RunAlgorithm:
    def __init__(self) -> None:
        challenge_sampling.reset()

    def result_for_evaluation(self, seed: int, hamiltonian_directory: str) -> tuple[Any, float]:
        energy_final = self.get_result(seed, hamiltonian_directory)
        total_shots = challenge_sampling.total_shots
        return energy_final, total_shots

    def get_result(self, seed: int, hamiltonian_directory: str) -> float:
        """
            param seed: the last letter in the Hamiltonian data file, taking one of the values 0,1,2,3,4
            param hamiltonian_directory: directory where hamiltonian data file exists
            return: calculated energy.
        """
        n_qubits = 28
        # n_qubits = 20
        # n_qubits = 12

        ham = problem_hamiltonian(n_qubits, seed, hamiltonian_directory)

        n_electrons = n_qubits // 2
        jw_hamiltonian = jordan_wigner(ham)
        qp_hamiltonian = operator_from_openfermion_op(jw_hamiltonian)

        # print([m for m in qp_hamiltonian])

        # print(qp_hamiltonian.keys())

        hf_bits = 2 ** n_electrons - 1
        comp_bases_sd = [
            ComputationalBasisState(n_qubits, bits=hf_bits ^ sum([1 << i for i in ies]) ^ sum([1 << j for j in js])) for m in range(0, 3) for ies in itertools.combinations(range(0, int(n_qubits/2)), m) for js in itertools.combinations(range(int(n_qubits/2), n_qubits), m)
        ]

        _, hf_energy = diagonalize_effective_ham(
            qp_hamiltonian, [ComputationalBasisState(n_qubits, bits=hf_bits)]
        )
        print("hf energy ", hf_energy)

        vec_cisd, val_cisd = diagonalize_effective_ham(
            qp_hamiltonian, comp_bases_sd
        )

        for th in [0, 0.015]:
            th_base = [b for i, b in enumerate(
                comp_bases_sd) if abs(vec_cisd[i]) > th]
            vec_th, val_th = diagonalize_effective_ham(
                qp_hamiltonian, th_base
            )
            print(f"{th}: {val_th}, {len(th_base)}")

        num_pickup, coeff_cutoff = 10000, 0.001
        print("original hamiltonian len", len(qp_hamiltonian))
        pt2_hamiltonian = round_hamiltonian(
            qp_hamiltonian, num_pickup, coeff_cutoff)
        print("rounded hamiltonian len", len(pt2_hamiltonian))

        cisd_coeffs = []
        for coef, base in zip(vec_th, th_base):
            if base.bits != hf_bits:
                virtual = [i for i in range(
                    int(n_qubits/2), n_qubits) if (1 << i) & base.bits]
                occupied = [i for i in range(
                    0, int(n_qubits/2)) if (1 << i) & base.bits == 0]
                cisd_coeffs.append((coef, virtual, occupied))

        cisd_coeffs = sorted(cisd_coeffs, key=lambda x: - abs(x[0]))

        mps_sampler = challenge_sampling.create_sampler()

        quant_pt2 = QuantumPT2(
            pt2_hamiltonian,
            # pool,
            cisd_coeffs=cisd_coeffs,
            n_qubits=n_qubits,
            n_ele_cas=n_electrons,
            sampler=mps_sampler,
            atol=1e-6,
            total_shots=10**7
        )
        results_dict = quant_pt2.run()
        print(results_dict)

        print("post processing")
        comp_basis_pt2, heavy_bits = pick_up_bits_from_counts(
            results_dict, n_qubits=n_qubits, n_ele=n_electrons, R_max=50000, threshold=0.00001)

        print(comp_basis_pt2)
        print("number of perturbation comp basis", len(comp_basis_pt2))

        perturb_hamiltonian = np.array(generate_truncated_hamiltonian_rectangle(
            qp_hamiltonian, comp_bases_sd, comp_basis_pt2).todense().real)
        perturb_hamiltonian_diag = np.array([generate_truncated_hamiltonian(
            qp_hamiltonian, [c]).todense()[0, 0] for c in comp_basis_pt2]).real

        # print(perturb_hamiltonian_diag)
        # print(perturb_hamiltonian)
        # print(perturb_hamiltonian_diag - val_cisd)
        # print(np.abs(perturb_hamiltonian.T @ vec_cisd))

        pt2_energy = np.sum(np.abs(perturb_hamiltonian.T @ vec_cisd)
                            ** 2 / (perturb_hamiltonian_diag - val_cisd))
        print(pt2_energy)

        return val_cisd - pt2_energy


if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result(seed=1, hamiltonian_directory="../hamiltonian"))
