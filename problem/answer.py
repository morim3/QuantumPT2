
import sys
from typing import Any
from openfermion import QubitOperator, jordan_wigner
from typing import Optional, Union, Tuple, List, Sequence, Mapping
from quri_parts.openfermion.operator import operator_from_openfermion_op
from quri_parts.circuit.transpile import RZSetTranspiler
# from quri_parts.algo.optimizer import SPSA, OptimizerStatus 
from quri_parts.core.operator import (
    pauli_label,
    Operator,
    PauliLabel,
    pauli_product,
    PAULI_IDENTITY,
)
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit, PauliRotation
from quri_parts.core.operator.representation import (
    BinarySymplecticVector,
    pauli_label_to_bsv,
    transition_amp_representation,
    transition_amp_comp_basis,
)
from quri_parts.core.state import ComputationalBasisState, ParametricCircuitQuantumState
import numpy as np
import scipy
from scipy.sparse import coo_matrix
import itertools

sys.path.append("../")
from utils.challenge_2024 import ChallengeSampling, ExceededError, problem_hamiltonian
from problem.custom_spsa import SPSA, HillClimbing 

challenge_sampling = ChallengeSampling()



class ADAPT_QSCI:

    def initialize_state(self, cisd_coeffs):

        self.param_values: list = []
        generators = []
        coeffs_list = []
        param_names = []

        # print(cisd_coeffs)
        for i, (coeffs, virtual, occupied) in enumerate(cisd_coeffs[::-1]):
            if len(virtual) == 1:
                # generators.append(([virtual[0], occupied[0]], (0, 1)))
                generators.append(f"X{virtual[0]} Y{occupied[0]}")
            else:
                # generators.append(([virtual[0], virtual[1], occupied[0], occupied[1]], (0, 0, 0, 1)))
                generators.append(f"X{virtual[0]} X{virtual[1]} X{occupied[0]} Y{occupied[1]}")


            coeffs_list.append(1.)
            param_names.append(f"param {i}")
            theta = np.arctan(coeffs * 2) * 2
            # theta = coeffs
            self.param_values.append(theta)

            
        return PauliRotationCircuit(generators, coeffs_list, param_names, self.n_qubits)

            
    def __init__(
        self,
        hamiltonian: Operator,
        cisd_coeffs,
        n_qubits: int,
        n_ele_cas: int,
        sampler,
        iter_max: int = 10,
        sampling_shots: int = 10**4,
        post_selected: bool = True,
        atol: float = 1e-5,
        max_num_converged: int = 1,
        final_sampling_shots_coeff: float = 1.0,
        check_duplicate: bool = True,
        reset_ignored_inx_mode: int = 10,
        hf_energy=None
    ):
        hf_bits = 2 ** n_ele_cas - 1
        self.initial_state = ComputationalBasisState(n_qubits, bits=hf_bits)
        self.hf_energy = hf_energy

        self.hamiltonian: Operator = hamiltonian
        self.n_qubits: int = n_qubits
        self.n_ele_cas: int = n_ele_cas
        self.iter_max: int = iter_max
        self.sampling_shots: int = sampling_shots
        self.atol: float = atol
        self.sampler = sampler
        self.post_selected: bool = post_selected
        self.check_duplicate: bool = check_duplicate
        # initialization

        self.ignored_gen_inx = []
        self.reset_ignored_inx_mode: int = reset_ignored_inx_mode if reset_ignored_inx_mode > 0 else iter_max
        # convergence
        assert max_num_converged >= 1
        self.final_sampling_shots: int = int(final_sampling_shots_coeff * sampling_shots)
        self.max_num_converged: int = max_num_converged
        self.num_converged: int = 0
        # results
        self.qsci_energy_history: list = []
        self.opt_energy_history: list = []
        self.raw_energy_history = []
        self.sampling_results_history = []
        self.opt_param_value_history = []
        self.corrected_energy = []

        self.pauli_rotation_circuit_qsci = self.initialize_state(cisd_coeffs)

    def run_qsci(self, circuit):
        counts = self.sampler(circuit, self.sampling_shots)
        comp_basis, heavy_bits = pick_up_bits_from_counts(
            counts=counts,
            n_qubits=self.n_qubits,
            R_max=num_basis_symmetry_adapted_cisd(self.n_qubits),
            threshold=1e-10,
            post_select=self.post_selected,
            n_ele=self.n_ele_cas,
        )
        if self.initial_state.bits not in np.array(heavy_bits):
            comp_basis.append(self.initial_state)

        vec_qsci, val_qsci = diagonalize_effective_ham(self.hamiltonian, comp_basis)

        hf_index = np.where(np.array(heavy_bits) == self.initial_state.bits)[0][0]
        return comp_basis, vec_qsci, val_qsci, vec_qsci[hf_index]

    def cost_fn(self, param_values):

        target_circuit = self.parametric_state_qsci.parametric_circuit.bind_parameters(param_values)
        transpiled_circuit = RZSetTranspiler()(target_circuit)

        comp_basis, vec_qsci, val_qsci, hf_vec_qsci = self.run_qsci(transpiled_circuit)

        print("param_values", param_values)
        print(f"num basis: {len(comp_basis)}")
        print(f"qsci energy for this param values: {val_qsci}")
        print(f"hf_energy: ", self.hf_energy, " hf_vec: ", hf_vec_qsci)
        print(f"davidson correction: {val_qsci + (1-hf_vec_qsci**2) * (val_qsci - self.hf_energy)}")

        self.corrected_energy.append(val_qsci + (1-hf_vec_qsci**2) * (val_qsci - self.hf_energy))
        self.qsci_energy_history.append(val_qsci)

        return val_qsci
        

    def run(self) -> float:

        self.parametric_state_qsci = prepare_parametric_state(self.initial_state, self.pauli_rotation_circuit_qsci())

        target_circuit = self.parametric_state_qsci.parametric_circuit.bind_parameters(self.param_values)
        transpiled_circuit = RZSetTranspiler()(target_circuit)
        self.comp_basis, _, val_qsci, hf_vec_qsci = self.run_qsci(transpiled_circuit)

        print("initial energy", val_qsci)
        print(f"initial basis: {[bin(b.bits)[2:].zfill(self.n_qubits) for b in self.comp_basis]}")
        print(f"num basis: {len(self.comp_basis)}")
        print(f"davidson correction: {val_qsci + (1-hf_vec_qsci**2) * (val_qsci - self.hf_energy)}")

        self.corrected_energy.append(val_qsci + (1-hf_vec_qsci**2) * (val_qsci - self.hf_energy))
        self.qsci_energy_history.append(val_qsci)

        optimizer = HillClimbing(val_qsci, c=0.001, ftol=10e-9)
        opt_state = optimizer.get_init_state(self.param_values)

        for itr in range(1, self.iter_max + 1):

            print(f"------ {itr} --------")
            try:
                opt_state = optimizer.step(opt_state, self.cost_fn)
            except ExceededError as e:
                print(str(e))
                # return np.min(self.qsci_energy_history)
                return np.min(self.corrected_energy)


        # return np.min(self.qsci_energy_history)
        return np.min(self.corrected_energy)


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

    # def add_new_gates(
    #     self, generator: str, coeff: float, param_name: str
    # ) -> LinearMappedUnboundParametricQuantumCircuit:
    #     self._reset()
    #     self.generetors_history.append(generator)
    #     for i, (g, n) in enumerate(zip(self.generators[::-1], self.param_names[::-1])):
    #         if is_equivalent(generator, g):
    #             self.fusion_mem = [-i]
    #             print(f"FUSED: {g, generator}")
    #             break
    #         elif is_commute(generator, g):
    #             continue
    #         else:
    #             break
    #     if not self.fusion_mem:
    #         self.generators.append(generator)
    #         self.coeffs.append(coeff)
    #         self.param_names.append(param_name)
    #     return self.construct_circuit()

    def delete_newest_gate(self) -> LinearMappedUnboundParametricQuantumCircuit:
        self._reset()
        self.generators = self.generators[:-1]
        self.coeffs = self.coeffs[:-1]
        self.param_names = self.param_names[:-1]
        return self.construct_circuit()

    def _reset(self):
        self.fusion_mem = []


def diagonalize_effective_ham(
    ham_qp: Operator, comp_bases_qp: list[ComputationalBasisState]
) -> Tuple[np.ndarray, np.ndarray]:
    effective_ham_sparse = generate_truncated_hamiltonian(ham_qp, comp_bases_qp)
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


def _add_term_from_bsv(
    bsvs: List[List[Tuple[int, int]]], ops: List[Operator]
) -> Operator:
    ret_op = Operator()
    op0_bsv, op1_bsv = bsvs[0], bsvs[1]
    op0, op1 = ops[0], ops[1]
    for i0, (pauli0, coeff0) in enumerate(op0.items()):
        for i1, (pauli1, coeff1) in enumerate(op1.items()):
            bitwise_string = str(
                bin(
                    (op0_bsv[i0][0] & op1_bsv[i1][1])
                    ^ (op0_bsv[i0][1] & op1_bsv[i1][0])
                )
            )
            if bitwise_string.count("1") % 2 == 1:
                pauli_prod_op, pauli_prod_phase = pauli_product(pauli0, pauli1)
                tot_coef = 2 * coeff0 * coeff1 * pauli_prod_phase
                ret_op.add_term(pauli_prod_op, tot_coef)
    return ret_op


def pauli_string_to_bsv(pauli_str: str) -> BinarySymplecticVector:
    return pauli_label_to_bsv(pauli_label(pauli_str))


def get_bsv(pauli: Union[PauliLabel, str]) -> BinarySymplecticVector:
    if isinstance(pauli, str):
        bsv = pauli_string_to_bsv(pauli)
    else:
        bsv = pauli_label_to_bsv(pauli)
    return bsv


def is_commute(pauli1: Union[PauliLabel, str], pauli2: Union[PauliLabel, str]) -> bool:
    bsv1 = get_bsv(pauli1)
    bsv2 = get_bsv(pauli2)
    x1_z2 = bsv1.x & bsv2.z
    z1_x2 = bsv1.z & bsv2.x
    is_bitwise_commute_str = str(bin(x1_z2 ^ z1_x2)).split("b")[-1]
    return sum(int(b) for b in is_bitwise_commute_str) % 2 == 0


def is_equivalent(
    pauli1: Union[PauliLabel, str], pauli2: Union[PauliLabel, str]
) -> bool:
    bsv1 = get_bsv(pauli1)
    bsv2 = get_bsv(pauli2)
    return bsv1.x == bsv2.x and bsv1.z == bsv2.z


def operator_bsv(op: Operator) -> List[Tuple[int, int]]:
    ret = []
    for pauli in op.keys():
        bsv_pauli = get_bsv(pauli)
        ret.append((bsv_pauli.x, bsv_pauli.z))
    return ret


def round_hamiltonian(op: Operator, num_pickup: int = None, coeff_cutoff: float = None):
    ret_op = Operator()
    if coeff_cutoff in [None, 0.0] and num_pickup is None:
        return op
    sorted_pauli = sorted(op.keys(), key=lambda x: abs(op[x]), reverse=True)
    if num_pickup is not None:
        sorted_pauli = sorted_pauli[:num_pickup]
    if coeff_cutoff is None:
        coeff_cutoff = 0
    for pauli in sorted_pauli:
        coeff = op[pauli]
        if abs(coeff) < coeff_cutoff:
            pass
        else:
            ret_op += Operator({pauli: coeff})
    return ret_op




def prepare_parametric_state(initial_state, ansatz):
    circuit = LinearMappedUnboundParametricQuantumCircuit(initial_state.qubit_count)
    circuit += initial_state.circuit
    circuit += ansatz
    return ParametricCircuitQuantumState(initial_state.qubit_count, circuit)


def key_sortedabsval(data: Union[list, dict, np.ndarray], round_: int = 5) -> dict:
    if isinstance(data, dict):
        sorted_keys = sorted(data.keys(), key=lambda x: abs(data[x]), reverse=True)
    else:
        sorted_keys = np.argsort(np.abs(data))[::-1]
    ret_dict = {}
    for k in sorted_keys:
        val = float(data[int(k)].real)
        assert np.isclose(val.imag, 0.0)
        ret_dict[int(k)] = round(val, round_)
    return ret_dict



def num_basis_symmetry_adapted_cisd(n_qubits: int):
    return (n_qubits**4 - 4 * n_qubits**3 + 20 * n_qubits**2 + 64) // 64
    # 8478


def pick_up_bits_from_counts(
    counts: Mapping[int, Union[int, float]],
    n_qubits,
    R_max=None,
    threshold=None,
    post_select=False,
    n_ele=None,
):
    sorted_keys = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
    if threshold is None:
        heavy_bits = sorted_keys
    else:
        heavy_bits = [bit for bit in sorted_keys if counts[bit] >= threshold]
    if post_select:
        assert n_ele is not None
        heavy_bits = [i for i in heavy_bits if bin(i).count("1") == n_ele]
    if R_max is not None:
        heavy_bits = heavy_bits[:R_max]
    comp_bases_qp = [
        ComputationalBasisState(n_qubits, bits=int(key)) for key in heavy_bits
    ]
    return comp_bases_qp, heavy_bits


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


        hf_bits = 2 ** n_electrons - 1
        comp_bases_qp = [
            ComputationalBasisState(n_qubits, bits=hf_bits ^ sum([1 << i for i in ies]) ^ sum([1 << j for j in js])) for m in range(0, 3) for ies in itertools.combinations(range(0, int(n_qubits/2)), m) for js in itertools.combinations(range(int(n_qubits/2), n_qubits), m)
        ]

        _, hf_energy = diagonalize_effective_ham(
            qp_hamiltonian, [ComputationalBasisState(n_qubits, bits=hf_bits)]
        )
        print("hf energy ", hf_energy)

        vec_qsci, val_qsci = diagonalize_effective_ham(
            qp_hamiltonian, comp_bases_qp
        )

        # print([(b.bits, v) for b, v in zip(comp_bases_qp, vec_qsci)])

        for th in [0, 0.01]:
            th_base = [b for i, b in enumerate(comp_bases_qp) if abs(vec_qsci[i]) > th]
            vec_qscith, val_qscith = diagonalize_effective_ham(
                qp_hamiltonian, th_base
            )
            print(f"{th}: {val_qscith}, {len(th_base)}")


        cisd_coeffs = []
        for coef, base in zip(vec_qscith, th_base):
            if base.bits != hf_bits:
                virtual = [i for i in range(int(n_qubits/2), n_qubits) if (1<<i) & base.bits]
                occupied = [i for i in range(0, int(n_qubits/2)) if (1<<i) & base.bits == 0]
                cisd_coeffs.append((coef, virtual, occupied))

        cisd_coeffs = sorted(cisd_coeffs, key=lambda x: - abs(x[0]))

        post_selection = True
        mps_sampler = challenge_sampling.create_sampler()

        adapt_qsci = ADAPT_QSCI(
            qp_hamiltonian,
            # pool,
            cisd_coeffs = cisd_coeffs,
            n_qubits=n_qubits,
            n_ele_cas=n_electrons,
            sampler=mps_sampler,
            iter_max=100000,
            post_selected=post_selection,
            sampling_shots=10**5,
            # sampling_shots=10**4,
            atol=1e-6,
            final_sampling_shots_coeff=1,
            max_num_converged=2000,
            hf_energy=hf_energy
        )
        res = adapt_qsci.run()
        return res


if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result(seed=1, hamiltonian_directory="../hamiltonian"))
