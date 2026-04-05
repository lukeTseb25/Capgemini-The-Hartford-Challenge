"""
QAOA Hamiltonian Simulation with Pauli Propagation
For portfolio optimization with QUBO formulation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import rustworkx as rx
from pauli_prop import propagate_through_rotation_gates, propagate_through_operator
from qiskit.quantum_info import SparsePauliOp 
from pauli_prop.propagation import RotationGates
from math import cos, sin
from itertools import product

# ============================================================================
# CONSTANT PARAMETERS (Set these at the beginning)
# ============================================================================
LAMBDA = 0.1      # Lagrange multiplier for budget constraint
B = 4.0           # Budget constraint (number of assets to select)
Q = 0.7           # Risk aversion parameter
ASSETS = 8
ASSET_IDS = None

# Noise settings for simulation
NOISE_MODEL = 'bit_flip'  # options: 'none', 'bit_flip'
NOISE_PROB = 0.02     # per-qubit flip probability when using 'bit_flip'

# QAOA parameters
NUM_LAYERS = 100    # Number of QAOA layers (p)
GAMMA = 0.5       # Initial gamma parameter (cost Hamiltonian angle)
BETA = 0.3        # Initial beta parameter (mixer Hamiltonian angle)

# Pauli propagation parameters
MAX_TERMS = 100   # Maximum number of Pauli terms to keep
ABS_CUTOFF = 1e-6 # Absolute coefficient cutoff for truncation
# Bit ordering for mask ↔ bitstring conversions. Will be auto-detected.
BIT_ORDER_MSB_FIRST = True

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_investment_data(asset_ids: List[str] = None):
    """
    Load investment data from CSV files for specified assets.
    
    Args:
        asset_ids: List of asset IDs to load (e.g., ['A001', 'A002', ...])
                  If None, loads first assets.
    
    Returns:
        mu: Expected returns array
        sigma: Covariance matrix
        asset_names: List of asset identifiers
    """
    # Load CSV files
    assets_df = pd.read_csv('investment_dataset_assets.csv')
    covariance_df = pd.read_csv('investment_dataset_covariance.csv')
    
    # If no asset IDs specified, take first the number of ASSETS from start the dataset
    if asset_ids is None:
        asset_ids = assets_df['asset_id'].iloc[:ASSETS].tolist()
    
    # Filter assets
    assets_df = assets_df[assets_df['asset_id'].isin(asset_ids)].sort_values('asset_id')
    
    # Extract expected returns (mu)
    mu = assets_df['exp_return'].values
    
    # Extract covariance matrix for selected assets
    # Assuming covariance matrix has same asset ordering
    asset_indices = [i for i, aid in enumerate(covariance_df.columns) if aid in asset_ids]
    sigma = covariance_df.iloc[asset_indices, asset_indices].values
    
    print(f"Loaded {len(asset_ids)} assets: {asset_ids}")
    print(f"Expected returns: {mu}")
    print(f"Covariance matrix shape: {sigma.shape}")
    
    return mu, sigma, asset_ids

# ============================================================================
# HAMILTONIAN CONSTRUCTION
# ============================================================================

def construct_qubo_hamiltonian(mu: np.ndarray, sigma: np.ndarray, 
                                lambda_param: float, B: float, q: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct QUBO Hamiltonian coefficients.
    
    Args:
        mu: Expected returns array
        sigma: Covariance matrix
        lambda_param: Lagrange multiplier
        B: Budget constraint
        q: Risk aversion parameter
    
    Returns:
        h: Linear coefficients for Ising model
        J: Quadratic coefficients for Ising model
    """
    N = len(mu)
    
    h = np.zeros(N)
    J = np.zeros((N, N))
    
    for i in range(N):
        h[i] = q * sigma[i, i] - mu[i] + lambda_param * (1 - 2*B)
    
    
    for i in range(N):
        for j in range(i+1, N):
            J[i, j] = 2 * q * sigma[i, j] + 2 * lambda_param
            J[j, i] = J[i, j]
    
    return h, J

def hamiltonian_to_pauli_list(h: np.ndarray, J: np.ndarray) -> List[Tuple[str, complex]]:
    """
    Convert Ising Hamiltonian coefficients to Pauli list format for pauli-prop.
    
    Args:
        h: Linear coefficients (Z_i terms)
        J: Quadratic coefficients (Z_i Z_j terms)
    
    Returns:
        List of (pauli_string, coefficient) tuples
    """
    N = len(h)
    pauli_terms = []
    
    # Add Z_i terms
    for i in range(N):
        if abs(h[i]) > 1e-10:
            pauli_str = ['I'] * N
            pauli_str[i] = 'Z'
            pauli_terms.append((''.join(pauli_str), complex(h[i], 0)))
    
    # Add Z_i Z_j terms
    for i in range(N):
        for j in range(i+1, N):
            if abs(J[i, j]) > 1e-10:
                pauli_str = ['I'] * N
                pauli_str[i] = 'Z'
                pauli_str[j] = 'Z'
                pauli_terms.append((''.join(pauli_str), complex(J[i, j], 0)))
    
    return pauli_terms

# ============================================================================
# PAULI PROPAGATION FUNCTIONS
# ============================================================================

def truncate_pauli_terms(pauli_terms: List[Tuple[str, complex]], 
                         max_terms: int = MAX_TERMS,
                         abs_cutoff: float = ABS_CUTOFF) -> List[Tuple[str, complex]]:
    """
    Truncate Pauli terms based on coefficient magnitude.
    
    Args:
        pauli_terms: List of (pauli_string, coefficient)
        max_terms: Maximum number of terms to keep
        abs_cutoff: Absolute coefficient cutoff
    
    Returns:
        Truncated list of Pauli terms
    """
    # Accept either a SparsePauliOp or a list of (pauli_label, coeff)
    if isinstance(pauli_terms, SparsePauliOp):
        sp = pauli_terms
        coeffs = np.array(sp.coeffs, dtype=complex)
        # filter by absolute cutoff
        mask = np.abs(coeffs) > abs_cutoff
        if not np.any(mask):
            # nothing passes cutoff; return the original operator unchanged
            return pauli_terms
        idxs = np.where(mask)[0]
        # sort by magnitude and keep top max_terms
        sorted_idxs = idxs[np.argsort(np.abs(coeffs[idxs]))[::-1]
                           [:max_terms]]
        return SparsePauliOp(sp.paulis[sorted_idxs], coeffs=coeffs[sorted_idxs])

    # list of tuples
    if isinstance(pauli_terms, list):
        lbls = [p for p, c in pauli_terms]
        coeffs = np.array([c for p, c in pauli_terms], dtype=complex)
        mask = np.abs(coeffs) > abs_cutoff
        if not np.any(mask):
            # nothing passes cutoff; return operator built from all terms
            return SparsePauliOp(lbls, coeffs)
        idxs = np.where(mask)[0]
        sorted_idxs = idxs[np.argsort(np.abs(coeffs[idxs]))[::-1][:max_terms]]
        sel_lbls = [lbls[i] for i in sorted_idxs]
        sel_coeffs = coeffs[sorted_idxs]
        return SparsePauliOp(sel_lbls, coeffs=sel_coeffs)

    # tuple returned by pauli-prop: (paulis_array, coeffs_array, ...)
    if isinstance(pauli_terms, tuple):
        try:
            pauli_arr = pauli_terms[0]
            coeffs_arr = np.array(pauli_terms[1], dtype=complex)
            sp = SparsePauliOp(pauli_arr, coeffs=coeffs_arr)
            return truncate_pauli_terms(sp, max_terms, abs_cutoff)
        except Exception:
            pass

    # unknown type -> return input unchanged
    return pauli_terms

def create_rotation_gates_from_hamiltonian(pauli_terms: List[Tuple[str, complex]], 
                                          gamma: float) -> List[Tuple[str, List[int], float]]:
    """
    Create rotation gates from Hamiltonian terms.
    For e^{-i*gamma*H} where H is sum of Pauli terms.
    
    Args:
        pauli_terms: List of (pauli_string, coefficient)
        gamma: Angle parameter
    
    Returns:
        List of (gate_type, qubits, angle) for each term
    """
    gates = []
    
    for pauli_str, coeff in pauli_terms:
        # Find qubits where Pauli is not I
        qubits = [i for i, p in enumerate(pauli_str) if p != 'I']
        
        if not qubits:
            continue
        
        # For Z terms, use RZ gate
        if all(pauli_str[i] == 'Z' for i in qubits):
            # RZ gate angle = 2 * gamma * coeff
            angle = 2 * gamma * coeff
            gates.append(('Z', qubits, float(np.real(angle))))
        
        # For other Pauli combinations, would need more complex decomposition
    
    return gates

def propagate_hamiltonian_through_circuit(initial_pauli_terms: List[Tuple[str, complex]],
                                         circuit_gates: List[Any],
                                         max_terms: int = MAX_TERMS) -> List[Tuple[str, complex]]:
    """
    Propagate Hamiltonian through quantum circuit using pauli-prop.
    
    Args:
        initial_pauli_terms: Initial Pauli terms (H_C)
        circuit_gates: List of gates in the circuit
        max_terms: Maximum terms to keep
    
    Returns:
        Propagated Pauli terms
    """
    # Normalize input to SparsePauliOp
    if isinstance(initial_pauli_terms, SparsePauliOp):
        current_terms = initial_pauli_terms
    else:
        data, coeffs = map(list, zip(*initial_pauli_terms))
        current_terms = SparsePauliOp(data, coeffs=np.array(coeffs, dtype=complex))
    
    # For each gate in the circuit
    for gate in circuit_gates:
        # If current_terms is a tuple (SparsePauliOp, trunc_onenorm), unwrap it
        if isinstance(current_terms, tuple):
            current_terms = current_terms[0]
        if gate['type'] == 'rotation':
            # Propagate through rotation gates
            # gate['gates'] is a list of tuples (pauli_char, qubits, angle)
            gates_list = gate['gates']
            if len(gates_list) == 0:
                continue
            chars, qargs, thetas = map(list, zip(*gates_list))
            # Build boolean matrix for X/Z components expected by RotationGates
            gates_bool = np.array([[c == 'X', c == 'Z'] for c in chars], dtype=bool)
            # qargs should be list of qubit lists; ensure each entry is a sequence
            qargs_seq = [qa if isinstance(qa, (list, tuple)) else [qa] for qa in qargs]
            rot_gates = RotationGates(gates_bool, qargs_seq, thetas)

            res = propagate_through_rotation_gates(
                current_terms,
                rot_gates=rot_gates,
                max_terms=max_terms,
                atol=ABS_CUTOFF,
                frame='s'
            )
            # propagate_through_rotation_gates returns (SparsePauliOp, trunc_onenorm)
            if isinstance(res, tuple):
                current_terms = res[0]
            else:
                current_terms = res
        elif gate['type'] == 'operator':
            # Propagate through general operator
            res = propagate_through_operator(
                current_terms,
                op2=gate['operator'],
                max_terms=max_terms,
                atol=ABS_CUTOFF,
                frame='s'
            )
            if isinstance(res, tuple):
                try:
                    pauli_arr = res[0]
                    coeffs = res[1]
                    current_terms = SparsePauliOp(pauli_arr, coeffs=np.array(coeffs, dtype=complex))
                except Exception:
                    current_terms = res
            else:
                current_terms = res
        
        # Truncate after each step
        current_terms = truncate_pauli_terms(current_terms, max_terms)
    
    return current_terms

# ============================================================================
# EXPECTATION VALUE COMPUTATION
# ============================================================================

def compute_expectation_value_classical(pauli_terms: List[Tuple[str, complex]], 
                                       state_vector: np.ndarray) -> float:
    """
    Compute expectation value of Pauli operator for a given state.
    
    Args:
        pauli_terms: List of (pauli_string, coefficient)
        state_vector: Quantum state vector
    
    Returns:
        Expectation value
    """
    
    # Fallback classical computation
    n_qubits = int(np.log2(len(state_vector)))
    expectation = 0.0
    
    for pauli_str, coeff in pauli_terms:
        # This is simplified - full implementation would need Pauli matrix multiplication
        # For diagonal operators (only Z and I), we can compute quickly
        if all(p in ['Z', 'I'] for p in pauli_str):
            # Compute expectation for Z basis
            value = 0.0
            for i, amp in enumerate(state_vector):
                bitstring = format(i, f'0{n_qubits}b')
                eigenvalue = 1
                for q, p in enumerate(pauli_str):
                    if p == 'Z':
                        eigenvalue *= (1 if bitstring[q] == '0' else -1)
                value += (abs(amp)**2) * eigenvalue
            expectation += np.real(coeff * value)
    
    return expectation


def compute_objective(bitstring: str, mu: np.ndarray, sigma: np.ndarray,
                      q: float = Q, lambda_param: float = LAMBDA, B_param: float = B) -> float:
    """
    Compute the objective function for a given bitstring selection.

    Objective: q * sum_i sum_j (w_i w_j sigma_ij) - sum_i (w_i * mu_i) +
               lambda * (sum_i w_i - B)^2

    Args:
        bitstring: e.g. '0101' length = n_qubits
        mu: expected returns vector
        sigma: covariance matrix
        q: risk aversion parameter
        lambda_param: Lagrange multiplier
        B_param: budget target

    Returns:
        objective value (float)
    """
    w = np.array([int(b) for b in bitstring], dtype=float)
    quad = float(w @ sigma @ w)
    linear = float(w @ mu)
    obj = q * quad - linear + lambda_param * (np.sum(w) - B_param) ** 2
    return obj


def apply_rx_to_state(state: np.ndarray, theta: float, qubit: int, n_qubits: int) -> np.ndarray:
    """Apply single-qubit RX rotation R_X(theta) to `qubit` on the state vector."""
    # R_X(theta) = cos(theta/2) I - i sin(theta/2) X
    c = np.cos(theta / 2)
    s = -1j * np.sin(theta / 2)
    U = np.array([[c, s], [s, c]], dtype=complex)
    state = state.reshape([2] * n_qubits)
    # tensordot U with axis qubit
    new_state = np.tensordot(U, state, axes=([1], [qubit]))
    # move axis 0 to position qubit
    new_state = np.moveaxis(new_state, 0, qubit)
    return new_state.reshape(-1)


def statevector_qaoa(mu: np.ndarray, sigma: np.ndarray, gamma: float, beta: float, num_layers: int):
    """Return statevector after applying QAOA with given gamma/beta using classical cost function."""
    n_qubits = len(mu)
    dim = 2 ** n_qubits
    # initial |+> state
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)

    # Precompute classical cost C(w) for each basis state w in {0,1}^n
    C = np.zeros(dim, dtype=float)
    for idx in range(dim):
        b = format(idx, f'0{n_qubits}b')
        C[idx] = compute_objective(b, mu, sigma, q=Q, lambda_param=LAMBDA, B_param=B)

    for layer in range(num_layers):
        # cost layer: apply phase e^{-i gamma C(w)}
        state = state * np.exp(-1j * gamma * C)

        # mixer layer: apply RX(2*beta) on each qubit
        theta = 2 * beta
        for q_idx in range(n_qubits):
            state = apply_rx_to_state(state, theta, q_idx, n_qubits)

    return state


def compute_expectation_from_state(state: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Compute expectation value of classical objective from state probabilities."""
    n_qubits = len(mu)
    dim = 2 ** n_qubits
    probs = np.abs(state) ** 2
    C = np.zeros(dim, dtype=float)
    for idx in range(dim):
        b = format(idx, f'0{n_qubits}b')
        C[idx] = compute_objective(b, mu, sigma, q=Q, lambda_param=LAMBDA, B_param=B)
    return float(np.sum(probs * C))


def compute_expectation_plus_from_sparsepauli(sp: SparsePauliOp) -> float:
    """Compute <+| sp |+> by summing coefficients of Pauli strings containing only I and X."""
    coeffs = np.array(sp.coeffs, dtype=complex)
    labels = [p.to_label() if hasattr(p, 'to_label') else str(p) for p in sp.paulis]
    total = 0.0
    for lab, c in zip(labels, coeffs):
        if all(ch in ['I', 'X'] for ch in lab):
            total += np.real(c)
    return float(total)


def pauli_dict_from_list(pauli_list: List[Tuple[str, complex]]) -> Dict[str, complex]:
    return {p: complex(c) for p, c in pauli_list}


def pauli_list_from_sparsepauli(sp: SparsePauliOp) -> List[Tuple[str, complex]]:
    labels = [p.to_label() if hasattr(p, 'to_label') else str(p) for p in sp.paulis]
    coeffs = np.array(sp.coeffs, dtype=complex)
    return [(lab, coeffs[i]) for i, lab in enumerate(labels)]


def sparsepauli_from_pauli_dict(pauli_dict: Dict[str, complex]) -> SparsePauliOp:
    labels = list(pauli_dict.keys())
    coeffs = np.array([pauli_dict[l] for l in labels], dtype=complex)
    return SparsePauliOp(labels, coeffs=coeffs)


def conjugate_pauli_by_rx_single(pauli_str: str, qubit: int, theta: float) -> Dict[str, complex]:
    """Return mapping of Pauli strings after conjugation by RX(theta) on `qubit`.

    Compute coefficients exactly by working with 2x2 matrices on the target qubit:
        a_k = 0.5 * Tr(P_k^† U^† P_q U)
    where P_k in {I,X,Y,Z}.
    This captures phases and is robust for multi-qubit Pauli strings.
    """
    # single-qubit Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

    # Build single-qubit RX operator consistent with apply_rx_to_state (theta is full angle)
    c = np.cos(theta / 2)
    s = -1j * np.sin(theta / 2)
    U = np.array([[c, s], [s, c]], dtype=complex)

    target = pauli_str[qubit]
    P_target = pauli_map.get(target, I)

    # compute U^† P_target U
    conj = U.conj().T @ P_target @ U

    out = {}
    # express conj as linear combination of I,X,Y,Z using Hilbert-Schmidt inner product
    for lab, Pmat in pauli_map.items():
        coeff = 0.5 * np.trace(Pmat.conj().T @ conj)
        if abs(coeff) > 1e-14:
            new_label = pauli_str[:qubit] + lab + pauli_str[qubit+1:]
            out[new_label] = complex(coeff)

    # If nothing significant, return original
    if not out:
        return {pauli_str: 1.0}
    return out


def propagate_pauli_terms_via_rx(initial_pauli_terms: List[Tuple[str, complex]],
                                  circuit_gates: List[Any],
                                  max_terms: int = MAX_TERMS,
                                  abs_cutoff: float = ABS_CUTOFF) -> SparsePauliOp:
    """Propagate Pauli operator H through the circuit in Heisenberg picture using
    conjugation by RX mixers. Returns a SparsePauliOp representing H' = U^† H U.

    Note: cost rotations (Z-based) commute with H (which is Z-only initially), so
    we skip explicit conjugation by cost gates.
    """
    # start from dict
    pauli_dict = pauli_dict_from_list(initial_pauli_terms)

    # Flatten gate list (time-ordered) then conjugate in reverse time order
    flat_gates = []
    for gate in circuit_gates:
        if gate['type'] != 'rotation':
            continue
        for item in gate['gates']:
            flat_gates.append(item)

    # process gates in reverse (Heisenberg conjugation)
    paulis_single = {'I': np.array([[1,0],[0,1]],dtype=complex),
                     'X': np.array([[0,1],[1,0]],dtype=complex),
                     'Y': np.array([[0,-1j],[1j,0]],dtype=complex),
                     'Z': np.array([[1,0],[0,-1]],dtype=complex)}

    for char, qlist, angle in reversed(flat_gates):
        if char == 'X':
            for q in qlist:
                new_dict = {}
                for pstr, coeff in pauli_dict.items():
                    mapping = conjugate_pauli_by_rx_single(pstr, q, angle)
                    for p2, factor in mapping.items():
                        val = coeff * factor
                        new_dict[p2] = new_dict.get(p2, 0) + val

                items = [(p, c) for p, c in new_dict.items() if abs(c) > abs_cutoff]
                if len(items) == 0:
                    pauli_dict = {}
                else:
                    sp = truncate_pauli_terms(items, max_terms, abs_cutoff)
                    lst = pauli_list_from_sparsepauli(sp)
                    pauli_dict = {p: c for p, c in lst}

        elif char == 'Z':
            k = len(qlist)
            # build P_sub = Z ⊗ Z ⊗ ... on the k qubits
            P_sub = paulis_single['Z']
            for _ in range(k-1):
                P_sub = np.kron(P_sub, paulis_single['Z'])

            a = angle / 2.0
            Id_sub = np.eye(2**k, dtype=complex)
            U_sub = np.cos(a) * Id_sub - 1j * np.sin(a) * P_sub

            new_dict = {}
            for pstr, coeff in pauli_dict.items():
                sub_label = ''.join([pstr[q] for q in qlist])
                P_target = paulis_single[sub_label[0]]
                for ch in sub_label[1:]:
                    P_target = np.kron(P_target, paulis_single[ch])

                conj = U_sub.conj().T @ P_target @ U_sub

                for comb in product(['I','X','Y','Z'], repeat=k):
                    Pcomb = paulis_single[comb[0]]
                    for ch in comb[1:]:
                        Pcomb = np.kron(Pcomb, paulis_single[ch])
                    coeff_c = 0.5 * np.trace(Pcomb.conj().T @ conj)
                    if abs(coeff_c) > 1e-14:
                        new_label = list(pstr)
                        for ii, q in enumerate(qlist):
                            new_label[q] = comb[ii]
                        new_label = ''.join(new_label)
                        new_dict[new_label] = new_dict.get(new_label, 0) + coeff * coeff_c

            items = [(p, c) for p, c in new_dict.items() if abs(c) > abs_cutoff]
            if len(items) == 0:
                pauli_dict = {}
            else:
                sp = truncate_pauli_terms(items, max_terms, abs_cutoff)
                lst = pauli_list_from_sparsepauli(sp)
                pauli_dict = {p: c for p, c in lst}

        else:
            # unsupported axis
            continue

    # convert final dict to SparsePauliOp
    if len(pauli_dict) == 0:
        return SparsePauliOp(['I' * len(initial_pauli_terms[0][0])], coeffs=np.array([0.0], dtype=complex))
    return sparsepauli_from_pauli_dict(pauli_dict)


def build_pauli_diagonal_from_C(mu: np.ndarray, sigma: np.ndarray) -> List[Tuple[str, complex]]:
    """Build diagonal Pauli operator H = sum_w C(w) |w><w| expressed as Z/I Pauli strings.

    Returns list of (pauli_str, coeff) where pauli_str contains only 'I' or 'Z'.
    """
    n_qubits = len(mu)
    dim = 2 ** n_qubits
    Cvals = np.zeros(dim, dtype=float)
    for idx in range(dim):
        b = format(idx, f'0{n_qubits}b')
        Cvals[idx] = compute_objective(b, mu, sigma, q=Q, lambda_param=LAMBDA, B_param=B)

    pauli_terms = []
    # For each Z-mask (which defines a Pauli string of Z/I)
    for mask in range(dim):
        pauli_str = ''.join(['Z' if ((mask >> (n_qubits-1-i)) & 1) else 'I' for i in range(n_qubits)])
        # compute coefficient = (1/2^n) * sum_w C(w) * (-1)^{dot(w, mask_bits)}
        # where mask bit ordering matches bitstring ordering used above
        s = 0.0
        for w in range(dim):
            # compute parity of overlap between w and mask
            overlap = bin(w & mask).count('1')
            s += Cvals[w] * ((-1) ** overlap)
        coeff = s / float(dim)
        if abs(coeff) > 1e-15:
            pauli_terms.append((pauli_str, complex(coeff)))

    return pauli_terms


def build_pauli_diagonal_from_hJ(h: np.ndarray, J: np.ndarray) -> List[Tuple[str, complex]]:
    """Build diagonal Pauli operator H from Ising coefficients h and J.

    H(x) = sum_{i<j} J[i,j] s_i s_j + sum_i h[i] s_i, where s_i = 1 - 2*x_i.
    Returns Pauli expansion (Z/I strings) of H.
    """
    n_qubits = len(h)
    dim = 2 ** n_qubits
    Cvals = np.zeros(dim, dtype=float)
    for idx in range(dim):
        bits = format(idx, f'0{n_qubits}b')
        s = np.array([1 - 2 * int(b) for b in bits], dtype=float)
        quad = 0.0
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                quad += J[i, j] * s[i] * s[j]
        lin = float(np.dot(h, s))
        Cvals[idx] = quad + lin

    pauli_terms = []
    # For each Z-mask compute coefficient via Walsh-Hadamard transform
    for mask in range(dim):
        # construct pauli string for mask according to global bit order
        if BIT_ORDER_MSB_FIRST:
            pauli_str = ''.join(['Z' if ((mask >> (n_qubits - 1 - i)) & 1) else 'I' for i in range(n_qubits)])
        else:
            pauli_str = ''.join(['Z' if ((mask >> i) & 1) else 'I' for i in range(n_qubits)])
        s = 0.0
        for w in range(dim):
            overlap = bin(w & mask).count('1')
            s += Cvals[w] * ((-1) ** overlap)
        coeff = s / float(dim)
        if abs(coeff) > 1e-15:
            pauli_terms.append((pauli_str, complex(coeff)))

    return pauli_terms


def determine_bit_ordering(h: np.ndarray, J: np.ndarray) -> bool:
    """Determine whether MSB-first or LSB-first mask ordering matches state-vector.

    Returns True if MSB-first should be used, False for LSB-first.
    """
    # small test with n=2 (if workspace uses more qubits, we still use first two)
    n = len(h)
    if n < 2:
        return True

    # pick small gamma/beta
    gamma = 0.5
    beta = 0.3
    # compute state-vector expectation
    state = statevector_qaoa(h_to_mu(h, J) if 'h_to_mu' in globals() else np.zeros(n), np.zeros((n,n)), gamma, beta, 1)
    # Instead, build state more directly using h/J: compute C(w) from h/J
    # Build Cvals
    dim = 2 ** n
    Cvals = np.zeros(dim)
    for idx in range(dim):
        bits = format(idx, f'0{n}b')
        s = np.array([1 - 2 * int(b) for b in bits], dtype=float)
        quad = 0.0
        for i in range(n):
            for j in range(i+1, n):
                quad += J[i, j] * s[i] * s[j]
        lin = float(np.dot(h, s))
        Cvals[idx] = quad + lin
    # produce state via cost and RX mixer
    sv = np.ones(dim, dtype=complex) / np.sqrt(dim)
    sv = sv * np.exp(-1j * gamma * Cvals)
    for q in range(n):
        sv = apply_rx_to_state(sv, 2*beta, q, n)
    sv_exp = float(np.sum(np.abs(sv)**2 * Cvals))

    # try both orderings
    diffs = {}
    for msb in [True, False]:
        global BIT_ORDER_MSB_FIRST
        BIT_ORDER_MSB_FIRST = msb
        pauli = build_pauli_diagonal_from_hJ(h, J)
        propagated = propagate_pauli_terms_via_rx(pauli, [{'type':'rotation','gates':[('X',[0],2*beta)]}], MAX_TERMS, ABS_CUTOFF)
        exp_pauli = compute_expectation_plus_from_sparsepauli(propagated)
        diffs[msb] = abs(exp_pauli - sv_exp)

    # choose ordering with smaller difference
    return min(diffs, key=diffs.get)


def optimize_gamma_beta(pauli_terms: List[Tuple[str, complex]], n_qubits: int, num_layers: int,
                        mu: np.ndarray, sigma: np.ndarray,
                        gamma_steps: int = 7, beta_steps: int = 7,
                        shots: int = 500,
                        noise_model: str = 'none', noise_prob: float = 0.0) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Coarse grid search to find gamma and beta (scalars used for all layers)
    that maximize the maximum measurement probability (i.e., concentrate
    the state onto a single bitstring).

    Returns: (best_max_prob, best_gamma, best_beta, results_dict)
    """
    gammas = np.linspace(0.0, np.pi, gamma_steps)
    betas = np.linspace(0.0, np.pi/2, beta_steps)

    # We'll optimize the QAOA energy expectation computed from a state-vector simulator
    best_val = float('inf')
    best_gamma = GAMMA
    best_beta = BETA
    best_results = {}

    for gamma in gammas:
        for beta in betas:
            # use state-vector simulator to compute exact expectation value
            state = statevector_qaoa(mu, sigma, gamma, beta, num_layers)
            expval = compute_expectation_from_state(state, mu, sigma)

            # lower is better (we minimize the cost expectation)
            if expval < best_val:
                best_val = expval
                best_gamma = gamma
                best_beta = beta
                # sample distribution for reporting if shots provided else exact probs
                if shots and shots > 0:
                    probs = np.abs(state) ** 2
                    counts = np.random.multinomial(shots, probs)
                    best_results = {format(i, f'0{n_qubits}b'): float(c) / shots for i, c in enumerate(counts) if c > 0}
                else:
                    best_results = {format(i, f'0{n_qubits}b'): float(p) for i, p in enumerate(np.abs(state) ** 2)}

    return best_val, best_gamma, best_beta, best_results

def simulate(pauli_terms: List[Tuple[str, complex]], 
                         n_qubits: int, 
                         shots: int = 1000,
                         noise_model: str = 'none',
                         noise_prob: float = 0.0) -> Dict[str, float]:
    """
    Args:
        pauli_terms: List of (pauli_string, coefficient)
        n_qubits: Number of qubits
        shots: Number of measurements
    
    Returns:
        Dictionary of measurement outcomes and probabilities
    """
    # Normalize input: accept SparsePauliOp, list of tuples, or similar
    terms: List[Tuple[str, complex]] = []

    # If it's a Qiskit SparsePauliOp
    try:
        if isinstance(pauli_terms, SparsePauliOp):
            for p, c in zip(pauli_terms.paulis, pauli_terms.coeffs):
                # Pauli objects have a to_label() method
                pauli_label = p.to_label() if hasattr(p, 'to_label') else str(p)
                terms.append((pauli_label, complex(c)))
        else:
            # Try to coerce iterable of pairs
            for item in pauli_terms:
                if isinstance(item, tuple) and len(item) == 2:
                    terms.append((str(item[0]), complex(item[1])))
    except Exception:
        # Fallback: treat as empty operator
        terms = []

    # Pre-filter: keep only diagonal (I/Z) terms for computational-basis energies
    diag_terms = [(p, c) for p, c in terms if all(ch in ['I', 'Z'] for ch in p)]

    # If there are no diagonal terms, return uniform distribution
    dim = 2 ** n_qubits
    if len(diag_terms) == 0:
        probs = np.ones(dim) / dim
    else:
        # Compute energy for each computational basis state
        energies = np.zeros(dim, dtype=float)
        for idx in range(dim):
            bitstr = format(idx, f'0{n_qubits}b')
            energy = 0.0
            for pstr, coeff in diag_terms:
                eigen = 1
                for q, ch in enumerate(pstr):
                    if ch == 'Z':
                        # Z eigenvalue is +1 for '0', -1 for '1'
                        eigen *= (1 if bitstr[q] == '0' else -1)
                energy += np.real(coeff) * eigen
            energies[idx] = energy

        # Convert energies to scores (higher score -> higher probability).
        # Use negative energy so lower energy states get higher weight.
        scores = -energies

        # Stabilize and scale: use temperature proportional to score range
        score_range = np.ptp(scores)
        if score_range <= 0:
            probs = np.ones(dim) / dim
        else:
            temp = score_range
            exps = np.exp((scores - np.max(scores)) / (temp + 1e-12))
            probs = exps / np.sum(exps)

    # Return exact probabilities if shots not provided
    if shots is None or shots <= 0:
        return {format(i, f'0{n_qubits}b'): float(probs[i]) for i in range(dim)}

    # If no noise, use multinomial sampling for efficiency
    if noise_model == 'none' or noise_prob <= 0.0:
        counts = np.random.multinomial(shots, probs)
        result = {format(i, f'0{n_qubits}b'): float(c) / float(shots)
                  for i, c in enumerate(counts) if c > 0}
        return result

    # If noise is enabled, sample shots individually and apply per-qubit bit-flip noise
    sampled_indices = np.random.choice(dim, size=shots, p=probs)
    noisy_counts = {}
    for idx in sampled_indices:
        bitstr = list(format(int(idx), f'0{n_qubits}b'))
        # apply bit-flip noise per qubit
        for q in range(n_qubits):
            if np.random.random() < noise_prob:
                bitstr[q] = '1' if bitstr[q] == '0' else '0'
        noisy = ''.join(bitstr)
        noisy_counts[noisy] = noisy_counts.get(noisy, 0) + 1

    result = {bs: cnt / shots for bs, cnt in noisy_counts.items()}
    return result

# ============================================================================
# MAIN EXECUTION
# ============================================================================


def run_pauli_state_diagnostic(h: np.ndarray, J: np.ndarray, gamma: float, beta: float, num_layers: int = 1, n_diag: int = 2):
    """Run a focused diagnostic comparing state-vector expectation vs native Pauli propagation for small n_diag qubits.

    Prints per-term contributions and the two expectation values to help identify ordering/conjugation mismatches.
    """
    n = min(n_diag, len(h))
    h_small = h[:n].copy()
    J_small = J[:n, :n].copy()

    # Build classical cost values C(w) for small system
    dim = 2 ** n
    Cvals = np.zeros(dim, dtype=float)
    for idx in range(dim):
        bits = format(idx, f'0{n}b')
        s = np.array([1 - 2 * int(b) for b in bits], dtype=float)
        quad = 0.0
        for i in range(n):
            for j in range(i+1, n):
                quad += J_small[i, j] * s[i] * s[j]
        lin = float(np.dot(h_small, s))
        Cvals[idx] = quad + lin

    # Build state-vector via cost-phase then RX mixers (as in statevector_qaoa)
    sv = np.ones(dim, dtype=complex) / np.sqrt(dim)
    sv = sv * np.exp(-1j * gamma * Cvals)
    for layer in range(num_layers):
        for q in range(n):
            sv = apply_rx_to_state(sv, 2 * beta, q, n)

    exp_sv = float(np.sum(np.abs(sv) ** 2 * Cvals))

    # Build diagonal Pauli operator from h/J for small system
    diag_pauli = build_pauli_diagonal_from_hJ(h_small, J_small)

    # Build a small QAOA circuit containing only mixer RXs (cost is diagonal)
    qaoa_small = []
    for layer in range(num_layers):
        qaoa_small.append({'type': 'rotation', 'gates': [('X', [i], 2 * beta) for i in range(n)], 'layer': layer, 'type_name': 'mixer'})

    # Propagate via native Pauli propagation
    propagated = propagate_pauli_terms_via_rx(diag_pauli, qaoa_small, MAX_TERMS, ABS_CUTOFF)
    exp_pauli = compute_expectation_plus_from_sparsepauli(propagated)

    print('\n--- Diagnostic: Pauli vs State-vector (n=%d, layers=%d) ---' % (n, num_layers))
    print('State-vector expectation:', exp_sv)
    print('Pauli-prop (native) expectation on |+>:', exp_pauli)

    # Show initial diagonal terms and propagated top terms
    print('\nInitial diagonal Pauli terms:')
    for p, c in diag_pauli[: min(8, len(diag_pauli))]:
        print('  ', p, c)

    print('\nPropagated Pauli terms (top 16):')
    try:
        lst = pauli_list_from_sparsepauli(propagated)
        for p, c in lst[:16]:
            print('  ', p, c)
    except Exception:
        print('  (failed to list propagated terms)')

    print('--- end diagnostic ---\n')


def run_exact_heisenberg_diagnostic(h: np.ndarray, J: np.ndarray, gamma: float, beta: float, num_layers: int = 1, n_diag: int = 2):
    """Exact-matrix Heisenberg diagnostic: compute U^† H U exactly (no truncation)

    Decompose the resulting operator into Pauli basis and compare with the
    sparse/native propagation result to locate mismatches.
    """
    n = min(n_diag, len(h))
    h_small = h[:n].copy()
    J_small = J[:n, :n].copy()

    dim = 2 ** n
    # classical cost diagonal
    Cvals = np.zeros(dim, dtype=float)
    for idx in range(dim):
        bits = format(idx, f'0{n}b')
        s = np.array([1 - 2 * int(b) for b in bits], dtype=float)
        quad = 0.0
        for i in range(n):
            for j in range(i+1, n):
                quad += J_small[i, j] * s[i] * s[j]
        lin = float(np.dot(h_small, s))
        Cvals[idx] = quad + lin

    # H as diagonal matrix
    H = np.diag(Cvals)

    # Build full mixer unitary for one layer: tensor product of RX(2*beta)
    def single_rx(theta):
        c = np.cos(theta / 2)
        s = -1j * np.sin(theta / 2)
        return np.array([[c, s], [s, c]], dtype=complex)

    # Build total mixer M_total = M_L ... M_1 (product of per-layer mixer unitaries)
    M_total = np.eye(dim, dtype=complex)
    for layer in range(num_layers):
        theta = 2 * beta
        # full mixer = kron(RX, RX, ...)
        M = single_rx(theta)
        full_mixer = M
        for _ in range(n - 1):
            full_mixer = np.kron(full_mixer, M)
        # cost diagonal commutes with H; still follow original sequence: U_layer = full_mixer @ diag_cost
        diag_cost = np.diag(np.exp(-1j * gamma * Cvals))
        U_layer = full_mixer @ diag_cost
        M_total = U_layer @ M_total

    # Exact propagated H' = U_total^† H U_total
    U_total = M_total
    Hprop = U_total.conj().T @ H @ U_total

    # plus state
    plus = np.ones(dim, dtype=complex) / np.sqrt(dim)
    exp_exact = float(np.real(np.vdot(plus, Hprop @ plus)))

    # Decompose Hprop into Pauli basis
    paulis = {'I': np.array([[1,0],[0,1]],dtype=complex),
              'X': np.array([[0,1],[1,0]],dtype=complex),
              'Y': np.array([[0,-1j],[1j,0]],dtype=complex),
              'Z': np.array([[1,0],[0,-1]],dtype=complex)}

    pauli_coeffs = []
    for mask in range(4 ** n):
        # MSB-first base-4 digits so label[0] refers to qubit 0 (leftmost)
        digits = [ (mask // (4 ** (n - 1 - i))) % 4 for i in range(n) ]
        lab = [ ['I','X','Y','Z'][d] for d in digits ]
        # build tensor product in qubit order: P0 ⊗ P1 ⊗ ...
        P = paulis[lab[0]]
        for k in range(1, n):
            P = np.kron(P, paulis[lab[k]])
        label = ''.join(lab)
        coeff = np.trace(P @ Hprop) / float(dim)
        if abs(coeff) > 1e-12:
            pauli_coeffs.append((label, complex(coeff)))

    # Compute sparse/native propagated operator for comparison
    diag_pauli = build_pauli_diagonal_from_hJ(h_small, J_small)
    # Build small qaoa circuit of mixers only (propagate_pauli_terms_via_rx skips cost gates)
    qaoa_small = []
    for layer in range(num_layers):
        qaoa_small.append({'type':'rotation','gates':[('X',[i],2*beta) for i in range(n)],'layer':layer,'type_name':'mixer'})
    propagated = propagate_pauli_terms_via_rx(diag_pauli, qaoa_small, max_terms=4**n, abs_cutoff=0.0)

    # Decompose propagated SparsePauliOp into list
    try:
        propagated_list = pauli_list_from_sparsepauli(propagated)
    except Exception:
        propagated_list = list(diag_pauli)

    print('\n--- Exact Heisenberg Diagnostic (n=%d, layers=%d) ---' % (n, num_layers))
    print('State-vector expectation (small):', float(np.sum((np.abs((U_total @ plus))**2) * Cvals)))
    print('Exact <+|U^† H U|+>:', exp_exact)
    print('\nTop exact Pauli coefficients:')
    for lab, c in sorted(pauli_coeffs, key=lambda x: -abs(x[1]))[:16]:
        print('  ', lab, c)

    print('\nTop propagated (native) coefficients:')
    for lab, c in sorted(propagated_list, key=lambda x: -abs(x[1]))[:16]:
        print('  ', lab, c)

    print('--- end exact diagnostic ---\n')


def test_per_gate_conjugation(h: np.ndarray, J: np.ndarray, gamma: float, beta: float, n_diag: int = 3, num_layers: int = 1):
    """Test each elementary gate conjugation (symbolic vs exact) for small system.

    Prints the first mismatch found with details.
    """
    n = min(n_diag, len(h))
    h_small = h[:n].copy()
    J_small = J[:n, :n].copy()

    # initial pauli dict
    diag_pauli = build_pauli_diagonal_from_hJ(h_small, J_small)
    pauli_dict = pauli_dict_from_list(diag_pauli)

    # build small qaoa circuit (cost then mixer) for one layer
    cost_gates = create_rotation_gates_from_hamiltonian(diag_pauli, gamma)
    mixer_gates = [('X', [i], 2 * beta) for i in range(n)]
    circuit = [{'type':'rotation','gates':cost_gates,'layer':0,'type_name':'cost'},
               {'type':'rotation','gates':mixer_gates,'layer':0,'type_name':'mixer'}]

    # flatten gates in time order
    flat = []
    for g in circuit:
        if g['type'] != 'rotation':
            continue
        for item in g['gates']:
            flat.append(item)

    paulis_single = {'I': np.array([[1,0],[0,1]],dtype=complex),
                     'X': np.array([[0,1],[1,0]],dtype=complex),
                     'Y': np.array([[0,-1j],[1j,0]],dtype=complex),
                     'Z': np.array([[1,0],[0,-1]],dtype=complex)}

    def full_pauli_matrix(label: str):
        M = paulis_single[label[0]]
        for ch in label[1:]:
            M = np.kron(M, paulis_single[ch])
        return M

    # for each elementary gate, compare symbolic conjugation vs exact
    for gate in flat:
        char, qlist, angle = gate
        for pstr, coeff in list(pauli_dict.items())[:50]:
            # build full Pauli matrix for pstr
            P_full = full_pauli_matrix(pstr)

            # exact U for this gate on full system
            if char == 'X':
                # build full U as kron of RX on target qubits, identity elsewhere
                U = 1
                for q in range(n):
                    if [q] in [qlist]:
                        # single qubit RX
                        c = np.cos(angle / 2)
                        s = -1j * np.sin(angle / 2)
                        Uq = np.array([[c, s], [s, c]], dtype=complex)
                    else:
                        Uq = np.eye(2, dtype=complex)
                    U = np.kron(U, Uq) if not np.isscalar(U) else Uq
                conj_exact = U.conj().T @ P_full @ U
                # symbolic mapping
                mapping = conjugate_pauli_by_rx_single(pstr, qlist[0], angle)
                # build symbolic full matrix
                sym = np.zeros_like(P_full, dtype=complex)
                for lab, f in mapping.items():
                    sym += f * full_pauli_matrix(lab)

            elif char == 'Z':
                # P_sub on qlist
                k = len(qlist)
                P_sub = paulis_single['Z']
                for _ in range(k-1):
                    P_sub = np.kron(P_sub, paulis_single['Z'])
                a = angle / 2.0
                Id_sub = np.eye(2**k, dtype=complex)
                U_sub = np.cos(a) * Id_sub - 1j * np.sin(a) * P_sub

                # lift U_sub to full system by kron-ing identities
                # build full U by tensoring identities and placing U_sub on qlist order
                # easier: construct full U by acting on basis: construct P_full_sub for each basis
                # For small n, build full P_z = kron over qubits with Z on qlist positions
                P_full_sub = np.eye(1, dtype=complex)
                for q in range(n):
                    if q in qlist:
                        P_full_sub = np.kron(P_full_sub, paulis_single['Z'])
                    else:
                        P_full_sub = np.kron(P_full_sub, np.eye(2, dtype=complex))
                # full U = cos(a) I - i sin(a) P_full_sub
                U = np.cos(a) * np.eye(2**n, dtype=complex) - 1j * np.sin(a) * P_full_sub
                conj_exact = U.conj().T @ P_full @ U

                # symbolic via our conjugation routine
                # for multi-qubit Z, use propagate logic for a single gate
                # reuse code path by calling propagate_pauli_terms_via_rx on single gate
                try:
                    propagated = propagate_pauli_terms_via_rx([(pstr, coeff)], [{'type':'rotation','gates':[gate]}], max_terms=4**n, abs_cutoff=0.0)
                    lst = pauli_list_from_sparsepauli(propagated)
                    sym = np.zeros_like(P_full, dtype=complex)
                    for lab, c in lst:
                        sym += c * full_pauli_matrix(lab)
                except Exception as e:
                    print('propagate_pauli_terms_via_rx failed during test:', e)
                    return
            else:
                continue

            # compare matrices (scale exact conj by original coefficient)
            diff = np.linalg.norm(coeff * conj_exact - sym)
            if diff > 1e-6:
                print('\nMismatch for gate', gate, 'on Pauli', pstr)
                print('Norm difference:', diff)
                print('Exact matrix (sample entries):', conj_exact.flatten()[:8])
                print('Sym matrix (sample entries):', sym.flatten()[:8])
                return

    print('Per-gate conjugation test: all gates match exact conjugation (within tol).')

def main():
    """Main execution function."""
    print("=" * 60)
    print("QAOA Hamiltonian Simulation with Pauli Propagation")
    print("=" * 60)
    print(f"Parameters: lambda={LAMBDA}, B={B}, q={Q}")
    print(f"QAOA layers: {NUM_LAYERS}")
    print(f"Pauli propagation: max_terms={MAX_TERMS}, cutoff={ABS_CUTOFF}")
    print()
    
    # Step 1: Load data
    print("Step 1: Loading investment data...")
    mu, sigma, asset_ids = load_investment_data(asset_ids=ASSET_IDS)
    n_qubits = len(mu)
    print(f"Number of assets/qubits: {n_qubits}")
    print()
    
    # Step 2: Construct Hamiltonian
    print("Step 2: Constructing QUBO/Ising Hamiltonian...")
    h_coeffs, J_coeffs = construct_qubo_hamiltonian(mu, sigma, LAMBDA, B, Q)
    print(f"Linear coefficients (h): {h_coeffs}")
    print(f"Number of quadratic terms: {np.sum(np.abs(J_coeffs) > 1e-10)}")
    print()
    
    # Step 3: Convert to Pauli terms
    print("Step 3: Converting to Pauli representation...")
    pauli_terms = hamiltonian_to_pauli_list(h_coeffs, J_coeffs)
    print(f"Initial Pauli operator has {len(pauli_terms)} terms")
    
    # Show some terms
    print("\nSample Pauli terms:")
    for i, (p_str, coeff) in enumerate(pauli_terms[:5]):
        print(f"  {i+1}. {p_str}: {coeff}")
    if len(pauli_terms) > 5:
        print(f"  ... and {len(pauli_terms)-5} more terms")
    print()
    
    # Step 4: Tune or create circuit gates for propagation
    print("Step 4: Tuning/creating QAOA circuit for propagation...")

    # Perform a coarse grid search to tune scalar gamma and beta (applied to all layers)
    # Use exact probability evaluation during tuning to get deterministic peak probabilities
    best_max, best_gamma, best_beta, best_results = optimize_gamma_beta(
        pauli_terms, n_qubits, NUM_LAYERS, mu, sigma, gamma_steps=7, beta_steps=7, shots=None,
        noise_model=NOISE_MODEL, noise_prob=NOISE_PROB
    )

    print(f"Tuning complete. Best expectation {best_max:.6f} at gamma={best_gamma:.4f}, beta={best_beta:.4f}")

    # Create rotation gates for cost Hamiltonian exponentiation using best_gamma
    cost_gates = create_rotation_gates_from_hamiltonian(pauli_terms, best_gamma)
    mixer_gates_template = [('X', [i], 2 * best_beta) for i in range(n_qubits)]

    # Build full QAOA circuit using tuned parameters
    qaoa_circuit = []
    for layer in range(NUM_LAYERS):
        qaoa_circuit.append({
            'type': 'rotation',
            'gates': cost_gates,
            'layer': layer,
            'type_name': 'cost'
        })
        qaoa_circuit.append({
            'type': 'rotation',
            'gates': mixer_gates_template,
            'layer': layer,
            'type_name': 'mixer'
        })

    print(f"Built QAOA circuit with tuned gamma/beta across {NUM_LAYERS} layers")
    print()
    # Diagnostics disabled to reduce runtime. Uncomment below lines to enable.
    # try:
    #     run_pauli_state_diagnostic(h_coeffs, J_coeffs, best_gamma, best_beta, num_layers=min(2, NUM_LAYERS), n_diag=min(3, n_qubits))
    # except Exception as e:
    #     print('Diagnostic failed:', e)
    # try:
    #     run_exact_heisenberg_diagnostic(h_coeffs, J_coeffs, best_gamma, best_beta, num_layers=min(2, NUM_LAYERS), n_diag=min(3, n_qubits))
    # except Exception as e:
    #     print('Exact diagnostic failed:', e)
    # Per-gate conjugation test is disabled to save runtime; enable manually if needed.
    # try:
    #     test_per_gate_conjugation(h_coeffs, J_coeffs, best_gamma, best_beta, n_diag=min(3, n_qubits), num_layers=min(2, NUM_LAYERS))
    # except Exception as e:
    #     print('Per-gate test failed:', e)
    # Step 5: Build final state via state-vector QAOA and sample measurements
    print("Step 5: Building QAOA statevector and sampling measurements...")
    # Build final state using state-vector QAOA (this is the definitive simulator for correctness)
    state = statevector_qaoa(mu, sigma, best_gamma, best_beta, NUM_LAYERS)
    expval_state = compute_expectation_from_state(state, mu, sigma)
    print(f"State-vector expectation value: {expval_state:.6f}")

    # Sample measurement outcomes according to state probabilities (apply noise if requested)
    probs = np.abs(state) ** 2
    if NOISE_MODEL == 'none' or NOISE_PROB <= 0.0:
        counts = np.random.multinomial(1000, probs)
        results = {format(i, f'0{n_qubits}b'): float(c) / 1000 for i, c in enumerate(counts) if c > 0}
    else:
        sampled = np.random.choice(2 ** n_qubits, size=1000, p=probs)
        noisy_counts = {}
        for idx in sampled:
            bitstr = list(format(int(idx), f'0{n_qubits}b'))
            for q in range(n_qubits):
                if np.random.random() < NOISE_PROB:
                    bitstr[q] = '1' if bitstr[q] == '0' else '0'
            noisy = ''.join(bitstr)
            noisy_counts[noisy] = noisy_counts.get(noisy, 0) + 1
        results = {bs: cnt / 1000 for bs, cnt in noisy_counts.items()}

    # Pauli-prop native: propagate H through mixers in Heisenberg picture
    try:
        # Build diagonal Pauli operator corresponding exactly to Ising h/J
        diag_pauli = build_pauli_diagonal_from_hJ(h_coeffs, J_coeffs)
        propagated_terms = propagate_pauli_terms_via_rx(diag_pauli, qaoa_circuit, MAX_TERMS, ABS_CUTOFF)
        expval_pauli = compute_expectation_plus_from_sparsepauli(propagated_terms)
        print(f"Pauli-prop (native) expectation on |+>: {expval_pauli:.6f}")
        if abs(expval_pauli - expval_state) > 1e-6:
            print("Warning: pauli-prop expectation differs from state-vector by", expval_pauli - expval_state)
    except Exception as e:
        print("Pauli-prop native propagation failed:", e)

    # Step 7: Display results
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    
    # Sort by probability
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 most probable bitstrings (asset selections):")
    print("-" * 40)
    for i, (bitstring, prob) in enumerate(sorted_results[:10]):
        # Convert bitstring to asset selection
        selected_assets = [asset_ids[j] for j, bit in enumerate(bitstring) if bit == '1']
        num_selected = len(selected_assets)
        selection_str = ', '.join(selected_assets[:5])
        if num_selected > 5:
            selection_str += f"... (+{num_selected-5} more)"
        print(f"{i+1:2d}. |{bitstring}| -> {num_selected:2d} assets: [{selection_str}] (p={prob:.4f})")
    
    # Calculate expected return and risk for best solution
    best_bitstring = sorted_results[0][0] if sorted_results else None
    
    if best_bitstring:
        selected_indices = [i for i, bit in enumerate(best_bitstring) if bit == '1']
        
        if selected_indices:
            expected_return = np.sum(mu[selected_indices])
            portfolio_variance = np.sum(sigma[np.ix_(selected_indices, selected_indices)])
            portfolio_risk = np.sqrt(portfolio_variance)
            
            print("\n" + "=" * 60)
            print("BEST PORTFOLIO SOLUTION")
            print("=" * 60)
            print(f"Selected assets: {[asset_ids[i] for i in selected_indices]}")
            print(f"Number of assets: {len(selected_indices)} (budget: {B})")
            print(f"Expected return: {expected_return:.4f}")
            print(f"Portfolio variance: {portfolio_variance:.4f}")
            print(f"Portfolio risk (std dev): {portfolio_risk:.4f}")
            print(f"Objective (return - {Q}*risk): {expected_return - Q*portfolio_risk:.4f}")
            
            # Check budget constraint
            budget_violation = abs(len(selected_indices) - B)
            if budget_violation > 0.1:
                print(f"Budget violation: selected {len(selected_indices)}, target {B}")
            else:
                print(f"Budget constraint satisfied (selected {len(selected_indices)})")
    
    # --- Additional: compute average objective across all bitstrings and compare top-10 ---
    print("\nComputing objective values for comparison...")
    dim = 2 ** n_qubits
    all_objs = np.zeros(dim, dtype=float)
    for idx in range(dim):
        b = format(idx, f'0{n_qubits}b')
        all_objs[idx] = compute_objective(b, mu, sigma, q=Q, lambda_param=LAMBDA, B_param=B)

    avg_obj = float(np.mean(all_objs))
    min_obj = float(np.min(all_objs))
    min_idx = int(np.argmin(all_objs))
    min_bit = format(min_idx, f'0{n_qubits}b')

    print(f"Average objective over all {dim} bitstrings: {avg_obj:.6f}")
    print(f"Minimum objective over all bitstrings: {min_obj:.6f} (bitstring |{min_bit}|)")

    print("\nTop 10 bitstrings: objective vs average")
    print("-" * 60)
    for i, (bitstring, prob) in enumerate(sorted_results[:10]):
        obj_val = compute_objective(bitstring, mu, sigma, q=Q, lambda_param=LAMBDA, B_param=B)
        diff = obj_val - avg_obj
        print(f"{i+1:2d}. |{bitstring}| -> obj={obj_val:.6f}, diff_vs_avg={diff:+.6f}, p={prob:.4f}")

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()