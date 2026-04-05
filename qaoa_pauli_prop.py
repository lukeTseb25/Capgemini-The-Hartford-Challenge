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

# ============================================================================
# CONSTANT PARAMETERS (Set these at the beginning)
# ============================================================================
LAMBDA = 0.1      # Lagrange multiplier for budget constraint
B = 4.0           # Budget constraint (number of assets to select)
Q = 0.7           # Risk aversion parameter
ASSETS = 8
ASSET_IDS = None

# Noise settings for simulation
NOISE_MODEL = 'none'  # options: 'none', 'bit_flip'
NOISE_PROB = 0.000     # per-qubit flip probability when using 'bit_flip'

# QAOA parameters
NUM_LAYERS = 10    # Number of QAOA layers (p)
GAMMA = 0.5       # Initial gamma parameter (cost Hamiltonian angle)
BETA = 0.3        # Initial beta parameter (mixer Hamiltonian angle)

# Pauli propagation parameters
MAX_TERMS = 100   # Maximum number of Pauli terms to keep
ABS_CUTOFF = 1e-6 # Absolute coefficient cutoff for truncation

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
    # Filter by absolute cutoff
    # Arrange paulis and coeffs into a list of tuples
    pcs = [(p, c) for p, c in zip(pauli_terms[0].paulis, pauli_terms[0].coeffs)]
    filtered = [(p, c) for p, c in pcs if abs(c) > abs_cutoff]
    
	# Make filtered into an Operator object
    filtered = SparsePauliOp([p for p, c in filtered], coeffs=np.array([c for p, c in filtered], dtype=complex))

    # Sort by magnitude and keep top max_terms
    if len(filtered) > max_terms:
        filtered.sort(key=lambda x: abs(x[1]), reverse=True)
        filtered = filtered[:max_terms]
    
    return filtered

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
    data, coeffs = map(list, zip(*initial_pauli_terms))
    current_terms = SparsePauliOp(data, coeffs=np.array(coeffs, dtype=complex))
    
    # For each gate in the circuit
    for gate in circuit_gates:
        if gate['type'] == 'rotation':
            # Propagate through rotation gates
            
			#Convert to rotation gates
            gates, qargs, thetas = map(list, zip(*gate['gates']))
			
			#Fix gates
            #Make rot_gates.gates a 2d array where each row is a list of 2 booleans for the X and Z components of the gate
            gates = np.array([[True if g == 'X' else False, True if g == 'Z' else False] for g in gates])

            rot_gates = RotationGates(gates, qargs, thetas)
            
            current_terms = propagate_through_rotation_gates(
                current_terms,
                rot_gates=rot_gates,
                max_terms=max_terms,
                atol=ABS_CUTOFF,
                frame='s'
            )
        elif gate['type'] == 'operator':
            # Propagate through general operator
            current_terms = propagate_through_operator(
                current_terms,
                op2=gate['operator'],
                max_terms=max_terms,
                atol=ABS_CUTOFF,
                frame='s'
            )
        
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

    best_max = -1.0
    best_gamma = GAMMA
    best_beta = BETA
    best_results = {}

    for gamma in gammas:
        # Precompute cost gates once per gamma
        cost_gates = create_rotation_gates_from_hamiltonian(pauli_terms, gamma)

        for beta in betas:
            # Build simple circuit using same gamma/beta across layers
            qaoa_circuit = []
            for layer in range(num_layers):
                qaoa_circuit.append({
                    'type': 'rotation',
                    'gates': cost_gates,
                    'layer': layer,
                    'type_name': 'cost'
                })
                mixer_gates = [('X', [i], 2 * beta) for i in range(n_qubits)]
                qaoa_circuit.append({
                    'type': 'rotation',
                    'gates': mixer_gates,
                    'layer': layer,
                    'type_name': 'mixer'
                })

            propagated = propagate_hamiltonian_through_circuit(pauli_terms, qaoa_circuit, MAX_TERMS)
            results = simulate(propagated, n_qubits, shots=shots, noise_model=noise_model, noise_prob=noise_prob)
            if results:
                max_p = max(results.values())
            else:
                max_p = 0.0

            if max_p > best_max:
                best_max = max_p
                best_gamma = gamma
                best_beta = beta
                best_results = results

    return best_max, best_gamma, best_beta, best_results

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

    print(f"Tuning complete. Best max probability {best_max:.4f} at gamma={best_gamma:.4f}, beta={best_beta:.4f}")

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
    
    # Step 5: Propagate through circuit (if pauli-prop available)
    print("Step 5: Propagating Hamiltonian through QAOA circuit...")
    
    propagated_terms = propagate_hamiltonian_through_circuit(
        pauli_terms, qaoa_circuit, MAX_TERMS
    )
    print(f"Propagated operator has {len(propagated_terms)} terms")
    
    # Show truncated terms
    print("\nTop propagated Pauli terms:")
    sorted_terms = sorted(propagated_terms, key=lambda x: abs(x.coeffs), reverse=True)
    for i, item in enumerate(sorted_terms[:10]):
        print(f"  {i+1}. {item.paulis[0]}: {item.coeffs[0]:.6f}")
    if len(sorted_terms) > 10:
        print(f"  ... and {len(sorted_terms)-10} more terms")
    
    print("Step 6: Simulating...")
    # Use sampling for final reporting (1000 shots)
    results = simulate(propagated_terms, n_qubits, shots=1000, noise_model=NOISE_MODEL, noise_prob=NOISE_PROB)

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