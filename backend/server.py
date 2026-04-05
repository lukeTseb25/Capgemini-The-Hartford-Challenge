"""
Flask API backend for QAOA Portfolio Optimization.
Wraps qaoa_pauli_prop.py and exposes it as a REST API for the Flutter app.
"""

import sys
import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent directory to path so we can import the QAOA module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qaoa_pauli_prop import (
    load_investment_data,
    construct_qubo_hamiltonian,
    hamiltonian_to_pauli_list,
    optimize_gamma_beta,
    statevector_qaoa,
    compute_objective,
)

app = Flask(__name__)
CORS(app)

# Change working directory to project root so CSV files are found
os.chdir(os.path.join(os.path.dirname(__file__), '..'))


@app.route('/api/assets', methods=['GET'])
def get_assets():
    """Return available asset IDs from the dataset."""
    import pandas as pd
    assets_df = pd.read_csv('investment_dataset_assets.csv')
    assets = assets_df[['asset_id', 'sector', 'exp_return', 'volatility']].to_dict(orient='records')
    return jsonify({'assets': assets})


@app.route('/api/run', methods=['POST'])
def run_simulation():
    """
    Run the QAOA simulation with user-provided parameters.

    Expected JSON body:
    {
        "lambda": 0.1,
        "budget": 4.0,
        "risk_aversion": 0.7,
        "num_assets": 8,
        "num_layers": 10,
        "max_terms": 100,
        "abs_cutoff": 1e-6,
        "noise_model": "none",
        "noise_prob": 0.0,
        "gamma_steps": 7,
        "beta_steps": 7,
        "shots": 1000
    }
    """
    data = request.get_json()

    # Extract parameters with defaults
    lambda_param = float(data.get('lambda', 0.1))
    budget = float(data.get('budget', 4.0))
    risk_aversion = float(data.get('risk_aversion', 0.7))
    num_assets = int(data.get('num_assets', 8))
    num_layers = int(data.get('num_layers', 10))
    noise_model = data.get('noise_model', 'none')
    noise_prob = float(data.get('noise_prob', 0.0))
    gamma_steps = int(data.get('gamma_steps', 7))
    beta_steps = int(data.get('beta_steps', 7))
    shots = int(data.get('shots', 1000))

    try:
        # Step 1: Load data
        import pandas as pd
        assets_df = pd.read_csv('investment_dataset_assets.csv')
        asset_ids = assets_df['asset_id'].iloc[:num_assets].tolist()
        mu, sigma, asset_ids = load_investment_data(asset_ids=asset_ids)
        n_qubits = len(mu)

        # Step 2: Construct Hamiltonian
        h_coeffs, J_coeffs = construct_qubo_hamiltonian(mu, sigma, lambda_param, budget, risk_aversion)

        # Step 3: Convert to Pauli terms
        pauli_terms = hamiltonian_to_pauli_list(h_coeffs, J_coeffs)

        # Step 4: Tune gamma/beta via grid search (using statevector, same as script)
        best_max, best_gamma, best_beta, _ = optimize_gamma_beta(
            pauli_terms, n_qubits, num_layers, mu, sigma,
            gamma_steps=gamma_steps, beta_steps=beta_steps, shots=None,
            noise_model=noise_model, noise_prob=noise_prob,
        )

        # Step 5: Build final statevector with tuned params (same as script's main())
        state = statevector_qaoa(mu, sigma, best_gamma, best_beta, num_layers)

        # Step 6: Sample from the real QAOA probability distribution
        probs = np.abs(state) ** 2
        if noise_model == 'none' or noise_prob <= 0.0:
            counts = np.random.multinomial(shots, probs)
            results = {format(i, f'0{n_qubits}b'): float(c) / shots
                       for i, c in enumerate(counts) if c > 0}
        else:
            sampled = np.random.choice(2 ** n_qubits, size=shots, p=probs)
            noisy_counts: dict = {}
            for idx in sampled:
                bitstr = list(format(int(idx), f'0{n_qubits}b'))
                for qi in range(n_qubits):
                    if np.random.random() < noise_prob:
                        bitstr[qi] = '1' if bitstr[qi] == '0' else '0'
                noisy = ''.join(bitstr)
                noisy_counts[noisy] = noisy_counts.get(noisy, 0) + 1
            results = {bs: cnt / shots for bs, cnt in noisy_counts.items()}

        # Step 8: Build response
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        top_results = []
        for bitstring, prob in sorted_results[:15]:
            selected = [asset_ids[j] for j, bit in enumerate(bitstring) if bit == '1']
            obj_val = compute_objective(bitstring, mu, sigma, q=risk_aversion,
                                       lambda_param=lambda_param, B_param=budget)

            selected_indices = [j for j, bit in enumerate(bitstring) if bit == '1']
            exp_ret = float(np.sum(mu[selected_indices])) if selected_indices else 0.0
            port_var = float(np.sum(sigma[np.ix_(selected_indices, selected_indices)])) if selected_indices else 0.0

            top_results.append({
                'bitstring': bitstring,
                'probability': float(prob),
                'selected_assets': selected,
                'num_selected': len(selected),
                'objective': float(obj_val),
                'expected_return': exp_ret,
                'portfolio_variance': port_var,
                'portfolio_risk': float(np.sqrt(port_var)) if port_var > 0 else 0.0,
            })

        # Compute global stats
        dim = 2 ** n_qubits
        all_objs = [compute_objective(format(idx, f'0{n_qubits}b'), mu, sigma,
                                      q=risk_aversion, lambda_param=lambda_param, B_param=budget)
                    for idx in range(dim)]
        avg_obj = float(np.mean(all_objs))
        min_obj = float(np.min(all_objs))
        min_bit = format(int(np.argmin(all_objs)), f'0{n_qubits}b')

        return jsonify({
            'success': True,
            'parameters': {
                'lambda': lambda_param,
                'budget': budget,
                'risk_aversion': risk_aversion,
                'num_assets': num_assets,
                'num_layers': num_layers,
                'best_gamma': float(best_gamma),
                'best_beta': float(best_beta),
                'best_max_prob': float(best_max),
                'shots': shots,
                'noise_model': noise_model,
                'noise_prob': noise_prob,
            },
            'results': top_results,
            'stats': {
                'avg_objective': avg_obj,
                'min_objective': min_obj,
                'min_bitstring': min_bit,
                'n_qubits': n_qubits,
                'total_bitstrings': dim,
            },
            'asset_ids': asset_ids,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
