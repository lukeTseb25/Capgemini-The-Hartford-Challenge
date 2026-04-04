from bloqade import squin
import cirq
from bloqade import cirq_utils
import networkx as nx

def build_qaoa_circuit(
    graph: nx.Graph, gamma: list[float], beta: list[float]
) -> cirq.Circuit:
    """Build a QAOA circuit for MaxCut on the given graph using squin"""
    n = len(graph.nodes)
    assert len(gamma) == len(beta), "Length of gamma and beta must be equal"

    # Prepare edge list for squin kernel
    edges = list(graph.edges)

    @squin.kernel
    def qaoa_kernel():
        q = squin.qalloc(n)

        # Initial Hadamard layer
        for i in range(n):
            squin.h(q[i])

        # QAOA layers
        for layer in range(len(gamma)):
            # Cost Hamiltonian: ZZ rotation for each edge
            # Using decomposition: exp(-i*gamma/2*Z⊗Z) = H → CZ → Rx(gamma) → CZ → H
            for edge in edges:
                u = edge[0]
                v = edge[1]
                squin.h(q[v])
                squin.cz(q[u], q[v])
                squin.rx(gamma[layer], q[v])
                squin.cz(q[u], q[v])
                squin.h(q[v])

            # Mixer Hamiltonian: Rx rotation on all qubits
            for i in range(n):
                squin.rx(2 * beta[layer], q[i])

    # Create LineQubits and emit circuit
    qubits = cirq.LineQubit.range(n)
    circuit = cirq_utils.emit_circuit(qaoa_kernel, circuit_qubits=qubits)

    # Convert to the native CZ gateset
    circuit2 = cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset())
    return circuit2


def build_qaoa_circuit_parallelized(
    graph: nx.Graph, gamma: list[float], beta: list[float]
) -> cirq.Circuit:
    """Build and parallelize a QAOA circuit for MaxCut on the given graph using squin"""
    n = len(graph.nodes)
    assert len(gamma) == len(beta), "Length of gamma and beta must be equal"

    # A smarter implementation would use the Misra–Gries algorithm,
    # which gives a guaranteed Δ+1 coloring, consistent with
    # Vizing's theorem for edge coloring.
    # However, networkx does not have an implementation of this algorithm,
    # so we use greedy coloring as an approximation. This does not guarantee
    # optimal depth, but works reasonably well in practice.
    linegraph = nx.line_graph(graph)
    best = 1e99
    for strategy in [
        "largest_first",
        "random_sequential",
        "smallest_last",
        "independent_set",
        "connected_sequential_bfs",
        "connected_sequential_dfs",
        "saturation_largest_first",
    ]:
        coloring: dict = nx.coloring.greedy_color(linegraph, strategy=strategy)
        num_colors = len(set(coloring.values()))
        if num_colors < best:
            best = num_colors
            best_coloring = coloring
    coloring: dict = best_coloring
    colors = [
        [edge for edge, color in coloring.items() if color == c]
        for c in set(coloring.values())
    ]

    # For QAOA MaxCut, we need exp(i*gamma/2*Z⊗Z) per edge.
    # We decompose this using CZ and single-qubit rotations:
    #
    # exp(-i*gamma/2*Z⊗Z)  =  -------o----------o-------
    #                                 |          |
    #                         -----H--o--Rx(g)--o--H----
    #
    # where Rx(gamma) = X^(gamma/pi) in Cirq notation.

    # To cancel repeated Hadamards, we can select which qubit
    # of each gate pair to apply the Hadamards on. The minimum
    # number of Hadamards is equal to the size of the minimum vertex cover
    # of the graph. Finding the minimum vertex cover is NP-hard,
    # but we can use a greedy MIS heuristic instead.
    # The complement of the MIS is a minimum vertex cover.
    mis = nx.algorithms.approximation.maximum_independent_set(graph)
    hadamard_qubits = set(graph.nodes) - set(mis)

    # Prepare data structures for squin kernel
    # Flatten color groups and create parallel lists for indices
    all_edges = []
    h_qubits = []
    for color_group in colors:
        for edge in color_group:
            all_edges.append(edge)
            u, v = edge
            if u in hadamard_qubits:
                h_qubits.append(u)
            else:
                h_qubits.append(v)

    # Build the circuit using squin
    @squin.kernel
    def qaoa_parallel_kernel():
        q = squin.qalloc(n)

        # Initial Hadamard layer
        for i in range(n):
            squin.h(q[i])

        # QAOA layers
        for layer in range(len(gamma)):
            # Cost Hamiltonian: process edges in order
            edge_start = 0
            for color_group in colors:
                group_size = len(color_group)

                # First Hadamard layer
                for i in range(group_size):
                    h_qubit = h_qubits[edge_start + i]
                    squin.h(q[h_qubit])

                # First CZ layer
                for i in range(group_size):
                    edge = color_group[i]
                    u = edge[0]
                    v = edge[1]
                    squin.cz(q[u], q[v])

                # Rotation layer (Rx)
                for i in range(group_size):
                    h_qubit = h_qubits[edge_start + i]
                    squin.rx(gamma[layer], q[h_qubit])

                # Second CZ layer
                for i in range(group_size):
                    edge = color_group[i]
                    u = edge[0]
                    v = edge[1]
                    squin.cz(q[u], q[v])

                # Second Hadamard layer
                for i in range(group_size):
                    h_qubit = h_qubits[edge_start + i]
                    squin.h(q[h_qubit])

                edge_start = edge_start + group_size

            # Mixer Hamiltonian: Rx rotation on all qubits
            for i in range(n):
                squin.rx(2 * beta[layer], q[i])

    # Create LineQubits and emit circuit
    qubits = cirq.LineQubit.range(n)
    circuit = cirq_utils.emit_circuit(qaoa_parallel_kernel, circuit_qubits=qubits)

    # This circuit will have some redundant doubly-repeated Hadamards that can be removed.
    # Lets do that now by merging single qubit gates to phased XZ gates, which is the native
    # single-qubit gate on neutral atoms.
    circuit2 = cirq.merge_single_qubit_moments_to_phxz(circuit)
    # Do any last optimizing...
    circuit3 = cirq.optimize_for_target_gateset(
        circuit2, gateset=cirq.CZTargetGateset()
    )

    return circuit3

print(build_qaoa_circuit_parallelized(nx.cycle_graph(6), gamma=[0.5], beta=[0.25]))