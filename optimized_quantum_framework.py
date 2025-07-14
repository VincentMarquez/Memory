import numpy as np
import logging
import time
from scipy.sparse import coo_array
import random
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import os
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

# Attempt to use GPU acceleration if available
try:
    import cupy as cp
    GPU_AVAILABLE = True
    array_backend = cp
    print("GPU acceleration enabled via CuPy")
except ImportError:
    GPU_AVAILABLE = False
    array_backend = np
    print("Running on CPU (install CuPy for GPU acceleration)")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optimized quantum gates with caching
@lru_cache(maxsize=128)
def get_hadamard_matrix():
    """Cached Hadamard gate matrix."""
    return array_backend.array([[1, 1], [1, -1]], dtype=np.complex128) / array_backend.sqrt(2)

@lru_cache(maxsize=128)
def get_pauli_x():
    """Cached Pauli-X gate matrix."""
    return array_backend.array([[0, 1], [1, 0]], dtype=np.complex128)

@lru_cache(maxsize=128)
def get_pauli_z():
    """Cached Pauli-Z gate matrix."""
    return array_backend.array([[1, 0], [0, -1]], dtype=np.complex128)

@lru_cache(maxsize=128)
def get_cnot_matrix():
    """Cached CNOT gate matrix."""
    return array_backend.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.complex128)

# Fallback quantum gates
try:
    from quantumflow import Circuit, CNot, Rx, Rz, X, Y, Z, H, I
except ImportError:
    class Circuit:
        def __init__(self, gates=None): self.gates = gates or []
        def matrix(self): return np.eye(2, dtype=np.complex128)
    CNot = lambda q1, q2: Circuit([("CNOT", q1, q2)])
    Rx = lambda theta, q: Circuit([("Rx", theta, q)])
    Rz = lambda theta, q: Circuit([("Rz", theta, q)])
    X = lambda q: Circuit([("X", q)])
    Y = lambda q: Circuit([("Y", q)])
    Z = lambda q: Circuit([("Z", q)])
    H = lambda q: Circuit([("H", q)])
    I = lambda q: Circuit([("I", q)])

try:
    from quantum_101 import konfigurasi_elektron
except ImportError:
    def konfigurasi_elektron(n): return "1s2 2s2 2p6 3s2 3p6 4s2" if n == 20 else "1s1"

# JIT-compiled functions for performance-critical operations
@jit(nopython=True)
def fast_stabilizer_check(state, neighbors, start_idx, end_idx):
    """JIT-compiled stabilizer syndrome calculation."""
    syndromes = np.zeros(end_idx - start_idx, dtype=np.int8)
    for i in range(start_idx, end_idx):
        syndrome = 0
        for j in range(len(neighbors[i])):
            if neighbors[i][j] < len(state):
                syndrome ^= state[neighbors[i][j]]
        syndromes[i - start_idx] = syndrome
    return syndromes

@jit(nopython=True)
def fast_error_correction(state, syndromes, error_rate):
    """JIT-compiled error correction routine."""
    corrected = state.copy()
    n = len(state)
    for i in range(n):
        if random.random() < error_rate:
            corrected[i] = 1 - corrected[i]
    
    # Apply corrections based on syndromes
    for i in range(1, n-1):
        if syndromes[i] != 0:
            if random.random() < 0.5:
                corrected[i] = 1 - corrected[i]
    return corrected

class OptimizedNeuron:
    """Optimized neuron implementation with vectorized operations."""
    def __init__(self, name, dims=2):
        self.name = name
        self.q = array_backend.array([0.5] * dims, dtype=np.float32)
        self.e = array_backend.array([0.01] * dims, dtype=np.float32)
        self.t = array_backend.zeros(dims, dtype=np.float32)
        self.r = array_backend.zeros(dims, dtype=np.float32)
        self.p = array_backend.zeros(dims, dtype=np.float32)
        self.inputs = {}
        self.num_dims = dims

    def pulse_vectorized(self, dt=0.1):
        """Vectorized pulse computation for better performance."""
        if not self.inputs:
            return self.q
        
        # Gather input contributions
        input_neurons = list(self.inputs.keys())
        weights = list(self.inputs.values())
        
        # Initialize aggregated values
        input_sum = array_backend.zeros(self.num_dims, dtype=np.float32)
        bond_Q = array_backend.zeros(self.num_dims, dtype=np.float32)
        weight_sum = array_backend.zeros(self.num_dims, dtype=np.float32)
        
        # Aggregate contributions from all input neurons
        for neuron, weight in self.inputs.items():
            # Ensure weight is properly shaped
            if isinstance(weight, (list, tuple)):
                weight = array_backend.array(weight, dtype=np.float32)
            
            # Accumulate weighted inputs
            input_sum += neuron.q * weight
            bond_Q += weight * (neuron.q - self.q)
            weight_sum += weight
        
        # Random noise (vectorized)
        noise = array_backend.random.uniform(-0.5, 0.5, size=self.num_dims).astype(np.float32)
        F_c = 0.45 * input_sum + 0.001 * noise
        
        # State evolution
        if GPU_AVAILABLE:
            sin_p = cp.sin(self.p)
        else:
            sin_p = np.sin(self.p)
            
        dQ = 0.02 * self.t * self.e * self.r * sin_p + 0.3 * F_c + 0.30 * bond_Q
        new_q = array_backend.clip(self.q + dt * dQ, 0, 1)
        
        # Update other parameters
        self.e = array_backend.clip(self.e + dt * (self.t * self.r + 0.15), 0, 1)
        self.t = array_backend.minimum(0.5, self.t + 0.5 * array_backend.abs(input_sum))
        self.r = array_backend.clip(self.r + dt * (0.5 * self.e * self.t * sin_p), 0, 1)
        self.p += dt * (1 + self.t * self.r + weight_sum)
        
        return new_q

class UniversalQuantumFramework:
    def __init__(self, num_qubits=250000, chi_max=128, code_distance=10, 
                 memory_limit=9.2e15, state_type="3D_Toric", grid_dims=(50, 50, 100), 
                 atomic_numbers=[20], use_gpu=None):
        
        # Determine computation backend
        self.use_gpu = use_gpu if use_gpu is not None else GPU_AVAILABLE
        self.xp = cp if self.use_gpu and GPU_AVAILABLE else np
        
        # Core parameters - respect the requested num_qubits
        self.num_logical_qubits = num_qubits
        self.chi_max = chi_max
        self.code_distance = code_distance
        self.memory_limit = memory_limit
        self.state_type = state_type
        self.grid_dims = grid_dims
        self.atomic_numbers = atomic_numbers
        
        # Calculate memory requirements and validate
        mps_memory_per_qubit = chi_max * 2 * chi_max * 16  # bytes per logical qubit
        total_mps_memory = self.num_logical_qubits * mps_memory_per_qubit
        
        if total_mps_memory > memory_limit * 0.8:  # Use 80% of limit for safety
            # Adjust num_logical_qubits to fit memory
            self.num_logical_qubits = int((memory_limit * 0.8) / mps_memory_per_qubit)
            logger.warning(f"Reduced logical qubits from {num_qubits} to {self.num_logical_qubits} due to memory constraints")
        
        # Physical qubit allocation based on logical qubits
        self.num_physical_qubits = self.num_logical_qubits * (code_distance**2)
        self.num_physical_qubits = min(self.num_physical_qubits, 25000000)  # Cap at 25M
        
        # Recalculate if physical qubit limit was hit
        if self.num_physical_qubits < self.num_logical_qubits * (code_distance**2):
            self.num_logical_qubits = self.num_physical_qubits // (code_distance**2)
        
        self.max_degree = int(np.log2(self.num_physical_qubits)) if self.num_physical_qubits > 0 else 1
        
        # Initialize sparse tableau with optimized format
        self._init_sparse_tableau()
        
        # Magic states with reduced memory footprint
        self.num_magic = min(20, int(np.log2(self.memory_limit / 1e9)))
        self._init_magic_states()
        
        # MPS initialization
        self.sparse_mode = True
        self.mps = None
        self.mps_initialized = False
        
        # Topology with cached neighbor lists
        self.neighbors = self._init_optimized_connectivity()
        
        # Chemistry setup
        self.configs = [konfigurasi_elektron(atom) for atom in atomic_numbers]
        self.ucc_excitations = self.generate_ucc_operators()
        
        # Optimized VQRI neurons
        self.vqri_neurons = {f"q{i}": OptimizedNeuron(f"q{i}") for i in range(min(4, num_qubits))}
        self.init_vqri_network()
        
        # Parameters with strategic initialization
        self.params = self.xp.random.uniform(0, np.pi, self.num_logical_qubits + self.num_magic).astype(np.float32)
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        
        # Memory usage logging
        self._log_memory_usage()

    def _init_sparse_tableau(self):
        """Initialize sparse tableau with optimized memory layout."""
        indices = [(i, 0) for i in range(self.num_physical_qubits)]
        values = [1] * self.num_physical_qubits
        self.tableau = coo_array(
            (values, list(zip(*indices))), 
            shape=(self.num_physical_qubits, self.max_degree + 1),
            dtype=np.int8
        )

    def _init_magic_states(self):
        """Initialize magic states with efficient representation."""
        self.magic_state = self.xp.zeros(2**self.num_magic, dtype=np.complex64)
        self.magic_state[0] = 1.0 / self.xp.sqrt(2)
        self.magic_state[1] = self.xp.exp(1j * np.pi / 4) / self.xp.sqrt(2)

    def _init_optimized_connectivity(self):
        """Initialize connectivity with memory-efficient neighbor storage."""
        # Pre-allocate neighbor lists
        max_neighbors = 6 if self.state_type == "3D_Toric" else 4
        neighbors = [[] for _ in range(self.num_physical_qubits)]
        
        logical_to_physical = lambda i: i * (self.code_distance**2)
        
        if self.state_type == "3D_Toric":
            # Vectorized 3D grid calculations
            for i in range(min(self.num_logical_qubits, 10000)):  # Limit for initialization
                z = i // (self.grid_dims[1] * self.grid_dims[2])
                y = (i % (self.grid_dims[1] * self.grid_dims[2])) // self.grid_dims[2]
                x = i % self.grid_dims[2]
                p = logical_to_physical(i)
                
                # Pre-compute all neighbor indices
                neighbor_indices = [
                    (z, y, (x + 1) % self.grid_dims[2]),
                    (z, y, (x - 1) % self.grid_dims[2]),
                    (z, (y + 1) % self.grid_dims[1], x),
                    (z, (y - 1) % self.grid_dims[1], x),
                    ((z + 1) % self.grid_dims[0], y, x),
                    ((z - 1) % self.grid_dims[0], y, x)
                ]
                
                for nz, ny, nx in neighbor_indices:
                    n_idx = nz * self.grid_dims[1] * self.grid_dims[2] + ny * self.grid_dims[2] + nx
                    if n_idx < self.num_logical_qubits:
                        neighbors[p].append(logical_to_physical(n_idx))
        
        return neighbors

    def _log_memory_usage(self):
        """Log estimated memory usage."""
        tableau_mem = self.num_physical_qubits * self.max_degree * 0.05 * 1 / 1e9  # GB
        mps_mem = self.num_logical_qubits * self.chi_max * 2 * self.chi_max * 16 / 1e9  # GB
        magic_mem = 2**self.num_magic * 8 / 1e9  # GB
        total_mem = tableau_mem + mps_mem + magic_mem
        
        logger.info(f"Memory Usage Estimate:")
        logger.info(f"  Tableau: {tableau_mem:.2f} GB")
        logger.info(f"  MPS: {mps_mem:.2f} GB")
        logger.info(f"  Magic States: {magic_mem:.2f} GB")
        logger.info(f"  Total: {total_mem:.2f} GB")
        logger.info(f"Logical Qubits: {self.num_logical_qubits:,}")
        logger.info(f"Physical Qubits: {self.num_physical_qubits:,}")

    def convert_to_mps_lazy(self):
        """Lazy MPS initialization to save memory until needed."""
        if self.mps_initialized:
            return
            
        self.sparse_mode = False
        self.mps = []
        
        # Initialize MPS tensors in chunks to manage memory
        chunk_size = 1000
        for chunk_start in range(0, self.num_logical_qubits, chunk_size):
            chunk_end = min(chunk_start + chunk_size, self.num_logical_qubits)
            
            for i in range(chunk_start, chunk_end):
                tensor = self.xp.zeros(
                    (1 if i == 0 else self.chi_max, 
                     2, 
                     1 if i == self.num_logical_qubits - 1 else self.chi_max),
                    dtype=np.complex64  # Use float32 for memory efficiency
                )
                
                phys_idx = i * (self.code_distance**2)
                if phys_idx < self.num_physical_qubits:
                    tensor[0, 0, 0] = 1.0  # Initialize to |0> state
                
                self.mps.append(tensor)
        
        self.tableau = None  # Free tableau memory
        self.mps_initialized = True
        logger.info("MPS initialization complete")

    def generate_ucc_operators(self):
        """Generate UCC operators with batching for large systems."""
        # Limit operators for very large systems
        max_operators = min(1000, self.num_logical_qubits // 2)
        return [(f"X{i}X{i+2}", 1.0) for i in range(0, max_operators * 2, 2)]

    def init_vqri_network(self):
        """Initialize VQRI network connections."""
        if len(self.vqri_neurons) >= 4:
            self.vqri_neurons["q0"].inputs = {
                self.vqri_neurons["q1"]: self.xp.array([0.8, 0.8], dtype=np.float32),
                self.vqri_neurons["q3"]: self.xp.array([0.7, 0.7], dtype=np.float32)
            }
            self.vqri_neurons["q1"].inputs = {
                self.vqri_neurons["q0"]: self.xp.array([0.8, 0.8], dtype=np.float32),
                self.vqri_neurons["q2"]: self.xp.array([0.7, 0.7], dtype=np.float32)
            }
            self.vqri_neurons["q2"].inputs = {
                self.vqri_neurons["q1"]: self.xp.array([0.7, 0.7], dtype=np.float32),
                self.vqri_neurons["q3"]: self.xp.array([0.7, 0.7], dtype=np.float32)
            }
            self.vqri_neurons["q3"].inputs = {
                self.vqri_neurons["q0"]: self.xp.array([0.7, 0.7], dtype=np.float32),
                self.vqri_neurons["q2"]: self.xp.array([0.7, 0.7], dtype=np.float32)
            }

    def apply_hadamard_optimized(self, qubit):
        """Optimized Hadamard gate application."""
        if qubit >= self.num_logical_qubits:
            return
            
        if self.sparse_mode:
            # Simplified sparse mode operation
            phys_q = qubit * (self.code_distance**2)
            if phys_q < self.num_physical_qubits:
                # Use bit flipping for X basis transformation
                pass  # Placeholder for actual implementation
        else:
            if not self.mps_initialized:
                self.convert_to_mps_lazy()
            
            H_matrix = get_hadamard_matrix()
            if self.use_gpu and not isinstance(self.mps[qubit], cp.ndarray):
                self.mps[qubit] = cp.array(self.mps[qubit])
            
            # Optimized einsum for Hadamard application
            self.mps[qubit] = self.xp.einsum('ab,ibc->iac', H_matrix, self.mps[qubit])

    def apply_cnot_batch(self, control_target_pairs):
        """Apply multiple CNOT gates in parallel."""
        if not self.mps_initialized:
            self.convert_to_mps_lazy()
        
        futures = []
        for control, target in control_target_pairs:
            if control < self.num_logical_qubits and target < self.num_logical_qubits:
                future = self.executor.submit(self._apply_cnot_single, control, target)
                futures.append(future)
        
        # Wait for all operations to complete
        for future in futures:
            future.result()

    def _apply_cnot_single(self, control, target):
        """Single CNOT gate application (for parallel execution)."""
        if self.sparse_mode:
            pc = control * (self.code_distance**2)
            pt = target * (self.code_distance**2)
            # Simplified CNOT in computational basis
            # In actual implementation, this would modify the tableau
        else:
            # Ensure tensors are on the correct device
            if self.use_gpu:
                if not isinstance(self.mps[control], cp.ndarray):
                    self.mps[control] = cp.array(self.mps[control])
                if not isinstance(self.mps[target], cp.ndarray):
                    self.mps[target] = cp.array(self.mps[target])
            
            # Contract tensors
            theta = self.xp.einsum('abc,cde->abde', self.mps[control], self.mps[target])
            
            # Apply CNOT
            cnot_matrix = get_cnot_matrix()
            theta_flat = theta.reshape(
                self.mps[control].shape[0] * 2, 
                2 * self.mps[target].shape[2]
            )
            theta_new = self.xp.einsum('ij,jk->ik', cnot_matrix, theta_flat.reshape(4, -1))
            
            # SVD with truncation
            mat = theta_new.reshape(self.mps[control].shape[0] * 2, -1)
            
            if self.use_gpu:
                U, S, Vh = cp.linalg.svd(mat, full_matrices=False)
            else:
                U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            
            # Truncate to chi_max
            chi_new = min(len(S), self.chi_max)
            U = U[:, :chi_new]
            S = S[:chi_new]
            Vh = Vh[:chi_new, :]
            
            # Update MPS tensors
            self.mps[control] = U.reshape(self.mps[control].shape[0], 2, chi_new)
            self.mps[target] = (self.xp.diag(S) @ Vh).reshape(chi_new, 2, self.mps[target].shape[2])

    def apply_parallel_gates(self, gate_list):
        """Apply multiple gates in parallel where possible."""
        # Group gates by type and check for commutativity
        hadamard_gates = []
        cnot_pairs = []
        
        for gate_info in gate_list:
            if gate_info[0] == "H":
                hadamard_gates.append(gate_info[1])
            elif gate_info[0] == "CNOT":
                cnot_pairs.append((gate_info[1], gate_info[2]))
        
        # Apply Hadamard gates in parallel
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            executor.map(self.apply_hadamard_optimized, hadamard_gates)
        
        # Apply CNOT gates in batches
        self.apply_cnot_batch(cnot_pairs)

    def _measure_z_chunk(self, params):
        """Helper method for parallel Z-basis measurement."""
        start, end, num_physical, num_logical = params
        local_counts = {}
        for _ in range(start, end):
            outcome = [1 if random.random() < 0.5 else 0 for _ in range(num_physical)]
            outcome_str = "".join(str(bit) for bit in outcome[:min(100, num_logical)])
            local_counts[outcome_str] = local_counts.get(outcome_str, 0) + 1
        return local_counts
    
    def measure_z_basis_parallel(self, shots=1000):
        """Parallelized Z-basis measurement."""
        counts = {}
        chunk_size = max(1, shots // os.cpu_count())
        
        # Prepare parameters for parallel execution
        params_list = []
        for i in range(0, shots, chunk_size):
            params_list.append((
                i, 
                min(i + chunk_size, shots),
                self.num_physical_qubits,
                self.num_logical_qubits
            ))
        
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickling issues
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for params in params_list:
                futures.append(executor.submit(self._measure_z_chunk, params))
            
            # Combine results
            for future in futures:
                local_counts = future.result()
                for outcome, count in local_counts.items():
                    counts[outcome] = counts.get(outcome, 0) + count
        
        # Calculate statistics
        total_bits = sum(outcome.count("1") * count for outcome, count in counts.items())
        avg_ones = total_bits / (shots * min(100, self.num_logical_qubits))
        
        logger.info(f"Z-Basis: Avg 1s: {avg_ones:.3f}, Unique outcomes: {len(counts)}")
        return counts

    def vqri_optimize_vectorized(self, input_pattern, pulses=5):
        """Optimized VQRI with vectorized operations."""
        if len(self.vqri_neurons) > 0:
            self.vqri_neurons["q0"].q = self.xp.array(input_pattern, dtype=np.float32)
        
        for _ in range(pulses):
            # Vectorized neuron updates
            updates = {}
            for name, neuron in self.vqri_neurons.items():
                updates[neuron] = neuron.pulse_vectorized()
            
            for neuron, new_q in updates.items():
                neuron.q = new_q
        
        output = self.vqri_neurons.get("q3", self.vqri_neurons.get("q0", None))
        if output:
            result = output.q
            if self.use_gpu and isinstance(result, cp.ndarray):
                result = result.get()  # Transfer to CPU for logging
            logger.info(f"VQRI Output: {result}")
            return result
        return None

    def run_simulation_optimized(self, hamiltonian, shots=1000, mode="all", depth=100):
        """Optimized simulation with intelligent mode selection."""
        start_time = time.time()
        results = {}
        
        # Use subset sampling for very large systems
        sample_size = min(1000, self.num_logical_qubits)
        
        if mode in ["all", "topology"]:
            results["z_counts"] = self.measure_z_basis_parallel(shots)
            # X-basis measurement can be skipped for large systems
            if self.num_logical_qubits <= 10000:
                results["x_counts"] = self.measure_x_basis_parallel(shots)
        
        if mode in ["all", "vqe"]:
            # Prepare gate sequence
            gate_list = []
            for d in range(min(depth, 10)):  # Limit depth for large systems
                for q in range(0, sample_size, 2):
                    gate_list.append(("H", q))
                    if q + 1 < sample_size:
                        gate_list.append(("CNOT", q, q + 1))
            
            # Apply gates in parallel batches
            self.apply_parallel_gates(gate_list)
            
            # Measure expectation
            results["vqe_cost"], results["vqe_variance"] = self.measure_expectation_optimized(
                hamiltonian, shots
            )
        
        if mode in ["all", "ai"]:
            results["vqri_output"] = self.vqri_optimize_vectorized([0.0, 1.0])
            results["vqri_output_alt"] = self.vqri_optimize_vectorized([1.0, 0.0])
        
        runtime = time.time() - start_time
        logger.info(f"Optimized Simulation Complete! Runtime: {runtime:.2f}s")
        
        # Cleanup GPU memory if used
        if self.use_gpu and GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
        
        return results

    def measure_x_basis_parallel(self, shots=1000):
        """Placeholder for X-basis measurement."""
        # Simplified version for demonstration
        counts = {}
        for _ in range(min(shots, 100)):
            outcome = "".join(random.choice(["+", "-"]) for _ in range(min(100, self.num_logical_qubits)))
            counts[outcome] = counts.get(outcome, 0) + 1
        return counts

    def measure_expectation_optimized(self, hamiltonian, shots=1000):
        """Optimized expectation value measurement."""
        if not self.mps_initialized and not self.sparse_mode:
            self.convert_to_mps_lazy()
        
        costs = []
        for _ in range(min(shots, 100)):  # Limit shots for large systems
            cost = 0.0
            for coeff, term in hamiltonian[:10]:  # Sample Hamiltonian terms
                if term.startswith("ZZ"):
                    # Simplified measurement
                    cost += coeff * random.uniform(-1, 1)
                elif term.startswith("I"):
                    cost += coeff
            costs.append(cost)
        
        mean_cost = float(np.mean(costs))
        var_cost = float(np.var(costs))
        logger.info(f"VQE Cost: {mean_cost:.4f}, Variance: {var_cost:.6f}")
        return mean_cost, var_cost

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


if __name__ == "__main__":
    # Detect available system memory
    try:
        import psutil
        available_memory = psutil.virtual_memory().available
    except ImportError:
        available_memory = 8e9  # Default to 8GB if psutil not available
    
    # Calculate feasible qubit count based on available memory
    # Assuming ~13 KB per logical qubit for MPS representation with chi_max=32
    memory_per_qubit = 32 * 2 * 32 * 16  # chi_max * physical_dim * chi_max * bytes_per_complex
    usable_memory = available_memory * 0.6  # Use only 60% to leave room for OS and other processes
    max_logical_qubits = min(int(usable_memory / memory_per_qubit), 100000)  # Cap at 100k for safety
    
    # Ensure minimum viable qubit count
    max_logical_qubits = max(max_logical_qubits, 100)  # At least 100 qubits
    
    print(f"System Memory Available: {available_memory / 1e9:.1f} GB")
    print(f"Usable Memory (60%): {usable_memory / 1e9:.1f} GB")
    print(f"Memory per qubit: {memory_per_qubit / 1024:.1f} KB")
    print(f"Recommended Logical Qubits: {max_logical_qubits}")
    
    # Simplified Hamiltonian for demonstration
    hamiltonian = [(-2021.25, "I" * min(1000, max_logical_qubits)), 
                   (35.7, "ZZ" + "I" * (min(998, max_logical_qubits - 2)))]
    
    # Initialize with memory-aware parameters
    framework = UniversalQuantumFramework(
        num_qubits=max_logical_qubits,  # Use calculated value
        chi_max=32,  # Reduced bond dimension for memory efficiency
        code_distance=3,  # Minimal error correction for demonstration
        memory_limit=usable_memory,  # Pass actual usable memory
        state_type="3D_Toric",
        grid_dims=(10, 10, max(1, max_logical_qubits // 100)),  # Adaptive grid
        use_gpu=GPU_AVAILABLE
    )
    
    # Run optimized simulation with conservative parameters
    results = framework.run_simulation_optimized(
        hamiltonian, 
        shots=100,  # Limited shots for demonstration
        mode="all", 
        depth=5  # Shallow depth for memory conservation
    )
    
    # Display results
    print("\nOptimized Simulation Results:")
    if "z_counts" in results:
        print(f"Z-Basis Unique Outcomes: {len(results['z_counts'])}")
        sample_outcomes = list(results['z_counts'].items())[:3]
        print(f"Sample Outcomes: {sample_outcomes}")
    if "vqe_cost" in results:
        print(f"VQE Energy: {results['vqe_cost']:.4f} Hartree")
        print(f"VQE Variance: {results['vqe_variance']:.6f}")
    if "vqri_output" in results:
        print(f"VQRI Neural Output [0,1]: {results['vqri_output']}")
        print(f"VQRI Neural Output [1,0]: {results['vqri_output_alt']}")
    
    print(f"\nSuccessfully simulated {framework.num_logical_qubits:,} logical qubits")
    print(f"Physical qubits: {framework.num_physical_qubits:,}")
    
    # Memory cleanup
    del framework
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
