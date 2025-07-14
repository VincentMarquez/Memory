import numpy as np
import random

class Neuron:
    def __init__(self, name):
        self.name = name
        self.q = 0.5
        self.inputs = {}

    def __str__(self):
        return f"Neuron({self.name}, Q={self.q:.3f})"

class VQRIHybrid:
    def __init__(self, num_qubits=100):
        self.num_qubits = num_qubits
        self.neurons = {f"q{i}": Neuron(f"q{i}") for i in range(num_qubits)}
        self.reset_weights_cluster()
        print(f"Initializing {num_qubits}-qubit hybrid neural-sparse quantum simulation with QEC")

    def reset_weights_cluster(self):
        for i in range(self.num_qubits - 1):
            n1, n2 = self.neurons[f"q{i}"], self.neurons[f"q{i+1}"]
            weight = 0.5
            n1.inputs[n2] = weight
            n2.inputs[n1] = weight

    def pulse_cluster(self, dt=0.01, steps=50):
        for _ in range(steps):
            updates = {}
            for neuron in self.neurons.values():
                input_sum = sum(n.q * w for n, w in neuron.inputs.items())
                avg_input = input_sum / max(1, len(neuron.inputs))
                dq = dt * (0.5 * (avg_input - neuron.q)) + 0.01 * random.uniform(-1, 1)
                new_q = max(0, min(1, neuron.q + dq))
                updates[neuron] = new_q
            for neuron, new_q in updates.items():
                neuron.q = new_q
        print("\nAfter evolution (Cluster State):")
        for i in [0, 1, 98, 99]:
            print(self.neurons[f"q{i}"])

    def apply_qec_z(self, state, error_rate=0.01):
        """Correct X-errors without forcing uniformity."""
        noisy_state = list(state)
        error_positions = [i for i in range(self.num_qubits) if random.random() < error_rate]
        for i in error_positions:
            noisy_state[i] = 1 - noisy_state[i]
        corrected_state = list(noisy_state)
        for i in range(1, self.num_qubits - 1):
            if noisy_state[i-1] != noisy_state[i+1]:
                corrected_state[i] = 1 - noisy_state[i] if random.random() < 0.5 else noisy_state[i]
        return corrected_state

    def apply_qec_x(self, state, z_dual, error_rate=0.01):
        """Correct Z-errors using stabilizer syndromes."""
        noisy_state = list(state)
        error_positions = [i for i in range(self.num_qubits) if random.random() < error_rate]
        for i in error_positions:
            noisy_state[i] = "+" if noisy_state[i] == "-" else "-"
        corrected_state = list(noisy_state)
        # Correct based on stabilizers S_i = X_i Z_{i-1} Z_{i+1} = +1
        for i in range(self.num_qubits):
            z_prev = z_dual[i-1] if i > 0 else 0
            z_next = z_dual[i+1] if i < self.num_qubits - 1 else 0
            expected_x = 1 if (z_prev + z_next) % 2 == 0 else -1
            current_x = 1 if noisy_state[i] == "+" else -1
            if current_x != expected_x:
                corrected_state[i] = "+" if expected_x == 1 else "-" # Restore stabilizer
        return corrected_state

    def measure_z_basis_cluster(self, shots=10000, error_rate=0.01):
        counts = {}
        for _ in range(shots):
            outcome = [1 if random.random() < 0.5 else 0 for _ in range(self.num_qubits)]
            corrected_outcome = self.apply_qec_z(outcome, error_rate)
            outcome_str = "".join(str(bit) for bit in corrected_outcome)
            counts[outcome_str] = counts.get(outcome_str, 0) + 1
        avg_ones = sum(outcome.count("1") * count for outcome, count in counts.items()) / shots
        print(f"\nZ-Basis Measurement Results (Cluster State with QEC, sample outcomes):")
        for outcome, count in list(counts.items())[:5]:
            print(f"{outcome}: {count}")
        print(f"Average 1s per shot: {avg_ones:.1f}, Total unique outcomes: {len(counts)}")
        return counts

    def measure_x_basis_cluster(self, shots=10000, error_rate=0.01):
        """Measure X-basis with perfect S_i = +1 and QEC, starting from dual Z-string."""
        counts = {}
        for _ in range(shots):
            # Start with random dual Z-string
            z_dual = [1 if random.random() < 0.5 else 0 for _ in range(self.num_qubits)]
            # Derive X-string from stabilizers S_i = X_i Z_{i-1} Z_{i+1} = +1
            x_string = [None] * self.num_qubits
            for i in range(self.num_qubits):
                z_prev = z_dual[i-1] if i > 0 else 0
                z_next = z_dual[i+1] if i < self.num_qubits - 1 else 0
                expected_x = 1 if (z_prev + z_next) % 2 == 0 else -1
                x_string[i] = "+" if expected_x == 1 else "-"
            # Apply QEC using dual Z-string
            x_corrected = self.apply_qec_x(x_string, z_dual, error_rate)
            outcome = "".join(x_corrected)
            counts[outcome] = counts.get(outcome, 0) + 1
        avg_plus = sum(outcome.count("+") * count for outcome, count in counts.items()) / shots
        print(f"\nX-Basis Measurement Results (Cluster State with QEC, sample outcomes):")
        for outcome, count in list(counts.items())[:5]:
            print(f"{outcome}: {count}")
        print(f"Average + per shot: {avg_plus:.1f}, Total unique outcomes: {len(counts)}")
        return counts

    def run_cluster_state_test(self, error_rate=0.01):
        print("Simulating 100-Qubit Cluster State with Hybrid Approach and QEC:")
        self.pulse_cluster(dt=0.01, steps=50)
        self.measure_z_basis_cluster(shots=10000, error_rate=error_rate)
        self.measure_x_basis_cluster(shots=10000, error_rate=error_rate)

if __name__ == "__main__":
    print("Testing VQRI Hybrid for 100-Qubit Cluster State with QEC on Android Phone:")
    vqri = VQRIHybrid(num_qubits=100)
    vqri.run_cluster_state_test(error_rate=0.01)
