"""
Consciousness index (Ψ_c) calculation
Part of the Information-Theoretic Reality Framework
Author: Dario Abece, 2025

Implements: Ψ_c = Φ · C · exp(-‖Ḃ‖/ε_stability)
"""

import numpy as np
from scipy.linalg import svd
from scipy.stats import entropy
from scipy.signal import welch

class ConsciousnessAnalyzer:
    """
    Calculate consciousness index from neural time series data.
    Based on the framework's boundary theory of consciousness.
    """
    
    def __init__(self, epsilon_stability=0.1, sampling_rate=1000):
        """
        Parameters:
        -----------
        epsilon_stability : float
            Stability parameter (default 0.1 for biological systems)
        sampling_rate : float
            Data sampling rate in Hz
        """
        self.epsilon_stability = epsilon_stability
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
    
    def compute_consciousness_index(self, data):
        """
        Compute consciousness index Ψ_c = Φ · C · stability
        
        Parameters:
        -----------
        data : array-like, shape (time_points, channels)
            Neural time series data
            
        Returns:
        --------
        dict with keys:
            - 'psi_c': Consciousness index
            - 'phi': Integrated information (proxy)
            - 'coupling': Coupling strength
            - 'stability': Temporal stability
            - 'gamma_power': Power in gamma band (38-42 Hz)
        """
        
        # Integrated information (simplified proxy)
        phi = self._compute_phi_proxy(data)
        
        # Coupling strength via effective dimensionality
        coupling = self._compute_coupling(data)
        
        # Temporal stability
        stability = self._compute_stability(data)
        
        # Gamma band analysis (prediction: peak at ~40 Hz)
        gamma_power = self._compute_gamma_power(data)
        
        # Combined index
        psi_c = phi * coupling * stability
        
        return {
            'psi_c': psi_c,
            'phi': phi,
            'coupling': coupling,
            'stability': stability,
            'gamma_power': gamma_power,
            'conscious_state': self._classify_state(psi_c)
        }
    
    def _compute_phi_proxy(self, data):
        """
        Simplified integrated information proxy.
        Uses entropy of eigenvalue distribution.
        """
        # Compute covariance
        cov = np.cov(data.T)
        
        # Eigenvalues for complexity
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros
        
        # Normalized entropy as proxy for Φ
        if len(eigenvalues) > 1:
            probs = eigenvalues / eigenvalues.sum()
            # Normalize by maximum possible entropy
            phi = entropy(probs) / np.log(len(eigenvalues))
        else:
            phi = 0
        
        return np.clip(phi, 0, 1)
    
    def _compute_coupling(self, data):
        """
        Compute coupling via participation ratio of singular values.
        Measures effective dimensionality of the system.
        """
        # SVD for effective dimensionality
        try:
            U, S, Vt = svd(data, full_matrices=False)
            
            # Participation ratio
            if len(S) > 0 and S[0] > 0:
                S_normalized = S / S[0]  # Normalize by largest singular value
                p = (S_normalized**2) / np.sum(S_normalized**2)
                eff_dim = 1 / np.sum(p**2)
                coupling = eff_dim / min(data.shape)
            else:
                coupling = 0
        except:
            coupling = 0
        
        return np.clip(coupling, 0, 1)
    
    def _compute_stability(self, data, window=100):
        """
        Compute temporal stability.
        Low rate of change = high stability.
        """
        if len(data) < window + 1:
            return 1.0
        
        rates = []
        for i in range(len(data) - window):
            # Compute change over window
            change = np.linalg.norm(data[i+window] - data[i])
            rate = change / (window * self.dt)
            rates.append(rate)
        
        if rates:
            mean_rate = np.mean(rates)
            # Exponential stability factor
            stability = np.exp(-mean_rate / self.epsilon_stability)
        else:
            stability = 1.0
        
        return np.clip(stability, 0, 1)
    
    def _compute_gamma_power(self, data):
        """
        Compute power in gamma band (38-42 Hz).
        Framework predicts consciousness correlates with ~40 Hz activity.
        """
        gamma_powers = []
        
        for channel in range(data.shape[1]):
            # Compute power spectral density
            freqs, psd = welch(data[:, channel], 
                              fs=self.sampling_rate,
                              nperseg=min(256, len(data)//4))
            
            # Extract gamma band (38-42 Hz)
            gamma_mask = (freqs >= 38) & (freqs <= 42)
            if gamma_mask.any():
                gamma_power = np.mean(psd[gamma_mask])
                gamma_powers.append(gamma_power)
        
        if gamma_powers:
            return np.mean(gamma_powers)
        else:
            return 0
    
    def _classify_state(self, psi_c):
        """
        Classify consciousness state based on Ψ_c value.
        Thresholds from Paper 3, Table 1 (theoretical).
        """
        if psi_c > 0.5:
            return "Awake/Conscious"
        elif psi_c > 0.2:
            return "REM/Dream"
        elif psi_c > 0.1:
            return "Light Sleep/Minimal Consciousness"
        elif psi_c > 0.05:
            return "Deep Sleep"
        else:
            return "Unconscious/Anesthetized"
    
    def analyze_anesthesia_transition(self, data_series, doses):
        """
        Analyze consciousness during anesthesia transition.
        Tests prediction: Ψ(D) = Ψ_0 · (1 - D/D_c)^β with β ≈ 0.5
        
        Parameters:
        -----------
        data_series : list of arrays
            Time series at different anesthetic doses
        doses : array-like
            Corresponding dose levels
            
        Returns:
        --------
        dict with scaling analysis
        """
        psi_values = []
        
        for data in data_series:
            result = self.compute_consciousness_index(data)
            psi_values.append(result['psi_c'])
        
        psi_values = np.array(psi_values)
        
        # Fit power law near critical dose
        # (Simplified - real implementation would need more sophisticated fitting)
        if len(psi_values) > 3:
            # Estimate critical dose where Ψ approaches 0
            D_c_estimate = doses[np.argmin(psi_values)] * 1.1
            
            # Compute scaling exponent
            valid_mask = doses < D_c_estimate
            if valid_mask.sum() > 2:
                x = np.log(1 - doses[valid_mask] / D_c_estimate)
                y = np.log(psi_values[valid_mask] / psi_values[0])
                
                # Linear fit in log-log space
                beta = np.polyfit(x[np.isfinite(x) & np.isfinite(y)], 
                                 y[np.isfinite(x) & np.isfinite(y)], 1)[0]
            else:
                beta = None
        else:
            beta = None
            D_c_estimate = None
        
        return {
            'psi_values': psi_values,
            'doses': doses,
            'critical_dose': D_c_estimate,
            'scaling_exponent': beta,
            'validates_prediction': beta is not None and abs(beta - 0.5) < 0.1
        }


# Example usage and testing
if __name__ == "__main__":
    print("Information-Theoretic Reality Framework")
    print("Consciousness Measure Calculator")
    print("Prediction: Ψ_c = Φ · C · exp(-‖Ḃ‖/ε)")
    print("-" * 40)
    
    # Generate synthetic neural data
    np.random.seed(42)
    time_points = 1000
    channels = 64
    sampling_rate = 1000  # Hz
    
    # Simulate different consciousness states
    def generate_neural_data(state='awake'):
        """Generate synthetic data mimicking different states."""
        t = np.linspace(0, time_points/sampling_rate, time_points)
        data = np.zeros((time_points, channels))
        
        if state == 'awake':
            # High complexity, strong 40 Hz component
            for i in range(channels):
                # Multiple frequency components
                data[:, i] = (np.sin(2 * np.pi * 40 * t + np.random.rand() * 2 * np.pi) * 0.5 +
                             np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi) * 0.3 +
                             np.random.randn(time_points) * 0.2)
        
        elif state == 'sleep':
            # Lower complexity, slower oscillations
            for i in range(channels):
                data[:, i] = (np.sin(2 * np.pi * 2 * t + np.random.rand() * 2 * np.pi) * 0.7 +
                             np.random.randn(time_points) * 0.1)
        
        elif state == 'anesthetized':
            # Very low complexity, minimal structure
            data = np.random.randn(time_points, channels) * 0.1
        
        return data
    
    # Test different states
    analyzer = ConsciousnessAnalyzer(sampling_rate=sampling_rate)
    
    print("\nTesting different consciousness states:")
    print("-" * 40)
    
    for state in ['awake', 'sleep', 'anesthetized']:
        data = generate_neural_data(state)
        result = analyzer.compute_consciousness_index(data)
        
        print(f"\nState: {state.upper()}")
        print(f"  Ψ_c (total): {result['psi_c']:.3f}")
        print(f"  Φ (integration): {result['phi']:.3f}")
        print(f"  C (coupling): {result['coupling']:.3f}")
        print(f"  Stability: {result['stability']:.3f}")
        print(f"  Gamma power: {result['gamma_power']:.2e}")
        print(f"  Classification: {result['conscious_state']}")
    
    # Test anesthesia transition
    print("\n" + "-" * 40)
    print("Testing anesthesia transition (β ≈ 0.5 prediction):")
    
    doses = np.linspace(0, 1, 5)  # Normalized doses
    data_series = []
    
    for dose in doses:
        # Simulate decreasing consciousness with dose
        complexity = 1 - dose
        data = generate_neural_data('awake') * complexity + \
                generate_neural_data('anesthetized') * dose
        data_series.append(data)
    
    transition_result = analyzer.analyze_anesthesia_transition(data_series, doses)
    
    print(f"\nAnesthesia transition analysis:")
    print(f"  Doses: {transition_result['doses']}")
    print(f"  Ψ_c values: {[f'{p:.3f}' for p in transition_result['psi_values']]}")
    if transition_result['scaling_exponent'] is not None:
        print(f"  Scaling exponent β: {transition_result['scaling_exponent']:.3f}")
        print(f"  Expected β: 0.5 ± 0.1")
        print(f"  Validates prediction: {transition_result['validates_prediction']}")
    else:
        print(f"  Insufficient data for scaling analysis")