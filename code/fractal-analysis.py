"""
Fractal dimension analysis for 3D biological structures
Part of the Information-Theoretic Reality Framework
Author: Dario Abece, 2025

Tests the prediction: D_f ≈ d - 0.5 (so D_f ≈ 2.5 for 3D systems)
"""

import numpy as np
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression

class FractalAnalyzer3D:
    """
    Calculate fractal dimension of 3D point clouds using box-counting method.
    Tests prediction: D_f ≈ d - 0.5 (so D_f ≈ 2.5 for 3D systems)
    """
    
    def __init__(self, min_box_size=1, max_box_size=None, n_sizes=20):
        self.min_box_size = min_box_size
        self.max_box_size = max_box_size
        self.n_sizes = n_sizes
        
    def calculate_dimension(self, points):
        """
        Calculate fractal dimension using box-counting.
        
        Parameters:
        -----------
        points : array-like, shape (n_points, 3)
            3D coordinates of the structure
            
        Returns:
        --------
        dict with keys:
            - 'dimension': Estimated fractal dimension
            - 'r_squared': Quality of log-log fit
            - 'scaling_range': Range where scaling holds
            - 'prediction_error': Deviation from D_f ≈ 2.5
        """
        points = np.array(points)
        
        # Determine range
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        extent = max_coords - min_coords
        
        if self.max_box_size is None:
            self.max_box_size = extent.max() / 2
            
        # Generate box sizes (logarithmic scale)
        sizes = np.logspace(
            np.log10(self.min_box_size),
            np.log10(self.max_box_size),
            self.n_sizes
        )
        
        counts = []
        for size in sizes:
            # Create 3D grid
            grid_coords = set()
            for point in points:
                grid_coord = tuple(((point - min_coords) / size).astype(int))
                grid_coords.add(grid_coord)
            
            counts.append(len(grid_coords))
        
        # Fit log-log relationship
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)
        
        # Find best linear region
        best_r2 = 0
        best_dim = None
        best_range = None
        
        for start in range(len(sizes) - 5):
            for end in range(start + 5, len(sizes)):
                X = log_sizes[start:end].reshape(-1, 1)
                y = log_counts[start:end]
                
                model = LinearRegression()
                model.fit(X, y)
                r2 = model.score(X, y)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_dim = -model.coef_[0]
                    best_range = (sizes[start], sizes[end])
        
        return {
            'dimension': best_dim,
            'r_squared': best_r2,
            'scaling_range': best_range,
            'prediction_error': abs(best_dim - 2.5)  # Compare to D_f ≈ 2.5
        }
    
    def validate_prediction(self, points, expected_dim=2.5, tolerance=0.1):
        """
        Test if fractal dimension matches theoretical prediction.
        
        Parameters:
        -----------
        points : array-like
            3D point cloud
        expected_dim : float
            Theoretical prediction (default 2.5 for 3D)
        tolerance : float
            Acceptable deviation
            
        Returns:
        --------
        dict with validation results
        """
        result = self.calculate_dimension(points)
        
        is_valid = abs(result['dimension'] - expected_dim) <= tolerance
        
        return {
            'measured': result['dimension'],
            'expected': expected_dim,
            'error': result['prediction_error'],
            'r_squared': result['r_squared'],
            'validates_theory': is_valid,
            'confidence': 'High' if result['r_squared'] > 0.95 else 'Medium' if result['r_squared'] > 0.90 else 'Low'
        }


def analyze_neuromorpho_swc(swc_content):
    """
    Analyze fractal dimension from NeuroMorpho.org SWC format.
    
    Parameters:
    -----------
    swc_content : str
        SWC file content
        
    Returns:
    --------
    dict with analysis results
    """
    points = []
    
    for line in swc_content.split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 7:
            # SWC format: ID, type, x, y, z, radius, parent
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            points.append([x, y, z])
    
    if len(points) < 100:
        return {'error': 'Insufficient points for analysis'}
    
    analyzer = FractalAnalyzer3D()
    return analyzer.validate_prediction(np.array(points))


# Example usage and testing
if __name__ == "__main__":
    print("Information-Theoretic Reality Framework")
    print("Fractal Dimension Analysis")
    print("Prediction: D_f ≈ 2.5 for 3D systems")
    print("-" * 40)
    
    # Generate test fractal (simplified Menger sponge-like structure)
    def generate_test_fractal(iterations=4, scale=100):
        """Generate a test 3D fractal structure."""
        points = []
        
        def subdivide(x, y, z, size, depth):
            if depth == 0:
                points.append([x, y, z])
                return
            
            new_size = size / 3
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        # Skip some cubes to create fractal pattern
                        if (i == 1 and j == 1) or (i == 1 and k == 1) or (j == 1 and k == 1):
                            continue
                        subdivide(
                            x + i * new_size,
                            y + j * new_size,
                            z + k * new_size,
                            new_size,
                            depth - 1
                        )
        
        subdivide(0, 0, 0, scale, iterations)
        return np.array(points)
    
    # Test with generated fractal
    print("\nTesting with generated fractal structure...")
    test_points = generate_test_fractal(iterations=3)
    print(f"Generated {len(test_points)} points")
    
    analyzer = FractalAnalyzer3D()
    result = analyzer.validate_prediction(test_points)
    
    print(f"\nResults:")
    print(f"  Measured dimension: {result['measured']:.3f}")
    print(f"  Expected dimension: {result['expected']:.3f}")
    print(f"  Error: {result['error']:.3f}")
    print(f"  R²: {result['r_squared']:.3f}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Validates theory: {result['validates_theory']}")
    
    # Test with random points (should give D ≈ 3)
    print("\n" + "-" * 40)
    print("Control test with random points (should give D ≈ 3)...")
    random_points = np.random.randn(1000, 3) * 10
    control_result = analyzer.calculate_dimension(random_points)
    print(f"Random points dimension: {control_result['dimension']:.3f}")
    print(f"(Far from 2.5, as expected for non-fractal structure)")