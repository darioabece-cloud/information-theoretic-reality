# Data Requirements for Framework Validation

This directory contains datasets and links to external data sources needed to test the framework's predictions. Due to size and licensing constraints, most data must be obtained separately.

## 1. Fractal Dimension Analysis (D_f ≈ 2.5 for 3D systems)

### Neural Morphology Data
- **Source**: [NeuroMorpho.org](http://neuromorpho.org)
- **Format**: SWC files (standard morphology format)
- **Requirements**: 
  - Minimum 100 points per structure
  - Complete 3D reconstructions
  - Multiple cell types for validation
- **Expected files**: `*.swc` in `data/neuromorpho/`

### Vascular Network Data
- **Source**: Medical imaging databases
- **Format**: 3D point clouds or voxel data
- **Requirements**:
  - Resolution < 100 μm
  - Volume > 1 cm³
  - Segmented vessel centerlines
- **Expected files**: `*.npy` or `*.mat` in `data/vascular/`

### Trabecular Bone Structure
- **Source**: μCT scan databases
- **Format**: 3D binary images or mesh files
- **Requirements**:
  - Isotropic voxel size < 50 μm
  - Segmented bone/marrow phases
- **Expected files**: `*.nii` or `*.stl` in `data/bone/`

## 2. Consciousness Measure Validation

### EEG/MEG Recordings
- **Source**: OpenNeuro, PhysioNet
- **Format**: EDF, FIF, or MAT files
- **Requirements**:
  - Sampling rate ≥ 1000 Hz
  - Minimum 64 channels
  - States: awake, sleep, anesthesia
  - Duration > 5 minutes per state
- **Expected files**: `*.edf` in `data/eeg/`

### Anesthesia Transition Data
- **Source**: Clinical research databases (requires IRB)
- **Format**: Continuous EEG with dose markers
- **Requirements**:
  - Dose-response curves
  - Behavioral responsiveness scores
  - Multiple subjects (N > 30)
- **Expected files**: `*.csv` with dose-response pairs

## 3. Branch Dynamics Testing

### Quantum Measurement Data
- **Source**: IBM Quantum, experimental labs
- **Format**: CSV or JSON with measurement outcomes
- **Requirements**:
  - Minimum 10⁶ trials
  - Known state preparation
  - Complexity metrics for each state
- **Expected files**: `quantum_measurements.csv`

### Control Data
- **Format**: Random number sequences for null hypothesis testing
- **Requirements**:
  - Same length as experimental data
  - Multiple independent runs
- **Expected files**: `control_*.csv`

## Data Processing Pipeline

```bash
# Example workflow
python code/fractal_analysis.py data/neuromorpho/*.swc
python code/consciousness_measure.py data/eeg/*.edf
python code/branch_dynamics_sim.py data/quantum_measurements.csv
```

## Publicly Available Datasets

### Immediate Testing (No IRB Required)
1. **NeuroMorpho.org**: 150,000+ neuron reconstructions
   - Direct download, citation required
   
2. **PhysioNet**: EEG/ECG recordings
   - Free access with PhysioNet account
   
3. **DRIVE Database**: Retinal vessel images (2D)
   - Good for validating D_f ≈ 1.5 prediction

### Requires Permission/IRB
1. Clinical anesthesia recordings
2. High-resolution brain imaging
3. Quantum device access

## Data Format Specifications

### SWC Format (Neuronal Morphology)
```
# SWC format
# id, type, x, y, z, radius, parent_id
1 1 0.0 0.0 0.0 1.0 -1
2 3 1.0 0.0 0.0 0.5 1
```

### Time Series Format (EEG/MEG)
```
# CSV format
# time, channel_1, channel_2, ..., channel_n
0.000, 0.123, -0.456, ..., 0.789
0.001, 0.234, -0.567, ..., 0.890
```

### Measurement Outcome Format
```
# CSV format
# trial, state_complexity, outcome, probability
1, 0.234, 0, 0.5
2, 0.567, 1, 0.5
```

## Notes

- All patient data must be de-identified
- Follow local IRB requirements for human subjects data
- Cite original data sources in publications
- Large files (>100MB) should not be committed to git

## Contact

For data access questions or to share relevant datasets, please open an issue on the GitHub repository.