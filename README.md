# Bayesian Neural Network Pruning

## Overview
This repository provides an implementation of Bayesian Neural Networks (BNNs) with pruning functionality, focusing on improving the efficiency and interpretability of neural networks while maintaining high predictive performance. The primary goals of this project are to:

- Leverage Bayesian inference for robust uncertainty quantification in model predictions.
- Employ signal-to-noise ratio-based methods for structured and unstructured pruning.
- Enhance computational efficiency through parallelized implementations.

This repository includes key scripts and notebooks for working with BNNs, applying MCMC sampling, and testing pruning techniques.

---

## Key Features

### Main Components

1. **Summary Notebook**
   - **File:** `Summary.ipynb`
   - Description: This notebook provides an overview of the implemented framework, experimental results, and performance analysis. It is the primary reference for understanding the repository’s capabilities.

2. **Signal-to-Noise Ratio Analysis**
   - **File:** `signal to noise ratio.ipynb`
   - Description: This notebook focuses on the signal-to-noise ratio-based pruning methodology. It explains the theory, demonstrates the implementation, and evaluates the impact of pruning on model performance.

3. **Bayesian Neural Network Implementation**
   - **Files:**
     - `BNN_mcmc.py`: Core implementation of Bayesian Neural Networks using Markov Chain Monte Carlo (MCMC) sampling.
     - `bnn_parallelelized.py`: Parallelized BNN training for improved computational efficiency.
     - `bnn_parallelelized_preprun.py`: Parallelized implementation with pre-pruning functionality.

4. **Regression and Classification Models**
   - **Files:**
     - `Bayesneuralnet_regcls.py`: Handles regression and classification tasks using BNNs.
     - `Bayesneuralnet_regcls_prun.py`: Incorporates pruning into regression and classification tasks.

5. **Convergence Testing**
   - **File:** `convergence_test.py`
   - Description: Tests convergence criteria for MCMC sampling to ensure robust posterior distributions.

### Supporting Resources

- **Data Files:**
  - `data.zip`: Contains datasets required for running experiments.

- **Visualization Assets:**
  - `framework_updated.jpg`: Overview of the implemented framework.
  - `trace_plots.jpg`: Example trace plots for convergence analysis.
  - `regression_resample.jpg`: Visualization of resampling results.

---

## Installation

### Requirements
Ensure you have the following installed:

- Python 3.8+


### Clone the Repository

```bash
git clone https://github.com/rvdeo/bayes_prun.git
cd bayes_prun
```

---

## Usage

1. **Train and Test BNN Models:**
   - Use `BNN_mcmc.py` for core Bayesian Neural Network implementations.
   - Use `Bayesneuralnet_regcls_prun.py` for regression/classification tasks with pruning.

2. **Perform Signal-to-Noise Ratio Pruning:**
   Use `signal to noise ratio.ipynb` to explore pruning techniques and their impact on model performance.


3. **Run Summary Analysis:**
   Open `Summary.ipynb` in Jupyter Notebook and execute the cells to understand the framework and results.

4. **Convergence Testing:**
   - Use `convergence_test.py` to ensure the MCMC sampling convergence for your BNN models.

---

## Contributions

Contributions are welcome! Please follow the standard GitHub workflow:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Submit a pull request.

---


## Contact

For any queries or discussions, please contact:

**Ratneel Deo**  
[GitHub Profile](https://github.com/rvdeo)
