# PINNs for Harmonic Oscillators

This repository implements Physics-Informed Neural Networks (PINNs) to solve the Ordinary Differential Equations (ODEs) governing both Simple Harmonic Motion (SHO) and Damped Harmonic Motion. By embedding the physical laws directly into the neural network's loss function, we allow the model to learn the trajectory of the oscillator without needing a labeled dataset—only the initial conditions and the governing equations.

## 1. Mathematical Background

### Simple Harmonic Oscillator (SHO)

The governing equation for an undamped oscillator is:

$$
\frac{d^2x}{dt^2} + \omega_0^2x = 0
$$

### Damped Harmonic Oscillator

The governing equation for a damped oscillator is:

$$
\frac{d^2x}{dt^2} + 2\gamma\frac{dx}{dt} + \omega_0^2x = 0
$$

Where:
- x : Position
- $\gamma$ : Damping coefficient
- $\omega_o$ : Natural frequency

## 2. Model Parameters & Hyperparameters

### Physics Parameters for Harmonic Oscillator
- ( $\omega_0 $) (Natural Frequency): 1.0

### Initial Conditions:
- x(0) =  0, 
- v(0) =  1

### Physics Parameters for Damped Harmonic Oscillator

- $\omega_0 $ (Natural Frequency): 10.0
- $\gamma  $ (Damping Ratio): 2

### Initial Conditions:
- $ x(0) = $ 0.1, 
- $ v(0) = $ 0

### Loss Weight Configuration
We use a weighted loss function to balance the Initial Condition (IC) and the Physics (ODE) residual:

$$
Loss_{total} = W_{ic} \cdot Loss_{ic} + W_{phys} \cdot Loss_{phys}
$$

| Case                | \( W_{ic} \) (Initial Condition) | \( W_{phys} \) (Physics Residual) |
|---------------------|----------------------------------|-----------------------------------|
| Harmonic Oscillator | 100                              | 10                                |
| Damped Oscillator   | 500                              | 10                                |

## 3. Optimization Strategy

During development, two optimization approaches were tested:

- **Adam Optimizer**: Good for initial convergence but often struggled to reach the high precision required for the oscillatory "wiggles" at the end of the time domain.
- **LBFGS Optimizer**: Performed significantly better for this second-order ODE. Its ability to use second-order derivatives allowed the PINN to "snap" to the true solution much faster and with higher accuracy.

### Conclusion:
Based on the results of the SHO case, the Damped Harmonic Oscillator was trained exclusively using LBFGS to ensure temporal accuracy across the entire domain.

## 4. Results and Visualizations

All generated plots can be found in the `/images` folder.

### SHO Comparison
Comparison between the analytical solution and the PINN prediction using LBFGS.

### Damped SHO Comparison
The PINN successfully captures the exponential decay and phase shift of the damped system.

### Loss vs. Epochs
The convergence profile showing the reduction in both physics and boundary residuals.

## 5. How to Run

1. Ensure you have PyTorch and Matplotlib installed.
2. Clone the repository:
   ```bash
   git clone https://github.com/rajdipthepower/Machine-Learning__Deep-Learning/new/main/Physics%20Informed%20Neural%20Network
3. Run the notebook: jupyter notebook SHO_PINN.ipynb
