Simulation Based Inference — Double Responses in Decision Tasks

A **Simulation-Based Inference (SBI)** project investigating **double responses in speeded decision-making tasks**, based on the work of Evans et al. (2020).

The project combines computational modeling, simulation, Bayesian inference, and real experimental data to study whether evidence-accumulation models can reproduce the occurrence of double responses.

## 📌 Project Overview

In many decision-making experiments, participants are asked to make a fast choice between two alternatives. Traditionally, models of decision making are evaluated using:

* **Choice** — which option was selected
* **Response Time (RT)** — how long the decision took

The paper by Evans et al. introduced an additional behavioral constraint: **double responding**.

A double response occurs when a participant gives an initial response and then produces a second response shortly afterward.

This project investigates this phenomenon using a **lateral-inhibition evidence accumulation model** and **Simulation-Based Inference (SBI)** with BayesFlow.

### Main objectives

1. Understand and implement the double-response mechanism.
2. Build a basic evidence-accumulation simulator.
3. Extend the model with lateral inhibition.
4. Optimize the simulator using vectorization.
5. Integrate the simulator with BayesFlow.
6. Define appropriate Bayesian priors.
7. Perform prior predictive checks.
8. Train neural posterior estimation models.
9. Apply the trained inference model to real experimental data.
10. Compare simulated and observed behavioral patterns.

---

## 📚 Reference

The project is based primarily on:

> Evans, N. J., Dutilh, G., Wagenmakers, E.-J., & van der Maas, H. L. J. (2020). *Double responding: A new constraint for models of speeded decision making*. Cognitive Psychology, 121, 101292.

The central idea is that double responses provide an additional constraint for evaluating models of speeded decision making.

---

## 🧠 Model

The project uses two competing evidence accumulators:

* Accumulator **A**
* Accumulator **B**

Each accumulator collects noisy evidence over time.

Important model parameters include:

| Parameter  | Description                 |
| ---------- | --------------------------- |
| `v_A`      | Drift rate of accumulator A |
| `v_B`      | Drift rate of accumulator B |
| `beta`     | Lateral inhibition strength |
| `b`        | Decision threshold          |
| `t0`       | Non-decision time           |
| `noise_sd` | Noise standard deviation    |
| `dt`       | Simulation time step        |
| `max_time` | Maximum simulation time     |

### Lateral inhibition

The accumulators inhibit each other during evidence accumulation. This creates competition between the two response alternatives and allows the model to reproduce more complex response behavior.

### Double-response rule

After the first accumulator reaches the decision threshold, the losing accumulator is monitored for a short period.

In this implementation:

```text
double-response window = 250 ms
```

If the losing accumulator reaches the threshold within this window, the trial is classified as a **double response**.

---

## 🔬 Simulation Pipeline

The project was developed incrementally.

### Step 1 — Basic RDM Simulator

A basic Random-Walk / Random-Diffusion style simulator was implemented to establish the fundamental evidence-accumulation process.

```text
step1_basic_rdm_simulator.py
```

This provided the foundation for later model development.

---

### Step 2 — Lateral-Inhibition Double-Response Model

The basic simulator was extended to include:

* Two competing accumulators
* Lateral inhibition
* Decision thresholds
* Non-decision time
* Double-response detection

```text
step2_lateral_inhibition_double_response.py
```

Example simulation configuration:

```text
v_A       = 1.2
v_B       = 0.9
beta      = 0.8
b         = 1.0
t0        = 0.25
noise_sd  = 1.0
n_trials  = 5000
```

Example results:

```text
Proportion choosing A  ≈ 0.569
Mean RT                ≈ 0.766 s
P(double response)     ≈ 0.152
```

---

### Step 3 — Vectorized Simulator

The simulation was then optimized using vectorized NumPy operations.

```text
step3_vectorized_simulator.py
```

For 5,000 trials, the vectorized implementation reduced simulation time substantially.

Example:

```text
Simulation time        ≈ 0.159 s
Proportion choosing A  ≈ 0.576
Mean RT                ≈ 0.767 s
P(double response)     ≈ 0.139
```

The optimized version achieved approximately a **21× speedup** compared with the original implementation.

---

### Step 4 — BayesFlow Simulator

The simulator was integrated with **BayesFlow** so that parameters could be sampled from prior distributions and used to generate simulated datasets.

```text
step4_bayesflow_priors_simulator.py
```

The simulator became the forward model for Simulation-Based Inference.

---

## 📊 Bayesian Priors

The SBI model uses prior distributions for the main model parameters.

The current prior setup includes truncated normal distributions:

| Parameter | Prior             |
| --------- | ----------------- |
| `v_A`     | Normal(3.0, 1.5)  |
| `v_B`     | Normal(1.5, 1.0)  |
| `beta`    | Normal(3.0, 1.5)  |
| `b`       | Normal(1.4, 0.4)  |
| `t0`      | Normal(0.3, 0.15) |

For `t0`, the prior is constrained to:

```text
0.05 ≤ t0 ≤ 0.60
```

These priors define the parameter space from which BayesFlow generates simulated datasets.

---

## 🔍 Prior Predictive Checks

Before fitting the model to real data, simulations are generated from the prior distribution.

This is called a **prior predictive check**.

The purpose is to determine whether the chosen priors produce behavior that is reasonably similar to the type of data observed experimentally.

Important summary statistics include:

* Choice proportions
* Response-time distributions
* Double-response probability
* Relationships between choice and RT
* Relationships between RT and double responses

If the prior predictive simulations produce unrealistic behavior, the priors should be reconsidered before training the SBI model.

---

## 🤖 Simulation-Based Inference

Traditional Bayesian inference can become difficult when the likelihood of a complex simulator cannot easily be calculated.

SBI solves this problem by learning the relationship between:

```text
Parameters → Simulated Data
```

A large number of parameter values are sampled from the prior, and the simulator generates corresponding datasets.

BayesFlow then learns to perform approximate Bayesian inference from these simulations.

The overall workflow is:

```text
Prior
  ↓
Sample model parameters
  ↓
Simulator
  ↓
Generate synthetic data
  ↓
Summary / neural network
  ↓
Train SBI model
  ↓
Real experimental data
  ↓
Approximate posterior
  ↓
Posterior parameter estimates
```

---

## 🧪 Real Data Analysis

After validating the simulator and training the SBI model, the trained inference pipeline is applied to real experimental data.

The real data contain information about:

* Initial choice
* Response time
* Double-response behavior

The observed response is encoded consistently with the original analysis.

In particular, the first response is represented as:

```text
0 = correct first response
1 = error
```

The trained SBI model can then estimate the posterior distribution of the underlying model parameters.

---

## ⚙️ Important Implementation Details

### Time-step consistency

The simulation originally used different time steps in different implementations.

This was aligned to:

```text
dt = 0.002 seconds
```

to ensure that the implementations could be compared fairly.

### Tie handling

When both accumulators reach the threshold at effectively the same time, ties are handled using a random coin flip rather than deterministically assigning the trial to accumulator A.

### Floating-point precision

Repeated addition of `dt` can produce values such as:

```text
0.25000000000000017
```

which can cause incorrect classification around the 250 ms double-response boundary.

The implementation therefore uses an integer simulation step:

```text
time = step * dt
```

to avoid this floating-point comparison problem.

---

## 📁 Project Structure

A simplified project structure is:

```text
.
├── step1_basic_rdm_simulator.py            # base RDM, dev/sanity check only
├── step2_lateral_inhibition_double_response.py
├── step3_vectorized_simulator.py           # fast, batched simulator used for training
├── step4_bayesflow_priors_simulator.py     # priors + BayesFlow simulator wrapper
├── step5_prior_predictive_check.py
├── step6_train_bayesflow.py                # initial training (0→40 epochs)
├── step6_continue_training.py              # continuation (40→60 epochs)
├── step6_continue_epoch60_to80.py          # continuation (60→80 epochs)
├── step6_preflight.py
├── step7_fit_real_data.py                  # posterior sampling + PPCs on real participants
│
├── Exp1/                                   # raw participant data (Dutilh et al., 2009)
├── checkpoints/                            # saved BayesFlow approximators (epoch 40/60/80)
├── step7_results/                          # posterior summaries, PPC figures, CSVs
├── model_configuration.json                # serialized network architecture
├── optimizer_configuration.json            # serialized optimizer / LR schedule
├── *.png                                   # diagnostic plots (recovery, SBC, contraction, loss)
│
└── report/                                 # LaTeX source and compiled report (see below)
```

The exact file structure may change as the project develops.

---

## 🛠️ Technologies

The project primarily uses:

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **JAX**
* **BayesFlow**
* **SciPy**
* **Jupyter Notebook**

The main development environment used during the project includes macOS and Python-based scientific computing tools.


## 📈 Expected Workflow

The recommended order for reproducing the project is:

```text
1. Basic simulation
       ↓
2. Lateral inhibition
       ↓
3. Double-response mechanism
       ↓
4. Vectorization / optimization
       ↓
5. Bayesian priors
       ↓
6. Prior predictive checks
       ↓
7. BayesFlow simulation pipeline
       ↓
8. SBI training
       ↓
9. Posterior predictive validation
       ↓
10. Real-data inference
```

---

## 🎯 Project Goals

The final goal is not simply to simulate double responses.

The main goal is to determine whether a computational evidence-accumulation model can explain the observed behavior and whether **Simulation-Based Inference** can recover meaningful model parameters from realistic experimental data.

The project therefore connects:

```text
Decision-Making Theory
        +
Computational Modeling
        +
Bayesian Statistics
        +
Simulation-Based Inference
        +
Neural Networks
        +
Real Experimental Data
```

---

## 📌 Key Takeaways

* Double responses provide an additional constraint for evaluating decision models.
* Lateral inhibition can produce realistic competition between response alternatives.
* Vectorization makes large-scale simulation much faster.
* Bayesian priors define the parameter space explored by SBI.
* Prior predictive checks help determine whether the chosen priors are reasonable.
* BayesFlow allows approximate Bayesian inference when an analytical likelihood is difficult or unavailable.
* Real experimental data can be used to estimate posterior distributions over the model parameters.

---

## 📖 Acknowledgements

This project was developed as part of a **Simulation-Based Inference** project at **TU Dortmund University**.

The conceptual foundation comes from the work of Evans et al. on double responding in speeded decision-making tasks.

---

## 📄 License

This project is intended for academic and educational purposes.

If you reuse the code or methodology, please cite the original paper and acknowledge this project.

---

## 👤 Author

**Showaibuzzaman Shihab**

Master's Student — Data Science
TU Dortmund University

**Project:** Simulation-Based Inference — Double Responses in Decision Tasks
