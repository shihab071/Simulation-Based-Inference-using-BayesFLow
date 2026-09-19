"""Priors and simulator packaging for BayesFlow."""

import numpy as np
from step3_vectorized_simulator import simulate_trials_filtered


def sample_prior(batch_size, rng=None):
    """Sample independent parameter values from the training priors."""
    if rng is None:
        rng = np.random.default_rng()

    def trunc_normal(mean, sd, lower=0.0, upper=np.inf, size=1):
        out = np.empty(size)
        for i in range(size):
            val = -1
            while val < lower or val > upper:
                val = rng.normal(mean, sd)
            out[i] = val
        return out

    v_A = trunc_normal(3.0, 1.5, lower=0.05, size=batch_size)
    v_B = trunc_normal(1.5, 1.0, lower=0.05, size=batch_size)
    beta = trunc_normal(3.0, 1.5, lower=0.0, size=batch_size)
    b = trunc_normal(1.4, 0.4, lower=0.3, size=batch_size)
    t0 = trunc_normal(0.3, 0.15, lower=0.05, upper=0.6, size=batch_size)

    return {"v_A": v_A, "v_B": v_B, "beta": beta, "b": b, "t0": t0}


def simulate_dataset(v_A, v_B, beta, b, t0, n_trials=500, rng=None):
    # RT exclusion applied via simulate_trials_filtered, matching how the
    # real data is filtered from step5 onward.
    result = simulate_trials_filtered(v_A, v_B, beta, b, t0, n_trials,
                                       rng=rng)
    choice_numeric = (result["choice"] == "B").astype(float)
    drt = np.nan_to_num(result["drt"], nan=0.0)
    return {
        "choice": choice_numeric,
        "rt": result["rt"].astype(float),
        "double_response": result["double_response"].astype(float),
        "drt": drt.astype(float),
    }


def bayesflow_simulator(batch_size, n_trials=500, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    params = sample_prior(batch_size, rng=rng)

    all_choice = np.zeros((batch_size, n_trials))
    all_rt = np.zeros((batch_size, n_trials))
    all_dr = np.zeros((batch_size, n_trials))
    all_drt = np.zeros((batch_size, n_trials))

    for i in range(batch_size):
        data = simulate_dataset(params["v_A"][i], params["v_B"][i],
                                 params["beta"][i], params["b"][i],
                                 params["t0"][i], n_trials=n_trials, rng=rng)
        all_choice[i] = data["choice"]
        all_rt[i] = data["rt"]
        all_dr[i] = data["double_response"]
        all_drt[i] = data["drt"]

    return {
        "v_A": params["v_A"], "v_B": params["v_B"], "beta": params["beta"],
        "b": params["b"], "t0": params["t0"],
        "choice": all_choice, "rt": all_rt,
        "double_response": all_dr, "drt": all_drt,
    }


if __name__ == "__main__":
    print("Testing prior sampling...")
    rng = np.random.default_rng(0)
    p = sample_prior(5, rng=rng)
    for k, v in p.items():
        print(f"  {k}: {np.round(v, 3)}")

    print("\nTesting one simulated dataset (100 trials)...")
    data = simulate_dataset(v_A=3.0, v_B=1.5, beta=1.0, b=1.0, t0=0.3,
                             n_trials=100, rng=rng)
    print(f"  Mean RT: {data['rt'].mean():.3f}")
    print(f"  P(choice=B): {data['choice'].mean():.3f}")
    print(f"  P(double response): {data['double_response'].mean():.3f}")

    print("\nTesting full batch simulator (batch_size=4, n_trials=200)...")
    batch = bayesflow_simulator(batch_size=4, n_trials=200, rng=rng)
    for i in range(4):
        print(f"  Sim {i}: v_A={batch['v_A'][i]:.2f}, v_B={batch['v_B'][i]:.2f}, "
              f"beta={batch['beta'][i]:.2f}, b={batch['b'][i]:.2f}, "
              f"t0={batch['t0'][i]:.2f}  ->  "
              f"meanRT={batch['rt'][i].mean():.3f}, "
              f"P(double)={batch['double_response'][i].mean():.3f}")

    print("\nRunning batch shape and value checks...")
    assert batch["v_A"].shape == (4,)
    assert batch["choice"].shape == (4, 200)
    assert batch["rt"].shape == (4, 200)
    assert batch["double_response"].shape == (4, 200)
    assert batch["drt"].shape == (4, 200)

    assert np.all(np.isfinite(batch["rt"]))
    assert np.all(batch["rt"] >= 0.25)
    assert np.all(np.isfinite(batch["drt"]))

    assert np.all(np.isin(batch["choice"], [0.0, 1.0]))
    assert np.all(np.isin(batch["double_response"], [0.0, 1.0]))

    no_double = batch["double_response"] == 0
    assert np.all(batch["drt"][no_double] == 0.0)

    has_double = batch["double_response"] == 1
    assert np.all(batch["drt"][has_double] >= 0.0)
    assert np.all(batch["drt"][has_double] <= 0.25)

    print("Batch shape and value checks passed.")

    print("\nRunning prior-bound checks (10,000 samples)...")
    prior_check = sample_prior(batch_size=10_000, rng=np.random.default_rng(123))

    assert np.all(prior_check["v_A"] >= 0.05)
    assert np.all(prior_check["v_B"] >= 0.05)
    assert np.all(prior_check["beta"] >= 0.0)
    assert np.all(prior_check["b"] >= 0.3)
    assert np.all((prior_check["t0"] >= 0.05) & (prior_check["t0"] <= 0.6))

    print("Prior-bound checks passed.")

    v_B_larger = np.mean(prior_check["v_B"] > prior_check["v_A"])
    print(f"\nP(v_B > v_A) under the prior: {v_B_larger:.3f} "
          f"(priors favor v_A > v_B on average but don't enforce it -- "
          f"worth examining in step5's prior predictive check)")

    # Sanity check at the real participants' trial count (10,000).
    print("\nSanity check at n_trials=10000 (matches the real participants)...")
    big = simulate_dataset(v_A=3.0, v_B=1.5, beta=3.0, b=1.4, t0=0.3,
                            n_trials=10000, rng=rng)
    print(f"  Mean RT: {big['rt'].mean():.3f}, "
          f"P(double response): {big['double_response'].mean():.4f}")

