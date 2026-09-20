"""Prior predictive check: accuracy, RT, and double-response rate."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from step3_vectorized_simulator import RT_EXCLUSION_MIN
from step4_bayesflow_priors_simulator import sample_prior, simulate_dataset

PARTICIPANT_FILES = {
    1: "Exp1/participant_1.csv",
    2: "Exp1/participant_2.csv",
    3: "Exp1/participant_3.csv",
    4: "Exp1/participant_4.csv",
}

N_DRAWS = 200
N_TRIALS = 2000


def compute_real_summary_stats():
    # Applies the same rt >= 0.25s exclusion used everywhere else, and
    # computes stats directly from the CSVs rather than hardcoding them.
    accuracy, mean_rt, p_double = {}, {}, {}
    for pid, filepath in PARTICIPANT_FILES.items():
        df = pd.read_csv(filepath)
        n_before = len(df)
        df = df[df["Time1"] >= RT_EXCLUSION_MIN].copy()

        label = f"P{pid} ({'speed' if pid in [1, 2] else 'accuracy'})"
        accuracy[label] = df["Correct"].astype(bool).mean()
        mean_rt[label] = df["Time1"].mean()
        p_double[label] = df["DoubleResp"].astype(bool).mean()

        print(f"  Participant {pid}: {n_before} trials -> "
              f"{len(df)} retained ({n_before - len(df)} removed for "
              f"rt < {RT_EXCLUSION_MIN}s)")

    return accuracy, mean_rt, p_double


def run_prior_predictive_check(n_draws=N_DRAWS, n_trials=N_TRIALS, seed=0):
    rng = np.random.default_rng(seed)
    params = sample_prior(n_draws, rng=rng)

    accuracy_vals = np.zeros(n_draws)
    mean_rt_vals = np.zeros(n_draws)
    p_double_vals = np.zeros(n_draws)

    for i in range(n_draws):
        data = simulate_dataset(params["v_A"][i], params["v_B"][i],
                                 params["beta"][i], params["b"][i],
                                 params["t0"][i], n_trials=n_trials, rng=rng)
        accuracy_vals[i] = 1.0 - data["choice"].mean()
        mean_rt_vals[i] = data["rt"].mean()
        p_double_vals[i] = data["double_response"].mean()

    return params, accuracy_vals, mean_rt_vals, p_double_vals


if __name__ == "__main__":
    print("Computing real participant summary statistics (filtered, "
          f"rt >= {RT_EXCLUSION_MIN}s)...")
    real_accuracy, real_mean_rt, real_p_double = compute_real_summary_stats()

    print(f"\nRunning prior predictive check with {N_DRAWS} draws, "
          f"{N_TRIALS} trials each...")
    params, accuracy_vals, mean_rt_vals, p_double_vals = run_prior_predictive_check()

    checks = [
        ("Accuracy", accuracy_vals, real_accuracy),
        ("Mean RT (s)", mean_rt_vals, real_mean_rt),
        ("P(double response)", p_double_vals, real_p_double),
    ]

    for label, sim_vals, real_dict in checks:
        print(f"\n{label}: simulated median={np.median(sim_vals):.4f}, "
              f"5th-95th pct=[{np.percentile(sim_vals, 5):.4f}, "
              f"{np.percentile(sim_vals, 95):.4f}]")
        for name, val in real_dict.items():
            pct_below = (sim_vals < val).mean() * 100
            print(f"  {name}: real={val:.4f}  "
                  f"({pct_below:.1f}% of simulated draws were below this)")

    # Follow-up from step4: v_B > v_A on ~19.5% of prior draws -- check
    # whether those draws behave noticeably differently.
    v_B_larger = params["v_B"] > params["v_A"]
    print(f"\nv_B > v_A on {v_B_larger.mean():.1%} of these {N_DRAWS} draws.")
    print(f"  Accuracy   -- v_A>v_B: {accuracy_vals[~v_B_larger].mean():.3f}   "
          f"v_B>v_A: {accuracy_vals[v_B_larger].mean():.3f}")
    print(f"  Mean RT    -- v_A>v_B: {mean_rt_vals[~v_B_larger].mean():.3f}   "
          f"v_B>v_A: {mean_rt_vals[v_B_larger].mean():.3f}")
    print(f"  P(double)  -- v_A>v_B: {p_double_vals[~v_B_larger].mean():.3f}   "
          f"v_B>v_A: {p_double_vals[v_B_larger].mean():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ["#e07a5f", "#3d5a80", "#81b29a", "#f2cc8f"]
    for ax, (label, sim_vals, real_dict) in zip(axes, checks):
        ax.hist(sim_vals, bins=30, color="#4a7c8c", alpha=0.7,
                label="Simulated (from priors)")
        for (name, val), c in zip(real_dict.items(), colors):
            ax.axvline(val, color=c, linestyle="--", linewidth=2, label=name)
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.legend(fontsize=7)
    fig.suptitle("Prior predictive check: accuracy, RT, and double response "
                  "(both real and simulated exclude rt < 0.25s)")
    fig.tight_layout()
    fig.savefig("prior_predictive_check.png", dpi=150)
    print("\nSaved plot to prior_predictive_check.png")