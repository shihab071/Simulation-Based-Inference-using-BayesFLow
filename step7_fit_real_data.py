"""Fit the trained LI model to the four empirical participants."""

import os
from pathlib import Path

# Set the backend before importing Keras.
os.environ.setdefault("KERAS_BACKEND", "jax")

# Import BayesFlow before loading the saved Keras model. This registers
# BayesFlow's custom layers and the ContinuousApproximator with Keras.
import bayesflow as bf
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step3_vectorized_simulator import (
    RT_EXCLUSION_MIN,
    simulate_trials_filtered,
)


# The first two participants completed the speed condition; the other two
# completed the accuracy condition. Each CSV originally contains 10,000 trials.
PARTICIPANT_FILES = {
    1: Path("Exp1/participant_1.csv"),
    2: Path("Exp1/participant_2.csv"),
    3: Path("Exp1/participant_3.csv"),
    4: Path("Exp1/participant_4.csv"),
}

PARAMETER_NAMES = ["v_A", "v_B", "beta", "b", "t0"]
PARAMETER_TITLES = {
    "v_A": r"Drift $v_A$",
    "v_B": r"Drift $v_B$",
    "beta": r"Inhibition $\beta$",
    "b": r"Threshold $b$",
    "t0": r"Non-decision time $t_0$",
}

# Epoch 60 is used for the main analysis. Epoch 80 had a lower training loss,
# but its SBC plots showed poorer calibration for beta. The epoch-80 model is
# kept as a possible sensitivity analysis rather than used for the main fit.
MODEL_PATH = Path("checkpoints/li_model_approximator_epoch60.keras")
RESULTS_DIR = Path("step7_results")

SEED = 2026
N_POSTERIOR_SAMPLES = 1_000
N_PPC_DRAWS = 100
RNG = np.random.default_rng(SEED)

REQUIRED_COLUMNS = {"Correct", "Time1", "DoubleResp", "Time2"}


def save_figure(fig, filename):
    """Save a figure and close it so repeated PPCs do not build up in memory."""
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def participant_label(pid):
    condition = "speed" if pid in (1, 2) else "accuracy"
    return f"P{pid} ({condition})"


def load_participant(filepath):
    """Load one participant and apply the same RT rule used in training."""
    if not filepath.is_file():
        raise FileNotFoundError(f"Participant file not found: {filepath}")

    df = pd.read_csv(filepath)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{filepath} is missing columns: {sorted(missing)}")

    n_before = len(df)
    df = df.loc[df["Time1"] >= RT_EXCLUSION_MIN].copy()
    n_removed = n_before - len(df)

    if df.empty:
        raise ValueError(f"No usable trials remain in {filepath}")

    print(
        f"  {filepath}: {n_before} trials -> {len(df)} retained "
        f"({n_removed} removed for RT < {RT_EXCLUSION_MIN}s)"
    )

    correct = df["Correct"].astype(bool).to_numpy()
    rt = df["Time1"].astype(float).to_numpy()
    double_response = df["DoubleResp"].astype(bool).to_numpy()
    drt_raw = df["Time2"].astype(float).to_numpy()

    # The simulator codes A as the correct response and B as the error response.
    # BayesFlow receives numeric choices: 0=A/correct and 1=B/error.
    choice = (~correct).astype(float)

    # Time2 is missing when there was no second response. The separate
    # DoubleResp indicator tells the network whether the zero is a placeholder.
    drt_for_network = np.nan_to_num(drt_raw, nan=0.0)

    return {
        "choice": choice[None, :],
        "rt": rt[None, :],
        "double_response": double_response.astype(float)[None, :],
        "drt": drt_for_network[None, :],
        "n_trials": len(df),
        # Keep convenient one-dimensional versions for the PPC plots.
        "correct": correct,
        "rt_raw": rt,
        "double_response_bool": double_response,
        "drt_raw": drt_raw,
    }


def sample_posterior(approximator, filepath, seed):
    """Draw posterior samples for one participant."""
    data = load_participant(filepath)
    conditions = {
        key: data[key]
        for key in ("choice", "rt", "double_response", "drt", "n_trials")
    }

    # batch_size=1 keeps the 10,000-trial summary pass memory-safe.
    posterior = approximator.sample(
        conditions=conditions,
        num_samples=N_POSTERIOR_SAMPLES,
        batch_size=1,
        seed=seed,
    )
    return posterior, data


def summarize_posterior(posterior):
    """Return posterior means, standard deviations, and 95% intervals."""
    summary = {}

    for name in PARAMETER_NAMES:
        samples = np.asarray(posterior[name]).reshape(-1)
        low, high = np.percentile(samples, [2.5, 97.5])
        summary[name] = {
            "mean": samples.mean(),
            "sd": samples.std(ddof=1),
            "ci_low": low,
            "ci_high": high,
            "samples": samples,
        }

        print(
            f"  {name}: mean={samples.mean():.3f}, "
            f"95% CI=[{low:.3f}, {high:.3f}]"
        )

    return summary


def simulate_posterior_predictive(summary, n_trials):
    """Simulate complete datasets from 100 randomly selected posterior draws."""
    n_available = len(summary["v_A"]["samples"])
    if N_PPC_DRAWS > n_available:
        raise ValueError("N_PPC_DRAWS cannot exceed N_POSTERIOR_SAMPLES")

    selected = RNG.choice(n_available, size=N_PPC_DRAWS, replace=False)
    simulations = []

    for index in selected:
        sim = simulate_trials_filtered(
            v_A=summary["v_A"]["samples"][index],
            v_B=summary["v_B"]["samples"][index],
            beta=summary["beta"]["samples"][index],
            b=summary["b"]["samples"][index],
            t0=summary["t0"]["samples"][index],
            n_trials=n_trials,
            rng=RNG,
        )
        simulations.append(sim)

    return simulations


def predictive_interval(values):
    """Return the mean and central 90% predictive interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    low, high = np.percentile(values, [5, 95])
    return values.mean(), low, high


def ppc_accuracy_and_mean_rt(pid, simulations, real_data):
    """Compare empirical accuracy and mean RT with posterior predictions."""
    real_accuracy = real_data["correct"].mean()
    real_mean_rt = real_data["rt_raw"].mean()

    sim_accuracy = np.array(
        [np.mean(sim["choice"] == "A") for sim in simulations]
    )
    sim_mean_rt = np.array([sim["rt"].mean() for sim in simulations])

    acc_mean, acc_low, acc_high = predictive_interval(sim_accuracy)
    rt_mean, rt_low, rt_high = predictive_interval(sim_mean_rt)

    print(
        f"\n  Accuracy: real={real_accuracy:.4f}, predicted mean={acc_mean:.4f}, "
        f"90% PI=[{acc_low:.4f}, {acc_high:.4f}]"
    )
    print(
        f"  Mean RT: real={real_mean_rt:.4f}, predicted mean={rt_mean:.4f}, "
        f"90% PI=[{rt_low:.4f}, {rt_high:.4f}]"
    )

    return [
        {
            "participant": pid,
            "statistic": "accuracy",
            "real": real_accuracy,
            "predicted_mean": acc_mean,
            "pi_5": acc_low,
            "pi_95": acc_high,
        },
        {
            "participant": pid,
            "statistic": "mean_rt",
            "real": real_mean_rt,
            "predicted_mean": rt_mean,
            "pi_5": rt_low,
            "pi_95": rt_high,
        },
    ]


def ppc_double_response_rates(pid, simulations, real_data):
    """Check overall and choice-conditional double-response rates."""
    real_correct = real_data["correct"]
    real_double = real_data["double_response_bool"]

    real_values = {
        "double_overall": real_double.mean(),
        "double_given_correct": real_double[real_correct].mean(),
        "double_given_error": real_double[~real_correct].mean(),
    }

    simulated_values = {key: [] for key in real_values}
    for sim in simulations:
        sim_correct = sim["choice"] == "A"
        sim_double = sim["double_response"]

        simulated_values["double_overall"].append(sim_double.mean())
        simulated_values["double_given_correct"].append(
            sim_double[sim_correct].mean() if sim_correct.any() else np.nan
        )
        simulated_values["double_given_error"].append(
            sim_double[~sim_correct].mean() if (~sim_correct).any() else np.nan
        )

    rows = []
    print()
    for key in real_values:
        mean, low, high = predictive_interval(simulated_values[key])
        print(
            f"  {key}: real={real_values[key]:.4f}, predicted mean={mean:.4f}, "
            f"90% PI=[{low:.4f}, {high:.4f}]"
        )
        rows.append(
            {
                "participant": pid,
                "statistic": key,
                "real": real_values[key],
                "predicted_mean": mean,
                "pi_5": low,
                "pi_95": high,
            }
        )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    display_labels = ["Overall", "Given correct", "Given error"]

    for position, key in enumerate(real_values):
        values = np.asarray(simulated_values[key], dtype=float)
        values = values[np.isfinite(values)]
        mean, low, high = predictive_interval(values)

        # Light jitter keeps overlapping posterior predictions visible.
        jitter = RNG.normal(0.0, 0.035, size=len(values))
        ax.scatter(
            position + jitter,
            values,
            color="#E76F51",
            alpha=0.25,
            s=18,
            label="Posterior predictions" if position == 0 else None,
        )
        ax.errorbar(
            position,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            color="#B23A48",
            capsize=5,
            label="Predicted mean and 90% PI" if position == 0 else None,
        )
        ax.scatter(
            position,
            real_values[key],
            marker="D",
            s=65,
            color="#277DA1",
            zorder=5,
            label="Observed" if position == 0 else None,
        )

    ax.set_xticks(range(3), display_labels)
    ax.set_ylabel("Probability of a double response")
    ax.set_title(f"{participant_label(pid)}: double-response PPC")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, f"ppc_participant{pid}_double_response.png")

    return rows


def ppc_rt_distributions(pid, simulations, real_data):
    """Plot RT distributions separately for correct and error responses."""
    real_rt = real_data["rt_raw"]
    real_correct = real_data["correct"]

    simulated_groups = {
        "Correct responses": [
            sim["rt"][sim["choice"] == "A"] for sim in simulations
        ],
        "Error responses": [
            sim["rt"][sim["choice"] == "B"] for sim in simulations
        ],
    }
    real_masks = {
        "Correct responses": real_correct,
        "Error responses": ~real_correct,
    }

    # All observed RTs fall below 2.5 s, and the simulator uses the same
    # decision-time cutoff. Shared bins make the two panels comparable.
    bins = np.linspace(RT_EXCLUSION_MIN, 2.5, 40)
    centers = (bins[:-1] + bins[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, label in zip(axes, simulated_groups):
        densities = np.array(
            [
                np.histogram(values, bins=bins, density=True)[0]
                for values in simulated_groups[label]
                if len(values) > 5
            ]
        )

        if densities.size == 0:
            raise RuntimeError(f"Too few simulated {label.lower()} for P{pid}")

        low, median, high = np.percentile(densities, [5, 50, 95], axis=0)
        ax.fill_between(
            centers,
            low,
            high,
            color="#E76F51",
            alpha=0.3,
            label="Predicted 90% band",
        )
        ax.plot(centers, median, color="#B23A48", label="Predicted median")
        ax.hist(
            real_rt[real_masks[label]],
            bins=bins,
            density=True,
            alpha=0.45,
            color="#277DA1",
            label="Observed",
        )
        ax.set_title(label)
        ax.set_xlabel("Response time (s)")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Density")
    fig.suptitle(f"{participant_label(pid)}: response-time PPC")
    fig.tight_layout()
    save_figure(fig, f"ppc_participant{pid}_rt.png")


def ppc_drt_distribution(pid, simulations, real_data):
    """Compare the delay between first and second responses."""
    real_drt = real_data["drt_raw"]
    real_drt = real_drt[np.isfinite(real_drt)]
    simulated_drt = [
        sim["drt"][sim["double_response"]] for sim in simulations
    ]

    print(
        f"\n  Double-response times: {len(real_drt)} observed; "
        f"{np.mean([len(x) for x in simulated_drt]):.1f} predicted per draw"
    )
    if len(real_drt) < 15:
        print("  Too few observed double responses for a stable DRT comparison.")

    bins = np.linspace(0.0, 0.25, 20)
    centers = (bins[:-1] + bins[1:]) / 2
    densities = np.array(
        [
            np.histogram(values, bins=bins, density=True)[0]
            for values in simulated_drt
            if len(values) > 3
        ]
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if densities.size:
        low, median, high = np.percentile(densities, [5, 50, 95], axis=0)
        ax.fill_between(
            centers,
            low,
            high,
            color="#E76F51",
            alpha=0.3,
            label="Predicted 90% band",
        )
        ax.plot(centers, median, color="#B23A48", label="Predicted median")

    if len(real_drt) > 3:
        ax.hist(
            real_drt,
            bins=bins,
            density=True,
            alpha=0.45,
            color="#277DA1",
            label="Observed",
        )

    ax.set_xlabel("Time after the first response (s)")
    ax.set_ylabel("Density")
    ax.set_title(f"{participant_label(pid)}: double-response-time PPC")
    if ax.has_data():
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, f"ppc_participant{pid}_drt.png")


def plot_parameter_estimates(all_summaries):
    """Plot posterior means and 95% credible intervals for all parameters."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.8))

    for ax, parameter in zip(axes, PARAMETER_NAMES):
        for position, pid in enumerate(PARTICIPANT_FILES):
            stats = all_summaries[pid][parameter]
            color = "#277DA1" if pid in (1, 2) else "#E76F51"
            ax.errorbar(
                stats["mean"],
                position,
                xerr=[
                    [stats["mean"] - stats["ci_low"]],
                    [stats["ci_high"] - stats["mean"]],
                ],
                fmt="o",
                color=color,
                capsize=4,
                markersize=7,
            )

        ax.set_yticks(
            range(4),
            [participant_label(pid) for pid in PARTICIPANT_FILES],
        )
        ax.invert_yaxis()
        ax.set_title(PARAMETER_TITLES[parameter])
        ax.set_xlabel("Posterior estimate (95% CI)")
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Participant-level parameter estimates")
    fig.tight_layout()
    save_figure(fig, "parameter_estimates_with_uncertainty.png")


def save_numeric_results(all_summaries, ppc_rows):
    """Save the values behind the figures for later use in the report."""
    posterior_rows = []
    for pid, participant_summary in all_summaries.items():
        for parameter, stats in participant_summary.items():
            posterior_rows.append(
                {
                    "participant": pid,
                    "condition": "speed" if pid in (1, 2) else "accuracy",
                    "parameter": parameter,
                    "mean": stats["mean"],
                    "sd": stats["sd"],
                    "ci_2.5": stats["ci_low"],
                    "ci_97.5": stats["ci_high"],
                }
            )

    pd.DataFrame(posterior_rows).to_csv(
        RESULTS_DIR / "posterior_parameter_summary.csv", index=False
    )
    pd.DataFrame(ppc_rows).to_csv(
        RESULTS_DIR / "posterior_predictive_summary.csv", index=False
    )
    print("  Saved numerical posterior and PPC summaries")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run this script from the project folder containing checkpoints/."
        )

    print(f"Loading trained approximator from {MODEL_PATH}...")
    print(f"BayesFlow version: {bf.__version__}")
    approximator = keras.saving.load_model(MODEL_PATH, compile=False)
    print("Model loaded.\n")

    all_summaries = {}
    all_ppc_rows = []

    for pid, filepath in PARTICIPANT_FILES.items():
        print("\n" + "=" * 64)
        print(participant_label(pid))
        print("=" * 64)

        posterior, real_data = sample_posterior(
            approximator,
            filepath,
            seed=SEED + pid,
        )
        summary = summarize_posterior(posterior)
        simulations = simulate_posterior_predictive(
            summary,
            n_trials=real_data["n_trials"],
        )

        all_ppc_rows.extend(
            ppc_accuracy_and_mean_rt(pid, simulations, real_data)
        )
        all_ppc_rows.extend(
            ppc_double_response_rates(pid, simulations, real_data)
        )
        ppc_rt_distributions(pid, simulations, real_data)
        ppc_drt_distribution(pid, simulations, real_data)

        all_summaries[pid] = summary

    plot_parameter_estimates(all_summaries)
    save_numeric_results(all_summaries, all_ppc_rows)

    print("\nStep 7 completed successfully.")
    print(f"All outputs are in: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()