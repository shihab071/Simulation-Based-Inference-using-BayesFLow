"""Train and validate the BayesFlow approximation for the LI model."""

import os

if not os.environ.get("KERAS_BACKEND"):
    os.environ["KERAS_BACKEND"] = "jax"

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import keras
import bayesflow as bf

from step3_vectorized_simulator import simulate_trials_filtered

# General configuration

SEED = 2026

PAR_KEYS = ["v_A", "v_B", "beta", "b", "t0"]
OBS_KEYS = ["choice", "rt", "double_response", "drt"]

PAR_LABELS = [
    r"$v_A$",
    r"$v_B$",
    r"$\beta$",
    r"$b$",
    r"$t_0$",
]

N_TRIALS_OPTIONS = [500, 1_000, 2_500, 5_000, 10_000]
N_TRIALS_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]

TRAINING_EPOCHS = 40
TRAINING_BATCH_SIZE = 4
NUM_BATCHES_PER_EPOCH = 400

VALIDATION_DATASETS = 200
VALIDATION_TRIALS = 10_000
POSTERIOR_SAMPLES = 500
VALIDATION_BATCH_SIZE = 2

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_NAME = "li_model_approximator"
FINAL_MODEL_PATH = CHECKPOINT_DIR / "li_model_approximator_final.keras"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Set both NumPy and Kerasseeds.
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

# The simulator uses this generator for within-trial noise.
SIMULATOR_RNG = np.random.default_rng(SEED)



# Prior and likelihood


def trunc_normal_scalar(mean, sd, lower=0.0, upper=np.inf):
    """Draw one value from a truncated normal distribution."""
    value = lower - 1.0

    while value < lower or value > upper:
        value = np.random.normal(mean, sd)

    return value


def prior():
    """Draw one parameter vector from the training prior."""
    return {
        "v_A": trunc_normal_scalar(
            mean=3.0,
            sd=1.5,
            lower=0.05,
        ),
        "v_B": trunc_normal_scalar(
            mean=1.5,
            sd=1.0,
            lower=0.05,
        ),
        "beta": trunc_normal_scalar(
            mean=3.0,
            sd=1.5,
            lower=0.0,
        ),
        "b": trunc_normal_scalar(
            mean=1.4,
            sd=0.4,
            lower=0.3,
        ),
        "t0": trunc_normal_scalar(
            mean=0.3,
            sd=0.15,
            lower=0.05,
            upper=0.6,
        ),
    }


def meta():
    """Choose one trial count for the current training batch."""
    n_trials = np.random.choice(
        N_TRIALS_OPTIONS,
        p=N_TRIALS_WEIGHTS,
    )

    return {"n_trials": int(n_trials)}


def validation_meta():
    """Use the real-data trial count for the main validation run."""
    return {"n_trials": VALIDATION_TRIALS}


def likelihood(v_A, v_B, beta, b, t0, n_trials):
    """Simulate one complete dataset for BayesFlow."""
    result = simulate_trials_filtered(
        v_A=v_A,
        v_B=v_B,
        beta=beta,
        b=b,
        t0=t0,
        n_trials=int(n_trials),
        rng=SIMULATOR_RNG,
    )

    # Choice A is coded as 0 and choice B as 1.
    choice_numeric = (
        result["choice"] == "B"
    ).astype(np.float64)

    # DRT is zero when no second response occurred. The separate
    # double_response variable tells the network whether DRT is observed.
    drt = np.nan_to_num(
        result["drt"],
        nan=0.0,
    ).astype(np.float64)

    return {
        "choice": choice_numeric,
        "rt": result["rt"].astype(np.float64),
        "double_response": result[
            "double_response"
        ].astype(np.float64),
        "drt": drt,
    }


# BayesFlow objects

def make_adapter():
    """Create the adapter used during training and inference."""
    return (
        bf.Adapter()
        .broadcast("n_trials", to="choice")
        .as_set(OBS_KEYS)
        .constrain("v_A", lower=0.05)
        .constrain("v_B", lower=0.05)
        .constrain("beta", lower=0.0)
        .constrain("b", lower=0.3)
        .constrain("t0", lower=0.05, upper=0.6)
        .sqrt("n_trials")
        .convert_dtype("float64", "float32")
        .concatenate(
            PAR_KEYS,
            into="inference_variables",
        )
        .concatenate(
            OBS_KEYS,
            into="summary_variables",
        )
        .rename(
            "n_trials",
            "inference_conditions",
        )
    )


def build_workflow(simulator):
    """Build the networks and complete BayesFlow workflow."""
    adapter = make_adapter()

    # Induced attention keeps the 10,000-trial datasets manageable.
    summary_network = bf.networks.SetTransformer(
        summary_dim=10,
        num_inducing_points=32,
    )

    inference_network = bf.networks.CouplingFlow(
        transform="spline",
    )

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=inference_network,
        summary_network=summary_network,
        standardize=[
            "inference_variables",
            "summary_variables",
        ],
        checkpoint_filepath=str(CHECKPOINT_DIR),
        checkpoint_name=CHECKPOINT_NAME,
        save_best_only=False,
    )

    return workflow, adapter, summary_network, inference_network


# Small utility functions

def check_finite(name, value):
    """Make sure a nested result contains no NaN or infinite values."""
    leaves = keras.tree.flatten(value)

    if not leaves:
        raise AssertionError(
            f"{name} did not contain any numeric arrays."
        )

    for leaf_number, leaf in enumerate(leaves):
        array = np.asarray(leaf)

        if not np.all(np.isfinite(array)):
            raise AssertionError(
                f"{name}, array {leaf_number}, contains "
                "NaN or infinite values."
            )


def print_shapes(name, value):
    """Print the array shapes in a nested BayesFlow result."""
    shapes = keras.tree.map_structure(
        lambda x: np.asarray(x).shape,
        value,
    )

    print(f"{name}: {shapes}")


def save_figure(figure, filename):
    """Save and close a diagnostic figure."""
    figure.savefig(
        filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)
    print(f"Saved {filename}")


def write_configuration(
    summary_network,
    inference_network,
):
    """Save the exact training and architecture configuration."""
    configuration = {
        "software": {
            "python_backend": keras.backend.backend(),
            "keras_version": keras.__version__,
            "bayesflow_version": getattr(
                bf,
                "__version__",
                "unknown",
            ),
        },
        "seed": SEED,
        "simulator": {
            "dt": 0.002,
            "noise_sd": 1.0,
            "double_response_window": 0.25,
            "max_time": 2.5,
            "minimum_retained_rt": 0.25,
        },
        "priors": {
            "v_A": {
                "distribution": "truncated normal",
                "mean": 3.0,
                "sd": 1.5,
                "lower": 0.05,
            },
            "v_B": {
                "distribution": "truncated normal",
                "mean": 1.5,
                "sd": 1.0,
                "lower": 0.05,
            },
            "beta": {
                "distribution": "truncated normal",
                "mean": 3.0,
                "sd": 1.5,
                "lower": 0.0,
            },
            "b": {
                "distribution": "truncated normal",
                "mean": 1.4,
                "sd": 0.4,
                "lower": 0.3,
            },
            "t0": {
                "distribution": "truncated normal",
                "mean": 0.3,
                "sd": 0.15,
                "lower": 0.05,
                "upper": 0.6,
            },
        },
        "training": {
            "epochs": TRAINING_EPOCHS,
            "batch_size": TRAINING_BATCH_SIZE,
            "batches_per_epoch": NUM_BATCHES_PER_EPOCH,
            "total_optimizer_updates": (
                TRAINING_EPOCHS
                * NUM_BATCHES_PER_EPOCH
            ),
            "total_simulated_datasets": (
                TRAINING_EPOCHS
                * NUM_BATCHES_PER_EPOCH
                * TRAINING_BATCH_SIZE
            ),
            "n_trials_options": N_TRIALS_OPTIONS,
            "n_trials_weights": N_TRIALS_WEIGHTS,
        },
        "validation": {
            "datasets": VALIDATION_DATASETS,
            "trials_per_dataset": VALIDATION_TRIALS,
            "posterior_samples_per_dataset": (
                POSTERIOR_SAMPLES
            ),
            "sampling_batch_size": (
                VALIDATION_BATCH_SIZE
            ),
        },
        "set_transformer": (
            summary_network.get_config()
        ),
        "coupling_flow": (
            inference_network.get_config()
        ),
    }

    with open(
        "model_configuration.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=2,
            default=str,
        )

    print("Saved model_configuration.json")

# Main training and validation

def main():
    print("=" * 72)
    print("BAYESFLOW TRAINING: LATERAL-INHIBITION MODEL")
    print("=" * 72)

    print(f"BayesFlow version: {getattr(bf, '__version__', 'unknown')}")
    print(f"Keras version: {keras.__version__}")
    print(f"Keras backend: {keras.backend.backend()}")
    print(f"Random seed: {SEED}")

    total_datasets = (
        TRAINING_EPOCHS
        * NUM_BATCHES_PER_EPOCH
        * TRAINING_BATCH_SIZE
    )

    total_updates = (
        TRAINING_EPOCHS
        * NUM_BATCHES_PER_EPOCH
    )

    print("\nTraining configuration:")
    print(f"  Epochs: {TRAINING_EPOCHS}")
    print(f"  Batch size: {TRAINING_BATCH_SIZE}")
    print(
        f"  Batches per epoch: "
        f"{NUM_BATCHES_PER_EPOCH}"
    )
    print(
        f"  Total simulated datasets: "
        f"{total_datasets:,}"
    )
    print(
        f"  Total optimizer updates: "
        f"{total_updates:,}"
    )
    print(
        f"  Trial-count options: "
        f"{N_TRIALS_OPTIONS}"
    )
    print(
        f"  Trial-count weights: "
        f"{N_TRIALS_WEIGHTS}"
    )

    simulator = bf.simulators.make_simulator(
        [prior, likelihood],
        meta_fn=meta,
    )

    print("\nTesting the simulator with four datasets...")
    simulator_test = simulator.sample(4)

    print_shapes(
        "Raw simulator shapes",
        simulator_test,
    )

    assert simulator_test["choice"].shape[0] == 4
    assert simulator_test["choice"].shape[-1] in (
        N_TRIALS_OPTIONS
    )
    assert np.all(
        np.isfinite(simulator_test["rt"])
    )
    assert np.all(
        simulator_test["rt"] >= 0.25
    )
    assert np.all(
        np.isfinite(simulator_test["drt"])
    )

    print("Simulator check passed.")

    (
        workflow,
        adapter,
        summary_network,
        inference_network,
    ) = build_workflow(simulator)

    print("\nSetTransformer configuration:")
    print(summary_network.get_config())

    print("\nCouplingFlow configuration:")
    print(inference_network.get_config())

    write_configuration(
        summary_network,
        inference_network,
    )

    print("\nTesting the adapter...")
    processed = adapter(simulator_test)

    print_shapes(
        "Adapter output shapes",
        processed,
    )

    assert (
        processed["summary_variables"].shape[-1]
        == 4
    )
    assert (
        processed["inference_conditions"].shape
        == (4, 1)
    )
    assert (
        processed["inference_variables"].shape
        == (4, 5)
    )

    check_finite(
        "Adapter output",
        processed,
    )

    print("Adapter check passed.")

    print("\n" + "=" * 72)
    print("STARTING FULL TRAINING")
    print("=" * 72)
    print(
        "Automatic checkpoints will be saved after each epoch."
    )

    training_start = time.perf_counter()

    history = workflow.fit_online(
        epochs=TRAINING_EPOCHS,
        batch_size=TRAINING_BATCH_SIZE,
        num_batches_per_epoch=(
            NUM_BATCHES_PER_EPOCH
        ),
    )

    training_seconds = (
        time.perf_counter() - training_start
    )

    print(
        f"\nTraining completed in "
        f"{training_seconds / 3600:.2f} hours."
    )

    if hasattr(history, "history"):
        history_values = history.history
    else:
        history_values = history

    loss_values = np.asarray(
        history_values["loss"],
        dtype=float,
    )

    if not np.all(np.isfinite(loss_values)):
        raise RuntimeError(
            "Training produced a NaN or infinite loss."
        )

    print(f"Final training loss: {loss_values[-1]:.4f}")

    loss_figure = bf.diagnostics.plots.loss(
        history
    )

    save_figure(
        loss_figure,
        "training_loss.png",
    )

    # Save the finished model immediately, before diagnostics.
    workflow.approximator.save(
        filepath=FINAL_MODEL_PATH
    )

    print(
        f"Saved final trained model to "
        f"{FINAL_MODEL_PATH}"
    )

    # Compare the final 20% of epochs with the preceding 20%.
    n_epochs = len(loss_values)

    previous_chunk = loss_values[
        int(n_epochs * 0.6):
        int(n_epochs * 0.8)
    ]

    final_chunk = loss_values[
        int(n_epochs * 0.8):
    ]

    loss_drop = (
        previous_chunk.mean()
        - final_chunk.mean()
    )

    print(
        f"\nMean loss during epochs "
        f"{int(n_epochs * 0.6) + 1}-"
        f"{int(n_epochs * 0.8)}: "
        f"{previous_chunk.mean():.4f}"
    )

    print(
        f"Mean loss during the final 20%: "
        f"{final_chunk.mean():.4f}"
    )

    print(
        f"Difference between the two periods: "
        f"{loss_drop:.4f}"
    )

    if loss_drop > 0.05:
        print(
            "Training loss was still declining noticeably "
            "during the final epochs."
        )
    else:
        print(
            "Training loss stabilized over the final epochs."
        )

    # Record the optimizer configuration after it has been built.
    optimizer = getattr(
        workflow.approximator,
        "optimizer",
        None,
    )

    if optimizer is not None:
        optimizer_config = optimizer.get_config()

        with open(
            "optimizer_configuration.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                optimizer_config,
                file,
                indent=2,
                default=str,
            )

        print("Saved optimizer_configuration.json")

    # Validation at the actual participant sample size


    print("\n" + "=" * 72)
    print("VALIDATION AT 10,000 TRIALS")
    print("=" * 72)

    validation_simulator = (
        bf.simulators.make_simulator(
            [prior, likelihood],
            meta_fn=validation_meta,
        )
    )

    print(
        f"Generating {VALIDATION_DATASETS} held-out "
        f"datasets with {VALIDATION_TRIALS:,} trials each..."
    )

    validation_start = time.perf_counter()

    validation_simulations = (
        validation_simulator.sample(
            VALIDATION_DATASETS
        )
    )

    validation_simulation_seconds = (
        time.perf_counter()
        - validation_start
    )

    print_shapes(
        "Validation simulator shapes",
        validation_simulations,
    )

    assert (
        validation_simulations["choice"].shape
        == (
            VALIDATION_DATASETS,
            VALIDATION_TRIALS,
        )
    )

    print(
        f"Validation simulation took "
        f"{validation_simulation_seconds / 60:.2f} minutes."
    )

    print(
        f"\nDrawing {POSTERIOR_SAMPLES} posterior "
        "samples per validation dataset..."
    )

    posterior_start = time.perf_counter()

    posterior_draws = workflow.sample(
        conditions=validation_simulations,
        num_samples=POSTERIOR_SAMPLES,
        batch_size=VALIDATION_BATCH_SIZE,
        seed=SEED,
    )

    posterior_seconds = (
        time.perf_counter()
        - posterior_start
    )

    print_shapes(
        "Posterior sample shapes",
        posterior_draws,
    )

    check_finite(
        "Posterior samples",
        posterior_draws,
    )

    for parameter in PAR_KEYS:
        expected_shape = (
            VALIDATION_DATASETS,
            POSTERIOR_SAMPLES,
            1,
        )

        assert (
            posterior_draws[parameter].shape
            == expected_shape
        ), (
            f"Unexpected posterior shape for "
            f"{parameter}: "
            f"{posterior_draws[parameter].shape}"
        )

    print(
        f"Posterior sampling took "
        f"{posterior_seconds / 60:.2f} minutes."
    )


    # Recovery and calibration plots
   

    print("\nCreating parameter-recovery plot...")

    recovery_figure = (
        bf.diagnostics.plots.recovery(
            estimates=posterior_draws,
            targets=validation_simulations,
            variable_keys=PAR_KEYS,
            variable_names=PAR_LABELS,
        )
    )

    save_figure(
        recovery_figure,
        "recovery.png",
    )

    print("\nCreating SBC rank histograms...")

    histogram_figure = (
        bf.diagnostics.plots.calibration_histogram(
            estimates=posterior_draws,
            targets=validation_simulations,
            variable_keys=PAR_KEYS,
            variable_names=PAR_LABELS,
        )
    )

    save_figure(
        histogram_figure,
        "sbc_rank_histogram.png",
    )

    print("\nCreating SBC ECDF plots...")

    ecdf_figure = (
        bf.diagnostics.plots.calibration_ecdf(
            estimates=posterior_draws,
            targets=validation_simulations,
            variable_keys=PAR_KEYS,
            variable_names=PAR_LABELS,
            difference=True,
            rank_type="distance",
        )
    )

    save_figure(
        ecdf_figure,
        "sbc_ecdf.png",
    )

    print(
        "\nCreating posterior contraction and "
        "z-score plots..."
    )

    contraction_figure = (
        bf.diagnostics.plots.z_score_contraction(
            estimates=posterior_draws,
            targets=validation_simulations,
            variable_keys=PAR_KEYS,
            variable_names=PAR_LABELS,
        )
    )

    save_figure(
        contraction_figure,
        "contraction_zscore.png",
    )

    
    # Reload check
    

    print("\n" + "=" * 72)
    print("CHECKING THE SAVED MODEL")
    print("=" * 72)

  
    loaded_approximator = keras.saving.load_model(
        FINAL_MODEL_PATH
    )

    reload_conditions = (
        validation_simulator.sample(1)
    )

    reload_samples = loaded_approximator.sample(
        conditions=reload_conditions,
        num_samples=20,
        batch_size=1,
        seed=SEED,
    )

    print_shapes(
        "Reloaded-model posterior shapes",
        reload_samples,
    )

    check_finite(
        "Reloaded-model posterior samples",
        reload_samples,
    )

    for parameter in PAR_KEYS:
        assert (
            reload_samples[parameter].shape
            == (1, 20, 1)
        )

    print("Saved-model reload check passed.")

    print("\n" + "=" * 72)
    print("STEP 6 COMPLETED SUCCESSFULLY")
    print("=" * 72)

    print("\nFiles created:")
    print("  training_loss.png")
    print("  recovery.png")
    print("  sbc_rank_histogram.png")
    print("  sbc_ecdf.png")
    print("  contraction_zscore.png")
    print("  model_configuration.json")
    print("  optimizer_configuration.json")
    print(
        f"  {CHECKPOINT_DIR / (CHECKPOINT_NAME + '.keras')}"
    )
    print(f"  {FINAL_MODEL_PATH}")

    print(
        "\nStep 7 should load "
        "li_model_approximator_final.keras."
    )


if __name__ == "__main__":
    main()