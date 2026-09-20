"""Continue the trained LI model from epoch 40 to epoch 60.

The saved .keras file contains the trained networks, adapter, and
standardization state. This script loads that complete model, gives it a new
optimizer schedule with a smaller learning rate, and trains for 20 more
epochs. The epoch-40 model is never overwritten.
"""

import os

# BayesFlow must know which Keras backend to use before Keras is imported.
if not os.environ.get("KERAS_BACKEND"):
    os.environ["KERAS_BACKEND"] = "jax"

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import keras
import bayesflow as bf

from step3_vectorized_simulator import simulate_trials_filtered


# Names used by the adapter and diagnostic plots.
PAR_KEYS = ["v_A", "v_B", "beta", "b", "t0"]
PAR_LABELS = [r"$v_A$", r"$v_B$", r"$\beta$", r"$b$", r"$t_0$"]

SEED = 2026

# Training still uses several dataset sizes. Ten percent of the simulated
# datasets contain 10,000 trials, matching the real participants.
N_TRIALS_OPTIONS = [500, 1_000, 2_500, 5_000, 10_000]
N_TRIALS_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]

# Continue for 20 epochs with a smaller learning rate than the original run.
STARTING_EPOCHS = 40
ADDITIONAL_EPOCHS = 20
CUMULATIVE_EPOCHS = STARTING_EPOCHS + ADDITIONAL_EPOCHS

FINE_TUNE_LEARNING_RATE = 1e-4
TRAINING_BATCH_SIZE = 4
NUM_BATCHES_PER_EPOCH = 400

# Use the same validation design as Step 6.
VALIDATION_DATASETS = 200
VALIDATION_TRIALS = 10_000
POSTERIOR_SAMPLES = 500
VALIDATION_BATCH_SIZE = 2


CHECKPOINT_DIR = Path("checkpoints")

# This is the model produced at the end of the original 40-epoch run.
SOURCE_MODEL_PATH = CHECKPOINT_DIR / "li_model_approximator_final.keras"

# The continuation uses separate filenames, leaving the source model intact.
CONTINUED_CHECKPOINT_NAME = (
    f"li_model_approximator_epoch{CUMULATIVE_EPOCHS}_checkpoint"
)
CONTINUED_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / f"{CONTINUED_CHECKPOINT_NAME}.keras"
)
CONTINUED_FINAL_MODEL_PATH = (
    CHECKPOINT_DIR / f"li_model_approximator_epoch{CUMULATIVE_EPOCHS}.keras"
)


# Set the random seeds once so the continuation can be reproduced.
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)
SIMULATOR_RNG = np.random.default_rng(SEED)


def trunc_normal_scalar(mean, sd, lower=0.0, upper=np.inf):
    """Draw one value from a truncated normal distribution."""
    value = lower - 1.0
    while value < lower or value > upper:
        value = np.random.normal(mean, sd)
    return value


def prior():
    """Sample one set of model parameters from the training priors."""
    return {
        "v_A": trunc_normal_scalar(3.0, 1.5, lower=0.05),
        "v_B": trunc_normal_scalar(1.5, 1.0, lower=0.05),
        "beta": trunc_normal_scalar(3.0, 1.5, lower=0.0),
        "b": trunc_normal_scalar(1.4, 0.4, lower=0.3),
        "t0": trunc_normal_scalar(0.3, 0.15, lower=0.05, upper=0.6),
    }


def meta():
    """Choose the trial count for one simulated training dataset."""
    n_trials = np.random.choice(N_TRIALS_OPTIONS, p=N_TRIALS_WEIGHTS)
    return {"n_trials": int(n_trials)}


def validation_meta():
    """Give every validation dataset the real-data trial count."""
    return {"n_trials": VALIDATION_TRIALS}


def likelihood(v_A, v_B, beta, b, t0, n_trials):
    """Simulate one dataset and return numeric inputs for BayesFlow."""
    result = simulate_trials_filtered(
        v_A, v_B, beta, b, t0, int(n_trials), rng=SIMULATOR_RNG
    )

    # A is coded as 0 and B as 1. Missing DRTs become zero; the separate
    # double-response flag tells the network whether that zero is a placeholder.
    return {
        "choice": (result["choice"] == "B").astype(np.float64),
        "rt": result["rt"].astype(np.float64),
        "double_response": result["double_response"].astype(np.float64),
        "drt": np.nan_to_num(result["drt"], nan=0.0).astype(np.float64),
    }


def save_figure(figure, filename):
    """Save a figure and close it to release memory."""
    figure.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {filename}")


def main():
    print("=" * 72)
    print(
        f"CONTINUING TRAINING: epoch {STARTING_EPOCHS} "
        f"-> epoch {CUMULATIVE_EPOCHS}"
    )
    print("=" * 72)
    print(f"BayesFlow version: {bf.__version__}")
    print(f"Keras version: {keras.__version__}")
    print(f"Keras backend: {keras.backend.backend()}")

    # Stop immediately if the original model cannot be found.
    if not SOURCE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Source model was not found: {SOURCE_MODEL_PATH}"
        )

    # Do not overwrite the output of an earlier continuation run.
    for output_path in [
        CONTINUED_CHECKPOINT_PATH,
        CONTINUED_FINAL_MODEL_PATH,
    ]:
        if output_path.exists():
            raise FileExistsError(
                f"Output already exists: {output_path}\n"
                "Move or rename it before starting another run."
            )

    print(f"\nLoading the trained model from {SOURCE_MODEL_PATH}...")

    # We only need the learned model, not the old optimizer schedule stored in
    # the .keras file. The new workflow below will create a fresh schedule.
    loaded_approximator = keras.saving.load_model(
        SOURCE_MODEL_PATH,
        compile=False,
    )

    simulator = bf.simulators.make_simulator(
        [prior, likelihood],
        meta_fn=meta,
    )

    # Reuse the adapter and networks stored inside the model. This avoids
    # accidentally changing the architecture or parameter transformations.
    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=loaded_approximator.adapter,
        inference_network=loaded_approximator.inference_network,
        summary_network=loaded_approximator.summary_network,
        standardize=["inference_variables", "summary_variables"],
        initial_learning_rate=FINE_TUNE_LEARNING_RATE,
        checkpoint_filepath=str(CHECKPOINT_DIR),
        checkpoint_name=CONTINUED_CHECKPOINT_NAME,
        save_best_only=False,
    )

    # BasicWorkflow creates a model during setup. Replace it with the complete
    # trained model loaded above, including its learned weights.
    workflow.approximator = loaded_approximator

    print("Model loaded successfully.")
    print(f"Fine-tuning learning rate: {FINE_TUNE_LEARNING_RATE}")
    print(f"Automatic checkpoint: {CONTINUED_CHECKPOINT_PATH}")
    print(f"Final continued model: {CONTINUED_FINAL_MODEL_PATH}")
    print(f"Original model will remain at: {SOURCE_MODEL_PATH}")

    print("\n" + "=" * 72)
    print("STARTING ADDITIONAL TRAINING")
    print("=" * 72)
    print(
        f"Training for {ADDITIONAL_EPOCHS} epochs "
        f"(batch_size={TRAINING_BATCH_SIZE}, "
        f"{NUM_BATCHES_PER_EPOCH} batches per epoch)..."
    )

    start = time.perf_counter()
    history = workflow.fit_online(
        epochs=ADDITIONAL_EPOCHS,
        batch_size=TRAINING_BATCH_SIZE,
        num_batches_per_epoch=NUM_BATCHES_PER_EPOCH,
        keep_optimizer=False,
    )
    elapsed = time.perf_counter() - start

    print(f"Additional training completed in {elapsed / 3600:.2f} hours.")

    loss_values = np.asarray(
        history.history["loss"]
        if hasattr(history, "history")
        else history["loss"],
        dtype=float,
    )
    if not np.all(np.isfinite(loss_values)):
        raise RuntimeError("Training produced a NaN or infinite loss.")

    print(f"Final loss this run: {loss_values[-1]:.4f}")

    loss_plot_name = f"training_loss_epoch{CUMULATIVE_EPOCHS}.png"
    save_figure(bf.diagnostics.plots.loss(history), loss_plot_name)

    # Save the continued model under a new name.
    workflow.approximator.save(filepath=CONTINUED_FINAL_MODEL_PATH)
    print(f"Saved continued model to {CONTINUED_FINAL_MODEL_PATH}")

    # This is a simple trend check, not a formal convergence test.
    n_epochs = len(loss_values)
    previous_chunk = loss_values[int(n_epochs * 0.6):int(n_epochs * 0.8)]
    final_chunk = loss_values[int(n_epochs * 0.8):]
    loss_drop = previous_chunk.mean() - final_chunk.mean()

    print(f"\nMean loss, previous 20%: {previous_chunk.mean():.4f}")
    print(f"Mean loss, final 20%: {final_chunk.mean():.4f}")
    print(f"Difference: {loss_drop:.4f}")
    if loss_drop > 0.05:
        print("Loss is still declining; check the diagnostics before deciding.")
    else:
        print("Loss is roughly stable over the final epochs.")

    print("\n" + "=" * 72)
    print(
        f"VALIDATION AT {VALIDATION_TRIALS:,} TRIALS "
        f"(epoch {CUMULATIVE_EPOCHS})"
    )
    print("=" * 72)

    validation_simulator = bf.simulators.make_simulator(
        [prior, likelihood],
        meta_fn=validation_meta,
    )

    print(f"Generating {VALIDATION_DATASETS} held-out datasets...")
    validation_start = time.perf_counter()
    validation_data = validation_simulator.sample(VALIDATION_DATASETS)
    validation_elapsed = time.perf_counter() - validation_start
    print(f"Validation simulation took {validation_elapsed / 60:.2f} minutes.")

    print(f"Drawing {POSTERIOR_SAMPLES} posterior samples per dataset...")
    sampling_start = time.perf_counter()
    posterior_draws = workflow.sample(
        conditions=validation_data,
        num_samples=POSTERIOR_SAMPLES,
        batch_size=VALIDATION_BATCH_SIZE,
        seed=SEED,
    )
    sampling_elapsed = time.perf_counter() - sampling_start
    print(f"Posterior sampling took {sampling_elapsed / 60:.2f} minutes.")

    print("\nCreating parameter-recovery plot...")
    save_figure(
        bf.diagnostics.plots.recovery(
            estimates=posterior_draws,
            targets=validation_data,
            variable_keys=PAR_KEYS,
            variable_names=PAR_LABELS,
        ),
        f"recovery_epoch{CUMULATIVE_EPOCHS}.png",
    )

    print("Creating SBC ECDF plot...")
    save_figure(
        bf.diagnostics.plots.calibration_ecdf(
            estimates=posterior_draws,
            targets=validation_data,
            variable_keys=PAR_KEYS,
            variable_names=PAR_LABELS,
            difference=True,
            rank_type="distance",
        ),
        f"sbc_ecdf_epoch{CUMULATIVE_EPOCHS}.png",
    )

    print("Creating posterior contraction and z-score plot...")
    save_figure(
        bf.diagnostics.plots.z_score_contraction(
            estimates=posterior_draws,
            targets=validation_data,
            variable_keys=PAR_KEYS,
            variable_names=PAR_LABELS,
        ),
        f"contraction_zscore_epoch{CUMULATIVE_EPOCHS}.png",
    )

    print("\n" + "=" * 72)
    print("CONTINUATION COMPLETED SUCCESSFULLY")
    print("=" * 72)
    print(f"Original model: {SOURCE_MODEL_PATH}")
    print(f"New model: {CONTINUED_FINAL_MODEL_PATH}")
    print("Review the new loss and calibration plots before training again.")


if __name__ == "__main__":
    main()