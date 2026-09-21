"""Memory preflight for the full BayesFlow training pipeline.

This runs a few real training updates at n_trials=10,000. It tests the
simulator, adapter, SetTransformer, CouplingFlow, backward pass, optimizer,
and posterior sampling before the long Step 6 training run.
"""

import os

if not os.environ.get("KERAS_BACKEND"):
    os.environ["KERAS_BACKEND"] = "jax"

import argparse
import sys
import time

import numpy as np
import keras
import bayesflow as bf

from step6_train_bayesflow import prior, likelihood


SEED = 2026
N_TRIALS = 10_000

PAR_KEYS = ["v_A", "v_B", "beta", "b", "t0"]
OBS_KEYS = ["choice", "rt", "double_response", "drt"]


def fixed_meta():
    """Force every preflight dataset to contain 10,000 trials."""
    return {"n_trials": N_TRIALS}


def make_adapter():
    """Create the same adapter used for final training."""
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
        .concatenate(PAR_KEYS, into="inference_variables")
        .concatenate(OBS_KEYS, into="summary_variables")
        .rename("n_trials", "inference_conditions")
    )


def make_preflight_workflow(simulator):
    """Build a disposable copy of the complete neural pipeline."""
    adapter = make_adapter()

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
        standardize=["inference_variables", "summary_variables"],
    )

    return workflow, adapter, summary_network, inference_network


def check_finite(name, value):
    """Check every numeric array contained in a nested result."""
    leaves = keras.tree.flatten(value)

    if not leaves:
        raise AssertionError(f"{name} did not contain any arrays.")

    for i, leaf in enumerate(leaves):
        array = np.asarray(leaf)

        if not np.all(np.isfinite(array)):
            raise AssertionError(
                f"{name}, array {i}, contains NaN or infinite values."
            )


def print_shapes(name, value):
    """Print the shapes of a nested dictionary or array."""
    try:
        shapes = keras.tree.map_structure(
            lambda x: np.asarray(x).shape,
            value,
        )
        print(f"{name}: {shapes}")
    except Exception:
        print(f"{name}: could not print nested shapes")


def run_preflight(batch_size, n_batches):
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)

    print("=" * 68)
    print("BAYESFLOW 10,000-TRIAL MEMORY PREFLIGHT")
    print("=" * 68)
    print(f"Python: {sys.version.split()[0]}")
    print(f"BayesFlow: {getattr(bf, '__version__', 'unknown')}")
    print(f"Keras: {keras.__version__}")
    print(f"Keras backend: {keras.backend.backend()}")
    print(f"Training batch size: {batch_size}")
    print(f"Forced trial count: {N_TRIALS}")
    print(f"Training batches in test: {n_batches}")

    simulator = bf.simulators.make_simulator(
        [prior, likelihood],
        meta_fn=fixed_meta,
    )

    workflow, adapter, summary_network, inference_network = (
        make_preflight_workflow(simulator)
    )

    print("\nSetTransformer configuration:")
    print(summary_network.get_config())

    print("\nCouplingFlow configuration:")
    print(inference_network.get_config())

    # First check the raw simulator without involving the network.
    print("\n1. Generating a full-size simulator batch...")
    start = time.perf_counter()
    raw_batch = simulator.sample(batch_size)
    simulation_time = time.perf_counter() - start

    print_shapes("Raw simulator output shapes", raw_batch)
    print(f"Simulation time: {simulation_time:.2f} seconds")

    assert raw_batch["choice"].shape == (batch_size, N_TRIALS)
    assert raw_batch["rt"].shape == (batch_size, N_TRIALS)
    assert raw_batch["double_response"].shape == (
        batch_size,
        N_TRIALS,
    )
    assert raw_batch["drt"].shape == (batch_size, N_TRIALS)

    assert np.all(np.isfinite(raw_batch["rt"]))
    assert np.all(raw_batch["rt"] >= 0.25)
    assert np.all(np.isfinite(raw_batch["drt"]))

    print("Raw simulator check passed.")

    # Check the exact tensors that will be sent to the neural networks.
    print("\n2. Checking the adapter output...")
    processed = adapter(raw_batch)

    print_shapes("Adapter output shapes", processed)

    summary_shape = np.asarray(
        processed["summary_variables"]
    ).shape

    parameter_shape = np.asarray(
        processed["inference_variables"]
    ).shape

    condition_shape = np.asarray(
        processed["inference_conditions"]
    ).shape

    assert summary_shape == (batch_size, N_TRIALS, 4), (
        f"Unexpected summary shape: {summary_shape}"
    )

    assert parameter_shape == (batch_size, 5), (
        f"Unexpected parameter shape: {parameter_shape}"
    )

    assert condition_shape[0] == batch_size, (
        f"Unexpected condition shape: {condition_shape}"
    )

    check_finite("Adapter output", processed)
    print("Adapter check passed.")

    # This is the important test. fit_online performs simulation, adapter
    # processing, a network forward pass, loss calculation, backpropagation,
    # and an optimizer update.
    print(
        "\n3. Running the complete neural-network training preflight..."
    )
    print(
        "This is the stage that will reveal whether the selected batch "
        "size fits in memory."
    )

    start = time.perf_counter()

    history = workflow.fit_online(
        epochs=1,
        batch_size=batch_size,
        num_batches_per_epoch=n_batches,
    )

    training_time = time.perf_counter() - start

    if hasattr(history, "history"):
        loss_values = np.asarray(
            history.history.get("loss", []),
            dtype=float,
        )
    else:
        loss_values = np.asarray(
            history.get("loss", []),
            dtype=float,
        )

    if loss_values.size == 0:
        raise AssertionError("Training finished but no loss was returned.")

    if not np.all(np.isfinite(loss_values)):
        raise AssertionError(
            f"Training produced a non-finite loss: {loss_values}"
        )

    print(f"Training time: {training_time:.2f} seconds")
    print(f"Loss values: {np.round(loss_values, 4)}")
    print("Forward pass, backward pass, and optimizer update passed.")

    # Posterior sampling follows a different network path, so test it too.
    print("\n4. Testing posterior sampling at 10,000 trials...")

    test_conditions = simulator.sample(1)

    start = time.perf_counter()

    posterior = workflow.sample(
        conditions=test_conditions,
        num_samples=100,
        batch_size=1,
        seed=SEED,
    )

    sampling_time = time.perf_counter() - start

    print_shapes("Posterior shapes", posterior)
    check_finite("Posterior samples", posterior)

    print(f"Posterior-sampling time: {sampling_time:.2f} seconds")
    print("Posterior-sampling check passed.")

    print("\n" + "=" * 68)
    print("PREFLIGHT PASSED")
    print("=" * 68)
    print(
        f"batch_size={batch_size} completed {n_batches} real training "
        f"updates at n_trials={N_TRIALS}."
    )
    print(
        "Close this process before starting Step 6 so the temporary model "
        "and JAX memory are released."
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Test the complete BayesFlow network at 10,000 trials."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of simulated datasets per training batch.",
    )

    parser.add_argument(
        "--batches",
        type=int,
        default=3,
        help="Number of real training updates to test.",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    if args.batches < 1:
        parser.error("--batches must be at least 1.")

    return args


if __name__ == "__main__":
    arguments = parse_arguments()

    try:
        run_preflight(
            batch_size=arguments.batch_size,
            n_batches=arguments.batches,
        )

    except MemoryError:
        print(
            "\nPREFLIGHT FAILED: Python reported insufficient memory.",
            file=sys.stderr,
        )
        print(
            "Close this process and repeat with --batch-size 2.",
            file=sys.stderr,
        )
        raise

    except Exception as error:
        error_text = str(error).lower()

        if (
            "out of memory" in error_text
            or "resource exhausted" in error_text
            or "allocation" in error_text
        ):
            print(
                "\nPREFLIGHT FAILED: the selected batch size appears to "
                "exceed available memory.",
                file=sys.stderr,
            )
            print(
                "Close this process and repeat with --batch-size 2.",
                file=sys.stderr,
            )

        raise