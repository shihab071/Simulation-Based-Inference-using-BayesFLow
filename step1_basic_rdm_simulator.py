"""Basic racing diffusion model without inhibition or double responses."""

import numpy as np
import matplotlib.pyplot as plt


def simulate_one_trial(
    v_A,
    v_B,
    b,
    t0,
    noise_sd=1.0,
    dt=0.001,
    max_time=5.0,
    rng=None,
):
    """Simulate one trial of a two-accumulator race."""
    if rng is None:
        rng = np.random.default_rng()

    # Starting evidence is fixed at zero in this simplified model.
    x_A = 0.0
    x_B = 0.0
    t = 0.0
    sqrt_dt = np.sqrt(dt)

    while t < max_time:
        x_A += v_A * dt + noise_sd * sqrt_dt * rng.normal()
        x_B += v_B * dt + noise_sd * sqrt_dt * rng.normal()
        t += dt

        if x_A >= b and x_B >= b:
            choice = "A" if rng.random() < 0.5 else "B"
            return choice, t + t0

        if x_A >= b:
            return "A", t + t0

        if x_B >= b:
            return "B", t + t0

    return None, np.nan


def simulate_many_trials(
    n_trials,
    v_A,
    v_B,
    b,
    t0,
    noise_sd=1.0,
    dt=0.001,
    max_time=5.0,
    seed=42,
):
    """Run several independent trials."""
    rng = np.random.default_rng(seed)
    choices = np.full(n_trials, "", dtype="<U1")
    rts = np.full(n_trials, np.nan)

    for trial in range(n_trials):
        choice, rt = simulate_one_trial(
            v_A=v_A,
            v_B=v_B,
            b=b,
            t0=t0,
            noise_sd=noise_sd,
            dt=dt,
            max_time=max_time,
            rng=rng,
        )

        if choice is not None:
            choices[trial] = choice
            rts[trial] = rt

    return choices, rts


if __name__ == "__main__":
    v_A = 1.2
    v_B = 0.8
    b = 1.0
    t0 = 0.25
    noise_sd = 1.0
    n_trials = 2_000

    print(
        f"Simulating {n_trials} trials with "
        f"v_A={v_A}, v_B={v_B}, b={b}, t0={t0} ..."
    )

    choices, rts = simulate_many_trials(
        n_trials=n_trials,
        v_A=v_A,
        v_B=v_B,
        b=b,
        t0=t0,
        noise_sd=noise_sd,
    )

    valid = np.isfinite(rts)
    n_timeouts = np.sum(~valid)

    choices = choices[valid]
    rts = rts[valid]

    if n_timeouts:
        print(f"Trials excluded after reaching max_time: {n_timeouts}")

    prop_A = np.mean(choices == "A")

    print(f"\nProportion choosing A: {prop_A:.3f}")
    print(f"Mean RT overall: {rts.mean():.3f} s")
    print(f"Mean RT for A choices: {rts[choices == 'A'].mean():.3f} s")
    print(f"Mean RT for B choices: {rts[choices == 'B'].mean():.3f} s")

    # Use common bins for both response distributions.
    bins = np.histogram_bin_edges(rts, bins=40)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        rts[choices == "A"],
        bins=bins,
        density=True,
        alpha=0.6,
        color="#3b6fa0",
        label="Choice A",
    )
    ax.hist(
        rts[choices == "B"],
        bins=bins,
        density=True,
        alpha=0.6,
        color="#c0553b",
        label="Choice B",
    )

    ax.set_xlabel("Response time (s)")
    ax.set_ylabel("Density")
    ax.set_title("Conditional RT distributions from the basic race model")
    ax.legend()

    fig.tight_layout()
    fig.savefig("basic_rdm_simulation.png", dpi=150)
    plt.close(fig)

    print("\nSaved plot to basic_rdm_simulation.png")