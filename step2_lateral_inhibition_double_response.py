"""Slow LI simulator used as a reference for the vectorized version."""

import numpy as np
import matplotlib.pyplot as plt


def simulate_one_trial(v_A, v_B, beta, b, t0, noise_sd=1.0, dt=0.002,
                        dr_window=0.25, max_time=2.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    x_A, x_B = 0.0, 0.0
    sqrt_dt = np.sqrt(dt)
    initial_choice = None
    initial_step = None
    n_steps = int(max_time / dt)

    for step in range(1, n_steps + 1):
        # Use the previous state for both accumulator updates.
        old_A, old_B = x_A, x_B
        x_A = old_A + (v_A - beta * old_B) * dt + noise_sd * sqrt_dt * rng.normal()
        x_B = old_B + (v_B - beta * old_A) * dt + noise_sd * sqrt_dt * rng.normal()

        crossed_A = x_A >= b
        crossed_B = x_B >= b

        if crossed_A and crossed_B:
            # A same-step crossing is recorded with DRT = 0.
            initial_choice = "A" if rng.random() < 0.5 else "B"
            return {"choice": initial_choice, "rt": step * dt + t0,
                    "double_response": True, "drt": 0.0}
        if crossed_A:
            initial_choice = "A"
            initial_step = step
            break
        if crossed_B:
            initial_choice = "B"
            initial_step = step
            break

    if initial_choice is None:
        return {"choice": None, "rt": np.nan, "double_response": False,
                "drt": None}

    rt = initial_step * dt + t0
    losing = "B" if initial_choice == "A" else "A"

    # Continue accumulation for 250 ms after the first response.
    n_extra_steps = int(dr_window / dt)
    for extra_step in range(1, n_extra_steps + 1):
        old_A, old_B = x_A, x_B
        x_A = old_A + (v_A - beta * old_B) * dt + noise_sd * sqrt_dt * rng.normal()
        x_B = old_B + (v_B - beta * old_A) * dt + noise_sd * sqrt_dt * rng.normal()

        losing_evidence = x_A if losing == "A" else x_B
        if losing_evidence >= b:
            return {"choice": initial_choice, "rt": rt,
                    "double_response": True, "drt": extra_step * dt}

    return {"choice": initial_choice, "rt": rt, "double_response": False,
            "drt": None}


def simulate_many_trials(n_trials, v_A, v_B, beta, b, t0, noise_sd=1.0,
                          dt=0.002, dr_window=0.25, max_time=2.5, seed=42):
    rng = np.random.default_rng(seed)
    return [simulate_one_trial(v_A, v_B, beta, b, t0, noise_sd, dt,
                                dr_window, max_time, rng=rng)
            for _ in range(n_trials)]


if __name__ == "__main__":
    v_A = 1.2
    v_B = 0.9
    beta = 0.8
    b = 1.0
    t0 = 0.25
    noise_sd = 1.0
    n_trials = 5000

    print(f"Simulating {n_trials} trials: v_A={v_A}, v_B={v_B}, beta={beta}, "
          f"b={b}, t0={t0} ...")
    results = simulate_many_trials(n_trials, v_A, v_B, beta, b, t0, noise_sd)

    results = [r for r in results if r["choice"] is not None]
    n_timeouts = n_trials - len(results)
    if n_timeouts:
        print(f"Timeouts: {n_timeouts}")

    choices = np.array([r["choice"] for r in results])
    rts = np.array([r["rt"] for r in results])
    dbl = np.array([r["double_response"] for r in results])

    prop_A = np.mean(choices == "A")
    prop_double = np.mean(dbl)
    prop_double_given_A = np.mean(dbl[choices == "A"])
    prop_double_given_B = np.mean(dbl[choices == "B"])

    print(f"\nProportion choosing A: {prop_A:.3f}")
    print(f"Mean RT: {rts.mean():.3f} s")
    print(f"Overall proportion of double responses: {prop_double:.4f}")
    print(f"  P(double response | initial choice = A): {prop_double_given_A:.4f}")
    print(f"  P(double response | initial choice = B): {prop_double_given_B:.4f}")

    drts = np.array([r["drt"] for r in results if r["double_response"]])
    if len(drts) > 5:
        assert drts.min() >= 0.0 and drts.max() <= 0.25
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(drts, bins=25, color="#6a4c93", alpha=0.8)
        ax.set_xlabel("Double response time (s, since initial response)")
        ax.set_ylabel("Count")
        ax.set_title(f"Simulated double-response-time distribution "
                      f"(n={len(drts)} double responses)")
        fig.tight_layout()
        fig.savefig("double_response_simulation.png", dpi=150)
        print(f"\nSaved plot to double_response_simulation.png "
              f"({len(drts)} double responses out of {n_trials} trials)")
    else:
        print("\nToo few double responses to plot.")