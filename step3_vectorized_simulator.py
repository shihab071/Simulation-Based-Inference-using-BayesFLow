"""Vectorized LI simulator used for BayesFlow training."""

import numpy as np
import time


def simulate_trials_vectorized(v_A, v_B, beta, b, t0, n_trials,
                                noise_sd=1.0, dt=0.002, dr_window=0.25,
                                max_time=2.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    x_A = np.zeros(n_trials)
    x_B = np.zeros(n_trials)

    decided = np.zeros(n_trials, dtype=bool)
    choice = np.full(n_trials, -1, dtype=int)
    decision_step = np.full(n_trials, -1, dtype=int)

    double_response = np.zeros(n_trials, dtype=bool)
    drt = np.full(n_trials, np.nan)

    sqrt_dt = np.sqrt(dt)
    n_decision_steps = int(round(max_time / dt))
    n_dr_steps = int(round(dr_window / dt))
    total_steps = n_decision_steps + n_dr_steps

    for current_step in range(1, total_steps + 1):
        noise_A = rng.normal(size=n_trials)
        noise_B = rng.normal(size=n_trials)
        # Use the previous state for both accumulator updates.
        old_A = x_A
        old_B = x_B
        x_A = old_A + (v_A - beta * old_B) * dt + noise_sd * sqrt_dt * noise_A
        x_B = old_B + (v_B - beta * old_A) * dt + noise_sd * sqrt_dt * noise_B

        can_decide = (~decided) & (current_step <= n_decision_steps)
        newly_either = can_decide & ((x_A >= b) | (x_B >= b))

        tie = newly_either & (x_A >= b) & (x_B >= b)
        tie_to_A = np.zeros(n_trials, dtype=bool)
        tie_indices = np.flatnonzero(tie)
        if tie_indices.size:
            tie_to_A[tie_indices] = rng.random(tie_indices.size) < 0.5
        newly_A = newly_either & (((x_A >= b) & ~tie) | tie_to_A)
        newly_B = newly_either & ~newly_A

        choice[newly_A] = 0
        choice[newly_B] = 1
        decision_step[newly_A | newly_B] = current_step
        decided[newly_A | newly_B] = True

        steps_since_decision = np.where(decided, current_step - decision_step, -1)
        losing_evidence = np.where(choice == 0, x_B, x_A)
        crossed_loser = (decided & (~double_response) &
                         (steps_since_decision <= n_dr_steps) &
                         (losing_evidence >= b))
        double_response[crossed_loser] = True
        drt[crossed_loser] = steps_since_decision[crossed_loser] * dt

        can_decide_later = (~decided) & (current_step < n_decision_steps)
        open_dr_window = decided & (~double_response) & (steps_since_decision < n_dr_steps)
        if not (can_decide_later | open_dr_window).any():
            break

    # Keep timeouts as missing values in the raw simulator output.
    # The filtered wrapper removes and replaces them.
    rt = np.where(decided, decision_step * dt + t0, np.nan)

    choice_letter = np.full(n_trials, "", dtype="<U1")
    choice_letter[choice == 0] = "A"
    choice_letter[choice == 1] = "B"

    return {
        "choice": choice_letter,
        "rt": rt,
        "double_response": double_response,
        "drt": drt,
    }


RT_EXCLUSION_MIN = 0.25


def simulate_trials_filtered(v_A, v_B, beta, b, t0, n_trials, noise_sd=1.0,
                              dt=0.002, dr_window=0.25, max_time=2.5,
                              rt_min=RT_EXCLUSION_MIN, rng=None, max_tries=15):
    if rng is None:
        rng = np.random.default_rng()

    collected = {"choice": [], "rt": [], "double_response": [], "drt": []}
    n_have = 0
    tries = 0
    survival_rate = 0.5
    while n_have < n_trials and tries < max_tries:
        n_needed = n_trials - n_have
        batch_n = int(np.ceil(n_needed / max(survival_rate, 1e-4) * 1.15))
        batch_n = min(max(batch_n, 20), 2_000_000)
        sim = simulate_trials_vectorized(v_A, v_B, beta, b, t0, batch_n,
                                          noise_sd=noise_sd, dt=dt,
                                          dr_window=dr_window,
                                          max_time=max_time, rng=rng)
        keep = sim["rt"] >= rt_min
        n_kept_this_round = int(keep.sum())
        survival_rate = max(n_kept_this_round / batch_n, 1e-4)
        for k in collected:
            collected[k].append(sim[k][keep])
        n_have = sum(len(x) for x in collected["rt"])
        tries += 1

    if n_have < n_trials:
        raise RuntimeError(
            f"Couldn't reach {n_trials} valid (rt >= {rt_min}s) trials "
            f"after {max_tries} attempts (only got {n_have})."
        )

    return {k: np.concatenate(v)[:n_trials] for k, v in collected.items()}


if __name__ == "__main__":
    v_A, v_B, beta, b, t0 = 1.2, 0.9, 0.8, 1.0, 0.25
    comparison_trials = 50_000

    print("Timing the vectorized simulator...")
    start = time.perf_counter()
    result = simulate_trials_vectorized(v_A, v_B, beta, b, t0, comparison_trials,
                                         rng=np.random.default_rng(0))
    elapsed = time.perf_counter() - start

    valid_fast = np.isfinite(result["rt"])
    fast_choices = result["choice"][valid_fast]
    fast_rts = result["rt"][valid_fast]
    fast_double = result["double_response"][valid_fast]
    n_timeouts_fast = comparison_trials - valid_fast.sum()

    print(f"Simulated {comparison_trials} trials in {elapsed:.3f}s "
          f"({comparison_trials/elapsed:.0f} trials/sec)")
    print(f"Timeouts: {n_timeouts_fast}")
    print(f"\nProportion choosing A: {np.mean(fast_choices == 'A'):.3f}")
    print(f"Mean RT: {fast_rts.mean():.3f} s")
    print(f"Overall P(double response): {fast_double.mean():.4f}")
    print(f"P(double | choice=A): {fast_double[fast_choices == 'A'].mean():.4f}")
    print(f"P(double | choice=B): {fast_double[fast_choices == 'B'].mean():.4f}")

    print("\n--- Comparing against the slow trial-by-trial version ---")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "step2", "step2_lateral_inhibition_double_response.py")
    step2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step2)

    start = time.perf_counter()
    slow_results = step2.simulate_many_trials(comparison_trials, v_A, v_B,
                                               beta, b, t0, seed=42)
    slow_elapsed = time.perf_counter() - start

    valid_slow_results = [r for r in slow_results if r["choice"] is not None]
    n_timeouts_slow = comparison_trials - len(valid_slow_results)
    slow_choices = np.array([r["choice"] for r in valid_slow_results])
    slow_rts = np.array([r["rt"] for r in valid_slow_results])
    slow_double = np.array([r["double_response"] for r in valid_slow_results])

    print(f"Slow version: {comparison_trials} trials in {slow_elapsed:.3f}s "
          f"({comparison_trials/slow_elapsed:.0f} trials/sec)")
    print(f"Timeouts: {n_timeouts_slow}")
    print(f"Slow version -- P(A)={np.mean(slow_choices=='A'):.3f}, "
          f"MeanRT={slow_rts.mean():.3f}, P(double)={slow_double.mean():.4f}")
    print(f"Fast version -- P(A)={np.mean(fast_choices=='A'):.3f}, "
          f"MeanRT={fast_rts.mean():.3f}, P(double)={fast_double.mean():.4f}")

    p_a_diff = abs(np.mean(slow_choices == "A") - np.mean(fast_choices == "A"))
    rt_diff = abs(slow_rts.mean() - fast_rts.mean())
    dr_diff = abs(slow_double.mean() - fast_double.mean())
    print(f"\nDifferences -- P(A): {p_a_diff:.4f}, Mean RT: {rt_diff:.4f}s, "
          f"P(DR): {dr_diff:.4f}")
    print(f"Speedup: {(slow_elapsed/comparison_trials) / (elapsed/comparison_trials):.0f}x faster")

    print("\n--- Testing the filtered wrapper ---")
    filtered = simulate_trials_filtered(
        v_A=v_A, v_B=v_B, beta=beta, b=b, t0=0.15, n_trials=5_000,
        rng=np.random.default_rng(123),
    )
    assert len(filtered["rt"]) == 5_000
    assert np.all(np.isfinite(filtered["rt"]))
    assert np.all(filtered["rt"] >= RT_EXCLUSION_MIN)
    has_double = filtered["double_response"]
    assert np.all(np.isnan(filtered["drt"][~has_double]))
    assert np.all(filtered["drt"][has_double] >= 0.0)
    assert np.all(filtered["drt"][has_double] <= 0.25)
    print("Filtered simulator checks passed.")