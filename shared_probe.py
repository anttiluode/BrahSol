"""Common-world evaluation: all fitted predictors face IDENTICAL held-out states.

Fits each policy separately, then tests predictions on a reference neural
medium driven by the SAME random stimulus history for every policy. This
controls evaluation state, but training distributions still differ.
"""
import argparse
import numpy as np
from brahsol import ACTIONS, Controller, NeuralMedium

POLICIES = ("random", "balanced", "sweep", "homeo", "active")


def shared_test(seed, policy, train_steps=100, test_steps=120, plasticity=True):
    world = NeuralMedium(seed, plasticity, "four")
    agent = Controller(policy, 4, seed+1717)
    y = world.last_y.copy()
    for t in range(train_steps):
        action = agent.choose(y, t)
        y_next = world.step(action)
        agent.model.update(y, action, y_next)
        y = y_next

    # Identical held-out environment for ALL policies on a given seed.
    ref = NeuralMedium(seed, plasticity, "four")
    prior_rng = np.random.default_rng(seed+33333)
    for _ in range(train_steps):
        ref.step(int(prior_rng.integers(ACTIONS)))
    heldout_y = ref.last_y.copy()
    action_rng = np.random.default_rng(seed+9999)
    errors = []
    persists = []
    for _ in range(test_steps):
        action = int(action_rng.integers(ACTIONS))
        prediction, _ = agent.model.predict(heldout_y, action)
        actual = ref.step(action)
        errors.extend((prediction-actual)**2)
        persists.extend((heldout_y-actual)**2)
        heldout_y = actual
    return float(np.sqrt(np.mean(errors))), float(np.sqrt(np.mean(persists)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--train", type=int, default=100)
    p.add_argument("--test", type=int, default=120)
    p.add_argument("--no-plasticity", action="store_true")
    args = p.parse_args()
    if args.seeds < 1:
        p.error("--seeds must be >= 1")
    outcomes = {policy: [] for policy in POLICIES}
    baselines = []
    for seed in range(args.seeds):
        for policy in POLICIES:
            score, baseline = shared_test(seed, policy, args.train,
                                          args.test, not args.no_plasticity)
            outcomes[policy].append(score)
            if policy == POLICIES[0]:
                baselines.append(baseline)
    print("Common-world held-out returns (same test environment per seed)")
    for policy in POLICIES:
        scores = np.asarray(outcomes[policy])
        print(f"{policy:8s} RMSE={scores.mean():.5f}  sd={scores.std(ddof=1):.5f}")
    print(f"persistence RMSE={np.mean(baselines):.5f}")
    for other in ("random", "balanced"):
        diff = np.array(outcomes[other])-np.array(outcomes["active"])
        print(f"active better than {other}: {int((diff>0).sum())}/{args.seeds} seeds; "
              f"mean paired gain={diff.mean():.5f}")


if __name__ == "__main__":
    main()
