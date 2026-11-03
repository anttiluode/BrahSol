"""BrahSol: causal write-gap-read route-memory pilot (NumPy only).

This is a deliberately engineered identifiability test, NOT learned probing
or evidence of emergent planning. Run: python route_probe.py --seeds 40
"""
import argparse
import numpy as np

def simulate(seed, mode="balanced", reset=False, delay=12, trials=500):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, trials)
    gains = np.array([[0.85, 0.15], [0.15, 0.85]])
    responses = np.empty(trials)
    ports = np.empty(trials, dtype=int)
    for i, label in enumerate(labels):
        port = i % 2 if mode == "balanced" else int(rng.integers(0, 2))
        ports[i] = port
        memory = 0.0 if mode == "none" else gains[label, port]
        for _ in range(delay):
            memory *= 0.94
        if reset:
            memory = 0.0
        responses[i] = 1 + 0.9*np.tanh(2*memory) + rng.normal(0, 0.12)
    cut = trials // 2
    means = np.array([
        [responses[:cut][(labels[:cut] == c) & (ports[:cut] == p)].mean()
         for p in range(2)] for c in range(2)])
    pred = np.array([np.argmin(np.abs(means[:, ports[i]] - responses[i]))
                     for i in range(cut, trials)])
    return float(np.mean(pred == labels[cut:])), float(np.mean(np.abs(means[0]-means[1])))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--delay", type=int, default=12)
    args = parser.parse_args()
    print("Synthetic write-gap-read pilot; held-out accuracy (NOT learned probing)")
    for mode, reset in [("balanced", False), ("random", False),
                        ("none", False), ("balanced", True)]:
        results = np.array([simulate(s, mode, reset, args.delay)
                            for s in range(args.seeds)])
        print(f"{mode:9s} reset={str(reset):5s} "
              f"accuracy={results[:,0].mean():.3f} +/- {results[:,0].std():.3f} "
              f"separation={results[:,1].mean():.3f}")

if __name__ == "__main__":
    main()
