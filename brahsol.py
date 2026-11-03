"""BrahSol: an active-ping experiment with a hidden, adapting, plastic medium.

Engineering toy, not a biophysical neuron. Requires NumPy.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

N = 24
ACTIONS = 8
TARGET_ENERGY = 0.016


class NeuralMedium:
    """Hidden recurrent units with adaptation, optional Hebbian plasticity."""
    def __init__(self, seed: int, plasticity: bool = True, sensor_mode: str = "four"):
        if sensor_mode not in ("four", "sum"):
            raise ValueError(sensor_mode)
        self.rng = np.random.default_rng(seed)
        self.plasticity = plasticity
        self.sensor_mode = sensor_mode
        self.h = np.zeros(N)
        self.adapt = np.zeros(N)
        self.theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        distance = np.abs(self.theta[:, None] - self.theta[None, :])
        distance = np.minimum(distance, 2*np.pi-distance)
        self.mask = ((distance < 1.4) |
                     (self.rng.random((N, N)) < 0.08)).astype(float)
        self.mask[np.diag_indices(N)] = 0
        raw = self.rng.normal(0, 1, (N, N))*self.mask
        self.W = raw/max(np.linalg.norm(raw, ord=2), 1e-9)*0.90
        self.W_initial = self.W.copy()
        self.bias = self.rng.normal(0, 0.035, N)
        self.input = np.array([
            0.95*np.exp(-0.5*(np.minimum(np.abs(self.theta-angle),
                 2*np.pi-np.abs(self.theta-angle))/0.45)**2)
            - 0.28*np.exp(-0.5*(np.minimum(np.abs(self.theta-(angle+np.pi)),
                 2*np.pi-np.abs(self.theta-(angle+np.pi)))/0.8)**2)
            for angle in np.linspace(0, 2*np.pi, ACTIONS, endpoint=False)
        ])
        centers = np.linspace(0, 2*np.pi, 4, endpoint=False)
        self.readout = np.array([
            np.exp(-0.5*(np.minimum(np.abs(self.theta-c),
                    2*np.pi-np.abs(self.theta-c))/0.65)**2)
            for c in centers
        ])
        self.readout /= self.readout.sum(axis=1, keepdims=True)
        self.last_y = np.zeros(self.channels)
        self.step_count = 0

    @property
    def channels(self):
        return 4 if self.sensor_mode == "four" else 1

    def step(self, action: int) -> np.ndarray:
        if not 0 <= action < ACTIONS:
            raise ValueError(action)
        old = self.h.copy()
        drive = self.W@old + self.input[action] + self.bias - 0.55*self.adapt
        self.h = 0.60*old + 0.40*np.tanh(drive)
        self.adapt = 0.965*self.adapt + 0.035*np.maximum(self.h, 0)
        if self.plasticity:
            dw = 0.0008*(np.outer(self.h, old) - 0.12*self.W)
            self.W = np.clip((self.W+dw)*self.mask, -0.32, 0.32)
        response = self.readout@self.h
        response += self.rng.normal(0, 0.012, 4)
        if self.sensor_mode == "sum":
            response = np.array([response.sum()])
        self.last_y = response
        self.step_count += 1
        return response.copy()

    def clone(self):
        return copy.deepcopy(self)

    def drift(self):
        return float(np.linalg.norm(self.W-self.W_initial) /
                     max(np.linalg.norm(self.W_initial), 1e-12))


class ReturnModel:
    """Independent regularized recursive least-squares model for each port.

    x'P x is an uncertainty surrogate, not calibrated Bayesian information gain.
    """
    def __init__(self, channels: int):
        self.channels = channels
        self.d = channels+1
        self.coef = np.zeros((ACTIONS, self.d, channels))
        self.P = np.tile(np.eye(self.d)[None, :, :]*12., (ACTIONS, 1, 1))
        self.counts = np.zeros(ACTIONS, dtype=int)

    def features(self, y):
        return np.r_[1.0, np.asarray(y, dtype=float)]

    def predict(self, y, action):
        x = self.features(y)
        mu = x@self.coef[action]
        v = max(0.0, float(x@self.P[action]@x))
        return mu, v

    def update(self, y_before, action, y_after):
        x = self.features(y_before)
        P = self.P[action]
        px = P@x
        gain = px/(1.0+float(x@px))
        error = y_after-x@self.coef[action]
        self.coef[action] += np.outer(gain, error)
        nextP = P-np.outer(gain, px)
        self.P[action] = 0.5*(nextP+nextP.T)
        self.counts[action] += 1


class Controller:
    def __init__(self, policy: str, channels: int, seed: int, beta=0.25):
        if policy not in ("random", "balanced", "sweep", "homeo", "active"):
            raise ValueError(policy)
        self.policy = policy
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        self.model = ReturnModel(channels)
        self.block = np.arange(ACTIONS)

    def choose(self, y, t: int) -> int:
        if self.policy == "random":
            return int(self.rng.integers(ACTIONS))
        if self.policy == "sweep":
            return t % ACTIONS
        if self.policy == "balanced" or t < ACTIONS*3:
            if t % ACTIONS == 0:
                self.block = self.rng.permutation(ACTIONS)
            return int(self.block[t % ACTIONS])
        scores = []
        for a in range(ACTIONS):
            mu, uncert = self.model.predict(y, a)
            energy = float(np.mean(np.square(mu)))
            homeo_cost = (energy-TARGET_ENERGY)**2
            curiosity = math.log1p(uncert)
            scores.append(-50.0*homeo_cost +
                          (self.beta*curiosity if self.policy == "active" else 0.0))
        return int(np.argmax(scores))


@dataclass
class Result:
    seed: int
    policy: str
    plasticity: bool
    sensor_mode: str
    train_steps: int
    test_steps: int
    test_rmse: float
    persistence_rmse: float
    relative_skill: float
    train_homeo_mse: float
    action_entropy: float
    action_coverage: int
    synaptic_drift: float


def run(seed=0, policy="active", plasticity=True, sensor_mode="four",
        train_steps=750, test_steps=240, beta=0.25, shuffled_returns=False):
    """Train by interaction, test frozen predictor on new random action sequence.

    Each policy is evaluated on its *own* end-of-training world clone. Since
    hidden states can differ, direct between-policy comparisons are limited.
    """
    if train_steps < 1 or test_steps < 1:
        raise ValueError("step counts must be positive")
    world = NeuralMedium(seed, plasticity, sensor_mode)
    agent = Controller(policy, world.channels, seed+1717, beta)
    last_y = world.last_y.copy()
    energies = []
    chosen = []
    buffer = deque(maxlen=400)
    shuffle_rng = np.random.default_rng(seed+3223)
    for t in range(train_steps):
        a = agent.choose(last_y, t)
        y = world.step(a)
        train_target = (buffer[int(shuffle_rng.integers(len(buffer)))].copy()
                        if shuffled_returns and len(buffer) > 20 else y)
        agent.model.update(last_y, a, train_target)
        buffer.append(y.copy())
        energies.append((float(np.mean(y**2))-TARGET_ENERGY)**2)
        chosen.append(a)
        last_y = y
    test_world = world.clone()
    test_rng = np.random.default_rng(seed+9999)
    y_before = last_y.copy()
    sqerr = []
    persist = []
    for _ in range(test_steps):
        a = int(test_rng.integers(ACTIONS))
        prediction, _ = agent.model.predict(y_before, a)
        y_after = test_world.step(a)
        sqerr.extend((prediction-y_after)**2)
        persist.extend((y_before-y_after)**2)
        y_before = y_after
    rmse = float(np.sqrt(np.mean(sqerr)))
    baseline = float(np.sqrt(np.mean(persist)))
    counts = np.bincount(chosen, minlength=ACTIONS)
    p = counts[counts > 0]/len(chosen)
    entropy = float(-np.sum(p*np.log2(p))/math.log2(ACTIONS))
    return Result(seed, policy, plasticity, sensor_mode, train_steps, test_steps,
                  rmse, baseline, 1.-rmse/max(baseline, 1e-12),
                  float(np.mean(energies)), entropy, int(np.count_nonzero(counts)),
                  world.drift())


def benchmark(seeds, train_steps=750, test_steps=240, plasticity="both",
              sensor_mode="both", beta=0.25, shuffled_returns=False):
    results = []
    modes = [False, True] if plasticity == "both" else [plasticity == "on"]
    sensors = ["four", "sum"] if sensor_mode == "both" else [sensor_mode]
    for pl in modes:
        for sm in sensors:
            for seed in seeds:
                for policy in ("random", "balanced", "sweep", "homeo", "active"):
                    results.append(run(seed, policy, pl, sm, train_steps, test_steps,
                                       beta, shuffled_returns))
    return results


def summarize(results):
    groups = {}
    for r in results:
        groups.setdefault((r.plasticity, r.sensor_mode, r.policy), []).append(r)
    lines = []
    for (pl, sm, pol), rs in sorted(groups.items()):
        vals = lambda key: np.array([getattr(r, key) for r in rs])
        lines.append({"plastic": pl, "sensors": sm, "policy": pol,
                      "n": len(rs),
                      "rmse": round(float(vals("test_rmse").mean()), 5),
                      "skill": round(float(vals("relative_skill").mean()), 4),
                      "homeo_mse": round(float(vals("train_homeo_mse").mean()), 6),
                      "action_entropy": round(float(vals("action_entropy").mean()), 3),
                      "drift": round(float(vals("synaptic_drift").mean()), 5)})
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train", type=int, default=750)
    parser.add_argument("--test", type=int, default=240)
    parser.add_argument("--plasticity", choices=["both", "on", "off"], default="both")
    parser.add_argument("--sensors", choices=["both", "four", "sum"], default="both")
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--shuffled-returns", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    rows = benchmark(range(args.seeds), args.train, args.test,
                     args.plasticity, args.sensors, args.beta, args.shuffled_returns)
    args.out.mkdir(exist_ok=True, parents=True)
    with (args.out/"runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])
    (args.out/"summary.json").write_text(json.dumps(summarize(rows), indent=2),
                                          encoding="utf-8")
    print("plastic sensors  policy   n    RMSE    skill    homeo_MSE entropy drift")
    for s in summarize(rows):
        print(f"{str(s['plastic']):7} {s['sensors']:7} {s['policy']:8} {s['n']:2d} "
              f"{s['rmse']:7.4f} {s['skill']:7.3f} {s['homeo_mse']:10.6f} "
              f"{s['action_entropy']:7.3f} {s['drift']:7.4f}")
    print(f"Wrote {args.out/'runs.csv'} and {args.out/'summary.json'}")


if __name__ == "__main__":
    main()
