# BrahSol v2 — changing-medium receipts (2026-11-03)

**Question:** can a controller learn the consequences of its own pings when its hidden neural medium changes, and do informative pings help it relearn faster?

## Reproducible experimental setup

- Hidden medium: 24 leaky recurrent units, activity adaptation, **optional Hebbian plasticity** (`eta=0.016`) and eight stimulation ports.
- Controller sees four spatially averaged return values, **not** hidden neurons or synaptic weights.
- Synthetic change at **tick 170**: reroute the eight afferent stimulus projections (ports 0–3 exchange projections with 4–7) **and** replace incoming recurrent synapses for one-third of hidden units. This is an engineered intervention, **not biological remodeling**.
- Same budget: **440 training pings**, **12 matched seeds**, fixed test every 12 ticks and immediately at the change point.
- Five stimulus strategies: uniform random, shuffled-balanced, greedy homeostatic, uncertainty-only, and combined active (homeostatic + uncertainty).
- Every policy uses an action-conditioned online linear return predictor; `gamma=0.985` discounts old samples, `gamma=1.0` does not.
- **Common-world control:** at each evaluation tick the exact same independently driven reference medium is cloned eight times, one for each candidate action. Predictions are frozen during evaluation. This controls the test state and action sequence across policies, but not the different states seen while each policy learns.

**Metric:** mean across seeds of held-out **one-step counterfactual RMSE**, lower is better. **Pre** = ticks 90–169; **early** = 170–209; **late** = 250–439. Each seed contributes equally. Noise is 0.012 on each of four sensors.

## Main result: plastic medium, afferent + recurrent shock enabled

| Strategy | Forgetting `gamma` | Pre RMSE | Early post-change | Late post-change |
|---|---:|---:|---:|---:|
| **Active** | **0.985** | 0.0416 | 0.1590 | **0.0483** |
| Random | 0.985 | 0.0458 | 0.1596 | 0.0561 |
| Balanced | 0.985 | 0.0529 | 0.1557 | 0.0693 |
| Uncertainty-only | 0.985 | 0.0429 | 0.1588 | 0.0523 |
| Homeostatic-only | 0.985 | 0.0778 | 0.1544 | 0.1119 |
| Active | 1.000 | **0.0307** | 0.1722 | 0.1008 |
| Random | 1.000 | 0.0358 | 0.1706 | 0.1004 |

At late recovery, active + forgetting is **14% lower RMSE than random + forgetting** (0.0483 vs 0.0561), winning on **11/12 paired seeds**. Immediately after the change, the active advantage is tiny (7/12 wins). Active with no forgetting performs no better than random with no forgetting after the change. This is not evidence of an intrinsically intelligent or biological controller: the active score is a designed leverage/coverage heuristic.

**The much larger effect is updating old expectations:** for the active strategy, forgetting reduces late post-shock RMSE from 0.1008 to 0.0483, though it slightly worsens accuracy before the shock.

## Necessary negative controls

### Plasticity switched **off**, shock still on

| Strategy | `gamma` | Pre | Early | Late |
|---|---:|---:|---:|---:|
| Active | 0.985 | 0.0372 | 0.1576 | **0.0418** |
| Random | 0.985 | 0.0415 | 0.1584 | 0.0484 |
| Active | 1.000 | **0.0279** | 0.1718 | 0.0977 |

This looks very similar to the plastic case. **The improvement does not require ongoing Hebbian synaptic learning.** It chiefly reflects the deliberately changed meaning of the stimulation ports, plus updating a return predictor. Rewiring itself still occurs as the engineered shock.

### Shock **off**, plasticity still on

| Strategy | `gamma` | Pre | Early-window (no shock) | Late |
|---|---:|---:|---:|---:|
| Active | 0.985 | 0.0416 | 0.0398 | 0.0422 |
| Active | 1.000 | **0.0307** | 0.0267 | **0.0248** |
| Random | 0.985 | 0.0458 | 0.0453 | 0.0507 |
| Random | 1.000 | 0.0358 | 0.0316 | 0.0274 |

When there is no sudden wiring change, keeping old evidence is **better**. Forgetting is a tradeoff, not an unconditional improvement. The corresponding frozen-synapse, no-shock experiment also ran: active late RMSE was 0.0352 with forgetting vs 0.0198 without.

## What changed since v1?

In v1, intrinsic synaptic plasticity was too weak to account for the active-versus-random result. V2 increases plasticity, measures weight drift, and introduces a reproducible external wiring intervention. The latter *must not* be confused with synapses spontaneously learning: it is an imposed change. It produces a clear error spike, allowing recovery to be measured.

The first attempt rewired **recurrent synapses only**. It changed internal weights strongly but hardly changed observable returns, so it was a weak challenge to a sensor-level predictor. V2 therefore deliberately reroutes **afferent synapses too**, so the same ping produces a different response.

Additional controls still needed: recurrent-only vs afferent-only intervention, continuous unannounced drift with no fixed intervention, out-of-distribution reference networks, matched action coverage, local perturbations, multiple observation budgets, and task-generalization beyond one-step linear prediction. A real changepoint-detection controller is **not** implemented: exponential forgetting continuously revises the predictor. Calling it active *change detection* would be incorrect.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python brahsol_v2.py --seeds 12 --steps 440 --change 170 --eval-every 12 --plasticity both --shock both --out results_v2
```

Produced files: `results_v2/runs.csv` (per seed and condition), `results_v2/eval_trace.csv` (time courses), `results_v2/summary.json` (summary means and standard deviations).

The experimental program is `brahsol_v2.py`. It uses v1's `NeuralMedium` initialization but overrides the transition dynamics, and requires only NumPy. Results are measurements from this deterministic-seeded synthetic environment—not evidence for how apical tufts identify their upstream neuronal partners.
